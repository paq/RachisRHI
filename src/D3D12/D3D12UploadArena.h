#pragma once

#include "Rachis/Error.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace rachis::d3d12
{

    inline constexpr std::uint64_t defaultUploadArenaSlabBytes = 64ULL * 1024ULL;
    inline constexpr std::size_t defaultUploadArenaMaxFreeSlabs = 4U;

    enum class UploadArenaAllocationKind : std::uint8_t
    {
        suballocated = 0,
        oversized = 1,
    };

    [[nodiscard]] constexpr std::uint64_t alignUploadArenaOffset(std::uint64_t offset, std::uint64_t alignment) noexcept
    {
        if (alignment <= 1U)
        {
            return offset;
        }
        const std::uint64_t mask = alignment - 1U;
        if (offset > std::numeric_limits<std::uint64_t>::max() - mask)
        {
            return std::numeric_limits<std::uint64_t>::max();
        }
        return (offset + mask) & ~mask;
    }

    template<typename SlabPayload>
    class UploadArena final : public std::enable_shared_from_this<UploadArena<SlabPayload>>
    {
    public:
        struct SlabSlot final
        {
            SlabPayload payload;
            std::uint64_t capacity = 0;
            std::uint64_t cursor = 0;
            std::size_t activeLeases = 0;
            bool current = false;

            SlabSlot(SlabPayload slabPayload,
                     std::uint64_t slabCapacity) noexcept(std::is_nothrow_move_constructible_v<SlabPayload>)
                : payload(std::move(slabPayload)), capacity(slabCapacity)
            {
            }
        };

        class Lease final
        {
        public:
            Lease() noexcept = default;

            Lease(std::shared_ptr<UploadArena> arena,
                  std::shared_ptr<SlabSlot> slot,
                  std::uint64_t offset,
                  std::uint64_t size) noexcept
                : arena_(std::move(arena)), slot_(std::move(slot)), offset_(offset), size_(size)
            {
            }

            ~Lease()
            {
                release();
            }

            Lease(const Lease&) = delete;
            Lease& operator=(const Lease&) = delete;

            Lease(Lease&& other) noexcept
                : arena_(std::move(other.arena_)), slot_(std::move(other.slot_)),
                  offset_(std::exchange(other.offset_, 0)), size_(std::exchange(other.size_, 0))
            {
            }

            Lease& operator=(Lease&& other) noexcept
            {
                if (this != &other)
                {
                    release();
                    arena_ = std::move(other.arena_);
                    slot_ = std::move(other.slot_);
                    offset_ = std::exchange(other.offset_, 0);
                    size_ = std::exchange(other.size_, 0);
                }
                return *this;
            }

            void release() noexcept
            {
                if (slot_ == nullptr)
                {
                    arena_.reset();
                    offset_ = 0;
                    size_ = 0;
                    return;
                }
                auto arena = std::move(arena_);
                auto slot = std::move(slot_);
                offset_ = 0;
                size_ = 0;
                if (arena != nullptr)
                {
                    arena->onLeaseReleased(std::move(slot));
                }
            }

            [[nodiscard]] bool valid() const noexcept
            {
                return slot_ != nullptr;
            }
            [[nodiscard]] const SlabPayload& payload() const noexcept
            {
                return slot_->payload;
            }
            [[nodiscard]] SlabPayload& payload() noexcept
            {
                return slot_->payload;
            }
            [[nodiscard]] std::uint64_t offset() const noexcept
            {
                return offset_;
            }
            [[nodiscard]] std::uint64_t size() const noexcept
            {
                return size_;
            }

        private:
            std::shared_ptr<UploadArena> arena_;
            std::shared_ptr<SlabSlot> slot_;
            std::uint64_t offset_ = 0;
            std::uint64_t size_ = 0;
        };

        struct Allocation final
        {
            UploadArenaAllocationKind kind = UploadArenaAllocationKind::oversized;
            Lease lease{};
        };

        struct Statistics final
        {
            std::size_t freeSlabCount = 0;
            std::size_t currentSlabActiveLeases = 0;
            std::uint64_t currentSlabCursor = 0;
            bool hasCurrentSlab = false;
        };

        [[nodiscard]] static std::shared_ptr<UploadArena> create(
            std::uint64_t slabCapacity = defaultUploadArenaSlabBytes,
            std::size_t maxFreeSlabs = defaultUploadArenaMaxFreeSlabs)
        {
            return std::shared_ptr<UploadArena>(new UploadArena(slabCapacity, maxFreeSlabs));
        }

        template<typename CreateSlabFn>
        [[nodiscard]] Result<Allocation> allocate(std::uint64_t size,
                                                  std::uint64_t alignment,
                                                  CreateSlabFn&& createSlab)
        {
            if (size == 0 || alignment == 0 || !std::has_single_bit(alignment) || alignment > slabCapacity_)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Upload-arena suballocation requires a non-empty size and a power-of-two alignment no larger "
                    "than one slab.",
                });
            }
            if (size > slabCapacity_)
            {
                return Allocation{.kind = UploadArenaAllocationKind::oversized};
            }

            std::shared_ptr<SlabSlot> excessToDestroy;
            std::unique_lock lock(mutex_);
            if (shuttingDown_)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The upload arena has already been shut down.",
                });
            }
            freeSlabs_.reserve(maxFreeSlabs_);

            if (auto fromCurrent = trySuballocateLocked(size, alignment))
            {
                return Allocation{
                    .kind = UploadArenaAllocationKind::suballocated,
                    .lease = std::move(*fromCurrent),
                };
            }

            excessToDestroy = detachCurrentSlabLocked();

            if (!freeSlabs_.empty())
            {
                currentSlab_ = std::move(freeSlabs_.back());
                freeSlabs_.pop_back();
                currentSlab_->current = true;
                currentSlab_->cursor = size;
                currentSlab_->activeLeases = 1;
                auto lease = Lease(this->shared_from_this(), currentSlab_, 0, size);
                lock.unlock();
                excessToDestroy.reset();
                return Allocation{
                    .kind = UploadArenaAllocationKind::suballocated,
                    .lease = std::move(lease),
                };
            }

            lock.unlock();
            excessToDestroy.reset();

            auto createdPayload = createSlab(slabCapacity_);
            if (!createdPayload)
            {
                return std::unexpected(std::move(createdPayload.error()));
            }
            auto newSlot = std::make_shared<SlabSlot>(std::move(*createdPayload), slabCapacity_);

            lock.lock();
            if (shuttingDown_)
            {
                lock.unlock();
                newSlot.reset();
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "The upload arena was shut down during slab creation.",
                });
            }

            if (auto racedCurrent = trySuballocateLocked(size, alignment))
            {
                if (freeSlabs_.size() < maxFreeSlabs_)
                {
                    freeSlabs_.push_back(std::move(newSlot));
                }
                lock.unlock();
                newSlot.reset();
                return Allocation{
                    .kind = UploadArenaAllocationKind::suballocated,
                    .lease = std::move(*racedCurrent),
                };
            }

            auto racedDetached = detachCurrentSlabLocked();
            newSlot->current = true;
            newSlot->cursor = size;
            newSlot->activeLeases = 1;
            currentSlab_ = newSlot;
            auto lease = Lease(this->shared_from_this(), std::move(newSlot), 0, size);
            lock.unlock();
            racedDetached.reset();
            return Allocation{
                .kind = UploadArenaAllocationKind::suballocated,
                .lease = std::move(lease),
            };
        }

        void shutdown() noexcept
        {
            std::shared_ptr<SlabSlot> detachedCurrent;
            std::vector<std::shared_ptr<SlabSlot>> detachedFree;
            {
                const std::scoped_lock lock(mutex_);
                shuttingDown_ = true;
                if (currentSlab_ != nullptr)
                {
                    currentSlab_->current = false;
                    detachedCurrent = std::move(currentSlab_);
                }
                detachedFree = std::move(freeSlabs_);
            }
            detachedCurrent.reset();
            detachedFree.clear();
        }

        [[nodiscard]] std::uint64_t slabCapacity() const noexcept
        {
            return slabCapacity_;
        }

        [[nodiscard]] Statistics statistics() const noexcept
        {
            const std::scoped_lock lock(mutex_);
            return Statistics{
                .freeSlabCount = freeSlabs_.size(),
                .currentSlabActiveLeases = currentSlab_ != nullptr ? currentSlab_->activeLeases : 0U,
                .currentSlabCursor = currentSlab_ != nullptr ? currentSlab_->cursor : 0ULL,
                .hasCurrentSlab = currentSlab_ != nullptr,
            };
        }

    private:
        UploadArena(std::uint64_t slabCapacity, std::size_t maxFreeSlabs)
            : slabCapacity_(slabCapacity), maxFreeSlabs_(maxFreeSlabs)
        {
        }

        [[nodiscard]] std::optional<Lease> trySuballocateLocked(std::uint64_t size, std::uint64_t alignment) noexcept
        {
            if (currentSlab_ == nullptr)
            {
                return std::nullopt;
            }
            const std::uint64_t alignedOffset = alignUploadArenaOffset(currentSlab_->cursor, alignment);
            if (alignedOffset > currentSlab_->capacity || size > currentSlab_->capacity - alignedOffset)
            {
                return std::nullopt;
            }
            currentSlab_->cursor = alignedOffset + size;
            ++currentSlab_->activeLeases;
            return Lease(this->shared_from_this(), currentSlab_, alignedOffset, size);
        }

        [[nodiscard]] std::shared_ptr<SlabSlot> detachCurrentSlabLocked() noexcept
        {
            if (currentSlab_ == nullptr)
            {
                return nullptr;
            }
            auto detached = std::move(currentSlab_);
            detached->current = false;
            if (detached->activeLeases == 0)
            {
                detached->cursor = 0;
                if (!shuttingDown_ && freeSlabs_.size() < maxFreeSlabs_)
                {
                    freeSlabs_.push_back(std::move(detached));
                    return nullptr;
                }
                return detached;
            }
            return nullptr;
        }

        void onLeaseReleased(std::shared_ptr<SlabSlot> slot) noexcept
        {
            std::shared_ptr<SlabSlot> toDestroy;
            {
                const std::scoped_lock lock(mutex_);
                if (slot->activeLeases == 0)
                {
                    std::terminate();
                }
                --slot->activeLeases;
                if (slot->activeLeases == 0)
                {
                    if (slot->current)
                    {
                        slot->cursor = 0;
                    }
                    else
                    {
                        slot->cursor = 0;
                        if (!shuttingDown_ && freeSlabs_.size() < maxFreeSlabs_)
                        {
                            freeSlabs_.push_back(std::move(slot));
                        }
                        else
                        {
                            toDestroy = std::move(slot);
                        }
                    }
                }
            }
            toDestroy.reset();
        }

        std::uint64_t slabCapacity_ = defaultUploadArenaSlabBytes;
        std::size_t maxFreeSlabs_ = defaultUploadArenaMaxFreeSlabs;
        mutable std::mutex mutex_;
        std::shared_ptr<SlabSlot> currentSlab_;
        std::vector<std::shared_ptr<SlabSlot>> freeSlabs_;
        bool shuttingDown_ = false;
    };

} // namespace rachis::d3d12
