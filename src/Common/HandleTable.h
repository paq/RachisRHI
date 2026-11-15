#pragma once

#include "Common/MonotonicIdentitySource.h"
#include "Rachis/Error.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <limits>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace rachis::detail
{

    template<typename HandleT>
    [[nodiscard]] MonotonicIdentitySource& handleOwnerTokenSource() noexcept
    {
        static MonotonicIdentitySource s_source;
        return s_source;
    }

    template<typename T, typename HandleT>
    class HandleTable final
    {
    public:
        HandleTable() noexcept : HandleTable(handleOwnerTokenSource<HandleT>())
        {
        }

        explicit HandleTable(MonotonicIdentitySource& ownerTokens) noexcept
            : ownerToken_(ownerTokenOrNone(ownerTokens.acquire().value_or(noOwnerToken)))
        {
        }

        explicit HandleTable(std::uint32_t ownerToken) noexcept : ownerToken_(ownerTokenOrNone(ownerToken))
        {
        }

        HandleTable(const HandleTable&) = delete;
        HandleTable& operator=(const HandleTable&) = delete;

        HandleTable(HandleTable&& other) noexcept(std::is_nothrow_move_constructible_v<std::deque<Slot>>)
            : ownerToken_(other.ownerToken_), slots_(std::move(other.slots_)),
              freeIndices_(std::move(other.freeIndices_)), liveCount_(other.liveCount_)
        {
            other.abandon();
        }

        HandleTable& operator=(HandleTable&& other) noexcept
        {
            if (this != &other)
            {
                slots_ = std::move(other.slots_);
                freeIndices_ = std::move(other.freeIndices_);
                ownerToken_ = other.ownerToken_;
                liveCount_ = other.liveCount_;
                other.abandon();
            }
            return *this;
        }

        ~HandleTable() = default;

        template<typename... Args>
        [[nodiscard]] Result<HandleT> emplace(Args&&... args)
        {
            if (ownerToken_ == noOwnerToken)
            {
                return std::unexpected(Error{
                    ErrorCode::resourceExhausted,
                    0,
                    "The RHI handle table has no owner token left for new handles.",
                });
            }

            try
            {
                if (!freeIndices_.empty())
                {
                    const auto index = freeIndices_.back();
                    auto& slot = slots_[index];
                    slot.value.emplace(std::forward<Args>(args)...);
                    freeIndices_.pop_back();
                    ++liveCount_;
                    return HandleT{encode(index, slot.generation)};
                }

                if (slots_.size() >= maximumSlotCount)
                {
                    return std::unexpected(Error{
                        ErrorCode::resourceExhausted,
                        0,
                        "The RHI handle table reached its slot capacity.",
                    });
                }

                const auto requiredCapacity = slots_.size() + 1;
                if (freeIndices_.capacity() < requiredCapacity)
                {
                    freeIndices_.reserve(nextFreeListCapacity(freeIndices_.capacity(), requiredCapacity));
                }
                const auto index = static_cast<std::uint32_t>(slots_.size());
                slots_.emplace_back(std::forward<Args>(args)...);
                ++liveCount_;
                return HandleT{encode(index, initialGeneration)};
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Could not allocate an RHI handle-table entry.",
                });
            }
        }

        [[nodiscard]] T* find(HandleT handle) noexcept
        {
            const auto decoded = decode(handle);
            if (!decoded.has_value())
            {
                return nullptr;
            }
            auto& slot = slots_[decoded->index];
            return slot.generation == decoded->generation && slot.value.has_value() ? &*slot.value : nullptr;
        }

        [[nodiscard]] const T* find(HandleT handle) const noexcept
        {
            const auto decoded = decode(handle);
            if (!decoded.has_value())
            {
                return nullptr;
            }
            const auto& slot = slots_[decoded->index];
            return slot.generation == decoded->generation && slot.value.has_value() ? &*slot.value : nullptr;
        }

        bool erase(HandleT handle) noexcept
        {
            const auto decoded = decode(handle);
            if (!decoded.has_value())
            {
                return false;
            }

            auto& slot = slots_[decoded->index];
            if (slot.generation != decoded->generation || !slot.value.has_value())
            {
                return false;
            }

            slot.value.reset();
            --liveCount_;
            recycle(decoded->index, slot);
            return true;
        }

        void clear() noexcept
        {
            for (std::uint32_t index = 0; index < slots_.size(); ++index)
            {
                auto& slot = slots_[index];
                if (!slot.value.has_value())
                {
                    continue;
                }
                slot.value.reset();
                recycle(index, slot);
            }
            liveCount_ = 0;
        }

        template<typename Function>
        void forEach(Function&& function)
        {
            auto callable = std::forward<Function>(function);
            for (std::uint32_t index = 0; index < slots_.size(); ++index)
            {
                auto& slot = slots_[index];
                if (slot.value.has_value())
                {
                    callable(HandleT{encode(index, slot.generation)}, *slot.value);
                }
            }
        }

        template<typename Function>
        void forEach(Function&& function) const
        {
            auto callable = std::forward<Function>(function);
            for (std::uint32_t index = 0; index < slots_.size(); ++index)
            {
                const auto& slot = slots_[index];
                if (slot.value.has_value())
                {
                    callable(HandleT{encode(index, slot.generation)}, *slot.value);
                }
            }
        }

        [[nodiscard]] std::size_t size() const noexcept
        {
            return liveCount_;
        }

        [[nodiscard]] std::size_t freeListCapacity() const noexcept
        {
            return freeIndices_.capacity();
        }

    private:
        static constexpr std::uint32_t indexBits = 24;
        static constexpr std::uint32_t generationBits = 16;
        static constexpr std::uint32_t ownerBits = 24;
        static constexpr std::uint64_t indexMask = (std::uint64_t{1} << indexBits) - 1;
        static constexpr std::uint64_t generationMask = (std::uint64_t{1} << generationBits) - 1;
        static constexpr std::uint64_t ownerMask = (std::uint64_t{1} << ownerBits) - 1;
        static constexpr std::uint32_t noOwnerToken = 0;
        static constexpr std::uint16_t initialGeneration = 1;
        static constexpr std::size_t initialFreeCapacity = 16;
        static constexpr std::size_t maximumSlotCount = static_cast<std::size_t>(indexMask);

        [[nodiscard]] static constexpr std::size_t nextFreeListCapacity(std::size_t currentCapacity,
                                                                        std::size_t requiredCapacity) noexcept
        {
            auto targetCapacity = std::max(initialFreeCapacity, currentCapacity);
            while (targetCapacity < requiredCapacity)
            {
                if (targetCapacity >= maximumSlotCount / 2)
                {
                    return maximumSlotCount;
                }
                targetCapacity *= 2;
            }
            return std::min(targetCapacity, maximumSlotCount);
        }

        struct Slot final
        {
            template<typename... Args>
            explicit Slot(Args&&... args) : value(std::in_place, std::forward<Args>(args)...)
            {
            }

            std::optional<T> value;
            std::uint16_t generation = initialGeneration;
        };

        struct DecodedHandle final
        {
            std::uint32_t index;
            std::uint16_t generation;
        };

        [[nodiscard]] static constexpr std::uint32_t ownerTokenOrNone(std::uint64_t token) noexcept
        {
            return token <= ownerMask ? static_cast<std::uint32_t>(token) : noOwnerToken;
        }

        [[nodiscard]] std::uint64_t encode(std::uint32_t index, std::uint16_t generation) const noexcept
        {
            return (static_cast<std::uint64_t>(ownerToken_) << (generationBits + indexBits)) |
                   (static_cast<std::uint64_t>(generation) << indexBits) | (static_cast<std::uint64_t>(index) + 1);
        }

        [[nodiscard]] std::optional<DecodedHandle> decode(HandleT handle) const noexcept
        {
            if (!handle.isValid())
            {
                return std::nullopt;
            }

            const auto identity = handle.identity;
            const auto owner = static_cast<std::uint32_t>((identity >> (generationBits + indexBits)) & ownerMask);
            const auto encodedIndex = identity & indexMask;
            const auto generation = static_cast<std::uint16_t>((identity >> indexBits) & generationMask);
            if (ownerToken_ == noOwnerToken || owner != ownerToken_ || encodedIndex == 0 || generation == 0)
            {
                return std::nullopt;
            }

            const auto index = static_cast<std::uint32_t>(encodedIndex - 1);
            if (index >= slots_.size())
            {
                return std::nullopt;
            }
            return DecodedHandle{index, generation};
        }

        void abandon() noexcept
        {
            ownerToken_ = noOwnerToken;
            slots_.clear();
            freeIndices_.clear();
            liveCount_ = 0;
        }

        void recycle(std::uint32_t index, Slot& slot) noexcept
        {
            if (slot.generation == std::numeric_limits<std::uint16_t>::max())
            {
                return;
            }
            ++slot.generation;
            freeIndices_.push_back(index);
        }

        std::uint32_t ownerToken_;
        std::deque<Slot> slots_;
        std::vector<std::uint32_t> freeIndices_;
        std::size_t liveCount_ = 0;
    };

} // namespace rachis::detail
