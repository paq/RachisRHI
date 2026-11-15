#pragma once

#include "C/CAbiTokenSource.h"
#include "Common/MonotonicIdentitySource.h"
#include "Rachis/rachis.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace rachis::cabi::detail
{

    inline constexpr std::uint32_t cAbiSlotIndexBits = 24;
    inline constexpr std::uint64_t cAbiSlotIndexMask = (std::uint64_t{1} << cAbiSlotIndexBits) - 1;

    inline constexpr std::size_t maxCAbiSlotCount = static_cast<std::size_t>(cAbiSlotIndexMask - 1);

    template<typename T, std::size_t MaxSlots = maxCAbiSlotCount>
    class CAbiSlotTable final
    {
        static_assert(std::is_nothrow_destructible_v<T>);
        static_assert(std::is_nothrow_move_constructible_v<T>);
        static_assert(MaxSlots >= 1 && MaxSlots <= maxCAbiSlotCount);

    public:
        CAbiSlotTable() noexcept = default;

        template<typename... Args>
        [[nodiscard]] rachis_handle emplace(rachis::detail::MonotonicIdentitySource& source, Args&&... args)
        {
            if (freeIndices_.empty())
            {
                if (slots_.size() >= MaxSlots)
                {
                    return 0;
                }
                const auto requiredCapacity = slots_.size() + 1;
                if (freeIndices_.capacity() < requiredCapacity)
                {
                    freeIndices_.reserve(nextCapacity(freeIndices_.capacity(), requiredCapacity));
                }
                if (slots_.capacity() < requiredCapacity)
                {
                    slots_.reserve(nextCapacity(slots_.capacity(), requiredCapacity));
                }
            }

            T constructed(std::forward<Args>(args)...);
            const auto tag = allocateCAbiToken(source);
            if (tag == 0)
            {
                return 0;
            }

            if (!freeIndices_.empty())
            {
                const auto index = freeIndices_.back();
                auto& slot = slots_[index];
                const auto handle = encode(index, tag);
                slot.value.emplace(std::move(constructed));
                slot.handle = handle;
                freeIndices_.pop_back();
                ++liveCount_;
                return handle;
            }

            const auto index = static_cast<std::uint32_t>(slots_.size());
            const auto handle = encode(index, tag);
            slots_.emplace_back(handle, std::in_place, std::move(constructed));
            ++liveCount_;
            return handle;
        }

        [[nodiscard]] T* find(rachis_handle handle) noexcept
        {
            const auto index = decodeIndex(handle);
            if (!index.has_value())
            {
                return nullptr;
            }
            auto& slot = slots_[*index];
            return slot.handle == handle && slot.value.has_value() ? &*slot.value : nullptr;
        }

        [[nodiscard]] const T* find(rachis_handle handle) const noexcept
        {
            const auto index = decodeIndex(handle);
            if (!index.has_value())
            {
                return nullptr;
            }
            const auto& slot = slots_[*index];
            return slot.handle == handle && slot.value.has_value() ? &*slot.value : nullptr;
        }

        bool erase(rachis_handle handle) noexcept
        {
            const auto index = decodeIndex(handle);
            if (!index.has_value())
            {
                return false;
            }
            auto& slot = slots_[*index];
            if (slot.handle != handle || !slot.value.has_value())
            {
                return false;
            }
            slot.value.reset();
            slot.handle = 0;
            freeIndices_.push_back(*index);
            --liveCount_;
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
                slot.handle = 0;
                freeIndices_.push_back(index);
            }
            liveCount_ = 0;
        }

        [[nodiscard]] std::size_t size() const noexcept
        {
            return liveCount_;
        }

        [[nodiscard]] std::size_t slotCapacity() const noexcept
        {
            return slots_.capacity();
        }

        [[nodiscard]] std::size_t freeListCapacity() const noexcept
        {
            return freeIndices_.capacity();
        }

    private:
        static constexpr std::size_t initialCapacity = std::min<std::size_t>(MaxSlots, 16);

        struct Slot final
        {
            template<typename... Args>
            explicit Slot(rachis_handle slotHandle, std::in_place_t, Args&&... args)
                : value(std::in_place, std::forward<Args>(args)...), handle(slotHandle)
            {
            }

            std::optional<T> value;
            rachis_handle handle = 0;
        };

        [[nodiscard]] static constexpr std::size_t nextCapacity(std::size_t currentCapacity,
                                                                std::size_t requiredCapacity) noexcept
        {
            auto targetCapacity = std::max(initialCapacity, currentCapacity);
            while (targetCapacity < requiredCapacity)
            {
                if (targetCapacity >= MaxSlots / 2)
                {
                    return MaxSlots;
                }
                targetCapacity *= 2;
            }
            return std::min(targetCapacity, MaxSlots);
        }

        [[nodiscard]] static constexpr rachis_handle encode(std::uint32_t index, std::uint64_t tag) noexcept
        {
            return (tag << cAbiSlotIndexBits) | (static_cast<std::uint64_t>(index) + 1);
        }

        [[nodiscard]] std::optional<std::uint32_t> decodeIndex(rachis_handle handle) const noexcept
        {
            if (handle == 0)
            {
                return std::nullopt;
            }
            const auto encodedIndex = handle & cAbiSlotIndexMask;
            const auto tag = handle >> cAbiSlotIndexBits;
            if (tag == 0 || encodedIndex == 0 || encodedIndex > MaxSlots)
            {
                return std::nullopt;
            }
            const auto index = static_cast<std::uint32_t>(encodedIndex - 1);
            if (index >= slots_.size())
            {
                return std::nullopt;
            }
            return index;
        }

        std::vector<Slot> slots_;
        std::vector<std::uint32_t> freeIndices_;
        std::size_t liveCount_ = 0;
    };

    struct CAbiSuspendedHandle final
    {
        rachis_handle handle = 0;
        std::uint64_t native = 0;
    };

    class CAbiBorrowedImageHandles final
    {
    public:
        constexpr void publish(rachis_handle texture, rachis_handle view) noexcept
        {
            handles_ = {texture, view};
        }

        template<typename TokenTable>
        void erase(TokenTable& tokens) noexcept
        {
            for (auto& handle : handles_)
            {
                if (handle != 0)
                {
                    static_cast<void>(tokens.erase(handle));
                    handle = 0;
                }
            }
        }

        template<typename TokenTable>
        [[nodiscard]] std::array<CAbiSuspendedHandle, 2> suspend(TokenTable& tokens) noexcept
        {
            std::array<CAbiSuspendedHandle, 2> suspended{};
            for (std::size_t i = 0; i < handles_.size(); ++i)
            {
                const auto handle = handles_[i];
                if (handle == 0)
                {
                    continue;
                }
                if (auto* const token = tokens.find(handle); token != nullptr)
                {
                    suspended[i] = CAbiSuspendedHandle{.handle = handle, .native = token->native};
                    token->native = 0;
                }
            }
            return suspended;
        }

        template<typename TokenTable>
        static void restoreSuspended(TokenTable& tokens, const std::array<CAbiSuspendedHandle, 2>& suspended) noexcept
        {
            for (const auto& entry : suspended)
            {
                if (entry.handle == 0)
                {
                    continue;
                }
                if (auto* const token = tokens.find(entry.handle); token != nullptr)
                {
                    token->native = entry.native;
                }
            }
        }

        template<typename TokenTable>
        void eraseSuspended(TokenTable& tokens, const std::array<CAbiSuspendedHandle, 2>& suspended) noexcept
        {
            for (const auto& entry : suspended)
            {
                if (entry.handle != 0)
                {
                    static_cast<void>(tokens.erase(entry.handle));
                }
            }
            for (std::size_t i = 0; i < handles_.size(); ++i)
            {
                if (handles_[i] == suspended[i].handle)
                {
                    handles_[i] = 0;
                }
            }
        }

        [[nodiscard]] constexpr std::array<rachis_handle, 2> handles() const noexcept
        {
            return handles_;
        }

    private:
        std::array<rachis_handle, 2> handles_{};
    };

} // namespace rachis::cabi::detail
