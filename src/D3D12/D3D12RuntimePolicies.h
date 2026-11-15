#pragma once

#include "Common/BindingPoolAccounting.h"
#include "Rachis/Binding.h"
#include "Rachis/Command.h"
#include "Rachis/Error.h"
#include "Rachis/Resource.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace rachis::d3d12
{

    inline constexpr std::int32_t kD3D12HresultOutOfMemory = static_cast<std::int32_t>(0x8007000EU);
    inline constexpr std::int32_t kDxgiErrorDeviceRemoved = static_cast<std::int32_t>(0x887A0005U);
    inline constexpr std::int32_t kDxgiErrorDeviceHung = static_cast<std::int32_t>(0x887A0006U);
    inline constexpr std::int32_t kDxgiErrorDeviceReset = static_cast<std::int32_t>(0x887A0007U);
    inline constexpr std::int32_t kDxgiErrorDriverInternalError = static_cast<std::int32_t>(0x887A0020U);

    [[nodiscard]] constexpr bool isD3D12DeviceLossHresult(std::int32_t hresult) noexcept
    {
        return hresult == kDxgiErrorDeviceRemoved || hresult == kDxgiErrorDeviceHung ||
               hresult == kDxgiErrorDeviceReset || hresult == kDxgiErrorDriverInternalError;
    }

    [[nodiscard]] constexpr ErrorCode classifyD3D12ResultErrorCode(
        std::int32_t hresult, ErrorCode fallback = ErrorCode::nativeFailure) noexcept
    {
        if (isD3D12DeviceLossHresult(hresult))
        {
            return ErrorCode::deviceLost;
        }
        if (hresult == kD3D12HresultOutOfMemory)
        {
            return ErrorCode::outOfMemory;
        }
        return fallback;
    }

    [[nodiscard]] inline Error formatD3D12HresultError(ErrorCode code, std::int32_t hresult, std::string_view operation)
    {
        constexpr std::string_view hexDigits = "0123456789ABCDEF";
        auto raw = static_cast<std::uint32_t>(hresult);
        std::array<char, 8> hex{};
        std::size_t firstNonZero = hex.size() - 1U;
        for (std::size_t index = hex.size(); index > 0; --index)
        {
            const auto nibble = raw & 0xFU;
            hex[index - 1U] = hexDigits[nibble];
            if (nibble != 0U)
            {
                firstNonZero = index - 1U;
            }
            raw >>= 4U;
        }

        std::string message;
        message.reserve(operation.size() + 32U);
        message.append(operation);
        message.append(" failed with HRESULT 0x");
        message.append(hex.data() + firstNonZero, hex.size() - firstNonZero);
        message.push_back('.');
        return Error{code, static_cast<std::int64_t>(hresult), std::move(message)};
    }

    [[nodiscard]] inline Error makeClassifiedD3D12HresultError(std::int32_t hresult,
                                                               std::string_view operation,
                                                               ErrorCode fallback = ErrorCode::nativeFailure)
    {
        return formatD3D12HresultError(classifyD3D12ResultErrorCode(hresult, fallback), hresult, operation);
    }

    enum class DredEnablementPolicy : std::uint8_t
    {
        systemControlled = 0,
        forcedOn = 1,
    };

    [[nodiscard]] constexpr DredEnablementPolicy selectDredEnablementPolicy(bool enableDred) noexcept
    {
        return enableDred ? DredEnablementPolicy::forcedOn : DredEnablementPolicy::systemControlled;
    }

    template<typename T>
    class BoundedFifoQueue final
    {
    public:
        BoundedFifoQueue() noexcept = default;

        void reserve(std::size_t capacity)
        {
            storage_.assign(capacity, T{});
            head_ = 0;
            tail_ = 0;
            count_ = 0;
        }

        [[nodiscard]] bool empty() const noexcept
        {
            return count_ == 0;
        }

        [[nodiscard]] std::size_t size() const noexcept
        {
            return count_;
        }

        [[nodiscard]] std::size_t capacity() const noexcept
        {
            return storage_.size();
        }

        [[nodiscard]] T& front() noexcept
        {
            if (count_ == 0)
            {
                std::terminate();
            }
            return storage_[head_];
        }

        [[nodiscard]] const T& front() const noexcept
        {
            if (count_ == 0)
            {
                std::terminate();
            }
            return storage_[head_];
        }

        void pop_front() noexcept
        {
            if (count_ == 0)
            {
                std::terminate();
            }
            storage_[head_] = T{};
            head_ = (head_ + 1) % storage_.size();
            --count_;
        }

        void push_back(T value) noexcept
        {
            if (count_ >= storage_.size())
            {
                std::terminate();
            }
            storage_[tail_] = std::move(value);
            tail_ = (tail_ + 1) % storage_.size();
            ++count_;
        }

        void clear() noexcept
        {
            for (auto& slot : storage_)
            {
                slot = T{};
            }
            head_ = 0;
            tail_ = 0;
            count_ = 0;
        }

    private:
        std::vector<T> storage_;
        std::size_t head_ = 0;
        std::size_t tail_ = 0;
        std::size_t count_ = 0;
    };

    template<typename UniqueHandle>
    class ReusableWaitEventPool final
    {
    public:
        explicit ReusableWaitEventPool(std::size_t maxCachedHandles = 64) noexcept : maxCachedHandles_(maxCachedHandles)
        {
        }

        ReusableWaitEventPool(const ReusableWaitEventPool&) = delete;
        ReusableWaitEventPool& operator=(const ReusableWaitEventPool&) = delete;
        ReusableWaitEventPool(ReusableWaitEventPool&&) = delete;
        ReusableWaitEventPool& operator=(ReusableWaitEventPool&&) = delete;

        [[nodiscard]] UniqueHandle tryAcquire() noexcept
        {
            const std::scoped_lock lock(mutex_);
            if (freeHandles_.empty())
            {
                return UniqueHandle{};
            }
            auto handle = std::move(freeHandles_.back());
            freeHandles_.pop_back();
            return handle;
        }

        void recycle(UniqueHandle handle) noexcept
        {
            if (!handle)
            {
                return;
            }
            const std::scoped_lock lock(mutex_);
            if (freeHandles_.size() >= maxCachedHandles_)
            {
                return;
            }
            try
            {
                freeHandles_.push_back(std::move(handle));
            }
            catch (const std::bad_alloc&)
            {
            }
        }

        [[nodiscard]] std::size_t cachedCount() const noexcept
        {
            const std::scoped_lock lock(mutex_);
            return freeHandles_.size();
        }

    private:
        mutable std::mutex mutex_;
        std::vector<UniqueHandle> freeHandles_;
        std::size_t maxCachedHandles_ = 64;
    };

    using BindingDescriptorCounts = detail::BindingDescriptorCounts;

    struct BindingPoolState final
    {
        std::mutex mutex;
        bool logicalAlive = true;
        std::uint32_t maxBindingSets = 0;
        std::uint32_t liveBindingSets = 0;
        BindingDescriptorCounts descriptorCapacity{};
        BindingDescriptorCounts liveDescriptors{};
        std::shared_ptr<void> descriptorBudgetReservation;
        std::string label;
    };

    struct BindingPoolSetLease final
    {
        explicit BindingPoolSetLease(std::shared_ptr<BindingPoolState> pool) noexcept : state(std::move(pool))
        {
        }

        ~BindingPoolSetLease()
        {
            if (!armed || state == nullptr)
            {
                return;
            }
            const std::scoped_lock lock(state->mutex);
            --state->liveBindingSets;
        }

        BindingPoolSetLease(const BindingPoolSetLease&) = delete;
        BindingPoolSetLease& operator=(const BindingPoolSetLease&) = delete;
        BindingPoolSetLease(BindingPoolSetLease&&) = delete;
        BindingPoolSetLease& operator=(BindingPoolSetLease&&) = delete;

        std::shared_ptr<BindingPoolState> state;
        std::uint64_t activeCommandRetentions = 0;
        bool armed = false;
    };

    struct BindingPoolVersionLease final
    {
        BindingPoolVersionLease(std::shared_ptr<BindingPoolState> pool,
                                std::shared_ptr<BindingPoolSetLease> setLease,
                                BindingDescriptorCounts requested) noexcept
            : state(std::move(pool)), poolSet(std::move(setLease)), counts(requested)
        {
        }

        ~BindingPoolVersionLease()
        {
            if (!armed || state == nullptr)
            {
                return;
            }
            const std::scoped_lock lock(state->mutex);
            for (std::size_t index = 0; index < counts.size(); ++index)
            {
                state->liveDescriptors[index] -= counts[index];
            }
        }

        BindingPoolVersionLease(const BindingPoolVersionLease&) = delete;
        BindingPoolVersionLease& operator=(const BindingPoolVersionLease&) = delete;
        BindingPoolVersionLease(BindingPoolVersionLease&&) = delete;
        BindingPoolVersionLease& operator=(BindingPoolVersionLease&&) = delete;

        std::shared_ptr<BindingPoolState> state;
        std::shared_ptr<BindingPoolSetLease> poolSet;
        BindingDescriptorCounts counts{};
        bool armed = false;
    };

    class BindingPoolCommandRetention final
    {
    public:
        BindingPoolCommandRetention() noexcept = default;

        ~BindingPoolCommandRetention()
        {
            reset();
        }

        BindingPoolCommandRetention(const BindingPoolCommandRetention&) = delete;
        BindingPoolCommandRetention& operator=(const BindingPoolCommandRetention&) = delete;

        BindingPoolCommandRetention(BindingPoolCommandRetention&& other) noexcept
            : poolSet_(std::move(other.poolSet_)), poolVersion_(std::move(other.poolVersion_)),
              armed_(std::exchange(other.armed_, false))
        {
        }

        BindingPoolCommandRetention& operator=(BindingPoolCommandRetention&& other) noexcept
        {
            if (this != &other)
            {
                reset();
                poolSet_ = std::move(other.poolSet_);
                poolVersion_ = std::move(other.poolVersion_);
                armed_ = std::exchange(other.armed_, false);
            }
            return *this;
        }

        void reset() noexcept
        {
            if (armed_ && poolSet_ != nullptr && poolSet_->state != nullptr)
            {
                const std::scoped_lock lock(poolSet_->state->mutex);
                --poolSet_->activeCommandRetentions;
            }
            poolVersion_.reset();
            poolSet_.reset();
            armed_ = false;
        }

        [[nodiscard]] bool armed() const noexcept
        {
            return armed_;
        }

        void arm(std::shared_ptr<BindingPoolSetLease> setLease,
                 std::shared_ptr<BindingPoolVersionLease> versionLease) noexcept
        {
            if (armed_)
            {
                std::terminate();
            }
            if (setLease == nullptr || setLease->state == nullptr)
            {
                poolVersion_ = std::move(versionLease);
                return;
            }
            {
                const std::scoped_lock lock(setLease->state->mutex);
                ++setLease->activeCommandRetentions;
            }
            poolSet_ = std::move(setLease);
            poolVersion_ = std::move(versionLease);
            armed_ = true;
        }

    private:
        std::shared_ptr<BindingPoolSetLease> poolSet_;
        std::shared_ptr<BindingPoolVersionLease> poolVersion_;
        bool armed_ = false;
    };

    [[nodiscard]] inline Result<std::shared_ptr<BindingPoolSetLease>> acquireBindingPoolSet(
        const std::shared_ptr<BindingPoolState>& state)
    {
        auto lease = std::make_shared<BindingPoolSetLease>(state);
        const std::scoped_lock lock(state->mutex);
        if (!state->logicalAlive)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The binding pool was destroyed before binding-set allocation completed.",
            });
        }
        if (state->liveBindingSets == state->maxBindingSets)
        {
            return std::unexpected(Error{
                ErrorCode::resourceExhausted,
                0,
                "The binding pool has no free physically live binding-set slots.",
            });
        }
        ++state->liveBindingSets;
        lease->armed = true;
        return lease;
    }

    [[nodiscard]] inline Result<std::shared_ptr<BindingPoolVersionLease>> acquireBindingPoolVersion(
        const std::shared_ptr<BindingPoolSetLease>& poolSet, const BindingDescriptorCounts& counts)
    {
        const auto& state = poolSet->state;
        auto lease = std::make_shared<BindingPoolVersionLease>(state, poolSet, counts);
        const std::scoped_lock lock(state->mutex);
        if (!state->logicalAlive)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The binding pool was destroyed before descriptor-version allocation completed.",
            });
        }
        for (std::size_t index = 0; index < counts.size(); ++index)
        {
            if (counts[index] > state->descriptorCapacity[index] - state->liveDescriptors[index])
            {
                return std::unexpected(Error{
                    ErrorCode::resourceExhausted,
                    0,
                    "The binding pool has insufficient per-type capacity for another immutable descriptor version.",
                });
            }
        }
        for (std::size_t index = 0; index < counts.size(); ++index)
        {
            state->liveDescriptors[index] += counts[index];
        }
        lease->armed = true;
        return lease;
    }

    [[nodiscard]] inline Status commitBindingPoolVersionUpdate(
        BindingPoolSetLease& poolSet,
        const std::shared_ptr<BindingPoolVersionLease>& currentVersion,
        BindingPoolVersionLease& nextVersion)
    {
        auto& state = *poolSet.state;
        const std::scoped_lock lock(state.mutex);
        if (!state.logicalAlive)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The binding pool was destroyed before descriptor-version allocation completed.",
            });
        }

        const bool currentArmed = currentVersion != nullptr && currentVersion->armed;
        auto committed =
            detail::commitBindingPoolVersionCapacity(state.descriptorCapacity,
                                                     state.liveDescriptors,
                                                     poolSet.activeCommandRetentions,
                                                     currentArmed,
                                                     currentArmed ? currentVersion->counts : BindingDescriptorCounts{},
                                                     nextVersion.counts);
        if (!committed)
        {
            return std::unexpected(std::move(committed.error()));
        }
        if (*committed == detail::BindingPoolVersionUpdate::reuseCurrentVersion)
        {
            currentVersion->armed = false;
        }
        nextVersion.armed = true;
        return {};
    }

    inline void revertBindingPoolVersionUpdate(BindingPoolSetLease& poolSet,
                                               const std::shared_ptr<BindingPoolVersionLease>& currentVersion,
                                               BindingPoolVersionLease& nextVersion)
    {
        auto& state = *poolSet.state;
        const std::scoped_lock lock(state.mutex);
        if (!nextVersion.armed)
        {
            return;
        }
        for (std::size_t index = 0; index < nextVersion.counts.size(); ++index)
        {
            state.liveDescriptors[index] -= nextVersion.counts[index];
        }
        nextVersion.armed = false;
        if (currentVersion != nullptr && !currentVersion->armed)
        {
            for (std::size_t index = 0; index < currentVersion->counts.size(); ++index)
            {
                state.liveDescriptors[index] += currentVersion->counts[index];
            }
            currentVersion->armed = true;
        }
    }

    inline constexpr std::uint32_t kDxgiSwapChainFlagFrameLatencyWaitableObject = 0x40U;
    inline constexpr std::uint32_t kDxgiSwapChainFlagAllowTearing = 0x800U;

    [[nodiscard]] constexpr std::uint32_t d3d12SwapchainFlags(bool allowTearing) noexcept
    {
        return kDxgiSwapChainFlagFrameLatencyWaitableObject | (allowTearing ? kDxgiSwapChainFlagAllowTearing : 0U);
    }

    class SwapchainLatencySlotState final
    {
    public:
        [[nodiscard]] constexpr bool needsWait(bool hasWaitableObject) const noexcept
        {
            return hasWaitableObject && !slotAcquired_;
        }

        constexpr void onWaitSucceeded() noexcept
        {
            slotAcquired_ = true;
        }

        constexpr void onCancel() noexcept
        {
        }

        constexpr void onPresent(bool acceptedByDxgi) noexcept
        {
            if (acceptedByDxgi)
            {
                slotAcquired_ = false;
            }
        }

        constexpr void onResize() noexcept
        {
        }

        constexpr void reset() noexcept
        {
            slotAcquired_ = false;
        }

        [[nodiscard]] constexpr bool hasAcquiredSlot() const noexcept
        {
            return slotAcquired_;
        }

    private:
        bool slotAcquired_ = false;
    };

    template<typename Values>
    void reserveForAppend(Values& values, std::size_t additional)
    {
        const auto required = values.size() + additional;
        if (required > values.capacity())
        {
            values.reserve((std::max)(required, values.capacity() * 2U));
        }
    }

    inline constexpr std::size_t noBatchConflict = (std::numeric_limits<std::size_t>::max)();
    inline constexpr std::size_t linearBatchScanLimit = 8;

    template<typename KeyOf>
    [[nodiscard]] std::size_t firstRepeatedKeyIndex(std::size_t count, KeyOf&& keyOf)
    {
        if (count <= linearBatchScanLimit)
        {
            for (std::size_t later = 1; later < count; ++later)
            {
                for (std::size_t earlier = 0; earlier < later; ++earlier)
                {
                    if (keyOf(earlier) == keyOf(later))
                    {
                        return later;
                    }
                }
            }
            return noBatchConflict;
        }

        std::vector<std::pair<std::uint64_t, std::size_t>> keyed;
        keyed.reserve(count);
        for (std::size_t index = 0; index < count; ++index)
        {
            keyed.emplace_back(keyOf(index), index);
        }
        std::ranges::sort(keyed);
        std::size_t first = noBatchConflict;
        for (std::size_t position = 1; position < keyed.size(); ++position)
        {
            if (keyed[position].first == keyed[position - 1U].first)
            {
                first = (std::min)(first, keyed[position].second);
            }
        }
        return first;
    }

    struct BatchSweepInterval final
    {
        std::uint64_t begin = 0;
        std::uint64_t end = 0;
    };

    struct BatchSweepKey final
    {
        std::uint64_t group = 0;
        BatchSweepInterval primary;
        BatchSweepInterval secondary{.begin = 0, .end = (std::numeric_limits<std::uint64_t>::max)()};
    };

    [[nodiscard]] constexpr bool batchSweepIntervalsMeet(BatchSweepInterval left, BatchSweepInterval right) noexcept
    {
        return left.begin < right.end && right.begin < left.end;
    }

    [[nodiscard]] constexpr bool batchSweepKeysMeet(const BatchSweepKey& left, const BatchSweepKey& right) noexcept
    {
        return left.group == right.group && batchSweepIntervalsMeet(left.primary, right.primary) &&
               batchSweepIntervalsMeet(left.secondary, right.secondary);
    }

    [[nodiscard]] constexpr BatchSweepInterval layeredBatchSweepInterval(std::uint32_t firstLayer,
                                                                         std::uint32_t layerCount,
                                                                         std::uint32_t first,
                                                                         std::uint32_t count) noexcept
    {
        constexpr std::uint64_t layerStride = std::uint64_t{1} << 32U;
        const std::uint64_t begin = firstLayer * layerStride + first;
        if (layerCount == 0 || count == 0)
        {
            return BatchSweepInterval{.begin = begin, .end = begin};
        }
        const std::uint64_t lastLayer = static_cast<std::uint64_t>(firstLayer) + layerCount - 1U;
        return BatchSweepInterval{.begin = begin, .end = lastLayer * layerStride + first + count};
    }

    [[nodiscard]] constexpr BatchSweepKey textureBoxSweepKey(std::uint32_t subresource,
                                                             Origin3D origin,
                                                             Extent3D extent) noexcept
    {
        return BatchSweepKey{
            .group = subresource,
            .primary = {.begin = origin.x, .end = static_cast<std::uint64_t>(origin.x) + extent.width},
            .secondary = layeredBatchSweepInterval(origin.z, extent.depth, origin.y, extent.height),
        };
    }

    [[nodiscard]] constexpr BatchSweepKey linearFootprintSweepKey(std::uint64_t offset,
                                                                  const TextureTransferLayout& transfer,
                                                                  std::uint64_t pitch) noexcept
    {
        pitch = (std::max)(pitch, std::uint64_t{1});
        const std::uint64_t firstRow = offset / pitch;
        if (transfer.requiredBufferSize <= offset)
        {
            return BatchSweepKey{.primary = {.begin = firstRow, .end = firstRow}};
        }
        const std::uint64_t column = offset % pitch;
        const bool rowsShareColumns = transfer.bytesPerRow == pitch && transfer.bytesPerImage % pitch == 0 &&
                                      transfer.rowDataSize <= pitch - column;
        return BatchSweepKey{
            .primary = {.begin = firstRow, .end = (transfer.requiredBufferSize - 1U) / pitch + 1U},
            .secondary = rowsShareColumns ? BatchSweepInterval{.begin = column, .end = column + transfer.rowDataSize}
                                          : BatchSweepInterval{.begin = 0, .end = pitch},
        };
    }

    struct BatchSweepAxisCost final
    {
        std::size_t blockVisits = 0;
        std::size_t blocks = 0;
    };

    [[nodiscard]] inline BatchSweepAxisCost batchSweepAxisCost(std::span<const BatchSweepKey> keys,
                                                               BatchSweepInterval BatchSweepKey::* axis)
    {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> starts;
        starts.reserve(keys.size());
        for (const auto& key : keys)
        {
            starts.emplace_back(key.group, (key.*axis).begin);
        }
        std::ranges::sort(starts);
        starts.erase(std::ranges::unique(starts).begin(), starts.end());
        BatchSweepAxisCost cost{.blocks = starts.size()};
        for (const auto& key : keys)
        {
            const auto interval = key.*axis;
            const auto first = std::ranges::upper_bound(starts, std::pair{key.group, interval.begin});
            const auto last = std::ranges::lower_bound(starts, std::pair{key.group, interval.end});
            if (first < last)
            {
                cost.blockVisits += static_cast<std::size_t>(last - first);
            }
        }
        return cost;
    }

    [[nodiscard]] inline bool sweepsSecondaryAxisFirst(std::span<const BatchSweepKey> keys)
    {
        const auto primary = batchSweepAxisCost(keys, &BatchSweepKey::primary);
        const auto secondary = batchSweepAxisCost(keys, &BatchSweepKey::secondary);
        return std::tie(secondary.blockVisits, secondary.blocks) < std::tie(primary.blockVisits, primary.blocks);
    }

    inline void orientBatchSweepKeys(std::span<BatchSweepKey> keys)
    {
        if (sweepsSecondaryAxisFirst(keys))
        {
            for (auto& key : keys)
            {
                std::swap(key.primary, key.secondary);
            }
        }
    }

    struct BatchSweepSteps final
    {
        std::size_t blocks = 0;
        std::size_t entries = 0;
    };

    class BatchSweepIndex final
    {
    public:
        struct Entry final
        {
            BatchSweepKey key;
            std::size_t index = 0;
            std::size_t blockEnd = 0;
            std::uint64_t reach = 0;
        };

        explicit BatchSweepIndex(std::span<const BatchSweepKey> keys, BatchSweepSteps* steps = nullptr) : steps_(steps)
        {
            entries_.reserve(keys.size());
            for (std::size_t index = 0; index < keys.size(); ++index)
            {
                entries_.push_back(Entry{.key = keys[index], .index = index});
            }
            std::ranges::sort(entries_, [](const Entry& left, const Entry& right) {
                return std::tie(left.key.group, left.key.primary.begin, left.key.secondary.begin, left.index) <
                       std::tie(right.key.group, right.key.primary.begin, right.key.secondary.begin, right.index);
            });
            for (std::size_t position = 0; position < entries_.size(); ++position)
            {
                auto& entry = entries_[position];
                const bool continuesBlock = position > 0 && sameBlock(entries_[position - 1U], entry);
                entry.reach = continuesBlock ? (std::max)(entries_[position - 1U].reach, entry.key.secondary.end)
                                             : entry.key.secondary.end;
            }
            for (std::size_t position = entries_.size(); position > 0; --position)
            {
                auto& entry = entries_[position - 1U];
                const bool continuesBlock = position < entries_.size() && sameBlock(entry, entries_[position]);
                entry.blockEnd = continuesBlock ? entries_[position].blockEnd : position;
            }
        }

        [[nodiscard]] std::size_t size() const noexcept
        {
            return entries_.size();
        }

        [[nodiscard]] const Entry& operator[](std::size_t position) const noexcept
        {
            return entries_[position];
        }

        [[nodiscard]] std::size_t firstBlockStartingFrom(std::uint64_t group, std::uint64_t begin) const noexcept
        {
            return static_cast<std::size_t>(std::ranges::partition_point(entries_,
                                                                         [&](const Entry& entry) {
                                                                             return std::tie(entry.key.group,
                                                                                             entry.key.primary.begin) <
                                                                                    std::tie(group, begin);
                                                                         }) -
                                            entries_.begin());
        }

        [[nodiscard]] std::size_t firstBlockStartingAfter(std::uint64_t group, std::uint64_t begin) const noexcept
        {
            return static_cast<std::size_t>(std::ranges::partition_point(entries_,
                                                                         [&](const Entry& entry) {
                                                                             return std::tie(entry.key.group,
                                                                                             entry.key.primary.begin) <=
                                                                                    std::tie(group, begin);
                                                                         }) -
                                            entries_.begin());
        }

        template<typename Visit>
        [[nodiscard]] bool anyMeetingLaterInBlock(std::size_t position, Visit&& visit) const
        {
            const auto& key = entries_[position].key;
            for (std::size_t next = position + 1U; next < entries_[position].blockEnd; ++next)
            {
                countStep(&BatchSweepSteps::entries);
                const auto& other = entries_[next];
                if (other.key.secondary.begin >= key.secondary.end)
                {
                    break;
                }
                if (batchSweepKeysMeet(key, other.key) && visit(other.index))
                {
                    return true;
                }
            }
            return false;
        }

        template<typename Visit>
        [[nodiscard]] bool anyMeetingInBlocksFrom(std::size_t position, const BatchSweepKey& key, Visit&& visit) const
        {
            while (position < entries_.size() && entries_[position].key.group == key.group &&
                   entries_[position].key.primary.begin < key.primary.end)
            {
                countStep(&BatchSweepSteps::blocks);
                if (anyMeetingInBlock(position, key, visit))
                {
                    return true;
                }
                position = entries_[position].blockEnd;
            }
            return false;
        }

    private:
        [[nodiscard]] static bool sameBlock(const Entry& left, const Entry& right) noexcept
        {
            return left.key.group == right.key.group && left.key.primary.begin == right.key.primary.begin;
        }

        void countStep(std::size_t BatchSweepSteps::* step) const noexcept
        {
            if (steps_ != nullptr)
            {
                ++(steps_->*step);
            }
        }

        template<typename Visit>
        [[nodiscard]] bool anyMeetingInBlock(std::size_t first, const BatchSweepKey& key, Visit& visit) const
        {
            const auto block = std::span(entries_).subspan(first, entries_[first].blockEnd - first);
            auto position = static_cast<std::size_t>(
                std::ranges::partition_point(
                    block, [&](const Entry& entry) { return entry.key.secondary.begin < key.secondary.end; }) -
                block.begin());
            while (position > 0)
            {
                countStep(&BatchSweepSteps::entries);
                const auto& entry = block[--position];
                if (entry.reach <= key.secondary.begin)
                {
                    break;
                }
                if (batchSweepKeysMeet(key, entry.key) && visit(entry.index))
                {
                    return true;
                }
            }
            return false;
        }

        std::vector<Entry> entries_;
        BatchSweepSteps* steps_ = nullptr;
    };

    template<typename KeyOf, typename Overlaps>
    [[nodiscard]] bool anyOverlappingPair(std::size_t count,
                                          KeyOf&& keyOf,
                                          Overlaps&& overlaps,
                                          BatchSweepSteps* steps = nullptr)
    {
        if (count <= linearBatchScanLimit)
        {
            for (std::size_t later = 1; later < count; ++later)
            {
                for (std::size_t earlier = 0; earlier < later; ++earlier)
                {
                    if (overlaps(earlier, later))
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        std::vector<BatchSweepKey> keys;
        keys.reserve(count);
        for (std::size_t index = 0; index < count; ++index)
        {
            keys.push_back(keyOf(index));
        }
        orientBatchSweepKeys(keys);
        const BatchSweepIndex index(keys, steps);
        for (std::size_t position = 0; position < index.size(); ++position)
        {
            const auto& entry = index[position];
            const auto visit = [&](std::size_t other) {
                return overlaps((std::min)(entry.index, other), (std::max)(entry.index, other));
            };
            if (index.anyMeetingLaterInBlock(position, visit) ||
                index.anyMeetingInBlocksFrom(entry.blockEnd, entry.key, visit))
            {
                return true;
            }
        }
        return false;
    }

    template<typename LeftKeyOf, typename RightKeyOf, typename Overlaps>
    [[nodiscard]] bool anyCrossOverlap(std::size_t count,
                                       LeftKeyOf&& leftKeyOf,
                                       RightKeyOf&& rightKeyOf,
                                       Overlaps&& overlaps,
                                       BatchSweepSteps* steps = nullptr)
    {
        if (count <= linearBatchScanLimit)
        {
            for (std::size_t left = 0; left < count; ++left)
            {
                for (std::size_t right = 0; right < count; ++right)
                {
                    if (overlaps(left, right))
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        std::vector<BatchSweepKey> keys;
        keys.reserve(count * 2U);
        for (std::size_t index = 0; index < count; ++index)
        {
            keys.push_back(leftKeyOf(index));
        }
        for (std::size_t index = 0; index < count; ++index)
        {
            keys.push_back(rightKeyOf(index));
        }
        orientBatchSweepKeys(keys);
        const auto allKeys = std::span<const BatchSweepKey>(keys);
        const BatchSweepIndex lefts(allKeys.first(count), steps);
        const BatchSweepIndex rights(allKeys.subspan(count), steps);
        for (std::size_t position = 0; position < lefts.size(); ++position)
        {
            const auto& left = lefts[position];
            const auto first = rights.firstBlockStartingFrom(left.key.group, left.key.primary.begin);
            if (rights.anyMeetingInBlocksFrom(
                    first, left.key, [&](std::size_t right) { return overlaps(left.index, right); }))
            {
                return true;
            }
        }
        for (std::size_t position = 0; position < rights.size(); ++position)
        {
            const auto& right = rights[position];
            const auto first = lefts.firstBlockStartingAfter(right.key.group, right.key.primary.begin);
            if (lefts.anyMeetingInBlocksFrom(
                    first, right.key, [&](std::size_t left) { return overlaps(left, right.index); }))
            {
                return true;
            }
        }
        return false;
    }

    template<typename T, std::size_t InlineCapacity>
    class SmallVector final
    {
        static_assert(InlineCapacity > 0, "SmallVector requires a non-zero inline capacity.");

    public:
        using value_type = T;
        using size_type = std::size_t;
        using iterator = T*;
        using const_iterator = const T*;

        SmallVector() noexcept = default;

        ~SmallVector() noexcept
        {
            destroyInline();
        }

        SmallVector(const SmallVector& other)
            requires std::is_copy_constructible_v<T>
        {
            if (other.heapStorage_.has_value())
            {
                heapStorage_.emplace(*other.heapStorage_);
            }
            else
            {
                InlineConstructionRollback rollback(*this);
                for (std::size_t index = 0; index < other.inlineSize_; ++index)
                {
                    std::construct_at(inlineSlotAddress(index), *other.inlineElement(index));
                    ++inlineSize_;
                }
                rollback.release();
            }
        }

        SmallVector& operator=(const SmallVector& other)
            requires std::is_copy_constructible_v<T>
        {
            if (this == &other)
            {
                return *this;
            }
            SmallVector copy(other);
            *this = std::move(copy);
            return *this;
        }

        SmallVector(SmallVector&& other) noexcept(std::is_nothrow_move_constructible_v<T>)
        {
            moveFrom(std::move(other));
        }

        SmallVector& operator=(SmallVector&& other) noexcept(std::is_nothrow_move_constructible_v<T>)
        {
            if (this != &other)
            {
                destroyInline();
                heapStorage_.reset();
                moveFrom(std::move(other));
            }
            return *this;
        }

        template<typename... Args>
        T& emplace_back(Args&&... args)
        {
            if (!heapStorage_.has_value() && inlineSize_ < InlineCapacity)
            {
                T* constructed = std::construct_at(inlineSlotAddress(inlineSize_), std::forward<Args>(args)...);
                ++inlineSize_;
                return *constructed;
            }
            if (!heapStorage_.has_value())
            {
                T appended(std::forward<Args>(args)...);
                spillToHeap(inlineSize_ * 2U);
                return heapStorage_->emplace_back(std::move(appended));
            }
            return heapStorage_->emplace_back(std::forward<Args>(args)...);
        }

        void push_back(const T& value)
            requires std::is_copy_constructible_v<T>
        {
            emplace_back(value);
        }

        void push_back(T&& value)
        {
            emplace_back(std::move(value));
        }

        void reserve(std::size_t requestedCapacity)
        {
            if (heapStorage_.has_value())
            {
                heapStorage_->reserve(requestedCapacity);
            }
            else if (requestedCapacity > InlineCapacity)
            {
                spillToHeap(requestedCapacity);
            }
        }

        void clear() noexcept
        {
            if (heapStorage_.has_value())
            {
                heapStorage_->clear();
            }
            else
            {
                destroyInline();
            }
        }

        void truncate(std::size_t newSize) noexcept
        {
            if (heapStorage_.has_value())
            {
                if (newSize < heapStorage_->size())
                {
                    heapStorage_->erase(heapStorage_->begin() + static_cast<std::ptrdiff_t>(newSize),
                                        heapStorage_->end());
                }
                return;
            }
            while (inlineSize_ > newSize)
            {
                --inlineSize_;
                std::destroy_at(inlineElement(inlineSize_));
            }
        }

        [[nodiscard]] bool usingInlineStorage() const noexcept
        {
            return !heapStorage_.has_value();
        }
        [[nodiscard]] std::size_t size() const noexcept
        {
            return heapStorage_.has_value() ? heapStorage_->size() : inlineSize_;
        }
        [[nodiscard]] bool empty() const noexcept
        {
            return size() == 0;
        }

        [[nodiscard]] T* data() noexcept
        {
            return heapStorage_.has_value() ? heapStorage_->data() : inlineData();
        }
        [[nodiscard]] const T* data() const noexcept
        {
            return heapStorage_.has_value() ? heapStorage_->data() : inlineData();
        }

        [[nodiscard]] iterator begin() noexcept
        {
            return data();
        }
        [[nodiscard]] const_iterator begin() const noexcept
        {
            return data();
        }
        [[nodiscard]] iterator end() noexcept
        {
            return data() + size();
        }
        [[nodiscard]] const_iterator end() const noexcept
        {
            return data() + size();
        }

        [[nodiscard]] T& operator[](std::size_t index) noexcept
        {
            return data()[index];
        }
        [[nodiscard]] const T& operator[](std::size_t index) const noexcept
        {
            return data()[index];
        }
        [[nodiscard]] T& back() noexcept
        {
            return data()[size() - 1U];
        }
        [[nodiscard]] const T& back() const noexcept
        {
            return data()[size() - 1U];
        }

        [[nodiscard]] std::span<T> span() noexcept
        {
            return {data(), size()};
        }
        [[nodiscard]] std::span<const T> span() const noexcept
        {
            return {data(), size()};
        }
        operator std::span<T>() noexcept
        {
            return span();
        }
        operator std::span<const T>() const noexcept
        {
            return span();
        }

    private:
        class InlineConstructionRollback final
        {
        public:
            explicit InlineConstructionRollback(SmallVector& owner) noexcept : owner_(&owner)
            {
            }

            InlineConstructionRollback(const InlineConstructionRollback&) = delete;
            InlineConstructionRollback& operator=(const InlineConstructionRollback&) = delete;
            InlineConstructionRollback(InlineConstructionRollback&&) = delete;
            InlineConstructionRollback& operator=(InlineConstructionRollback&&) = delete;

            ~InlineConstructionRollback() noexcept
            {
                if (owner_ != nullptr)
                {
                    owner_->destroyInline();
                }
            }

            void release() noexcept
            {
                owner_ = nullptr;
            }

        private:
            SmallVector* owner_;
        };

        [[nodiscard]] T* inlineSlotAddress(std::size_t index) noexcept
        {
            return reinterpret_cast<T*>(inlineBytes_.data() + index * sizeof(T));
        }

        [[nodiscard]] const T* inlineSlotAddress(std::size_t index) const noexcept
        {
            return reinterpret_cast<const T*>(inlineBytes_.data() + index * sizeof(T));
        }

        [[nodiscard]] T* inlineElement(std::size_t index) noexcept
        {
            return std::launder(inlineSlotAddress(index));
        }

        [[nodiscard]] const T* inlineElement(std::size_t index) const noexcept
        {
            return std::launder(inlineSlotAddress(index));
        }

        [[nodiscard]] T* inlineData() noexcept
        {
            return inlineSize_ == 0 ? inlineSlotAddress(0) : inlineElement(0);
        }

        [[nodiscard]] const T* inlineData() const noexcept
        {
            return inlineSize_ == 0 ? inlineSlotAddress(0) : inlineElement(0);
        }

        void destroyInline() noexcept
        {
            while (inlineSize_ > 0)
            {
                --inlineSize_;
                std::destroy_at(inlineElement(inlineSize_));
            }
        }

        void moveFrom(SmallVector&& other) noexcept(std::is_nothrow_move_constructible_v<T>)
        {
            if (other.heapStorage_.has_value())
            {
                heapStorage_ = std::move(other.heapStorage_);
                other.heapStorage_.reset();
            }
            else
            {
                InlineConstructionRollback rollback(*this);
                for (std::size_t index = 0; index < other.inlineSize_; ++index)
                {
                    std::construct_at(inlineSlotAddress(index), std::move_if_noexcept(*other.inlineElement(index)));
                    ++inlineSize_;
                }
                rollback.release();
                other.destroyInline();
            }
        }

        void spillToHeap(std::size_t targetCapacity)
        {
            std::vector<T> spilled;
            spilled.reserve((std::max)(targetCapacity, inlineSize_));
            for (std::size_t index = 0; index < inlineSize_; ++index)
            {
                spilled.push_back(std::move_if_noexcept(*inlineElement(index)));
            }
            destroyInline();
            heapStorage_.emplace(std::move(spilled));
        }

        alignas(T) std::array<std::byte, sizeof(T) * InlineCapacity> inlineBytes_{};
        std::size_t inlineSize_ = 0;
        std::optional<std::vector<T>> heapStorage_{};
    };

    template<typename T, std::size_t InlineCapacity>
    inline void deduplicateSharedPointers(SmallVector<std::shared_ptr<T>, InlineCapacity>& values)
    {
        std::size_t writeIndex = 0;
        for (std::size_t readIndex = 0; readIndex < values.size(); ++readIndex)
        {
            if (values[readIndex] != nullptr)
            {
                if (writeIndex != readIndex)
                {
                    values[writeIndex] = std::move(values[readIndex]);
                }
                ++writeIndex;
            }
        }
        values.truncate(writeIndex);
        if (values.size() <= 1U)
        {
            return;
        }

        const auto lessByAddress = [](const std::shared_ptr<T>& lhs, const std::shared_ptr<T>& rhs) noexcept {
            return std::less<const T*>{}(lhs.get(), rhs.get());
        };
        bool strictlySorted = true;
        for (std::size_t index = 1; index < values.size(); ++index)
        {
            if (!lessByAddress(values[index - 1U], values[index]))
            {
                strictlySorted = false;
                break;
            }
        }
        if (strictlySorted)
        {
            return;
        }
        auto activeSpan = values.span();
        std::ranges::sort(activeSpan, lessByAddress);
        const auto duplicateTail = std::ranges::unique(activeSpan);
        values.truncate(static_cast<std::size_t>(duplicateTail.begin() - activeSpan.begin()));
    }

    inline constexpr std::size_t kD3D12ResidencyQueueCount = 3;
    inline constexpr std::uint64_t kD3D12DeviceRemovedFenceValue = (std::numeric_limits<std::uint64_t>::max)();

    using D3D12ResidencyReadyThresholds = std::array<std::uint64_t, kD3D12ResidencyQueueCount>;

    [[nodiscard]] constexpr std::size_t d3d12ResidencyQueueIndex(QueueType queueType) noexcept
    {
        return static_cast<std::size_t>(queueType);
    }

    template<typename QueryCompletedFenceFn>
    [[nodiscard]] constexpr bool promoteCompletedResidency(
        ResidencyState& state,
        D3D12ResidencyReadyThresholds& readyAfter,
        QueryCompletedFenceFn&& queryCompletedFence) noexcept(noexcept(queryCompletedFence(std::size_t{0})))
    {
        if (state != ResidencyState::pending)
        {
            return state == ResidencyState::resident;
        }
        for (std::size_t index = 0; index < readyAfter.size(); ++index)
        {
            const auto required = readyAfter[index];
            if (required == 0)
            {
                continue;
            }
            const auto completed = queryCompletedFence(index);
            if (completed != kD3D12DeviceRemovedFenceValue && completed >= required)
            {
                state = ResidencyState::resident;
                readyAfter = {};
                return true;
            }
        }
        return false;
    }

    template<typename QueryCompletedFenceFn>
    [[nodiscard]] inline Status validateResidencySubmission(ResidencyState& state,
                                                            D3D12ResidencyReadyThresholds& readyAfter,
                                                            QueueType submissionQueue,
                                                            std::span<const TimelinePoint> waits,
                                                            QueryCompletedFenceFn&& queryCompletedFence)
    {
        if (state == ResidencyState::resident || state == ResidencyState::automatic ||
            promoteCompletedResidency(state, readyAfter, queryCompletedFence))
        {
            return {};
        }
        if (state == ResidencyState::evicted)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A submitted command buffer references an evicted resource.",
            });
        }

        const auto submissionIndex = d3d12ResidencyQueueIndex(submissionQueue);
        bool ordered = submissionIndex < readyAfter.size() && readyAfter[submissionIndex] != 0;
        for (const auto wait : waits)
        {
            const auto waitIndex = d3d12ResidencyQueueIndex(wait.queueType());
            ordered = ordered || (waitIndex < readyAfter.size() && readyAfter[waitIndex] != 0 &&
                                  wait.value() >= readyAfter[waitIndex]);
        }
        if (!ordered)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A cross-queue submission that uses a pending resource must wait on its make-resident timeline point.",
            });
        }
        return {};
    }

    template<typename QueryCompletedFenceFn>
    constexpr void propagateResidencySubmission(
        ResidencyState& state,
        D3D12ResidencyReadyThresholds& readyAfter,
        QueueType submissionQueue,
        std::uint64_t submittedValue,
        QueryCompletedFenceFn&& queryCompletedFence) noexcept(noexcept(promoteCompletedResidency(state,
                                                                                                 readyAfter,
                                                                                                 queryCompletedFence)))
    {
        if (submittedValue == 0 || state != ResidencyState::pending ||
            promoteCompletedResidency(state, readyAfter, queryCompletedFence))
        {
            return;
        }
        const auto submissionIndex = d3d12ResidencyQueueIndex(submissionQueue);
        if (submissionIndex >= readyAfter.size())
        {
            return;
        }
        auto& threshold = readyAfter[submissionIndex];
        if (threshold == 0 || submittedValue < threshold)
        {
            threshold = submittedValue;
        }
    }

    [[nodiscard]] constexpr bool isD3D12CommandPoolIdleFromInFlight(std::size_t totalEntries,
                                                                    std::size_t availableCount,
                                                                    std::size_t quarantinedCount,
                                                                    std::size_t recordingCount) noexcept
    {
        return availableCount <= totalEntries && quarantinedCount <= totalEntries - availableCount &&
               recordingCount == totalEntries - availableCount - quarantinedCount;
    }

    template<typename CloseCommandListFn,
             typename ResetAllocatorFn,
             typename ResetCommandListFn,
             typename ClearEntryRetentionsFn,
             typename SetAllocatorCleanFn,
             typename ActivateFreshRecordingFn,
             typename QuarantineAndDetachFn,
             typename ResetFrontendStateFn>
    [[nodiscard]] inline Status executeD3D12RecordingReset(CloseCommandListFn&& closeCommandList,
                                                           ResetAllocatorFn&& resetAllocator,
                                                           ResetCommandListFn&& resetCommandList,
                                                           ClearEntryRetentionsFn&& clearEntryRetentions,
                                                           SetAllocatorCleanFn&& setAllocatorClean,
                                                           ActivateFreshRecordingFn&& activateFreshRecording,
                                                           QuarantineAndDetachFn&& quarantineAndDetach,
                                                           ResetFrontendStateFn&& resetFrontendState)
    {
        const auto failAndQuarantine =
            [&](std::int32_t hresult, std::string_view operation, ErrorCode fallback) -> Status {
            clearEntryRetentions();
            setAllocatorClean(false);
            quarantineAndDetach();
            resetFrontendState();
            return std::unexpected(makeClassifiedD3D12HresultError(hresult, operation, fallback));
        };

        const std::int32_t closeResult = closeCommandList();
        if (closeResult < 0)
        {
            return failAndQuarantine(
                closeResult, "ID3D12GraphicsCommandList::Close(command reset)", ErrorCode::validationError);
        }

        clearEntryRetentions();
        const std::int32_t allocatorResult = resetAllocator();
        if (allocatorResult < 0)
        {
            return failAndQuarantine(allocatorResult, "ID3D12CommandAllocator::Reset", ErrorCode::nativeFailure);
        }
        setAllocatorClean(true);

        const std::int32_t commandListResult = resetCommandList();
        if (commandListResult < 0)
        {
            return failAndQuarantine(commandListResult, "ID3D12GraphicsCommandList::Reset", ErrorCode::nativeFailure);
        }

        setAllocatorClean(false);
        activateFreshRecording();
        resetFrontendState();
        return {};
    }

    template<typename EntryPtr, typename IsAllocatorCleanFn, typename SetAllocatorCleanFn, typename ResetAllocatorFn>
    [[nodiscard]] inline Status resetD3D12IdleCommandPoolEntries(BoundedFifoQueue<EntryPtr>& available,
                                                                 std::vector<EntryPtr>& quarantined,
                                                                 IsAllocatorCleanFn&& isAllocatorClean,
                                                                 SetAllocatorCleanFn&& setAllocatorClean,
                                                                 ResetAllocatorFn&& resetAllocator)
    {
        const auto idleCount = available.size();
        if (idleCount == 0)
        {
            return {};
        }
        try
        {
            quarantined.reserve(quarantined.size() + 1U);
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to reserve command-pool quarantine storage.",
            });
        }
        for (std::size_t index = 0; index < idleCount; ++index)
        {
            auto entry = available.front();
            available.pop_front();
            if (!isAllocatorClean(*entry))
            {
                const std::int32_t result = resetAllocator(*entry);
                if (result < 0)
                {
                    setAllocatorClean(*entry, false);
                    quarantined.push_back(entry);
                    return std::unexpected(makeClassifiedD3D12HresultError(
                        result, "ID3D12CommandAllocator::Reset(command pool)", ErrorCode::nativeFailure));
                }
                setAllocatorClean(*entry, true);
            }
            available.push_back(entry);
        }
        return {};
    }

    template<typename EntryPtr,
             typename ClearEntryRetentionsFn,
             typename IsAllocatorCleanFn,
             typename SetAllocatorCleanFn,
             typename ResetAllocatorFn,
             typename ResetCommandListFn>
    [[nodiscard]] inline Result<EntryPtr> acquireAndReopenD3D12PoolEntry(BoundedFifoQueue<EntryPtr>& available,
                                                                         std::vector<EntryPtr>& quarantined,
                                                                         ClearEntryRetentionsFn&& clearEntryRetentions,
                                                                         IsAllocatorCleanFn&& isAllocatorClean,
                                                                         SetAllocatorCleanFn&& setAllocatorClean,
                                                                         ResetAllocatorFn&& resetAllocator,
                                                                         ResetCommandListFn&& resetCommandList)
    {
        if (available.empty())
        {
            return std::unexpected(Error{
                ErrorCode::resourceExhausted,
                0,
                "The bounded command pool is exhausted.",
            });
        }
        try
        {
            quarantined.reserve(quarantined.size() + 1U);
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to reserve command-pool quarantine storage.",
            });
        }

        auto entry = available.front();
        available.pop_front();
        clearEntryRetentions(*entry);
        if (!isAllocatorClean(*entry))
        {
            const std::int32_t resetResult = resetAllocator(*entry);
            if (resetResult < 0)
            {
                setAllocatorClean(*entry, false);
                quarantined.push_back(entry);
                return std::unexpected(makeClassifiedD3D12HresultError(
                    resetResult, "ID3D12CommandAllocator::Reset", ErrorCode::nativeFailure));
            }
        }
        setAllocatorClean(*entry, false);
        const std::int32_t listResult = resetCommandList(*entry);
        if (listResult < 0)
        {
            quarantined.push_back(entry);
            return std::unexpected(makeClassifiedD3D12HresultError(
                listResult, "ID3D12GraphicsCommandList::Reset", ErrorCode::nativeFailure));
        }
        return entry;
    }

} // namespace rachis::d3d12
