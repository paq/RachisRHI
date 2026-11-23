#pragma once

#include "Common/PresentTimingHistory.h"

#include <cstdint>
#include <limits>
#include <span>

namespace rachis::d3d12
{

    class PresentationProgressTracker final
    {
    public:
        void start(std::uint64_t generation, std::uint32_t nativePresentCount) noexcept
        {
            supported_ = true;
            generation_ = generation;
            lastNativeCount_ = nativePresentCount;
            submitted_ = 0;
            displayed_ = 0;
            timings_.reset();
        }

        void disable() noexcept
        {
            supported_ = false;
            submitted_ = 0;
            displayed_ = 0;
            timings_.reset();
        }

        void beginGeneration(std::uint64_t generation, std::uint32_t nativePresentCount) noexcept
        {
            if (!supported_)
            {
                return;
            }
            if (static_cast<std::uint32_t>(nativePresentCount - lastNativeCount_) > maxLag)
            {
                disable();
                return;
            }
            generation_ = generation;
            lastNativeCount_ = nativePresentCount;
            submitted_ = 0;
            displayed_ = 0;
            timings_.reset();
        }

        void presented(std::uint32_t nativePresentCount) noexcept
        {
            if (!supported_)
            {
                return;
            }
            const auto advanced = static_cast<std::uint32_t>(nativePresentCount - lastNativeCount_);
            if (advanced > maxLag || submitted_ > std::numeric_limits<std::uint64_t>::max() - advanced)
            {
                disable();
                return;
            }
            submitted_ += advanced;
            lastNativeCount_ = nativePresentCount;
            timings_.submitted(submitted_);
        }

        void observed(std::uint32_t displayedPresentCount) noexcept
        {
            if (!supported_)
            {
                return;
            }
            const auto lag = behind(displayedPresentCount);

            if (lag > maxLag || lag > submitted_)
            {
                return;
            }
            const auto candidate = submitted_ - lag;
            if (candidate > displayed_)
            {
                displayed_ = candidate;
            }
        }

        void observedFrame(std::uint32_t presentCount,
                           std::uint32_t presentRefreshCount,
                           std::uint32_t syncRefreshCount,
                           std::uint64_t syncTime) noexcept
        {
            if (!supported_)
            {
                return;
            }
            measureRefresh(syncRefreshCount, syncTime);
            observed(presentCount);
            const auto lag = behind(presentCount);
            if (lag > maxLag || lag >= submitted_ || syncTime == 0)
            {
                return;
            }
            const auto sequence = submitted_ - lag;
            const auto vblanksSince = static_cast<std::uint32_t>(syncRefreshCount - presentRefreshCount);
            auto flags = PresentTimingFlags::displayed;
            auto time = syncTime;
            if (vblanksSince != 0)
            {
                const auto back = static_cast<std::uint64_t>(vblanksSince) * refreshTicks_;
                if (vblanksSince > maxRefreshGap || refreshTicks_ == 0 || back >= syncTime)
                {
                    timings_.settleBefore(sequence);
                    return;
                }
                time = syncTime - back;
                flags = flags | PresentTimingFlags::estimated;
            }
            timings_.settle(sequence, time, flags);
            timings_.settleBefore(sequence);
        }

        [[nodiscard]] std::uint64_t refreshTicks() const noexcept
        {
            return refreshTicks_;
        }

        [[nodiscard]] detail::PresentTimingHistory::Snapshot readTimings(
            std::uint64_t firstSequence, const detail::PresentTimingSink& records) const noexcept
        {
            return timings_.read(firstSequence, records);
        }

        [[nodiscard]] detail::PresentTimingHistory::Snapshot readTimings(
            std::uint64_t firstSequence, std::span<PresentTiming> records) const noexcept
        {
            return timings_.read(firstSequence, records);
        }

        [[nodiscard]] bool supported() const noexcept
        {
            return supported_;
        }
        [[nodiscard]] std::uint64_t generation() const noexcept
        {
            return supported_ ? generation_ : 0;
        }
        [[nodiscard]] std::uint64_t submittedSequence() const noexcept
        {
            return submitted_;
        }
        [[nodiscard]] std::uint64_t displayedSequence() const noexcept
        {
            return displayed_;
        }

    private:
        static constexpr std::uint32_t maxLag = std::numeric_limits<std::int32_t>::max();
        static constexpr std::uint32_t maxRefreshGap = 600;

        void measureRefresh(std::uint32_t syncRefreshCount, std::uint64_t syncTime) noexcept
        {
            const auto refreshes = static_cast<std::uint32_t>(syncRefreshCount - lastSyncRefreshCount_);
            if (lastSyncTime_ != 0 && syncTime > lastSyncTime_ && refreshes != 0 && refreshes <= maxRefreshGap)
            {
                refreshTicks_ = (syncTime - lastSyncTime_ + refreshes / 2) / refreshes;
            }
            if (syncTime != 0 && refreshes != 0)
            {
                lastSyncRefreshCount_ = syncRefreshCount;
                lastSyncTime_ = syncTime;
            }
        }

        [[nodiscard]] std::uint32_t behind(std::uint32_t nativePresentCount) const noexcept
        {
            return static_cast<std::uint32_t>(lastNativeCount_ - nativePresentCount);
        }

        bool supported_ = false;
        std::uint64_t generation_ = 0;
        std::uint32_t lastNativeCount_ = 0;
        std::uint64_t submitted_ = 0;
        std::uint64_t displayed_ = 0;
        std::uint32_t lastSyncRefreshCount_ = 0;
        std::uint64_t lastSyncTime_ = 0;
        std::uint64_t refreshTicks_ = 0;
        detail::PresentTimingHistory timings_;
    };

} // namespace rachis::d3d12
