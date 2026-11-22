#pragma once

#include "Common/PresentTimingHistory.h"

#include <atomic>
#include <cmath>
#include <cstdint>

#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

namespace rachis::metal
{
    // The presented-handler outcomes of one swapchain generation.
    class MetalPresentationFeedback final
    {
    public:
        MetalPresentationFeedback() noexcept
        {
#ifdef __APPLE__
            mach_timebase_info_data_t timebase{};
            if (mach_timebase_info(&timebase) == KERN_SUCCESS && timebase.numer != 0 && timebase.denom != 0)
            {
                ticksPerSecond_ = 1.0e9 * static_cast<double>(timebase.denom) / static_cast<double>(timebase.numer);
            }
#endif
        }

        void submitted(std::uint64_t sequence) noexcept
        {
            submitted_.store(sequence, std::memory_order_release);
            history_.submitted(sequence);
        }

        // presentedTime is CAMetalDrawable.presentedTime: CACurrentMediaTime seconds, 0 when nothing was shown.
        // surfaceVisible says whether the surface could be seen when the handler ran.
        void presented(std::uint64_t sequence, double presentedTime, bool surfaceVisible = true) noexcept
        {
            if (!(presentedTime > 0.0))
            {
                history_.settle(sequence, 0, notShownFlags(sequence, surfaceVisible));
                return;
            }
            history_.settle(sequence, toTicks(presentedTime), PresentTimingFlags::displayed);
            auto previous = displayed_.load(std::memory_order_relaxed);
            while (previous < sequence && !displayed_.compare_exchange_weak(
                                              previous, sequence, std::memory_order_release, std::memory_order_relaxed))
            {
            }
        }

        [[nodiscard]] std::uint64_t displayedSequence() const noexcept
        {
            return displayed_.load(std::memory_order_acquire);
        }

        [[nodiscard]] detail::PresentTimingHistory::Snapshot readTimings(
            std::uint64_t firstSequence, std::span<PresentTiming> records) const noexcept
        {
            return history_.read(firstSequence, records);
        }

        // mach_absolute_time ticks per second.
        [[nodiscard]] std::uint64_t ticksPerSecond() const noexcept
        {
            return static_cast<std::uint64_t>(std::llround(ticksPerSecond_));
        }

    private:
        [[nodiscard]] PresentTimingFlags notShownFlags(std::uint64_t sequence, bool surfaceVisible) const noexcept
        {
            if (!surfaceVisible)
            {
                return PresentTimingFlags::hidden;
            }
            // Core Animation drops a drawable that a newer one replaced before the next refresh.
            return sequence < submitted_.load(std::memory_order_acquire) ? PresentTimingFlags::dropped
                                                                         : PresentTimingFlags::none;
        }

        [[nodiscard]] std::uint64_t toTicks(double seconds) const noexcept
        {
            return static_cast<std::uint64_t>(std::llround(seconds * ticksPerSecond_));
        }

        std::atomic<std::uint64_t> displayed_{0};
        std::atomic<std::uint64_t> submitted_{0};
        // mach_absolute_time ticks per second; nanoseconds where the header is built for portable tests.
        double ticksPerSecond_ = 1.0e9;
        detail::PresentTimingHistory history_;
    };
} // namespace rachis::metal
