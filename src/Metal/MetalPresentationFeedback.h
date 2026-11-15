#pragma once

#include <atomic>
#include <cstdint>

namespace rachis::metal
{
    class MetalPresentationFeedback final
    {
    public:
        void presented(std::uint64_t sequence, double presentedTime) noexcept
        {
            if (!(presentedTime > 0.0))
            {
                return;
            }
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

    private:
        std::atomic<std::uint64_t> displayed_{0};
    };
} // namespace rachis::metal
