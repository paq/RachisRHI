#pragma once

#include "D3D12/D3D12TestHooks.h"

#include <dxgi1_6.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

namespace rachis::test
{
    class ScopedD3D12FrameStatistics final
    {
    public:
        enum class Source
        {
            native,
            zero,
            disjoint,
            currentPresent,
        };

        explicit ScopedD3D12FrameStatistics(Source source = Source::native) noexcept : source_(source)
        {
            previousScope_ = current_;
            current_ = this;
            previous_ = d3d12::test::nativeFrameStatisticsReader().exchange(read);
        }

        ~ScopedD3D12FrameStatistics()
        {
            d3d12::test::nativeFrameStatisticsReader().store(previous_);
            current_ = previousScope_;
        }

        ScopedD3D12FrameStatistics(const ScopedD3D12FrameStatistics&) = delete;
        ScopedD3D12FrameStatistics& operator=(const ScopedD3D12FrameStatistics&) = delete;

        void source(Source value) noexcept
        {
            source_ = value;
        }

        void beginGeneration() noexcept
        {
            baseline_.reset();
            observedSequence_ = 0;
        }

        [[nodiscard]] std::uint64_t observedSequence() const noexcept
        {
            return observedSequence_;
        }

        [[nodiscard]] HRESULT lastResult() const noexcept
        {
            return lastResult_;
        }

    private:
        static std::int32_t read(void* nativeSwapchain, void* nativeStatistics) noexcept
        {
            auto& self = *current_;
            auto& swapchain = *static_cast<IDXGISwapChain4*>(nativeSwapchain);
            auto& statistics = *static_cast<DXGI_FRAME_STATISTICS*>(nativeStatistics);
            UINT submitted = 0;
            const auto countResult = swapchain.GetLastPresentCount(&submitted);
            if (!self.baseline_ && SUCCEEDED(countResult))
            {
                self.baseline_ = submitted;
            }
            statistics = {};
            switch (self.source_)
            {
            case Source::native:
                self.lastResult_ = swapchain.GetFrameStatistics(&statistics);
                break;
            case Source::zero:
                self.lastResult_ = S_OK;
                break;
            case Source::disjoint:
                self.lastResult_ = DXGI_ERROR_FRAME_STATISTICS_DISJOINT;
                break;
            case Source::currentPresent:
                statistics.PresentCount = submitted;
                self.lastResult_ = countResult;
                break;
            }
            if (SUCCEEDED(self.lastResult_) && SUCCEEDED(countResult) && self.baseline_)
            {
                const auto sequence = static_cast<std::uint32_t>(statistics.PresentCount - *self.baseline_);
                const auto submittedSequence = static_cast<std::uint32_t>(submitted - *self.baseline_);
                if (sequence <= submittedSequence &&
                    sequence <= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
                {
                    self.observedSequence_ = std::max(self.observedSequence_, static_cast<std::uint64_t>(sequence));
                }
            }
            return static_cast<std::int32_t>(self.lastResult_);
        }

        inline static ScopedD3D12FrameStatistics* current_ = nullptr;
        ScopedD3D12FrameStatistics* previousScope_ = nullptr;
        d3d12::test::NativeFrameStatisticsReader previous_ = nullptr;
        Source source_;
        std::optional<std::uint32_t> baseline_;
        std::uint64_t observedSequence_ = 0;
        HRESULT lastResult_ = S_OK;
    };
} // namespace rachis::test
