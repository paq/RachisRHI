#pragma once

#include "Rachis/Command.h"
#include "Rachis/Resource.h"

#include <cstdint>
#include <string_view>
#include <vector>

namespace rachis
{

    enum class SurfaceKind : std::uint8_t
    {
        win32,
        wayland,
        xcb,
        caMetalLayer,
        win32Composition,
    };

    struct SurfaceDesc final
    {
        SurfaceKind kind = SurfaceKind::win32;

        void* windowHandle = nullptr;
        void* displayHandle = nullptr;
        std::string_view label = {};
    };

    enum class PresentMode : std::uint8_t
    {
        fifo,
        immediate,
    };

    enum class PresentModeMask : std::uint8_t
    {
        none = 0,
        fifo = 1U << 0U,
        immediate = 1U << 1U,
    };

    [[nodiscard]] constexpr PresentModeMask operator|(PresentModeMask left, PresentModeMask right) noexcept
    {
        return static_cast<PresentModeMask>(static_cast<std::uint8_t>(left) | static_cast<std::uint8_t>(right));
    }

    constexpr PresentModeMask& operator|=(PresentModeMask& left, PresentModeMask right) noexcept
    {
        left = left | right;
        return left;
    }

    [[nodiscard]] constexpr bool supportsPresentMode(PresentModeMask mask, PresentMode mode) noexcept
    {
        const auto required = [&]() {
            switch (mode)
            {
            case PresentMode::fifo:
                return PresentModeMask::fifo;
            case PresentMode::immediate:
                return PresentModeMask::immediate;
            }
            return PresentModeMask::none;
        }();
        return (static_cast<std::uint8_t>(mask) & static_cast<std::uint8_t>(required)) != 0;
    }

    enum class SurfaceColorSpace : std::uint8_t
    {
        srgbNonlinear,
        extendedSrgbLinear,
        hdr10St2084,
    };

    struct SurfaceFormat final
    {
        Format format = Format::unknown;
        SurfaceColorSpace colorSpace = SurfaceColorSpace::srgbNonlinear;

        friend constexpr bool operator==(const SurfaceFormat&, const SurfaceFormat&) noexcept = default;
    };

    struct SurfaceCapabilities final
    {
        std::uint32_t minImageCount = 0;
        std::uint32_t maxImageCount = 0;
        std::uint32_t minWidth = 0;
        std::uint32_t minHeight = 0;
        std::uint32_t maxWidth = 0;
        std::uint32_t maxHeight = 0;
        std::uint32_t preferredWidth = 0;
        std::uint32_t preferredHeight = 0;
        TextureUsage usages = TextureUsage::none;
        PresentModeMask presentModes = PresentModeMask::none;
        std::vector<SurfaceFormat> formats = {};
    };

    struct SwapchainDesc final
    {
        SurfaceHandle surface = {};
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::uint32_t imageCount = 3;
        Format format = Format::bgra8Unorm;
        SurfaceColorSpace colorSpace = SurfaceColorSpace::srgbNonlinear;
        TextureUsage usage = TextureUsage::renderTarget;
        PresentMode presentMode = PresentMode::fifo;
        std::string_view label = {};
    };

    struct AcquiredSwapchainImage final
    {
        TextureHandle texture = {};
        TextureViewHandle renderTargetView = {};
        TextureLayout initialLayout = TextureLayout::present;
        PipelineStage initialStages = PipelineStage::none;
        Access initialAccess = Access::none;
        TextureLayout requiredFinalLayout = TextureLayout::present;
        PipelineStage requiredFinalStages = PipelineStage::none;
        Access requiredFinalAccess = Access::none;
        bool contentsDefined = false;
        std::uint32_t imageIndex = 0;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        Format format = Format::unknown;
        SurfaceColorSpace colorSpace = SurfaceColorSpace::srgbNonlinear;
        std::uint64_t generation = 0;
    };

    enum class PresentStatus : std::uint8_t
    {
        presented,
        occluded,
        surfaceOutOfDate,
    };

    struct SwapchainPresentationInfo final
    {
        Format format = Format::unknown;
        SurfaceColorSpace colorSpace = SurfaceColorSpace::srgbNonlinear;
        PresentMode presentMode = PresentMode::fifo;
        TextureUsage usage = TextureUsage::none;
        std::uint32_t imageCount = 0;
        std::uint32_t framesInFlightBound = 0;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        bool hdr = false;
    };

    struct SwapchainFrameTiming final
    {
        std::uint64_t lastAcquireNs = 0;
        std::uint64_t lastPresentSubmitNs = 0;
        std::uint64_t lastPresentIntervalNs = 0;
        std::uint32_t presentsCompleted = 0;
        std::uint32_t inFlightPresents = 0;
        bool hidden = false;

        bool presentationSupported = false;
        std::uint64_t presentationGeneration = 0;
        std::uint64_t submittedPresentationSequence = 0;
        std::uint64_t displayedPresentationSequence = 0;
    };

} // namespace rachis
