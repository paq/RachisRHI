#include "LinuxSurfaceTestHost.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include "Vulkan/VulkanTestHooks.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace
{

    using rachis::test::WaylandTestWindow;
    using rachis::test::WindowExtent;
    using rachis::test::XcbTestWindow;

    constexpr WindowExtent initialExtent{.width = 128, .height = 96};

    struct HostWindow final
    {
        rachis::SurfaceKind kind = rachis::SurfaceKind::wayland;
        void* window = nullptr;
        void* display = nullptr;
        std::string name;
    };

    class OpenedWindow final
    {
    public:
        [[nodiscard]] static OpenedWindow wayland()
        {
            if (!WaylandTestWindow::compositorReachable())
            {
                SKIP("No Wayland compositor accepts connections from this process.");
            }
            OpenedWindow opened;
            std::string error;
            opened.wayland_ = WaylandTestWindow::open("Rachis Vulkan Wayland contract", error);
            INFO(error);
            REQUIRE(opened.wayland_ != nullptr);
            return opened;
        }

        [[nodiscard]] static OpenedWindow xcb()
        {
            const char* display = std::getenv("DISPLAY");
            if (display == nullptr || display[0] == '\0')
            {
                SKIP("No X server is configured for this process: DISPLAY is unset.");
            }
            OpenedWindow opened;
            std::string error;
            opened.xcb_ = XcbTestWindow::open("Rachis Vulkan XCB contract", initialExtent, error);
            INFO(error);
            REQUIRE(opened.xcb_ != nullptr);
            return opened;
        }

        [[nodiscard]] HostWindow host() const
        {
            if (wayland_ != nullptr)
            {
                return {
                    .kind = rachis::SurfaceKind::wayland,
                    .window = wayland_->surface(),
                    .display = wayland_->display(),
                    .name = "wayland",
                };
            }
            return {
                .kind = rachis::SurfaceKind::xcb,
                .window = xcb_->window(),
                .display = xcb_->connection(),
                .name = "xcb",
            };
        }

        [[nodiscard]] std::optional<WindowExtent> liveExtent() const
        {
            if (xcb_ == nullptr)
            {
                return std::nullopt;
            }
            return xcb_->geometry();
        }

        [[nodiscard]] WindowExtent resizeHost(WindowExtent extent)
        {
            if (xcb_ == nullptr)
            {
                return extent;
            }
            return xcb_->resize(extent);
        }

        void pump()
        {
            if (wayland_ != nullptr)
            {
                REQUIRE(wayland_->roundtrip());
            }
        }

        [[nodiscard]] int connectionError() const noexcept
        {
            return wayland_ != nullptr ? wayland_->connectionError() : xcb_->connectionError();
        }

    private:
        std::unique_ptr<WaylandTestWindow> wayland_;
        std::unique_ptr<XcbTestWindow> xcb_;
    };

    void captureErrors(const rachis::DiagnosticMessage& message, void* userData) noexcept
    {
        if (message.severity == rachis::DiagnosticSeverity::error ||
            message.severity == rachis::DiagnosticSeverity::corruption)
        {
            static_cast<std::atomic_uint64_t*>(userData)->fetch_add(1, std::memory_order_relaxed);
        }
    }

    [[nodiscard]] rachis::Device requireHardwareDevice(std::atomic_uint64_t& errors)
    {
        auto desc = rachis::test::vulkanDeviceDesc();
        desc.diagnosticCallback = &captureErrors;
        desc.diagnosticUserData = &errors;
        auto created = rachis::createDevice(desc);
        if (!created)
        {
            INFO(created.error().message);
            REQUIRE(created);
        }
        auto device = std::move(*created);
        const auto adapter = rachis::test::requireAdapterInfo(device);
        REQUIRE(adapter.backend == rachis::Backend::vulkan);
        REQUIRE(adapter.type != rachis::AdapterType::software);
        return device;
    }

    [[nodiscard]] rachis::SurfaceHandle requireSurface(rachis::Device& device,
                                                       const HostWindow& host,
                                                       const char* label)
    {
        auto surface = device.createSurface({
            .kind = host.kind,
            .windowHandle = host.window,
            .displayHandle = host.display,
            .label = label,
        });
        if (!surface)
        {
            INFO(surface.error().message);
            REQUIRE(surface);
        }
        return *surface;
    }

    [[nodiscard]] rachis::SurfaceCapabilities requireCapabilities(rachis::Device& device, rachis::SurfaceHandle surface)
    {
        auto capabilities = device.querySurfaceCapabilities(surface);
        if (!capabilities)
        {
            INFO(capabilities.error().message);
            REQUIRE(capabilities);
        }
        return std::move(*capabilities);
    }

    [[nodiscard]] bool isSrgb(rachis::Format format) noexcept
    {
        return format == rachis::Format::bgra8UnormSrgb || format == rachis::Format::rgba8UnormSrgb;
    }

    [[nodiscard]] std::optional<rachis::SurfaceFormat> firstFormat(const rachis::SurfaceCapabilities& capabilities,
                                                                   bool srgb)
    {
        for (const auto& format : capabilities.formats)
        {
            if (format.colorSpace == rachis::SurfaceColorSpace::srgbNonlinear && isSrgb(format.format) == srgb)
            {
                return format;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] WindowExtent swapchainExtent(const rachis::SurfaceCapabilities& capabilities) noexcept
    {
        return {
            .width = capabilities.preferredWidth == 0 ? initialExtent.width : capabilities.preferredWidth,
            .height = capabilities.preferredHeight == 0 ? initialExtent.height : capabilities.preferredHeight,
        };
    }

    [[nodiscard]] rachis::SwapchainHandle requireSwapchain(
        rachis::Device& device,
        rachis::SurfaceHandle surface,
        const rachis::SurfaceCapabilities& capabilities,
        rachis::SurfaceFormat format,
        rachis::PresentMode mode,
        rachis::TextureUsage usage = rachis::TextureUsage::renderTarget)
    {
        const auto extent = swapchainExtent(capabilities);
        auto swapchain = device.createSwapchain({
            .surface = surface,
            .width = extent.width,
            .height = extent.height,
            .imageCount = (std::min)(capabilities.maxImageCount, (std::max)(capabilities.minImageCount, 2U)),
            .format = format.format,
            .colorSpace = format.colorSpace,
            .usage = usage,
            .presentMode = mode,
            .label = "Linux WSI contract",
        });
        if (!swapchain)
        {
            INFO(swapchain.error().message);
            REQUIRE(swapchain);
        }
        return *swapchain;
    }

    [[nodiscard]] rachis::Result<rachis::AcquiredSwapchainImage> acquireFrame(rachis::Device& device,
                                                                              rachis::SwapchainHandle swapchain)
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (true)
        {
            auto acquired = device.acquireNextImage(swapchain);
            if (acquired || acquired.error().code != rachis::ErrorCode::notReady ||
                std::chrono::steady_clock::now() >= deadline)
            {
                return acquired;
            }
        }
    }

    [[nodiscard]] rachis::Result<rachis::TimelinePoint> clearAcquired(rachis::Device& device,
                                                                      const rachis::AcquiredSwapchainImage& acquired,
                                                                      rachis::ClearColor color,
                                                                      std::optional<rachis::BufferHandle> readback)
    {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        if (!command)
        {
            return std::unexpected(std::move(command.error()));
        }
        auto toColor = device.textureBarrier(*command,
                                             {
                                                 .texture = acquired.texture,
                                                 .stagesBefore = acquired.initialStages,
                                                 .stagesAfter = rachis::PipelineStage::renderTarget,
                                                 .accessBefore = acquired.initialAccess,
                                                 .accessAfter = rachis::Access::renderTarget,
                                                 .layoutBefore = acquired.initialLayout,
                                                 .layoutAfter = rachis::TextureLayout::renderTarget,
                                             });
        if (!toColor)
        {
            return std::unexpected(std::move(toColor.error()));
        }
        const std::array attachments{
            rachis::ColorAttachment{
                .view = acquired.renderTargetView,
                .loadOperation = rachis::LoadOperation::clear,
                .clearValue = color,
            },
        };
        if (auto began = device.beginRendering(*command, {.colorAttachments = attachments}); !began)
        {
            return std::unexpected(std::move(began.error()));
        }
        if (auto ended = device.endRendering(*command); !ended)
        {
            return std::unexpected(std::move(ended.error()));
        }
        auto finalStages = rachis::PipelineStage::renderTarget;
        auto finalAccess = rachis::Access::renderTarget;
        auto finalLayout = rachis::TextureLayout::renderTarget;
        if (readback)
        {
            auto toCopy = device.textureBarrier(*command,
                                                {
                                                    .texture = acquired.texture,
                                                    .stagesBefore = finalStages,
                                                    .stagesAfter = rachis::PipelineStage::copy,
                                                    .accessBefore = finalAccess,
                                                    .accessAfter = rachis::Access::copySource,
                                                    .layoutBefore = finalLayout,
                                                    .layoutAfter = rachis::TextureLayout::copySource,
                                                });
            if (!toCopy)
            {
                return std::unexpected(std::move(toCopy.error()));
            }
            auto copied = device.copyTextureToBuffer(*command,
                                                     {
                                                         .buffer = *readback,
                                                         .texture = acquired.texture,
                                                         .extent = {acquired.width, acquired.height, 1},
                                                     });
            if (!copied)
            {
                return std::unexpected(std::move(copied.error()));
            }
            finalStages = rachis::PipelineStage::copy;
            finalAccess = rachis::Access::copySource;
            finalLayout = rachis::TextureLayout::copySource;
        }
        auto toPresent = device.textureBarrier(*command,
                                               {
                                                   .texture = acquired.texture,
                                                   .stagesBefore = finalStages,
                                                   .stagesAfter = acquired.requiredFinalStages,
                                                   .accessBefore = finalAccess,
                                                   .accessAfter = acquired.requiredFinalAccess,
                                                   .layoutBefore = finalLayout,
                                                   .layoutAfter = acquired.requiredFinalLayout,
                                               });
        if (!toPresent)
        {
            return std::unexpected(std::move(toPresent.error()));
        }
        return device.submit(std::move(*command));
    }

    rachis::AcquiredSwapchainImage presentFrame(rachis::Device& device,
                                                rachis::SwapchainHandle swapchain,
                                                rachis::ClearColor color = {0.2F, 0.1F, 0.4F, 1.0F})
    {
        auto acquired = acquireFrame(device, swapchain);
        if (!acquired)
        {
            INFO(acquired.error().message);
            REQUIRE(acquired);
        }
        auto submitted = clearAcquired(device, *acquired, color, std::nullopt);
        if (!submitted)
        {
            INFO(submitted.error().message);
            REQUIRE(submitted);
        }
        auto presented = device.present(swapchain, *submitted);
        if (!presented)
        {
            INFO(presented.error().message);
            REQUIRE(presented);
        }
        CHECK(*presented == rachis::PresentStatus::presented);
        return *acquired;
    }

    struct ReadbackPixel final
    {
        std::uint8_t gray = 0;
        std::uint8_t alpha = 0;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
    };

    [[nodiscard]] ReadbackPixel presentReadbackFrame(rachis::Device& device, rachis::SwapchainHandle swapchain)
    {
        auto acquired = acquireFrame(device, swapchain);
        REQUIRE(acquired);
        auto layout = device.queryTextureTransferLayout(acquired->texture,
                                                        {
                                                            .extent = {acquired->width, acquired->height, 1},
                                                        });
        REQUIRE(layout);
        auto readback = device.createBuffer({
            .size = layout->requiredBufferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Linux WSI readback",
        });
        REQUIRE(readback);
        auto submitted = clearAcquired(device, *acquired, {0.5F, 0.5F, 0.5F, 1.0F}, *readback);
        if (!submitted)
        {
            INFO(submitted.error().message);
            REQUIRE(submitted);
        }
        auto presented = device.present(swapchain, *submitted);
        REQUIRE(presented);
        CHECK(*presented == rachis::PresentStatus::presented);
        REQUIRE(device.wait(*submitted).has_value());
        auto mapped = device.mapBuffer(*readback);
        REQUIRE(mapped);
        const ReadbackPixel pixel{
            .gray = std::to_integer<std::uint8_t>(mapped->data[0]),
            .alpha = std::to_integer<std::uint8_t>(mapped->data[3]),
            .width = acquired->width,
            .height = acquired->height,
        };
        REQUIRE(device.unmapBuffer(*readback).has_value());
        REQUIRE(device.destroyBuffer(*readback).has_value());
        return pixel;
    }

    [[nodiscard]] std::filesystem::path evidenceDir()
    {
#ifdef RACHIS_PARITY_EVIDENCE_DIR
        return {RACHIS_PARITY_EVIDENCE_DIR};
#else
        return std::filesystem::temp_directory_path() / "rachis-vulkan-parity";
#endif
    }

    void writeReport(const std::string& fileName, const std::string& body)
    {
        const auto path = evidenceDir() / fileName;
        std::error_code error;
        std::filesystem::create_directories(path.parent_path(), error);
        std::ofstream stream(path, std::ios::binary);
        stream << body;
    }

    [[nodiscard]] std::string formatName(rachis::Format format)
    {
        switch (format)
        {
        case rachis::Format::bgra8Unorm:
            return "bgra8Unorm";
        case rachis::Format::bgra8UnormSrgb:
            return "bgra8UnormSrgb";
        case rachis::Format::rgba8Unorm:
            return "rgba8Unorm";
        case rachis::Format::rgba8UnormSrgb:
            return "rgba8UnormSrgb";
        case rachis::Format::rgba16Float:
            return "rgba16Float";
        default:
            return "format" + std::to_string(static_cast<int>(format));
        }
    }

    [[nodiscard]] std::string colorSpaceName(rachis::SurfaceColorSpace space)
    {
        switch (space)
        {
        case rachis::SurfaceColorSpace::srgbNonlinear:
            return "srgbNonlinear";
        case rachis::SurfaceColorSpace::extendedSrgbLinear:
            return "extendedSrgbLinear";
        case rachis::SurfaceColorSpace::hdr10St2084:
            return "hdr10St2084";
        }
        return "colorSpace" + std::to_string(static_cast<int>(space));
    }

    [[nodiscard]] std::vector<double> presentPacedFrames(rachis::Device& device,
                                                         OpenedWindow& window,
                                                         rachis::SwapchainHandle swapchain,
                                                         int frames)
    {
        auto info = device.querySwapchainPresentation(swapchain);
        REQUIRE(info);
        std::vector<double> intervals;
        auto before = device.querySwapchainFrameTiming(swapchain);
        REQUIRE(before);
        for (int frame = 0; frame < frames; ++frame)
        {
            INFO("paced frame " << frame);
            static_cast<void>(presentFrame(device, swapchain, {0.1F, 0.2F, 0.3F, 1.0F}));
            auto timing = device.querySwapchainFrameTiming(swapchain);
            REQUIRE(timing);
            CHECK(timing->inFlightPresents <= info->framesInFlightBound);
            CHECK(timing->presentsCompleted == before->presentsCompleted + static_cast<std::uint32_t>(frame + 1));
            CHECK_FALSE(timing->hidden);
            if (frame > 0)
            {
                intervals.push_back(static_cast<double>(timing->lastPresentIntervalNs) / 1.0e6);
            }
            window.pump();
        }
        return intervals;
    }

    [[nodiscard]] std::string intervalSummary(std::vector<double> intervals)
    {
        if (intervals.empty())
        {
            return "none";
        }
        std::ranges::sort(intervals);
        const auto percentile = [&intervals](double fraction) {
            const auto index = static_cast<std::size_t>(fraction * static_cast<double>(intervals.size() - 1));
            return intervals[index];
        };
        std::ostringstream text;
        text << "samples=" << intervals.size() << " medianMs=" << percentile(0.5) << " p90Ms=" << percentile(0.9)
             << " maxMs=" << intervals.back();
        return text.str();
    }

    void checkVerifiedPresentation(OpenedWindow& window)
    {
        const auto host = window.host();
        std::atomic_uint64_t errors{0};
        auto device = requireHardwareDevice(errors);
        const auto adapter = rachis::test::requireAdapterInfo(device);

        CHECK_FALSE(device.createSurface({.kind = host.kind, .displayHandle = host.display}).has_value());
        const auto surface = requireSurface(device, host, "Linux WSI verified presentation");
        const auto capabilities = requireCapabilities(device, surface);
        CHECK(capabilities.minImageCount >= 1);
        CHECK(rachis::supportsPresentMode(capabilities.presentModes, rachis::PresentMode::fifo));
        REQUIRE_FALSE(capabilities.formats.empty());
        REQUIRE(hasAny(capabilities.usages, rachis::TextureUsage::copySource));
        if (const auto live = window.liveExtent())
        {
            CHECK(capabilities.preferredWidth == live->width);
            CHECK(capabilities.preferredHeight == live->height);
        }
        else
        {
            CHECK(capabilities.preferredWidth == 0);
            CHECK(capabilities.preferredHeight == 0);
        }

        const auto extent = swapchainExtent(capabilities);
        auto zero = device.createSwapchain({
            .surface = surface,
            .width = 0,
            .height = 0,
            .imageCount = capabilities.minImageCount,
            .format = capabilities.formats.front().format,
        });
        REQUIRE_FALSE(zero.has_value());
        CHECK(zero.error().code == rachis::ErrorCode::invalidArgument);
        auto depth = device.createSwapchain({
            .surface = surface,
            .width = extent.width,
            .height = extent.height,
            .imageCount = capabilities.minImageCount,
            .format = rachis::Format::d32Float,
        });
        REQUIRE_FALSE(depth.has_value());
        CHECK(depth.error().code == rachis::ErrorCode::unsupported);

        const auto linear = firstFormat(capabilities, false);
        REQUIRE(linear);
        const auto swapchain = requireSwapchain(device,
                                                surface,
                                                capabilities,
                                                *linear,
                                                rachis::PresentMode::fifo,
                                                rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource);
        auto second = device.createSwapchain({
            .surface = surface,
            .width = extent.width,
            .height = extent.height,
            .imageCount = capabilities.minImageCount,
            .format = linear->format,
        });
        REQUIRE_FALSE(second.has_value());
        CHECK(second.error().code == rachis::ErrorCode::invalidState);

        auto info = device.querySwapchainPresentation(swapchain);
        REQUIRE(info);
        CHECK(info->format == linear->format);
        CHECK(info->presentMode == rachis::PresentMode::fifo);
        CHECK_FALSE(info->hdr);
        CHECK(info->width == extent.width);
        CHECK(info->height == extent.height);

        const auto linearPixel = presentReadbackFrame(device, swapchain);
        CAPTURE(static_cast<int>(linearPixel.gray), static_cast<int>(linearPixel.alpha));
        CHECK(linearPixel.width == extent.width);
        CHECK(linearPixel.height == extent.height);
        CHECK(linearPixel.gray >= 125);
        CHECK(linearPixel.gray <= 129);
        CHECK(linearPixel.alpha == 255);
        window.pump();

        const auto fifoIntervals = presentPacedFrames(device, window, swapchain, 16);
        REQUIRE(device.destroySwapchain(swapchain).has_value());

        std::string srgbLine = "srgb=unavailable";
        if (const auto srgb = firstFormat(capabilities, true))
        {
            const auto srgbSwapchain =
                requireSwapchain(device,
                                 surface,
                                 capabilities,
                                 *srgb,
                                 rachis::PresentMode::fifo,
                                 rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource);
            const auto srgbPixel = presentReadbackFrame(device, srgbSwapchain);
            CAPTURE(static_cast<int>(srgbPixel.gray));
            CHECK(srgbPixel.gray >= 186);
            CHECK(srgbPixel.gray <= 190);
            REQUIRE(device.destroySwapchain(srgbSwapchain).has_value());
            srgbLine = "srgb=" + formatName(srgb->format) + " clear0.5=" + std::to_string(srgbPixel.gray);
            window.pump();
        }

        std::string immediateLine = "immediate=unsupported";
        if (rachis::supportsPresentMode(capabilities.presentModes, rachis::PresentMode::immediate))
        {
            const auto immediate =
                requireSwapchain(device, surface, capabilities, *linear, rachis::PresentMode::immediate);
            auto immediateInfo = device.querySwapchainPresentation(immediate);
            REQUIRE(immediateInfo);
            CHECK(immediateInfo->presentMode == rachis::PresentMode::immediate);
            immediateLine = "immediate " + intervalSummary(presentPacedFrames(device, window, immediate, 16));
            REQUIRE(device.destroySwapchain(immediate).has_value());
        }

        REQUIRE(device.destroySurface(surface).has_value());
        CHECK(errors.load() == 0);
        CHECK(device.diagnosticCounts().errors == 0);

        std::ostringstream formats;
        for (const auto& format : capabilities.formats)
        {
            formats << ' ' << formatName(format.format) << '/' << colorSpaceName(format.colorSpace);
        }
        writeReport("vulkan-linux-" + host.name + "-present.txt",
                    "windowSystem=" + host.name + "\nadapter=" + adapter.name +
                        " vendor=" + std::to_string(adapter.vendorId) + " device=" + std::to_string(adapter.deviceId) +
                        "\nimages=" + std::to_string(info->imageCount) +
                        " minImages=" + std::to_string(capabilities.minImageCount) + " maxImages=" +
                        std::to_string(capabilities.maxImageCount) + "\nextent=" + std::to_string(extent.width) + "x" +
                        std::to_string(extent.height) + " preferred=" + std::to_string(capabilities.preferredWidth) +
                        "x" + std::to_string(capabilities.preferredHeight) + "\nformats=" + formats.str() +
                        "\nlinear=" + formatName(linear->format) + " clear0.5=" + std::to_string(linearPixel.gray) +
                        " alpha=" + std::to_string(linearPixel.alpha) + "\n" + srgbLine + "\nfifo " +
                        intervalSummary(fifoIntervals) + "\n" + immediateLine + "\n");
    }

    void checkResizeSuspendAndReplacement(OpenedWindow& window)
    {
        const auto host = window.host();
        std::atomic_uint64_t errors{0};
        auto device = requireHardwareDevice(errors);
        const auto surface = requireSurface(device, host, "Linux WSI resize");
        const auto capabilities = requireCapabilities(device, surface);
        const auto format = capabilities.formats.front();
        const auto swapchain = requireSwapchain(device, surface, capabilities, format, rachis::PresentMode::fifo);
        const auto first = presentFrame(device, swapchain);
        const auto extent = swapchainExtent(capabilities);

        const auto live = window.resizeHost({.width = extent.width + 32, .height = extent.height + 16});
        REQUIRE(live.width > 0);
        REQUIRE(live.height > 0);
        if (window.liveExtent())
        {
            const auto reported = requireCapabilities(device, surface);
            CHECK(reported.preferredWidth == live.width);
            CHECK(reported.preferredHeight == live.height);
            REQUIRE(device.resizeSwapchain(swapchain, extent.width, extent.height).has_value());
        }
        else
        {
            REQUIRE(device.resizeSwapchain(swapchain, live.width, live.height).has_value());
        }
        CHECK_FALSE(device.destroyTexture(first.texture).has_value());
        const auto resized = presentFrame(device, swapchain);
        CHECK(resized.width == live.width);
        CHECK(resized.height == live.height);
        CHECK(resized.generation != first.generation);
        window.pump();

        REQUIRE(device.resizeSwapchain(swapchain, 0, 0).has_value());
        auto suspended = device.acquireNextImage(swapchain);
        REQUIRE_FALSE(suspended.has_value());
        CHECK(suspended.error().code == rachis::ErrorCode::invalidState);
        REQUIRE(device.resizeSwapchain(swapchain, live.width, live.height).has_value());
        const auto resumed = presentFrame(device, swapchain);
        CHECK(resumed.generation != resized.generation);
        window.pump();

        auto acquired = acquireFrame(device, swapchain);
        REQUIRE(acquired);
        REQUIRE_FALSE(device.destroySwapchain(swapchain).has_value());
        REQUIRE(device.cancelSwapchainImage(swapchain).has_value());
        REQUIRE(device.destroySwapchain(swapchain).has_value());
        REQUIRE(device.destroySurface(surface).has_value());

        const auto replacementSurface = requireSurface(device, host, "Linux WSI replacement");
        const auto replacementCapabilities = requireCapabilities(device, replacementSurface);
        const auto replacement =
            requireSwapchain(device, replacementSurface, replacementCapabilities, format, rachis::PresentMode::fifo);
        static_cast<void>(presentFrame(device, replacement));
        REQUIRE(device.destroySwapchain(replacement).has_value());
        REQUIRE(device.destroySurface(replacementSurface).has_value());
        CHECK(errors.load() == 0);
        CHECK(device.diagnosticCounts().errors == 0);
    }

    void checkOneSwapchainPerWindow(OpenedWindow& window)
    {
        const auto host = window.host();
        std::atomic_uint64_t errors{0};
        auto device = requireHardwareDevice(errors);
        const auto first = requireSurface(device, host, "Linux WSI first surface");
        const auto capabilities = requireCapabilities(device, first);
        const auto format = capabilities.formats.front();
        const auto live = requireSwapchain(device, first, capabilities, format, rachis::PresentMode::fifo);
        static_cast<void>(presentFrame(device, live));
        window.pump();

        const auto second = requireSurface(device, host, "Linux WSI second surface");
        const auto extent = swapchainExtent(capabilities);
        const rachis::SwapchainDesc secondDesc{
            .surface = second,
            .width = extent.width,
            .height = extent.height,
            .imageCount = (std::min)(capabilities.maxImageCount, (std::max)(capabilities.minImageCount, 2U)),
            .format = format.format,
            .colorSpace = format.colorSpace,
            .presentMode = rachis::PresentMode::fifo,
        };
        auto refused = device.createSwapchain(secondDesc);
        REQUIRE_FALSE(refused.has_value());
        CHECK(refused.error().code == rachis::ErrorCode::invalidState);
        CHECK(refused.error().message == "The native window already has a live swapchain.");
        CHECK(window.connectionError() == 0);
        window.pump();
        CHECK(window.connectionError() == 0);

        static_cast<void>(presentFrame(device, live));
        window.pump();
        REQUIRE(device.destroySwapchain(live).has_value());

        const auto replacement = requireSwapchain(device, second, capabilities, format, rachis::PresentMode::fifo);
        static_cast<void>(presentFrame(device, replacement));
        window.pump();
        CHECK(window.connectionError() == 0);
        auto firstRefused = device.createSwapchain({
            .surface = first,
            .width = extent.width,
            .height = extent.height,
            .imageCount = secondDesc.imageCount,
            .format = format.format,
            .colorSpace = format.colorSpace,
            .presentMode = rachis::PresentMode::fifo,
        });
        REQUIRE_FALSE(firstRefused.has_value());
        CHECK(firstRefused.error().code == rachis::ErrorCode::invalidState);
        REQUIRE(device.destroySwapchain(replacement).has_value());
        REQUIRE(device.destroySurface(second).has_value());
        REQUIRE(device.destroySurface(first).has_value());
        window.pump();
        CHECK(window.connectionError() == 0);
        CHECK(errors.load() == 0);
        CHECK(device.diagnosticCounts().errors == 0);
    }

    void checkDeviceLossRecovery(OpenedWindow& window)
    {
        const auto host = window.host();
        {
            std::atomic_uint64_t errors{0};
            auto lost = requireHardwareDevice(errors);
            const auto surface = requireSurface(lost, host, "Linux WSI lost device");
            const auto capabilities = requireCapabilities(lost, surface);
            const auto swapchain =
                requireSwapchain(lost, surface, capabilities, capabilities.formats.front(), rachis::PresentMode::fifo);
            static_cast<void>(presentFrame(lost, swapchain));

            auto acquired = acquireFrame(lost, swapchain);
            REQUIRE(acquired);
            auto submitted = clearAcquired(lost, *acquired, {0.2F, 0.1F, 0.4F, 1.0F}, std::nullopt);
            REQUIRE(submitted);
            rachis::vulkan::test::nextPresentResult().store(static_cast<std::int32_t>(VK_ERROR_DEVICE_LOST),
                                                            std::memory_order_relaxed);
            auto failed = lost.present(swapchain, *submitted);
            REQUIRE_FALSE(failed.has_value());
            CHECK(failed.error().code == rachis::ErrorCode::deviceLost);
            CHECK(lost.operationalState() == rachis::DeviceOperationalState::lost);
            CHECK(errors.load() == 1);
        }

        std::atomic_uint64_t errors{0};
        auto replacement = requireHardwareDevice(errors);
        const auto surface = requireSurface(replacement, host, "Linux WSI replacement device");
        const auto capabilities = requireCapabilities(replacement, surface);
        const auto linear = firstFormat(capabilities, false);
        REQUIRE(linear);
        const auto swapchain = requireSwapchain(replacement,
                                                surface,
                                                capabilities,
                                                *linear,
                                                rachis::PresentMode::fifo,
                                                rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource);
        const auto pixel = presentReadbackFrame(replacement, swapchain);
        CAPTURE(static_cast<int>(pixel.gray));
        CHECK(pixel.gray >= 125);
        CHECK(pixel.gray <= 129);
        REQUIRE(replacement.destroySwapchain(swapchain).has_value());
        REQUIRE(replacement.destroySurface(surface).has_value());
        CHECK(errors.load() == 0);
        CHECK(replacement.diagnosticCounts().errors == 0);
    }

    void checkPresentationSoak(OpenedWindow& window, int frames)
    {
        const auto host = window.host();
        std::atomic_uint64_t errors{0};
        auto device = requireHardwareDevice(errors);
        const auto surface = requireSurface(device, host, "Linux WSI soak");
        const auto capabilities = requireCapabilities(device, surface);
        const auto mode = rachis::supportsPresentMode(capabilities.presentModes, rachis::PresentMode::immediate)
                              ? rachis::PresentMode::immediate
                              : rachis::PresentMode::fifo;
        const auto swapchain = requireSwapchain(device, surface, capabilities, capabilities.formats.front(), mode);
        const auto base = swapchainExtent(capabilities);

        const auto started = std::chrono::steady_clock::now();
        std::vector<double> intervals;
        std::uint64_t generation = 0;
        int resizes = 0;
        for (int frame = 0; frame < frames; ++frame)
        {
            INFO("soak frame " << frame);
            if (frame > 0 && frame % 60 == 0)
            {
                const auto step = static_cast<std::uint32_t>((frame / 60) % 4) * 8;
                const auto live = window.resizeHost({.width = base.width + step, .height = base.height + step});
                REQUIRE(device.resizeSwapchain(swapchain, live.width, live.height).has_value());
                ++resizes;
            }
            const auto presented = presentFrame(device, swapchain);
            if (frame % 60 == 0)
            {
                CHECK(presented.generation != generation);
                generation = presented.generation;
            }
            auto timing = device.querySwapchainFrameTiming(swapchain);
            REQUIRE(timing);
            if (frame % 60 != 0)
            {
                intervals.push_back(static_cast<double>(timing->lastPresentIntervalNs) / 1.0e6);
            }
            if (frame % 30 == 0)
            {
                window.pump();
            }
        }
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        REQUIRE(device.waitIdle().has_value());
        REQUIRE(device.destroySwapchain(swapchain).has_value());
        REQUIRE(device.destroySurface(surface).has_value());
        CHECK(errors.load() == 0);
        CHECK(device.diagnosticCounts().errors == 0);

        std::ostringstream text;
        text << "windowSystem=" << host.name
             << "\nmode=" << (mode == rachis::PresentMode::immediate ? "immediate" : "fifo") << "\nframes=" << frames
             << " resizes=" << resizes << " elapsedSeconds=" << elapsed << '\n'
             << intervalSummary(std::move(intervals)) << '\n';
        writeReport("vulkan-linux-" + host.name + "-soak.txt", text.str());
    }

} // namespace

TEST_CASE("Vulkan presents verified SDR frames to a mapped Wayland toplevel",
          "[size:medium][requires:InteractiveDesktop][requires:Wayland][requires:HardwareVulkan][rhi][vulkan][linux]["
          "surface][present]")
{
    auto window = OpenedWindow::wayland();
    checkVerifiedPresentation(window);
}

TEST_CASE("Vulkan presents verified SDR frames to a mapped XCB window",
          "[size:medium][requires:InteractiveDesktop][requires:X11][requires:HardwareVulkan][rhi][vulkan][linux]["
          "surface][present]")
{
    auto window = OpenedWindow::xcb();
    checkVerifiedPresentation(window);
}

TEST_CASE("Vulkan Wayland swapchains resize, suspend at zero extent, and accept a replacement surface",
          "[size:medium][requires:InteractiveDesktop][requires:Wayland][requires:HardwareVulkan][rhi][vulkan][linux]["
          "resize]")
{
    auto window = OpenedWindow::wayland();
    checkResizeSuspendAndReplacement(window);
}

TEST_CASE("Vulkan XCB swapchains follow the live window extent, suspend at zero extent, and accept a replacement "
          "surface",
          "[size:medium][requires:InteractiveDesktop][requires:X11][requires:HardwareVulkan][rhi][vulkan][linux]["
          "resize]")
{
    auto window = OpenedWindow::xcb();
    checkResizeSuspendAndReplacement(window);
}

TEST_CASE("Vulkan refuses a second live swapchain on the same Wayland surface",
          "[size:medium][requires:InteractiveDesktop][requires:Wayland][requires:HardwareVulkan][rhi][vulkan][linux]["
          "surface][present]")
{
    auto window = OpenedWindow::wayland();
    checkOneSwapchainPerWindow(window);
}

TEST_CASE("Vulkan refuses a second live swapchain on the same XCB window",
          "[size:medium][requires:InteractiveDesktop][requires:X11][requires:HardwareVulkan][rhi][vulkan][linux]["
          "surface][present]")
{
    auto window = OpenedWindow::xcb();
    checkOneSwapchainPerWindow(window);
}

TEST_CASE("A replacement Vulkan device presents to the Wayland toplevel after injected present loss",
          "[size:medium][requires:InteractiveDesktop][requires:Wayland][requires:HardwareVulkan][rhi][vulkan][linux]["
          "present][loss]")
{
    auto window = OpenedWindow::wayland();
    checkDeviceLossRecovery(window);
}

TEST_CASE("A replacement Vulkan device presents to the XCB window after injected present loss",
          "[size:medium][requires:InteractiveDesktop][requires:X11][requires:HardwareVulkan][rhi][vulkan][linux]["
          "present][loss]")
{
    auto window = OpenedWindow::xcb();
    checkDeviceLossRecovery(window);
}

TEST_CASE("Vulkan Wayland presentation soak resizes and presents 600 frames without errors",
          "[size:medium][requires:InteractiveDesktop][requires:Wayland][requires:HardwareVulkan][rhi][vulkan][linux]["
          "present][soak]")
{
    auto window = OpenedWindow::wayland();
    checkPresentationSoak(window, 600);
}

TEST_CASE("Vulkan XCB presentation soak resizes and presents 600 frames without errors",
          "[size:medium][requires:InteractiveDesktop][requires:X11][requires:HardwareVulkan][rhi][vulkan][linux]["
          "present][soak]")
{
    auto window = OpenedWindow::xcb();
    checkPresentationSoak(window, 600);
}
