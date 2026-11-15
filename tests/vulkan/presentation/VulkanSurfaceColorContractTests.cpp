#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include <catch2/catch_test_macros.hpp>

#include <Windows.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <utility>

namespace
{

    class TestWindow final
    {
    public:
        explicit TestWindow(bool visible)
            : window_(CreateWindowExW(0,
                                      L"STATIC",
                                      L"Rachis Vulkan surface color",
                                      WS_OVERLAPPEDWINDOW,
                                      CW_USEDEFAULT,
                                      CW_USEDEFAULT,
                                      96,
                                      64,
                                      nullptr,
                                      nullptr,
                                      GetModuleHandleW(nullptr),
                                      nullptr))
        {
            if (visible && window_ != nullptr)
            {
                static_cast<void>(ShowWindow(window_, SW_SHOW));
            }
        }

        ~TestWindow()
        {
            if (window_ != nullptr)
            {
                static_cast<void>(DestroyWindow(window_));
            }
        }

        TestWindow(const TestWindow&) = delete;
        TestWindow& operator=(const TestWindow&) = delete;
        TestWindow(TestWindow&&) = delete;
        TestWindow& operator=(TestWindow&&) = delete;

        [[nodiscard]] HWND get() const noexcept
        {
            return window_;
        }

    private:
        HWND window_ = nullptr;
    };

    [[nodiscard]] rachis::DeviceDesc vulkanDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::vulkan,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
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

    [[nodiscard]] rachis::Result<rachis::TimelinePoint> clearAndCopy(rachis::Device& device,
                                                                     const rachis::AcquiredSwapchainImage& acquired,
                                                                     rachis::BufferHandle readback)
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
                .clearValue = {0.5F, 0.5F, 0.5F, 1.0F},
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
        auto toCopy = device.textureBarrier(*command,
                                            {
                                                .texture = acquired.texture,
                                                .stagesBefore = rachis::PipelineStage::renderTarget,
                                                .stagesAfter = rachis::PipelineStage::copy,
                                                .accessBefore = rachis::Access::renderTarget,
                                                .accessAfter = rachis::Access::copySource,
                                                .layoutBefore = rachis::TextureLayout::renderTarget,
                                                .layoutAfter = rachis::TextureLayout::copySource,
                                            });
        if (!toCopy)
        {
            return std::unexpected(std::move(toCopy.error()));
        }
        auto copied = device.copyTextureToBuffer(*command,
                                                 {
                                                     .buffer = readback,
                                                     .texture = acquired.texture,
                                                     .extent = {acquired.width, acquired.height, 1},
                                                 });
        if (!copied)
        {
            return std::unexpected(std::move(copied.error()));
        }
        auto toPresent = device.textureBarrier(*command,
                                               {
                                                   .texture = acquired.texture,
                                                   .stagesBefore = rachis::PipelineStage::copy,
                                                   .stagesAfter = acquired.requiredFinalStages,
                                                   .accessBefore = rachis::Access::copySource,
                                                   .accessAfter = acquired.requiredFinalAccess,
                                                   .layoutBefore = rachis::TextureLayout::copySource,
                                                   .layoutAfter = acquired.requiredFinalLayout,
                                               });
        if (!toPresent)
        {
            return std::unexpected(std::move(toPresent.error()));
        }
        return device.submit(std::move(*command));
    }

} // namespace

TEST_CASE("Vulkan SDR swapchain stores declared transfer and opaque alpha",
          "[size:medium][requires:InteractiveDesktop][requires:HardwareVulkan][rhi][vulkan][color]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireAdapterInfo(device).backend == rachis::Backend::vulkan);

    TestWindow const window(true);
    REQUIRE(window.get() != nullptr);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::win32,
        .windowHandle = window.get(),
        .label = "Vulkan SDR color",
    });
    REQUIRE(surface);
    auto capabilities = device.querySurfaceCapabilities(*surface);
    REQUIRE(capabilities);
    REQUIRE(hasAny(capabilities->usages, rachis::TextureUsage::copySource));

    const auto width = capabilities->preferredWidth == 0 ? 96 : capabilities->preferredWidth;
    const auto height = capabilities->preferredHeight == 0 ? 64 : capabilities->preferredHeight;
    const auto linear = firstFormat(*capabilities, false);
    const auto srgb = firstFormat(*capabilities, true);
    REQUIRE(linear);

    auto hdr = device.createSwapchain({
        .surface = *surface,
        .width = width,
        .height = height,
        .imageCount = capabilities->minImageCount,
        .format = rachis::Format::bgra8Unorm,
        .colorSpace = rachis::SurfaceColorSpace::hdr10St2084,
        .usage = rachis::TextureUsage::renderTarget,
    });
    if (std::none_of(
            capabilities->formats.begin(), capabilities->formats.end(), [](const rachis::SurfaceFormat& format) {
                return format.colorSpace == rachis::SurfaceColorSpace::hdr10St2084;
            }))
    {
        REQUIRE_FALSE(hdr.has_value());
        CHECK(hdr.error().code == rachis::ErrorCode::unsupported);
    }

    auto swapchain = device.createSwapchain({
        .surface = *surface,
        .width = width,
        .height = height,
        .imageCount = (std::min)(capabilities->maxImageCount, (std::max)(capabilities->minImageCount, 2U)),
        .format = linear->format,
        .colorSpace = linear->colorSpace,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource,
        .presentMode = rachis::PresentMode::fifo,
        .label = "Vulkan linear SDR",
    });
    REQUIRE(swapchain);
    auto info = device.querySwapchainPresentation(*swapchain);
    REQUIRE(info);
    CHECK(info->format == linear->format);
    CHECK(info->colorSpace == rachis::SurfaceColorSpace::srgbNonlinear);
    CHECK(info->presentMode == rachis::PresentMode::fifo);
    CHECK_FALSE(info->hdr);
    CHECK(info->framesInFlightBound == info->imageCount);

    auto acquired = device.acquireNextImage(*swapchain);
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
    });
    REQUIRE(readback);
    auto submitted = clearAndCopy(device, *acquired, *readback);
    REQUIRE(submitted);
    auto presented = device.present(*swapchain, *submitted);
    REQUIRE(presented);
    CHECK(*presented == rachis::PresentStatus::presented);
    REQUIRE(device.wait(*submitted).has_value());
    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped);
    const auto gray = std::to_integer<std::uint8_t>(mapped->data[0]);
    const auto alpha = std::to_integer<std::uint8_t>(mapped->data[3]);
    CAPTURE(gray, alpha);
    CHECK(gray >= 125);
    CHECK(gray <= 129);
    CHECK(alpha == 255);
    REQUIRE(device.unmapBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());

    if (srgb)
    {
        auto srgbSwapchain = device.createSwapchain({
            .surface = *surface,
            .width = width,
            .height = height,
            .imageCount = (std::min)(capabilities->maxImageCount, (std::max)(capabilities->minImageCount, 2U)),
            .format = srgb->format,
            .colorSpace = srgb->colorSpace,
            .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource,
            .presentMode = rachis::PresentMode::fifo,
            .label = "Vulkan sRGB SDR",
        });
        REQUIRE(srgbSwapchain);
        auto srgbAcquired = device.acquireNextImage(*srgbSwapchain);
        REQUIRE(srgbAcquired);
        auto srgbLayout =
            device.queryTextureTransferLayout(srgbAcquired->texture,
                                              {
                                                  .extent = {srgbAcquired->width, srgbAcquired->height, 1},
                                              });
        REQUIRE(srgbLayout);
        auto srgbRead = device.createBuffer({
            .size = srgbLayout->requiredBufferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
        });
        REQUIRE(srgbRead);
        auto srgbSubmitted = clearAndCopy(device, *srgbAcquired, *srgbRead);
        REQUIRE(srgbSubmitted);
        REQUIRE(device.present(*srgbSwapchain, *srgbSubmitted).has_value());
        REQUIRE(device.wait(*srgbSubmitted).has_value());
        auto srgbMapped = device.mapBuffer(*srgbRead);
        REQUIRE(srgbMapped);
        const auto srgbGray = std::to_integer<std::uint8_t>(srgbMapped->data[0]);
        CAPTURE(srgbGray);
        CHECK(srgbGray >= 186);
        CHECK(srgbGray <= 190);
        REQUIRE(device.unmapBuffer(*srgbRead).has_value());
        REQUIRE(device.destroyBuffer(*srgbRead).has_value());
        REQUIRE(device.destroySwapchain(*srgbSwapchain).has_value());
    }

    REQUIRE(device.destroySurface(*surface).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Vulkan FIFO and immediate presents stay within the image-count bound",
          "[size:medium][requires:InteractiveDesktop][requires:HardwareVulkan][rhi][vulkan][color]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    const auto runMode = [&](rachis::PresentMode mode, bool visible) {
        TestWindow const window(visible);
        REQUIRE(window.get() != nullptr);
        auto surface = device.createSurface({
            .kind = rachis::SurfaceKind::win32,
            .windowHandle = window.get(),
        });
        REQUIRE(surface);
        auto capabilities = device.querySurfaceCapabilities(*surface);
        REQUIRE(capabilities);
        if (!rachis::supportsPresentMode(capabilities->presentModes, mode))
        {
            REQUIRE(device.destroySurface(*surface).has_value());
            return;
        }
        const auto width = capabilities->preferredWidth == 0 ? 96 : capabilities->preferredWidth;
        const auto height = capabilities->preferredHeight == 0 ? 64 : capabilities->preferredHeight;
        const auto format = capabilities->formats.front();
        auto swapchain = device.createSwapchain({
            .surface = *surface,
            .width = width,
            .height = height,
            .imageCount = (std::min)(capabilities->maxImageCount, (std::max)(capabilities->minImageCount, 2U)),
            .format = format.format,
            .colorSpace = format.colorSpace,
            .usage = rachis::TextureUsage::renderTarget,
            .presentMode = mode,
        });
        REQUIRE(swapchain);
        auto info = device.querySwapchainPresentation(*swapchain);
        REQUIRE(info);
        CHECK(info->presentMode == mode);
        for (int frame = 0; frame < 8; ++frame)
        {
            auto acquired = device.acquireNextImage(*swapchain);
            REQUIRE(acquired);
            auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
            REQUIRE(command);
            REQUIRE(device
                        .textureBarrier(*command,
                                        {
                                            .texture = acquired->texture,
                                            .stagesBefore = acquired->initialStages,
                                            .stagesAfter = rachis::PipelineStage::renderTarget,
                                            .accessBefore = acquired->initialAccess,
                                            .accessAfter = rachis::Access::renderTarget,
                                            .layoutBefore = acquired->initialLayout,
                                            .layoutAfter = rachis::TextureLayout::renderTarget,
                                        })
                        .has_value());
            const std::array attachments{
                rachis::ColorAttachment{
                    .view = acquired->renderTargetView,
                    .loadOperation = rachis::LoadOperation::clear,
                    .clearValue = {0.1F, 0.2F, 0.3F, 1.0F},
                },
            };
            REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
            REQUIRE(device.endRendering(*command).has_value());
            REQUIRE(device
                        .textureBarrier(*command,
                                        {
                                            .texture = acquired->texture,
                                            .stagesBefore = rachis::PipelineStage::renderTarget,
                                            .stagesAfter = acquired->requiredFinalStages,
                                            .accessBefore = rachis::Access::renderTarget,
                                            .accessAfter = acquired->requiredFinalAccess,
                                            .layoutBefore = rachis::TextureLayout::renderTarget,
                                            .layoutAfter = acquired->requiredFinalLayout,
                                        })
                        .has_value());
            auto submitted = device.submit(std::move(*command));
            REQUIRE(submitted);
            auto presented = device.present(*swapchain, *submitted);
            REQUIRE(presented);
            if (visible)
            {
                CHECK(*presented == rachis::PresentStatus::presented);
            }
            auto timing = device.querySwapchainFrameTiming(*swapchain);
            REQUIRE(timing);
            CHECK(timing->inFlightPresents <= info->framesInFlightBound);
            CHECK(std::cmp_equal(timing->presentsCompleted, frame + 1));
            CHECK(timing->hidden == !visible);
            if (!visible)
            {
                CHECK(*presented == rachis::PresentStatus::occluded);
            }
        }
        REQUIRE(device.destroySwapchain(*swapchain).has_value());
        REQUIRE(device.destroySurface(*surface).has_value());
    };

    runMode(rachis::PresentMode::fifo, true);
    runMode(rachis::PresentMode::immediate, true);
    runMode(rachis::PresentMode::fifo, false);
    CHECK(device.diagnosticCounts().errors == 0);
}
