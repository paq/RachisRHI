#include "MetalSurfaceTestHost.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <utility>

namespace
{

    [[nodiscard]] rachis::DeviceDesc metalDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
    }

    struct LayerHost final
    {
        void* layer = nullptr;
        void* view = nullptr;

        explicit LayerHost(std::uint32_t width, std::uint32_t height)
            : layer(rxMetalTestCreateLayer(width, height)), view(rxMetalTestCreateView(layer))
        {
        }

        ~LayerHost()
        {
            rxMetalTestRelease(view);
            rxMetalTestRelease(layer);
        }

        LayerHost(const LayerHost&) = delete;
        LayerHost& operator=(const LayerHost&) = delete;
        LayerHost(LayerHost&&) = delete;
        LayerHost& operator=(LayerHost&&) = delete;
    };

    [[nodiscard]] bool isSrgb(rachis::Format format) noexcept
    {
        return format == rachis::Format::bgra8UnormSrgb || format == rachis::Format::rgba8UnormSrgb;
    }

    [[nodiscard]] rachis::SwapchainDesc colorSwapchain(rachis::SurfaceHandle surface,
                                                       std::uint32_t width,
                                                       std::uint32_t height,
                                                       rachis::Format format) noexcept
    {
        return rachis::SwapchainDesc{
            .surface = surface,
            .width = width,
            .height = height,
            .imageCount = 1,
            .format = format,
            .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource,
            .presentMode = rachis::PresentMode::fifo,
            .label = "Metal SDR color",
        };
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
        auto layout = device.queryTextureTransferLayout(acquired.texture,
                                                        {
                                                            .extent = {acquired.width, acquired.height, 1},
                                                        });
        if (!layout)
        {
            return std::unexpected(std::move(layout.error()));
        }
        const rachis::BufferTextureCopyRegion region{
            .bytesPerRow = layout->bytesPerRow,
            .rowsPerImage = layout->rowsPerImage,
            .extent = {.width = acquired.width, .height = acquired.height, .depth = 1},
        };
        auto copied = device.copyTextureToBuffer(*command, acquired.texture, readback, std::array{region});
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

TEST_CASE("Metal SDR swapchain stores declared transfer and opaque alpha", "[size:large][rhi][metal][color][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    LayerHost const host(96, 64);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = host.layer,
        .displayHandle = host.view,
        .label = "SDR color",
    });
    REQUIRE(surface);
    auto capabilities = device.querySurfaceCapabilities(*surface);
    REQUIRE(capabilities);
    REQUIRE(rachis::hasAny(capabilities->usages, rachis::TextureUsage::copySource));
    REQUIRE(rachis::hasAny(capabilities->usages, rachis::TextureUsage::renderTarget));

    auto hdr = device.createSwapchain({
        .surface = *surface,
        .width = 96,
        .height = 64,
        .imageCount = 1,
        .format = rachis::Format::bgra8Unorm,
        .colorSpace = rachis::SurfaceColorSpace::hdr10St2084,
        .usage = rachis::TextureUsage::renderTarget,
        .presentMode = rachis::PresentMode::fifo,
    });
    REQUIRE_FALSE(hdr);
    CHECK(hdr.error().code == rachis::ErrorCode::unsupported);

    auto edr = device.createSwapchain({
        .surface = *surface,
        .width = 96,
        .height = 64,
        .imageCount = 1,
        .format = rachis::Format::bgra8Unorm,
        .colorSpace = rachis::SurfaceColorSpace::extendedSrgbLinear,
        .usage = rachis::TextureUsage::renderTarget,
        .presentMode = rachis::PresentMode::fifo,
    });
    REQUIRE_FALSE(edr);
    CHECK(edr.error().code == rachis::ErrorCode::unsupported);

    const auto readOne = [&](rachis::Format format, std::uint8_t minGray, std::uint8_t maxGray) {
        auto swapchain = device.createSwapchain(colorSwapchain(*surface, 96, 64, format));
        REQUIRE(swapchain);
        auto info = device.querySwapchainPresentation(*swapchain);
        REQUIRE(info);
        CHECK(info->format == format);
        CHECK(info->colorSpace == rachis::SurfaceColorSpace::srgbNonlinear);
        CHECK(info->presentMode == rachis::PresentMode::fifo);
        CHECK_FALSE(info->hdr);

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
        CAPTURE(format, gray, alpha, isSrgb(format));
        CHECK(gray >= minGray);
        CHECK(gray <= maxGray);
        CHECK(alpha == 255);
        REQUIRE(device.unmapBuffer(*readback).has_value());
        REQUIRE(device.destroyBuffer(*readback).has_value());
        REQUIRE(device.destroySwapchain(*swapchain).has_value());
    };

    readOne(rachis::Format::bgra8Unorm, 125, 129);
    readOne(rachis::Format::bgra8UnormSrgb, 186, 190);
    REQUIRE(device.destroySurface(*surface).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal FIFO presents stay within the native occupancy bound and record display timing",
          "[size:large][rhi][metal][color][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    LayerHost const host(80, 60);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = host.layer,
        .displayHandle = host.view,
        .label = "fifo pacing",
    });
    REQUIRE(surface);
    auto swapchain = device.createSwapchain({
        .surface = *surface,
        .width = 80,
        .height = 60,
        .imageCount = 1,
        .format = rachis::Format::bgra8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .presentMode = rachis::PresentMode::fifo,
        .label = "fifo",
    });
    REQUIRE(swapchain);
    auto info = device.querySwapchainPresentation(*swapchain);
    REQUIRE(info);
    CHECK(info->presentMode == rachis::PresentMode::fifo);

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
        CHECK(*presented == rachis::PresentStatus::presented);
        auto timing = device.querySwapchainFrameTiming(*swapchain);
        REQUIRE(timing);
        CHECK(timing->inFlightPresents <= info->framesInFlightBound);
        CHECK_FALSE(timing->hidden);
        if (frame >= 1)
        {
            CHECK(timing->lastPresentIntervalNs > 0);
        }
    }
    REQUIRE(device.waitIdle().has_value());
    auto completed = device.querySwapchainFrameTiming(*swapchain);
    REQUIRE(completed);
    CHECK(completed->presentsCompleted >= 1);
    CHECK(completed->inFlightPresents == 0);
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal hidden surfaces reject acquire without waiting on nextDrawable",
          "[size:large][rhi][metal][color][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    LayerHost const host(64, 48);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = host.layer,
        .displayHandle = host.view,
        .label = "hidden pacing",
    });
    REQUIRE(surface);
    auto swapchain = device.createSwapchain({
        .surface = *surface,
        .width = 64,
        .height = 48,
        .imageCount = 1,
        .format = rachis::Format::bgra8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .presentMode = rachis::PresentMode::fifo,
    });
    REQUIRE(swapchain);
    rxMetalTestSetViewHidden(host.view, true);
    const auto started = std::chrono::steady_clock::now();
    for (int attempt = 0; attempt < 8; ++attempt)
    {
        auto acquired = device.acquireNextImage(*swapchain);
        REQUIRE_FALSE(acquired);
        CHECK(acquired.error().code == rachis::ErrorCode::notReady);
        auto timing = device.querySwapchainFrameTiming(*swapchain);
        REQUIRE(timing);
        CHECK(timing->hidden);
    }
    const auto elapsed = std::chrono::steady_clock::now() - started;
    CHECK(elapsed < std::chrono::milliseconds(250));
    rxMetalTestSetViewHidden(host.view, false);
    auto restored = device.acquireNextImage(*swapchain);
    REQUIRE(restored);
    REQUIRE(device.cancelSwapchainImage(*swapchain).has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}
