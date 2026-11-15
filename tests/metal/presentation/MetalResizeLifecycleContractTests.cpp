#include "Metal/MetalTestHooks.h"
#include "MetalSurfaceTestHost.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <string>
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

    [[nodiscard]] rachis::SwapchainDesc swapchainDesc(rachis::SurfaceHandle surface,
                                                      std::uint32_t width,
                                                      std::uint32_t height) noexcept
    {
        return rachis::SwapchainDesc{
            .surface = surface,
            .width = width,
            .height = height,
            .imageCount = 1,
            .format = rachis::Format::bgra8Unorm,
            .usage = rachis::TextureUsage::renderTarget,
            .presentMode = rachis::PresentMode::fifo,
            .label = "Metal resize",
        };
    }

    [[nodiscard]] rachis::Result<rachis::TimelinePoint> drawAcquired(rachis::Device& device,
                                                                     const rachis::AcquiredSwapchainImage& acquired)
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
                .clearValue = {0.2F, 0.1F, 0.4F, 1.0F},
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
        auto toPresent = device.textureBarrier(*command,
                                               {
                                                   .texture = acquired.texture,
                                                   .stagesBefore = rachis::PipelineStage::renderTarget,
                                                   .stagesAfter = acquired.requiredFinalStages,
                                                   .accessBefore = rachis::Access::renderTarget,
                                                   .accessAfter = acquired.requiredFinalAccess,
                                                   .layoutBefore = rachis::TextureLayout::renderTarget,
                                                   .layoutAfter = acquired.requiredFinalLayout,
                                               });
        if (!toPresent)
        {
            return std::unexpected(std::move(toPresent.error()));
        }
        return device.submit(std::move(*command));
    }

} // namespace

TEST_CASE("Metal swapchain resize bumps generation and retires borrowed images",
          "[size:large][rhi][metal][resize][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    LayerHost const host(96, 64);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = host.layer,
        .displayHandle = host.view,
        .label = "resize surface",
    });
    REQUIRE(surface);
    auto swapchain = device.createSwapchain(swapchainDesc(*surface, 96, 64));
    REQUIRE(swapchain);

    auto first = device.acquireNextImage(*swapchain);
    REQUIRE(first);
    const auto firstGeneration = first->generation;
    const auto firstTexture = first->texture;
    auto submitted = drawAcquired(device, *first);
    REQUIRE(submitted);
    auto presented = device.present(*swapchain, *submitted);
    REQUIRE(presented);
    CHECK(*presented == rachis::PresentStatus::presented);

    REQUIRE(device.resizeSwapchain(*swapchain, 112, 72).has_value());
    CHECK_FALSE(device.destroyTexture(firstTexture).has_value());

    auto second = device.acquireNextImage(*swapchain);
    REQUIRE(second);
    CHECK(second->width == 112);
    CHECK(second->height == 72);
    CHECK(second->generation != firstGeneration);
    auto secondSubmit = drawAcquired(device, *second);
    REQUIRE(secondSubmit);
    auto secondPresent = device.present(*swapchain, *secondSubmit);
    REQUIRE(secondPresent);
    CHECK(*secondPresent == rachis::PresentStatus::presented);

    REQUIRE(device.resizeSwapchain(*swapchain, 112, 72).has_value());
    auto refreshed = device.acquireNextImage(*swapchain);
    REQUIRE(refreshed);
    CHECK(refreshed->generation != second->generation);
    REQUIRE(device.cancelSwapchainImage(*swapchain).has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal swapchain resize follows the live CAMetalLayer backing size", "[size:large][rhi][metal][resize][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    LayerHost const host(96, 64);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = host.layer,
        .displayHandle = host.view,
        .label = "live resize surface",
    });
    REQUIRE(surface);
    auto swapchain = device.createSwapchain(swapchainDesc(*surface, 96, 64));
    REQUIRE(swapchain);

    auto first = device.acquireNextImage(*swapchain);
    REQUIRE(first);
    auto submitted = drawAcquired(device, *first);
    REQUIRE(submitted);
    REQUIRE(device.present(*swapchain, *submitted).has_value());

    rxMetalTestSetDrawableSize(host.layer, 160, 90);
    auto live = device.querySurfaceCapabilities(*surface);
    REQUIRE(live);
    CHECK(live->preferredWidth == 160);
    CHECK(live->preferredHeight == 90);
    auto stale = device.acquireNextImage(*swapchain);
    REQUIRE_FALSE(stale);
    CHECK(stale.error().code == rachis::ErrorCode::invalidState);

    REQUIRE(device.resizeSwapchain(*swapchain, live->preferredWidth, live->preferredHeight).has_value());
    auto second = device.acquireNextImage(*swapchain);
    REQUIRE(second);
    CHECK(second->width == 160);
    CHECK(second->height == 90);
    auto secondSubmit = drawAcquired(device, *second);
    REQUIRE(secondSubmit);
    REQUIRE(device.present(*swapchain, *secondSubmit).has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal swapchain suspends zero-size and hidden surfaces without waiting on nextDrawable",
          "[size:large][rhi][metal][resize][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    LayerHost const host(96, 64);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = host.layer,
        .displayHandle = host.view,
        .label = "hidden surface",
    });
    REQUIRE(surface);
    auto swapchain = device.createSwapchain(swapchainDesc(*surface, 96, 64));
    REQUIRE(swapchain);

    auto marker = device.createBuffer({
        .size = 16,
        .usage = rachis::BufferUsage::copyDestination,
        .label = "resize scene marker",
    });
    REQUIRE(marker);

    REQUIRE(device.resizeSwapchain(*swapchain, 0, 0).has_value());
    auto suspendedCaps = device.querySurfaceCapabilities(*surface);
    REQUIRE(suspendedCaps);
    CHECK(suspendedCaps->preferredWidth == 0);
    CHECK(suspendedCaps->preferredHeight == 0);
    auto suspended = device.acquireNextImage(*swapchain);
    REQUIRE_FALSE(suspended.has_value());
    CHECK(suspended.error().code == rachis::ErrorCode::invalidState);
    auto timing = device.querySwapchainFrameTiming(*swapchain);
    REQUIRE(timing);
    CHECK(timing->hidden);

    REQUIRE(device.resizeSwapchain(*swapchain, 96, 64).has_value());
    rxMetalTestSetViewHidden(host.view, true);
    auto hiddenCaps = device.querySurfaceCapabilities(*surface);
    REQUIRE(hiddenCaps);
    CHECK(hiddenCaps->preferredWidth == 0);
    CHECK(hiddenCaps->preferredHeight == 0);
    auto hiddenAcquire = device.acquireNextImage(*swapchain);
    REQUIRE_FALSE(hiddenAcquire);
    CHECK(hiddenAcquire.error().code == rachis::ErrorCode::notReady);

    rxMetalTestSetViewHidden(host.view, false);
    auto restored = device.acquireNextImage(*swapchain);
    REQUIRE(restored);
    CHECK(restored->width == 96);
    CHECK(restored->height == 64);
    auto submitted = drawAcquired(device, *restored);
    REQUIRE(submitted);
    auto presented = device.present(*swapchain, *submitted);
    REQUIRE(presented);
    CHECK(*presented == rachis::PresentStatus::presented);
    REQUIRE(device.destroyBuffer(*marker).has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal swapchain acquire uses Retina backing pixels after contentsScale changes",
          "[size:large][rhi][metal][resize][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    LayerHost const host(96, 64);
    rxMetalTestSetContentsScale(host.layer, 2.0);
    rxMetalTestSetDrawableSize(host.layer, 192, 128);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = host.layer,
        .displayHandle = host.view,
        .label = "retina surface",
    });
    REQUIRE(surface);
    auto caps = device.querySurfaceCapabilities(*surface);
    REQUIRE(caps);
    CHECK(caps->preferredWidth == 192);
    CHECK(caps->preferredHeight == 128);
    auto swapchain = device.createSwapchain(swapchainDesc(*surface, 192, 128));
    REQUIRE(swapchain);
    auto acquired = device.acquireNextImage(*swapchain);
    REQUIRE(acquired);
    CHECK(acquired->width == 192);
    CHECK(acquired->height == 128);
    auto submitted = drawAcquired(device, *acquired);
    REQUIRE(submitted);
    REQUIRE(device.present(*swapchain, *submitted).has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}
