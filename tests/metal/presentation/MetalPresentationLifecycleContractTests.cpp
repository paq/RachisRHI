#include "Metal/MetalTestHooks.h"
#include "MetalSurfaceTestHost.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifndef RACHIS_TESTING
#error Metal presentation acceptance injection must stay test-only.
#endif

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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

    [[nodiscard]] rachis::SwapchainDesc fifoDesc(rachis::SurfaceHandle surface,
                                                 std::uint32_t width,
                                                 std::uint32_t height,
                                                 bool copySource) noexcept
    {
        return rachis::SwapchainDesc{
            .surface = surface,
            .width = width,
            .height = height,
            .imageCount = 1,
            .format = rachis::Format::bgra8Unorm,
            .usage = copySource ? rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource
                                : rachis::TextureUsage::renderTarget,
            .presentMode = rachis::PresentMode::fifo,
            .label = "Metal present accept",
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
                .clearValue = {0.2F, 0.4F, 0.6F, 1.0F},
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

TEST_CASE("Metal presentation lifecycle repeats acquire present resize hide close and injection on Apple GPU",
          "[size:large][rhi][metal][present][acceptance][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    REQUIRE(device.adapterInfo().has_value());
    const auto adapter = *device.adapterInfo();
    REQUIRE(adapter.backend == rachis::Backend::metal);
    REQUIRE(adapter.vendorId == 0x106b);
    REQUIRE(adapter.type != rachis::AdapterType::software);
    CAPTURE(adapter.name, adapter.vendorId, adapter.deviceId);

    const auto baseline = device.captureResourceMemorySnapshot();
    REQUIRE(baseline);
    const auto started = std::chrono::steady_clock::now();

    constexpr int cycles = 3;
    for (int cycle = 0; cycle < cycles; ++cycle)
    {
        CAPTURE(cycle);
        LayerHost const host(96, 64);
        auto surface = device.createSurface({
            .kind = rachis::SurfaceKind::caMetalLayer,
            .windowHandle = host.layer,
            .displayHandle = host.view,
            .label = "accept surface",
        });
        REQUIRE(surface);
        auto swapchain = device.createSwapchain(fifoDesc(*surface, 96, 64, true));
        REQUIRE(swapchain);

        auto first = device.acquireNextImage(*swapchain);
        REQUIRE(first);
        CHECK(first->imageIndex == 0);
        CHECK_FALSE(device.destroyTexture(first->texture).has_value());
        auto submitted = drawAcquired(device, *first);
        REQUIRE(submitted);
        auto presented = device.present(*swapchain, *submitted);
        REQUIRE(presented);
        CHECK(*presented == rachis::PresentStatus::presented);
        CHECK_FALSE(device.present(*swapchain, {}).has_value());

        REQUIRE(device.resizeSwapchain(*swapchain, 112, 72).has_value());
        auto resized = device.acquireNextImage(*swapchain);
        REQUIRE(resized);
        CHECK(resized->width == 112);
        CHECK(resized->height == 72);
        CHECK(resized->generation != first->generation);
        auto resizedSubmit = drawAcquired(device, *resized);
        REQUIRE(resizedSubmit);
        REQUIRE(device.present(*swapchain, *resizedSubmit).has_value());

        auto acquired = device.acquireNextImage(*swapchain);
        REQUIRE(acquired);
        REQUIRE(device.cancelSwapchainImage(*swapchain).has_value());
        CHECK_FALSE(device.cancelSwapchainImage(*swapchain).has_value());

        {
            rachis::metal::ScopedFailNextDrawable const failDrawable;
            auto timedOut = device.acquireNextImage(*swapchain);
            REQUIRE_FALSE(timedOut);
            CHECK(timedOut.error().code == rachis::ErrorCode::notReady);
        }

        rxMetalTestSetViewHidden(host.view, true);
        auto hidden = device.acquireNextImage(*swapchain);
        REQUIRE_FALSE(hidden);
        CHECK(hidden.error().code == rachis::ErrorCode::notReady);
        auto hiddenTiming = device.querySwapchainFrameTiming(*swapchain);
        REQUIRE(hiddenTiming);
        CHECK(hiddenTiming->hidden);
        rxMetalTestSetViewHidden(host.view, false);

        auto restored = device.acquireNextImage(*swapchain);
        REQUIRE(restored);
        auto restoredSubmit = drawAcquired(device, *restored);
        REQUIRE(restoredSubmit);
        REQUIRE(device.present(*swapchain, *restoredSubmit).has_value());

        REQUIRE(device.waitIdle().has_value());
        auto timing = device.querySwapchainFrameTiming(*swapchain);
        REQUIRE(timing);
        CHECK(timing->inFlightPresents == 0);
        CHECK(timing->presentsCompleted >= 1);
        CHECK_FALSE(timing->hidden);

        REQUIRE(device.destroySwapchain(*swapchain).has_value());
        REQUIRE(device.destroySurface(*surface).has_value());
        CHECK(device.diagnosticCounts().errors == 0);
    }

    REQUIRE(device.waitIdle().has_value());
    device.collectGarbage();
    const auto released = device.captureResourceMemorySnapshot();
    REQUIRE(released);
    CHECK(released->liveAllocationCount == baseline->liveAllocationCount);
    CHECK(released->liveAllocationBytes == baseline->liveAllocationBytes);
    CHECK(std::chrono::steady_clock::now() - started < std::chrono::seconds(30));
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal host close during an acquired lease returns surface-out-of-date and releases the drawable",
          "[size:large][rhi][metal][present][acceptance][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    LayerHost const host(80, 60);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = host.layer,
        .displayHandle = host.view,
        .label = "close during present",
    });
    REQUIRE(surface);
    auto swapchain = device.createSwapchain(fifoDesc(*surface, 80, 60, false));
    REQUIRE(swapchain);
    auto acquired = device.acquireNextImage(*swapchain);
    REQUIRE(acquired);
    CHECK(device.hasAcquiredSwapchainImage(*swapchain));
    REQUIRE(rachis::metal::notifySurfaceHostClose(*surface));
    CHECK_FALSE(device.hasAcquiredSwapchainImage(*swapchain));
    auto presented = device.present(*swapchain, {});
    REQUIRE(presented);
    CHECK(*presented == rachis::PresentStatus::surfaceOutOfDate);
    CHECK_FALSE(device.present(*swapchain, {}).has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    CHECK_FALSE(device.destroySurface(*surface).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal host close immediately drops acquired swapchain leases and returns native drawables to the layer pool",
          "[size:large][rhi][metal][present][acceptance][gpu][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    LayerHost const host(80, 60);

    std::vector<rachis::SwapchainHandle> unreleasedSwapchains;
    unreleasedSwapchains.reserve(4);

    for (int iteration = 0; iteration < 4; ++iteration)
    {
        CAPTURE(iteration);
        auto surface = device.createSurface({
            .kind = rachis::SurfaceKind::caMetalLayer,
            .windowHandle = host.layer,
            .displayHandle = host.view,
            .label = "host-close pool recycle",
        });
        REQUIRE(surface);
        auto swapchain = device.createSwapchain(fifoDesc(*surface, 80, 60, false));
        REQUIRE(swapchain);

        auto acquired = device.acquireNextImage(*swapchain);
        REQUIRE(acquired);
        CHECK(device.hasAcquiredSwapchainImage(*swapchain));

        REQUIRE(rachis::metal::notifySurfaceHostClose(*surface));
        CHECK_FALSE(device.hasAcquiredSwapchainImage(*swapchain));
        CHECK_FALSE(device.cancelSwapchainImage(*swapchain).has_value());

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command);
        auto staleBarrier = device.textureBarrier(*command,
                                                  {
                                                      .texture = acquired->texture,
                                                      .stagesBefore = acquired->initialStages,
                                                      .stagesAfter = rachis::PipelineStage::renderTarget,
                                                      .accessBefore = acquired->initialAccess,
                                                      .accessAfter = rachis::Access::renderTarget,
                                                      .layoutBefore = acquired->initialLayout,
                                                      .layoutAfter = rachis::TextureLayout::renderTarget,
                                                  });
        REQUIRE_FALSE(staleBarrier);
        CHECK(staleBarrier.error().code == rachis::ErrorCode::invalidArgument);

        unreleasedSwapchains.push_back(*swapchain);
    }

    REQUIRE(device.resizeSwapchain(unreleasedSwapchains.front(), 0, 0).has_value());

    for (const auto swapchain : unreleasedSwapchains)
    {
        REQUIRE(device.destroySwapchain(swapchain).has_value());
    }
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal device loss during presentation is terminal and a replacement device can present",
          "[size:large][rhi][metal][present][acceptance][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    LayerHost const host(64, 48);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = host.layer,
        .displayHandle = host.view,
        .label = "loss surface",
    });
    REQUIRE(surface);
    auto swapchain = device.createSwapchain(fifoDesc(*surface, 64, 48, false));
    REQUIRE(swapchain);
    auto acquired = device.acquireNextImage(*swapchain);
    REQUIRE(acquired);
    auto submitted = drawAcquired(device, *acquired);
    REQUIRE(submitted);
    REQUIRE(device.present(*swapchain, *submitted).has_value());

    auto injected = rachis::metal::injectMetalDeviceLoss();
    REQUIRE_FALSE(injected);
    CHECK(injected.error().code == rachis::ErrorCode::deviceLost);
    CHECK(device.operationalState() == rachis::DeviceOperationalState::lost);
    auto afterLoss = device.acquireNextImage(*swapchain);
    REQUIRE_FALSE(afterLoss);
    CHECK(afterLoss.error().code == rachis::ErrorCode::deviceLost);

    auto replacement = rachis::createDevice(metalDesc());
    REQUIRE(replacement);
    auto recovered = std::move(*replacement);
    LayerHost const recoveredHost(64, 48);
    auto recoveredSurface = recovered.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = recoveredHost.layer,
        .displayHandle = recoveredHost.view,
        .label = "replacement surface",
    });
    REQUIRE(recoveredSurface);
    auto recoveredSwapchain = recovered.createSwapchain(fifoDesc(*recoveredSurface, 64, 48, false));
    REQUIRE(recoveredSwapchain);
    auto recoveredImage = recovered.acquireNextImage(*recoveredSwapchain);
    REQUIRE(recoveredImage);
    auto recoveredSubmit = drawAcquired(recovered, *recoveredImage);
    REQUIRE(recoveredSubmit);
    auto recoveredPresent = recovered.present(*recoveredSwapchain, *recoveredSubmit);
    REQUIRE(recoveredPresent);
    CHECK(*recoveredPresent == rachis::PresentStatus::presented);
    REQUIRE(recovered.destroySwapchain(*recoveredSwapchain).has_value());
    REQUIRE(recovered.destroySurface(*recoveredSurface).has_value());
    CHECK(recovered.diagnosticCounts().errors == 0);
}
