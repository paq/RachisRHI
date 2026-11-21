#include "Metal/MetalTestHooks.h"
#include "MetalSurfaceTestHost.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <future>
#include <optional>
#include <thread>
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

    struct WorkerFrame final
    {
        std::optional<rachis::ErrorCode> acquireError;
        bool acquired = false;
        bool presented = false;
    };

    [[nodiscard]] bool recordClearFrame(rachis::Device& device,
                                        rachis::CommandBuffer& command,
                                        const rachis::AcquiredSwapchainImage& image)
    {
        if (!device
                 .textureBarrier(command,
                                 {
                                     .texture = image.texture,
                                     .stagesBefore = image.initialStages,
                                     .stagesAfter = rachis::PipelineStage::renderTarget,
                                     .accessBefore = image.initialAccess,
                                     .accessAfter = rachis::Access::renderTarget,
                                     .layoutBefore = image.initialLayout,
                                     .layoutAfter = rachis::TextureLayout::renderTarget,
                                 })
                 .has_value())
        {
            return false;
        }
        const std::array attachments{
            rachis::ColorAttachment{
                .view = image.renderTargetView,
                .loadOperation = rachis::LoadOperation::clear,
                .storeOperation = rachis::StoreOperation::store,
                .clearValue = {0.25F, 0.5F, 0.75F, 1.0F},
            },
        };
        if (!device.beginRendering(command, {.colorAttachments = attachments}).has_value() ||
            !device.endRendering(command).has_value())
        {
            return false;
        }
        return device
            .textureBarrier(command,
                            {
                                .texture = image.texture,
                                .stagesBefore = rachis::PipelineStage::renderTarget,
                                .stagesAfter = image.requiredFinalStages,
                                .accessBefore = rachis::Access::renderTarget,
                                .accessAfter = image.requiredFinalAccess,
                                .layoutBefore = rachis::TextureLayout::renderTarget,
                                .layoutAfter = image.requiredFinalLayout,
                            })
            .has_value();
    }

    [[nodiscard]] WorkerFrame presentFrameOnWorker(rachis::Device& device, rachis::SwapchainHandle swapchain)
    {
        WorkerFrame frame;
        std::thread worker([&]() {
            auto image = device.acquireNextImage(swapchain);
            if (!image)
            {
                frame.acquireError = image.error().code;
                return;
            }
            frame.acquired = true;
            auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
            if (!command || !recordClearFrame(device, *command, *image))
            {
                static_cast<void>(device.cancelSwapchainImage(swapchain));
                return;
            }
            auto submitted = device.submit(std::move(*command));
            if (!submitted)
            {
                static_cast<void>(device.cancelSwapchainImage(swapchain));
                return;
            }
            auto presented = device.present(swapchain, *submitted);
            frame.presented = presented.has_value() && *presented == rachis::PresentStatus::presented;
        });
        worker.join();
        return frame;
    }

} // namespace

TEST_CASE("Metal surfaces require a host-owned CAMetalLayer and reject foreign kinds",
          "[size:large][rhi][metal][surface][validation]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    auto win32 = device.createSurface({.kind = rachis::SurfaceKind::win32, .windowHandle = &device});
    REQUIRE_FALSE(win32);
    CHECK(win32.error().code == rachis::ErrorCode::unsupported);

    auto missing = device.createSurface({.kind = rachis::SurfaceKind::caMetalLayer});
    REQUIRE_FALSE(missing);
    CHECK(missing.error().code == rachis::ErrorCode::invalidArgument);

    void* const notLayer = rxMetalTestCreateNonLayerObject();
    auto wrongType = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = notLayer,
        .label = "not a layer",
    });
    rxMetalTestRelease(notLayer);
    REQUIRE_FALSE(wrongType);
    CHECK(wrongType.error().code == rachis::ErrorCode::invalidArgument);

    void* const layer = rxMetalTestCreateLayer(64, 48);
    void* const notView = rxMetalTestCreateNonLayerObject();
    auto wrongView = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = layer,
        .displayHandle = notView,
        .label = "bad view",
    });
    rxMetalTestRelease(notView);
    rxMetalTestRelease(layer);
    REQUIRE_FALSE(wrongView);
    CHECK(wrongView.error().code == rachis::ErrorCode::invalidArgument);
}

TEST_CASE("Metal surfaces reject a CAMetalLayer bound to a different device",
          "[size:large][rhi][metal][surface][validation]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    void* const nativeDevice = rachis::metal::boundNativeMetalDeviceHandle();
    REQUIRE(nativeDevice != nullptr);
    void* const foreign = rxMetalTestCreateLayerOnOtherDevice(nativeDevice, 32, 24);
    if (foreign == nullptr)
    {
        SKIP("Host exposes a single Metal device");
    }
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = foreign,
        .label = "foreign device layer",
    });
    rxMetalTestRelease(foreign);
    REQUIRE_FALSE(surface);
    CHECK(surface.error().code == rachis::ErrorCode::invalidArgument);
}

TEST_CASE("Metal CAMetalLayer surfaces publish drawable size and survive host detach",
          "[size:large][rhi][metal][surface][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    void* const layer = rxMetalTestCreateLayer(128, 96);
    void* const view = rxMetalTestCreateView(layer);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = layer,
        .displayHandle = view,
        .label = "Metal host layer",
    });
    REQUIRE(surface);
    auto caps = device.querySurfaceCapabilities(*surface);
    REQUIRE(caps);
    CHECK(caps->preferredWidth == 128);
    CHECK(caps->preferredHeight == 96);
    CHECK(rachis::hasAny(caps->usages, rachis::TextureUsage::renderTarget));
    CHECK(rachis::supportsPresentMode(caps->presentModes, rachis::PresentMode::fifo));
    CHECK(rachis::supportsPresentMode(caps->presentModes, rachis::PresentMode::immediate));
    REQUIRE_FALSE(caps->formats.empty());

    rxMetalTestSetDrawableSize(layer, 256, 144);
    auto resized = device.querySurfaceCapabilities(*surface);
    REQUIRE(resized);
    CHECK(resized->preferredWidth == 256);
    CHECK(resized->preferredHeight == 144);

    REQUIRE(device.destroySurface(*surface).has_value());
    auto stale = device.querySurfaceCapabilities(*surface);
    REQUIRE_FALSE(stale);
    CHECK(stale.error().code == rachis::ErrorCode::invalidArgument);

    auto replacement = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = layer,
        .displayHandle = view,
        .label = "Metal replacement layer",
    });
    REQUIRE(replacement);
    REQUIRE(device.destroySurface(*replacement).has_value());
    rxMetalTestRelease(view);
    rxMetalTestRelease(layer);
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal surfaces accept a host-owned layer without a view", "[size:large][rhi][metal][surface][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    void* const layer = rxMetalTestCreateLayer(64, 48);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = layer,
        .label = "layer only",
    });
    REQUIRE(surface);
    auto caps = device.querySurfaceCapabilities(*surface);
    REQUIRE(caps);
    CHECK(caps->preferredWidth == 64);
    CHECK(caps->preferredHeight == 48);
    REQUIRE(device.destroySurface(*surface).has_value());
    rxMetalTestRelease(layer);
}

TEST_CASE("Metal surfaces accept host close from another thread while queries run",
          "[size:large][rhi][metal][surface][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    void* const layer = rxMetalTestCreateLayer(96, 64);
    void* const view = rxMetalTestCreateView(layer);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = layer,
        .displayHandle = view,
        .label = "close race",
    });
    REQUIRE(surface);

    std::atomic<bool> closed{false};
    std::atomic<bool> closeSucceeded{false};
    std::thread closer([&]() {
        const auto closedSurface = rachis::metal::notifySurfaceHostClose(*surface);
        closeSucceeded.store(closedSurface.has_value(), std::memory_order_release);
        closed.store(true, std::memory_order_release);
    });
    while (!closed.load(std::memory_order_acquire))
    {
        auto caps = device.querySurfaceCapabilities(*surface);
        if (!caps)
        {
            CHECK((caps.error().code == rachis::ErrorCode::invalidArgument ||
                   caps.error().code == rachis::ErrorCode::notReady));
        }
    }
    closer.join();
    CHECK(closeSucceeded.load(std::memory_order_acquire));
    auto stale = device.querySurfaceCapabilities(*surface);
    REQUIRE_FALSE(stale);
    CHECK(stale.error().code == rachis::ErrorCode::invalidArgument);
    rxMetalTestRelease(view);
    rxMetalTestRelease(layer);
}

TEST_CASE("Metal surface test host runs worker-thread calls on the main thread and starts AppKit there",
          "[size:large][rhi][metal][surface][concurrency]")
{
    std::atomic<bool> finished{false};
    void* layer = nullptr;
    void* view = nullptr;
    std::thread worker([&] {
        layer = rxMetalTestCreateLayer(32, 24);
        view = layer == nullptr ? nullptr : rxMetalTestCreateView(layer);
        finished.store(true, std::memory_order_release);
    });
    while (!finished.load(std::memory_order_acquire))
    {
        rxMetalTestPumpMainRunLoop(0.01);
    }
    worker.join();
    CHECK(rxMetalTestAppKitStartedOnMainThread());
    CHECK(layer != nullptr);
    CHECK(view != nullptr);
    rxMetalTestRelease(view);
    rxMetalTestRelease(layer);
}

TEST_CASE("Metal surface test host skips a worker-thread call instead of deadlocking while the main thread is blocked",
          "[size:large][rhi][metal][surface][concurrency]")
{
    std::promise<void*> blockedCall;
    auto blockedResult = blockedCall.get_future();
    std::thread blockedWorker([&blockedCall] { blockedCall.set_value(rxMetalTestCreateLayer(32, 24)); });
    const bool returnedWhileBlocked = blockedResult.wait_for(std::chrono::seconds(15)) == std::future_status::ready;
    CHECK(returnedWhileBlocked);
    while (blockedResult.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    {
        rxMetalTestPumpMainRunLoop(0.01);
    }
    blockedWorker.join();
    void* const skipped = blockedResult.get();
    CHECK(skipped == nullptr);
    rxMetalTestRelease(skipped);

    std::atomic<bool> finished{false};
    void* layer = nullptr;
    std::thread worker([&] {
        layer = rxMetalTestCreateLayer(32, 24);
        finished.store(true, std::memory_order_release);
    });
    while (!finished.load(std::memory_order_acquire))
    {
        rxMetalTestPumpMainRunLoop(0.01);
    }
    worker.join();
    CHECK(rxMetalTestAppKitStartedOnMainThread());
    CHECK(layer != nullptr);
    rxMetalTestRelease(layer);
}

TEST_CASE("Metal acquire and present on a worker thread read the host NSView only on the main thread",
          "[size:large][rhi][metal][surface][gpu][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    void* const layer = rxMetalTestCreateLayer(64, 48);
    void* const view = rxMetalTestCreateProbeView(layer);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = layer,
        .displayHandle = view,
        .label = "worker presentation",
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
        .label = "worker drawable",
    });
    REQUIRE(swapchain);

    const auto visible = presentFrameOnWorker(device, *swapchain);
    CHECK(visible.acquired);
    CHECK(visible.presented);

    rxMetalTestSetViewHidden(view, true);
    rxMetalTestDrainMainQueue();
    const auto hidden = presentFrameOnWorker(device, *swapchain);
    CHECK_FALSE(hidden.acquired);
    REQUIRE(hidden.acquireError.has_value());
    CHECK(*hidden.acquireError == rachis::ErrorCode::notReady);

    rxMetalTestSetViewHidden(view, false);
    rxMetalTestDrainMainQueue();
    const auto shown = presentFrameOnWorker(device, *swapchain);
    CHECK(shown.acquired);
    CHECK(shown.presented);

    CHECK(rxMetalTestProbeViewOffMainAccesses() == 0);
    REQUIRE(device.waitIdle().has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
    rxMetalTestRelease(view);
    rxMetalTestRelease(layer);
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal surfaces leave the CAMetalLayer unchanged until a swapchain configures it",
          "[size:large][rhi][metal][surface][gpu][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    void* const layer = rxMetalTestCreateUnconfiguredLayer(64, 48);
    const auto initial = rxMetalTestQueryLayerState(layer);
    REQUIRE_FALSE(initial.hasDevice);
    REQUIRE(initial.rgba16Float);
    REQUIRE_FALSE(initial.framebufferOnly);
    REQUIRE(initial.name.empty());

    auto probe = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = layer,
        .label = "probe surface",
    });
    REQUIRE(probe);
    auto caps = device.querySurfaceCapabilities(*probe);
    REQUIRE(caps);
    const auto probed = rxMetalTestQueryLayerState(layer);
    CHECK_FALSE(probed.hasDevice);
    CHECK(probed.rgba16Float);
    CHECK_FALSE(probed.framebufferOnly);
    CHECK(probed.name.empty());
    REQUIRE(device.destroySurface(*probe).has_value());

    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = layer,
        .label = "configured surface",
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
        .label = "configured drawable",
    });
    REQUIRE(swapchain);
    const auto configured = rxMetalTestQueryLayerState(layer);
    CHECK(configured.hasDevice);
    CHECK(configured.bgra8Unorm);
    CHECK(configured.framebufferOnly);
    CHECK(configured.displaySyncEnabled);
    CHECK(configured.name == "configured surface");

    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
    rxMetalTestRelease(layer);
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal immediate swapchains turn off display sync and keep it off across resize",
          "[size:large][rhi][metal][surface][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    void* const layer = rxMetalTestCreateUnconfiguredLayer(64, 48);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = layer,
        .label = "immediate surface",
    });
    REQUIRE(surface);
    auto caps = device.querySurfaceCapabilities(*surface);
    REQUIRE(caps);
    REQUIRE(rachis::supportsPresentMode(caps->presentModes, rachis::PresentMode::immediate));

    const rachis::SwapchainDesc immediateDesc{
        .surface = *surface,
        .width = 64,
        .height = 48,
        .imageCount = 1,
        .format = rachis::Format::bgra8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .presentMode = rachis::PresentMode::immediate,
        .label = "immediate drawable",
    };
    auto swapchain = device.createSwapchain(immediateDesc);
    REQUIRE(swapchain);
    CHECK_FALSE(rxMetalTestQueryLayerState(layer).displaySyncEnabled);
    auto info = device.querySwapchainPresentation(*swapchain);
    REQUIRE(info);
    CHECK(info->presentMode == rachis::PresentMode::immediate);

    auto image = device.acquireNextImage(*swapchain);
    REQUIRE(image);
    REQUIRE(device.cancelSwapchainImage(*swapchain).has_value());

    rxMetalTestSetDrawableSize(layer, 32, 24);
    REQUIRE(device.resizeSwapchain(*swapchain, 32, 24).has_value());
    CHECK_FALSE(rxMetalTestQueryLayerState(layer).displaySyncEnabled);
    REQUIRE(device.destroySwapchain(*swapchain).has_value());

    auto fifoDesc = immediateDesc;
    fifoDesc.width = 32;
    fifoDesc.height = 24;
    fifoDesc.presentMode = rachis::PresentMode::fifo;
    auto fifo = device.createSwapchain(fifoDesc);
    REQUIRE(fifo);
    CHECK(rxMetalTestQueryLayerState(layer).displaySyncEnabled);
    REQUIRE(device.destroySwapchain(*fifo).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
    rxMetalTestRelease(layer);
    CHECK(device.diagnosticCounts().errors == 0);
}
