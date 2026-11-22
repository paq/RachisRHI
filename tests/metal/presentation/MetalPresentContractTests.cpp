#include "Metal/MetalTestHooks.h"
#include "MetalSurfaceTestHost.h"
#include "Rachis/Device.h"
#include "Rachis/TestSupport/PresentInjection.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <mach/mach.h>

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

    constexpr auto timedAcquireSlack = std::chrono::milliseconds(20);

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
            .label = "Metal drawable",
        };
    }

    enum class DrawableUse : std::uint8_t
    {
        renderTarget,
        resolveDestination,
        copySource,
    };

    struct DrawableUseResources final
    {
        rachis::TextureHandle msaa{};
        rachis::TextureViewHandle msaaView{};
        rachis::BufferHandle readback{};
    };

    [[nodiscard]] bool createDrawableUseResources(rachis::Device& device,
                                                  std::uint32_t width,
                                                  std::uint32_t height,
                                                  DrawableUseResources& resources)
    {
        auto msaa = device.createTexture({
            .width = width,
            .height = height,
            .sampleCount = 4,
            .format = rachis::Format::bgra8Unorm,
            .usage = rachis::TextureUsage::renderTarget,
            .label = "drawable use msaa scene",
        });
        if (!msaa)
        {
            return false;
        }
        resources.msaa = *msaa;
        auto msaaView = device.createTextureView({
            .texture = *msaa,
            .usage = rachis::TextureViewUsage::renderTarget,
        });
        if (!msaaView)
        {
            return false;
        }
        resources.msaaView = *msaaView;
        auto readback = device.createBuffer({
            .size = 256,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "drawable use readback",
        });
        if (!readback)
        {
            return false;
        }
        resources.readback = *readback;
        return true;
    }

    [[nodiscard]] bool destroyDrawableUseResources(rachis::Device& device, const DrawableUseResources& resources)
    {
        const bool buffer = !resources.readback.isValid() || device.destroyBuffer(resources.readback).has_value();
        const bool view = !resources.msaaView.isValid() || device.destroyTextureView(resources.msaaView).has_value();
        const bool texture = !resources.msaa.isValid() || device.destroyTexture(resources.msaa).has_value();
        return buffer && view && texture;
    }

    [[nodiscard]] bool barrierDrawable(rachis::Device& device,
                                       rachis::CommandBuffer& command,
                                       rachis::TextureHandle texture,
                                       rachis::PipelineStage stagesBefore,
                                       rachis::PipelineStage stagesAfter,
                                       rachis::Access accessBefore,
                                       rachis::Access accessAfter,
                                       rachis::TextureLayout layoutBefore,
                                       rachis::TextureLayout layoutAfter)
    {
        return device
            .textureBarrier(command,
                            {
                                .texture = texture,
                                .stagesBefore = stagesBefore,
                                .stagesAfter = stagesAfter,
                                .accessBefore = accessBefore,
                                .accessAfter = accessAfter,
                                .layoutBefore = layoutBefore,
                                .layoutAfter = layoutAfter,
                            })
            .has_value();
    }

    [[nodiscard]] bool clearView(rachis::Device& device, rachis::CommandBuffer& command, rachis::TextureViewHandle view)
    {
        const std::array attachments{
            rachis::ColorAttachment{
                .view = view,
                .loadOperation = rachis::LoadOperation::clear,
                .storeOperation = rachis::StoreOperation::store,
                .clearValue = {0.25F, 0.5F, 0.75F, 1.0F},
            },
        };
        return device.beginRendering(command, {.colorAttachments = attachments}).has_value() &&
               device.endRendering(command).has_value();
    }

    [[nodiscard]] bool resolveIntoDrawable(rachis::Device& device,
                                           rachis::CommandBuffer& command,
                                           const rachis::AcquiredSwapchainImage& image,
                                           const DrawableUseResources& resources)
    {
        return barrierDrawable(device,
                               command,
                               resources.msaa,
                               rachis::PipelineStage::none,
                               rachis::PipelineStage::renderTarget,
                               rachis::Access::none,
                               rachis::Access::renderTarget,
                               rachis::TextureLayout::undefined,
                               rachis::TextureLayout::renderTarget) &&
               clearView(device, command, resources.msaaView) &&
               barrierDrawable(device,
                               command,
                               resources.msaa,
                               rachis::PipelineStage::renderTarget,
                               rachis::PipelineStage::resolve,
                               rachis::Access::renderTarget,
                               rachis::Access::resolveSource,
                               rachis::TextureLayout::renderTarget,
                               rachis::TextureLayout::resolveSource) &&
               barrierDrawable(device,
                               command,
                               image.texture,
                               image.initialStages,
                               rachis::PipelineStage::resolve,
                               image.initialAccess,
                               rachis::Access::resolveDestination,
                               image.initialLayout,
                               rachis::TextureLayout::resolveDestination) &&
               device.resolveTexture(command, {.source = resources.msaa, .destination = image.texture}).has_value();
    }

    [[nodiscard]] bool copyFromDrawable(rachis::Device& device,
                                        rachis::CommandBuffer& command,
                                        const rachis::AcquiredSwapchainImage& image,
                                        const DrawableUseResources& resources)
    {
        const rachis::BufferTextureCopyRegion pixel{
            .bytesPerRow = 256,
            .rowsPerImage = 1,
            .extent = {.width = 1, .height = 1, .depth = 1},
        };
        return barrierDrawable(device,
                               command,
                               image.texture,
                               image.initialStages,
                               rachis::PipelineStage::copy,
                               image.initialAccess,
                               rachis::Access::copySource,
                               image.initialLayout,
                               rachis::TextureLayout::copySource) &&
               device.copyTextureToBuffer(command, image.texture, resources.readback, std::array{pixel}).has_value();
    }

    [[nodiscard]] bool recordDrawableUse(rachis::Device& device,
                                         rachis::CommandBuffer& command,
                                         const rachis::AcquiredSwapchainImage& image,
                                         DrawableUse use,
                                         const DrawableUseResources& resources)
    {
        auto stage = rachis::PipelineStage::renderTarget;
        auto access = rachis::Access::renderTarget;
        auto layout = rachis::TextureLayout::renderTarget;
        bool recorded = false;
        switch (use)
        {
        case DrawableUse::renderTarget:
            recorded = barrierDrawable(device,
                                       command,
                                       image.texture,
                                       image.initialStages,
                                       stage,
                                       image.initialAccess,
                                       access,
                                       image.initialLayout,
                                       layout) &&
                       clearView(device, command, image.renderTargetView);
            break;
        case DrawableUse::resolveDestination:
            stage = rachis::PipelineStage::resolve;
            access = rachis::Access::resolveDestination;
            layout = rachis::TextureLayout::resolveDestination;
            recorded = resolveIntoDrawable(device, command, image, resources);
            break;
        case DrawableUse::copySource:
            stage = rachis::PipelineStage::copy;
            access = rachis::Access::copySource;
            layout = rachis::TextureLayout::copySource;
            recorded = copyFromDrawable(device, command, image, resources);
            break;
        }
        return recorded && barrierDrawable(device,
                                           command,
                                           image.texture,
                                           stage,
                                           image.requiredFinalStages,
                                           access,
                                           image.requiredFinalAccess,
                                           layout,
                                           image.requiredFinalLayout);
    }

    [[nodiscard]] rachis::SwapchainDesc readableSwapchainDesc(rachis::SurfaceHandle surface,
                                                              std::uint32_t width,
                                                              std::uint32_t height) noexcept
    {
        auto desc = swapchainDesc(surface, width, height);
        desc.usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource;
        return desc;
    }

    [[nodiscard]] std::size_t currentTaskPortCount() noexcept
    {
        mach_port_name_array_t names = nullptr;
        mach_msg_type_number_t nameCount = 0;
        mach_port_type_array_t types = nullptr;
        mach_msg_type_number_t typeCount = 0;
        if (mach_port_names(mach_task_self(), &names, &nameCount, &types, &typeCount) != KERN_SUCCESS)
        {
            return 0;
        }
        vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(names), nameCount * sizeof(*names));
        vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(types), typeCount * sizeof(*types));
        return nameCount;
    }

} // namespace

TEST_CASE("Metal swapchains require one BGRA image on a live CAMetalLayer",
          "[size:large][rhi][metal][present][validation]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    LayerHost const host(64, 48);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = host.layer,
        .displayHandle = host.view,
        .label = "present surface",
    });
    REQUIRE(surface);

    auto tooMany = device.createSwapchain(rachis::SwapchainDesc{
        .surface = *surface,
        .width = 64,
        .height = 48,
        .imageCount = 3,
        .format = rachis::Format::bgra8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .presentMode = rachis::PresentMode::fifo,
    });
    REQUIRE_FALSE(tooMany);
    CHECK(tooMany.error().code == rachis::ErrorCode::invalidArgument);

    auto immediate = device.createSwapchain(rachis::SwapchainDesc{
        .surface = *surface,
        .width = 64,
        .height = 48,
        .imageCount = 1,
        .format = rachis::Format::bgra8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .presentMode = rachis::PresentMode::immediate,
    });
    REQUIRE(immediate);
    REQUIRE(device.destroySwapchain(*immediate).has_value());

    auto swapchain = device.createSwapchain(swapchainDesc(*surface, 64, 48));
    REQUIRE(swapchain);
    auto duplicate = device.createSwapchain(swapchainDesc(*surface, 64, 48));
    REQUIRE_FALSE(duplicate);
    CHECK(duplicate.error().code == rachis::ErrorCode::invalidState);
    CHECK_FALSE(device.destroySurface(*surface).has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
}

TEST_CASE("Metal layers accept one live swapchain across every surface on them",
          "[size:large][rhi][metal][present][validation]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    LayerHost const host(64, 48);
    LayerHost const otherHost(64, 48);
    const auto surfaceDesc = [](const LayerHost& layerHost) {
        return rachis::SurfaceDesc{
            .kind = rachis::SurfaceKind::caMetalLayer,
            .windowHandle = layerHost.layer,
            .displayHandle = layerHost.view,
        };
    };
    auto first = device.createSurface(surfaceDesc(host));
    REQUIRE(first);
    auto second = device.createSurface(surfaceDesc(host));
    REQUIRE(second);
    auto other = device.createSurface(surfaceDesc(otherHost));
    REQUIRE(other);

    auto live = device.createSwapchain(swapchainDesc(*first, 64, 48));
    REQUIRE(live);
    auto refused = device.createSwapchain(swapchainDesc(*second, 64, 48));
    REQUIRE_FALSE(refused);
    CHECK(refused.error().code == rachis::ErrorCode::invalidState);
    CHECK(refused.error().message == "The native window already has a live swapchain.");
    auto otherSwapchain = device.createSwapchain(swapchainDesc(*other, 64, 48));
    REQUIRE(otherSwapchain);
    REQUIRE(device.resizeSwapchain(*live, 32, 24).has_value());
    auto stillRefused = device.createSwapchain(swapchainDesc(*second, 64, 48));
    REQUIRE_FALSE(stillRefused);
    CHECK(stillRefused.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.destroySwapchain(*live).has_value());

    auto replacement = device.createSwapchain(swapchainDesc(*second, 64, 48));
    REQUIRE(replacement);
    auto firstRefused = device.createSwapchain(swapchainDesc(*first, 64, 48));
    REQUIRE_FALSE(firstRefused);
    CHECK(firstRefused.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.destroySwapchain(*replacement).has_value());

    auto recreated = device.createSwapchain(swapchainDesc(*first, 64, 48));
    REQUIRE(recreated);
    REQUIRE(device.destroySwapchain(*recreated).has_value());
    REQUIRE(device.destroySwapchain(*otherSwapchain).has_value());
    REQUIRE(device.destroySurface(*first).has_value());
    REQUIRE(device.destroySurface(*second).has_value());
    REQUIRE(device.destroySurface(*other).has_value());
}

TEST_CASE("Metal acquire/cancel/reacquire releases the drawable instead of starving the pool",
          "[size:large][rhi][metal][present][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    LayerHost const host(96, 64);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = host.layer,
        .displayHandle = host.view,
        .label = "cancel surface",
    });
    REQUIRE(surface);
    auto swapchain = device.createSwapchain(swapchainDesc(*surface, 96, 64));
    REQUIRE(swapchain);

    for (int cycle = 0; cycle < 8; ++cycle)
    {
        auto acquired = device.acquireNextImage(*swapchain);
        REQUIRE(acquired);
        CHECK(acquired->imageIndex == 0);
        CHECK(acquired->width == 96);
        CHECK(acquired->height == 64);
        CHECK(acquired->format == rachis::Format::bgra8Unorm);
        REQUIRE(device.cancelSwapchainImage(*swapchain).has_value());
        CHECK_FALSE(device.cancelSwapchainImage(*swapchain).has_value());
    }

    {
        rachis::metal::ScopedFailNextDrawable const failDrawable;
        auto timedOut = device.acquireNextImage(*swapchain);
        REQUIRE_FALSE(timedOut);
        CHECK(timedOut.error().code == rachis::ErrorCode::notReady);
    }

    {
        rachis::metal::ScopedFailNextDrawable const unconsumedDrawableFailure;
    }

    auto recovered = device.acquireNextImage(*swapchain);
    REQUIRE(recovered);
    REQUIRE(device.cancelSwapchainImage(*swapchain).has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
}

TEST_CASE("Metal timed acquire returns timeout within its bound while presents hold every drawable",
          "[size:large][rhi][metal][present][gpu][regression]")
{
    auto feedbackGate = std::make_shared<rachis::metal::MetalCommitFeedbackGate>();
    struct ScopedGateRelease final
    {
        std::shared_ptr<rachis::metal::MetalCommitFeedbackGate> gate;
        ~ScopedGateRelease() noexcept
        {
            gate->release();
        }
    } const releaseOnExit{feedbackGate};

    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    LayerHost const host(64, 48);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = host.layer,
        .displayHandle = host.view,
        .label = "timed acquire surface",
    });
    REQUIRE(surface);
    auto swapchain = device.createSwapchain(swapchainDesc(*surface, 64, 48));
    REQUIRE(swapchain);

    for (int frame = 0; frame < 3; ++frame)
    {
        auto acquired = device.acquireNextImage(*swapchain);
        REQUIRE(acquired);
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command);
        REQUIRE(device.insertDebugMarker(*command, {.label = "Held frame"}).has_value());
        if (frame == 0)
        {
            rachis::metal::holdNextCommitFeedback(feedbackGate);
        }
        auto rendered = device.submit(std::move(*command));
        REQUIRE(rendered);
        auto presented = device.present(*swapchain, *rendered);
        REQUIRE(presented);
        CHECK(*presented == rachis::PresentStatus::presented);
    }
    REQUIRE(feedbackGate->waitForEntry(std::chrono::seconds(2)));
    auto timing = device.querySwapchainFrameTiming(*swapchain);
    REQUIRE(timing);
    REQUIRE(timing->inFlightPresents == 3);

    for (const auto timeout :
         {std::chrono::milliseconds(0), std::chrono::milliseconds(20), std::chrono::milliseconds(50)})
    {
        const auto started = std::chrono::steady_clock::now();
        auto acquired = device.acquireNextImage(*swapchain, timeout);
        const auto elapsed = std::chrono::steady_clock::now() - started;
        REQUIRE_FALSE(acquired);
        CHECK(acquired.error().code == rachis::ErrorCode::timeout);
        CHECK(elapsed < timeout + timedAcquireSlack);
        CHECK_FALSE(device.hasAcquiredSwapchainImage(*swapchain));
    }

    rxMetalTestSetViewHidden(host.view, true);
    auto hidden = device.acquireNextImage(*swapchain, std::chrono::milliseconds(20));
    REQUIRE_FALSE(hidden);
    CHECK(hidden.error().code == rachis::ErrorCode::notReady);
    rxMetalTestSetViewHidden(host.view, false);

    feedbackGate->release();
    auto recovered = device.acquireNextImage(*swapchain, std::chrono::seconds(3));
    REQUIRE(recovered);
    CHECK(recovered->width == 64);
    CHECK(recovered->height == 48);
    REQUIRE(device.cancelSwapchainImage(*swapchain).has_value());

    auto untimed = device.acquireNextImage(*swapchain);
    REQUIRE(untimed);
    REQUIRE(device.cancelSwapchainImage(*swapchain).has_value());
    auto unbounded = device.acquireNextImage(*swapchain, std::chrono::nanoseconds::max());
    REQUIRE(unbounded);
    REQUIRE(device.cancelSwapchainImage(*swapchain).has_value());
    REQUIRE(device.waitIdle().has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
}

TEST_CASE("Metal timed acquire keeps a drawable that arrives late and drops it on resize",
          "[size:large][rhi][metal][present][gpu][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    LayerHost const host(64, 48);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = host.layer,
        .displayHandle = host.view,
        .label = "late drawable surface",
    });
    REQUIRE(surface);
    auto swapchain = device.createSwapchain(swapchainDesc(*surface, 64, 48));
    REQUIRE(swapchain);

    struct AcquireGateRelease final
    {
        std::shared_ptr<rachis::metal::MetalCommitFeedbackGate> gate;

        ~AcquireGateRelease()
        {
            gate->release();
            rachis::metal::holdNextDrawableAcquire(nullptr);
        }
    };
    {
        const AcquireGateRelease release{std::make_shared<rachis::metal::MetalCommitFeedbackGate>()};
        rachis::metal::holdNextDrawableAcquire(release.gate);
        for (const auto timeout : {std::chrono::milliseconds(20), std::chrono::milliseconds(50)})
        {
            const auto started = std::chrono::steady_clock::now();
            auto late = device.acquireNextImage(*swapchain, timeout);
            const auto elapsed = std::chrono::steady_clock::now() - started;
            REQUIRE_FALSE(late);
            CHECK(late.error().code == rachis::ErrorCode::timeout);
            CHECK(elapsed < timeout + timedAcquireSlack);
        }
        auto polled = device.acquireNextImage(*swapchain, std::chrono::milliseconds(0));
        REQUIRE_FALSE(polled);
        CHECK(polled.error().code == rachis::ErrorCode::timeout);
        REQUIRE(release.gate->waitForEntry(std::chrono::seconds(10)));
    }
    auto kept = device.acquireNextImage(*swapchain, std::chrono::seconds(3));
    REQUIRE(kept);
    REQUIRE(device.cancelSwapchainImage(*swapchain).has_value());

    {
        const AcquireGateRelease release{std::make_shared<rachis::metal::MetalCommitFeedbackGate>()};
        rachis::metal::holdNextDrawableAcquire(release.gate);
        auto late = device.acquireNextImage(*swapchain, std::chrono::milliseconds(0));
        REQUIRE_FALSE(late);
        CHECK(late.error().code == rachis::ErrorCode::timeout);
        REQUIRE(release.gate->waitForEntry(std::chrono::seconds(10)));
    }
    rxMetalTestSetDrawableSize(host.layer, 32, 24);
    REQUIRE(device.resizeSwapchain(*swapchain, 32, 24).has_value());
    auto resized = device.acquireNextImage(*swapchain, std::chrono::seconds(3));
    REQUIRE(resized);
    CHECK(resized->width == 32);
    CHECK(resized->height == 24);
    REQUIRE(device.cancelSwapchainImage(*swapchain).has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
}

TEST_CASE("Metal timed acquire returns timeout within its bound while another acquire holds the swapchain",
          "[size:large][rhi][metal][present][gpu][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    LayerHost const host(64, 48);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = host.layer,
        .displayHandle = host.view,
        .label = "contended acquire surface",
    });
    REQUIRE(surface);
    auto swapchain = device.createSwapchain(swapchainDesc(*surface, 64, 48));
    REQUIRE(swapchain);

    struct AcquireGateRelease final
    {
        std::shared_ptr<rachis::metal::MetalCommitFeedbackGate> gate;

        ~AcquireGateRelease()
        {
            gate->release();
            rachis::metal::holdNextDrawableAcquire(nullptr);
        }
    };

    std::optional<rachis::Result<rachis::AcquiredSwapchainImage>> parkedAcquire;
    {
        std::jthread parked;
        const AcquireGateRelease release{std::make_shared<rachis::metal::MetalCommitFeedbackGate>()};
        rachis::metal::holdNextDrawableAcquire(release.gate);
        parked = std::jthread([&] { parkedAcquire.emplace(device.acquireNextImage(*swapchain)); });
        REQUIRE(release.gate->waitForEntry(std::chrono::seconds(10)));

        // The parked acquire holds the swapchain until its drawable arrives, so these acquires wait only for the
        // swapchain lock, and that wait has to end at their deadline.
        for (const auto timeout :
             {std::chrono::milliseconds(0), std::chrono::milliseconds(20), std::chrono::milliseconds(50)})
        {
            const auto started = std::chrono::steady_clock::now();
            auto contended = device.acquireNextImage(*swapchain, timeout);
            const auto elapsed = std::chrono::steady_clock::now() - started;
            REQUIRE_FALSE(contended);
            CHECK(contended.error().code == rachis::ErrorCode::timeout);
            CHECK(elapsed < timeout + timedAcquireSlack);
        }
    }

    REQUIRE(parkedAcquire.has_value());
    if (parkedAcquire->has_value())
    {
        REQUIRE(device.cancelSwapchainImage(*swapchain).has_value());
    }
    auto recovered = device.acquireNextImage(*swapchain, std::chrono::seconds(3));
    REQUIRE(recovered);
    REQUIRE(device.cancelSwapchainImage(*swapchain).has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
}

TEST_CASE("Metal acquireNextImage releases the device lock while it waits for a drawable",
          "[size:large][rhi][metal][present][gpu][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    LayerHost const host(64, 48);
    LayerHost const otherHost(32, 24);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = host.layer,
        .displayHandle = host.view,
        .label = "blocked acquire surface",
    });
    REQUIRE(surface);
    auto otherSurface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = otherHost.layer,
        .displayHandle = otherHost.view,
        .label = "unblocked acquire surface",
    });
    REQUIRE(otherSurface);
    auto swapchain = device.createSwapchain(swapchainDesc(*surface, 64, 48));
    REQUIRE(swapchain);
    auto otherSwapchain = device.createSwapchain(swapchainDesc(*otherSurface, 32, 24));
    REQUIRE(otherSwapchain);

    bool destroyDuringWait = false;
    SECTION("a cancel on the same swapchain waits for the acquire")
    {
        destroyDuringWait = false;
    }
    SECTION("a destroy of the same swapchain waits for the acquire")
    {
        destroyDuringWait = true;
    }

    struct AcquireGateRelease final
    {
        std::shared_ptr<rachis::metal::MetalCommitFeedbackGate> gate;

        ~AcquireGateRelease()
        {
            gate->release();
            rachis::metal::holdNextDrawableAcquire(nullptr);
        }
    };

    std::optional<rachis::Result<rachis::AcquiredSwapchainImage>> acquired;
    std::optional<rachis::Result<rachis::BufferHandle>> buffer;
    std::optional<rachis::Result<rachis::SwapchainFrameTiming>> timing;
    std::optional<rachis::Result<rachis::AcquiredSwapchainImage>> otherAcquired;
    std::optional<rachis::Status> otherCancelled;
    std::optional<rachis::Status> serialized;
    std::atomic_bool probed{false};
    std::atomic_bool serializedReturned{false};
    bool parked = false;
    bool probedWhileParked = false;
    bool serializedWhileParked = true;
    {
        std::jthread acquirer;
        std::jthread serializer;
        std::jthread prober;
        const AcquireGateRelease release{std::make_shared<rachis::metal::MetalCommitFeedbackGate>()};
        rachis::metal::holdNextDrawableAcquire(release.gate);
        acquirer = std::jthread([&] { acquired.emplace(device.acquireNextImage(*swapchain)); });
        parked = release.gate->waitForEntry(std::chrono::seconds(10));
        serializer = std::jthread([&] {
            serialized.emplace(destroyDuringWait ? device.destroySwapchain(*swapchain)
                                                 : device.cancelSwapchainImage(*swapchain));
            serializedReturned.store(true, std::memory_order_release);
        });
        prober = std::jthread([&] {
            buffer.emplace(device.createBuffer({
                .size = 256,
                .usage = rachis::BufferUsage::copyDestination,
                .memoryType = rachis::MemoryType::readback,
                .label = "created during a blocked acquire",
            }));
            timing.emplace(device.querySwapchainFrameTiming(*swapchain));
            otherAcquired.emplace(device.acquireNextImage(*otherSwapchain));
            if (otherAcquired->has_value())
            {
                otherCancelled.emplace(device.cancelSwapchainImage(*otherSwapchain));
            }
            probed.store(true, std::memory_order_release);
        });
        for (int poll = 0; poll < 500 && !probed.load(std::memory_order_acquire); ++poll)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        probedWhileParked = probed.load(std::memory_order_acquire);
        serializedWhileParked = serializedReturned.load(std::memory_order_acquire);
    }

    REQUIRE(parked);
    CHECK(probedWhileParked);
    CHECK_FALSE(serializedWhileParked);
    REQUIRE(buffer.has_value());
    REQUIRE(buffer->has_value());
    REQUIRE(device.destroyBuffer(**buffer).has_value());
    REQUIRE(timing.has_value());
    CHECK(timing->has_value());
    REQUIRE(otherAcquired.has_value());
    REQUIRE(otherAcquired->has_value());
    REQUIRE(otherCancelled.has_value());
    CHECK(otherCancelled->has_value());
    REQUIRE(acquired.has_value());
    REQUIRE(acquired->has_value());
    CHECK((*acquired)->width == 64);
    CHECK((*acquired)->height == 48);
    REQUIRE(serialized.has_value());
    REQUIRE(serialized->has_value());
    CHECK_FALSE(device.cancelSwapchainImage(*swapchain).has_value());
    if (destroyDuringWait)
    {
        auto replacement = device.createSwapchain(swapchainDesc(*surface, 64, 48));
        REQUIRE(replacement);
        auto image = device.acquireNextImage(*replacement);
        REQUIRE(image);
        REQUIRE(device.cancelSwapchainImage(*replacement).has_value());
        REQUIRE(device.destroySwapchain(*replacement).has_value());
    }
    else
    {
        REQUIRE(device.destroySwapchain(*swapchain).has_value());
    }
    REQUIRE(device.destroySwapchain(*otherSwapchain).has_value());
    REQUIRE(device.destroySurface(*otherSurface).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
}

TEST_CASE("ScopedPresentInjection resets unconsumed failure on scope exit and injects present device loss when active",
          "[size:large][rhi][metal][present][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    LayerHost const host(80, 60);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = host.layer,
        .displayHandle = host.view,
        .label = "scoped present injection",
    });
    REQUIRE(surface);
    auto swapchain = device.createSwapchain(swapchainDesc(*surface, 80, 60));
    REQUIRE(swapchain);

    auto first = device.acquireNextImage(*swapchain);
    REQUIRE(first);
    auto cleanPresent = device.present(*swapchain, {});
    REQUIRE(cleanPresent);
    CHECK(*cleanPresent == rachis::PresentStatus::presented);

    auto second = device.acquireNextImage(*swapchain);
    REQUIRE(second);
    {
        rachis::test::ScopedPresentInjection const injectedLoss(-1);
        auto failedPresent = device.present(*swapchain, {});
        REQUIRE_FALSE(failedPresent);
        CHECK(failedPresent.error().code == rachis::ErrorCode::deviceLost);
    }
}

TEST_CASE("Metal present retires the public lease and completes GPU work before destroy",
          "[size:large][rhi][metal][present][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    LayerHost const host(80, 60);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = host.layer,
        .displayHandle = host.view,
        .label = "present surface",
    });
    REQUIRE(surface);
    auto swapchain = device.createSwapchain(swapchainDesc(*surface, 80, 60));
    REQUIRE(swapchain);

    auto acquired = device.acquireNextImage(*swapchain);
    REQUIRE(acquired);
    auto second = device.acquireNextImage(*swapchain);
    REQUIRE_FALSE(second);
    CHECK(second.error().code == rachis::ErrorCode::invalidState);

    auto presented = device.present(*swapchain, {});
    REQUIRE(presented);
    CHECK(*presented == rachis::PresentStatus::presented);
    CHECK_FALSE(device.present(*swapchain, {}).has_value());

    REQUIRE(device.waitIdle().has_value());
    auto timing = device.querySwapchainFrameTiming(*swapchain);
    REQUIRE(timing);
    CHECK(timing->presentsCompleted >= 1);
    CHECK(timing->inFlightPresents == 0);

    auto again = device.acquireNextImage(*swapchain);
    REQUIRE(again);
    REQUIRE(device.present(*swapchain, {}).has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal rejects destroying the borrowed render-target view of an acquired drawable",
          "[size:large][rhi][metal][present][gpu][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    LayerHost const host(64, 48);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = host.layer,
        .displayHandle = host.view,
        .label = "borrowed view surface",
    });
    REQUIRE(surface);
    auto swapchain = device.createSwapchain(swapchainDesc(*surface, 64, 48));
    REQUIRE(swapchain);

    auto image = device.acquireNextImage(*swapchain);
    REQUIRE(image);
    auto destroyedView = device.destroyTextureView(image->renderTargetView);
    REQUIRE_FALSE(destroyedView);
    CHECK(destroyedView.error().code == rachis::ErrorCode::invalidState);
    auto destroyedTexture = device.destroyTexture(image->texture);
    REQUIRE_FALSE(destroyedTexture);
    CHECK(destroyedTexture.error().code == rachis::ErrorCode::invalidState);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(recordDrawableUse(device, *command, *image, DrawableUse::renderTarget, DrawableUseResources{}));
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    auto presented = device.present(*swapchain, *submitted);
    REQUIRE(presented);
    CHECK(*presented == rachis::PresentStatus::presented);

    auto released = device.destroyTextureView(image->renderTargetView);
    REQUIRE_FALSE(released);
    CHECK(released.error().code == rachis::ErrorCode::invalidArgument);

    REQUIRE(device.waitIdle().has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal presents retire only after the commit feedback of earlier graphics work",
          "[size:large][rhi][metal][present][gpu][regression]")
{
    auto feedbackGate = std::make_shared<rachis::metal::MetalCommitFeedbackGate>();
    struct ScopedGateRelease final
    {
        std::shared_ptr<rachis::metal::MetalCommitFeedbackGate> gate;
        ~ScopedGateRelease() noexcept
        {
            gate->release();
        }
    } const releaseOnExit{feedbackGate};

    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    LayerHost const host(80, 60);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = host.layer,
        .displayHandle = host.view,
        .label = "feedback-gated present surface",
    });
    REQUIRE(surface);
    auto swapchain = device.createSwapchain(swapchainDesc(*surface, 80, 60));
    REQUIRE(swapchain);

    auto acquired = device.acquireNextImage(*swapchain);
    REQUIRE(acquired);
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device.insertDebugMarker(*command, {.label = "Feedback-gated graphics work"}).has_value());
    rachis::metal::holdNextCommitFeedback(feedbackGate);
    auto rendered = device.submit(std::move(*command));
    REQUIRE(rendered);
    auto presented = device.present(*swapchain, *rendered);
    REQUIRE(presented);
    CHECK(*presented == rachis::PresentStatus::presented);
    REQUIRE(feedbackGate->waitForEntry(std::chrono::seconds(2)));

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    auto held = device.querySwapchainFrameTiming(*swapchain);
    while (held && held->inFlightPresents != 0 && std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        device.collectGarbage();
        held = device.querySwapchainFrameTiming(*swapchain);
    }
    REQUIRE(held);
    CHECK(held->inFlightPresents == 1);
    CHECK(held->presentsCompleted == 0);
    CHECK(device.completedValue(rachis::QueueType::graphics) < rendered->value());

    feedbackGate->release();
    REQUIRE(device.waitIdle().has_value());
    auto retired = device.querySwapchainFrameTiming(*swapchain);
    REQUIRE(retired);
    CHECK(retired->inFlightPresents == 0);
    CHECK(retired->presentsCompleted == 1);
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal swapchains support multi-submit and batch-submit rendering passes on one acquired drawable",
          "[size:large][rhi][metal][present][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    constexpr std::uint32_t width = 64;
    constexpr std::uint32_t height = 48;
    LayerHost const host(width, height);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = host.layer,
        .displayHandle = host.view,
        .label = "multi-submit present surface",
    });
    REQUIRE(surface);
    auto swapchain = device.createSwapchain(rachis::SwapchainDesc{
        .surface = *surface,
        .width = width,
        .height = height,
        .imageCount = 1,
        .format = rachis::Format::bgra8Unorm,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource,
        .presentMode = rachis::PresentMode::fifo,
        .label = "Metal multi-submit drawable",
    });
    REQUIRE(swapchain);

    auto pool = device.createCommandPool({
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 4,
        .label = "Multi-submit graphics pool",
    });
    REQUIRE(pool);

    auto frame1 = device.acquireNextImage(*swapchain);
    REQUIRE(frame1);

    auto layout = device.queryTextureTransferLayout(frame1->texture,
                                                    {
                                                        .extent = {frame1->width, frame1->height, 1},
                                                    });
    REQUIRE(layout);
    auto readback = device.createBuffer({
        .size = layout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Multi-submit readback",
    });
    REQUIRE(readback);

    auto pass1 = device.acquireCommandBuffer(*pool);
    auto pass2 = device.acquireCommandBuffer(*pool);
    REQUIRE(pass1);
    REQUIRE(pass2);

    REQUIRE(device
                .textureBarrier(*pass1,
                                {
                                    .texture = frame1->texture,
                                    .stagesBefore = frame1->initialStages,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessBefore = frame1->initialAccess,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutBefore = frame1->initialLayout,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array clearAttachments{
        rachis::ColorAttachment{
            .view = frame1->renderTargetView,
            .loadOperation = rachis::LoadOperation::clear,
            .storeOperation = rachis::StoreOperation::store,
            .clearValue = {0.25F, 0.5F, 0.75F, 1.0F},
        },
    };
    REQUIRE(device.beginRendering(*pass1, {.colorAttachments = clearAttachments}).has_value());
    REQUIRE(device.endRendering(*pass1).has_value());

    const std::array loadAttachments{
        rachis::ColorAttachment{
            .view = frame1->renderTargetView,
            .loadOperation = rachis::LoadOperation::load,
            .storeOperation = rachis::StoreOperation::store,
        },
    };
    REQUIRE(device.beginRendering(*pass2, {.colorAttachments = loadAttachments}).has_value());
    REQUIRE(device.endRendering(*pass2).has_value());
    REQUIRE(device
                .textureBarrier(*pass2,
                                {
                                    .texture = frame1->texture,
                                    .stagesBefore = rachis::PipelineStage::renderTarget,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::renderTarget,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    const rachis::BufferTextureCopyRegion copyRegion{
        .bytesPerRow = layout->bytesPerRow,
        .rowsPerImage = layout->rowsPerImage,
        .extent = {.width = frame1->width, .height = frame1->height, .depth = 1},
    };
    REQUIRE(device.copyTextureToBuffer(*pass2, frame1->texture, *readback, std::array{copyRegion}).has_value());
    REQUIRE(device
                .textureBarrier(*pass2,
                                {
                                    .texture = frame1->texture,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = frame1->requiredFinalStages,
                                    .accessBefore = rachis::Access::copySource,
                                    .accessAfter = frame1->requiredFinalAccess,
                                    .layoutBefore = rachis::TextureLayout::copySource,
                                    .layoutAfter = frame1->requiredFinalLayout,
                                })
                .has_value());

    auto firstPoint = device.submit(std::move(*pass1));
    REQUIRE(firstPoint);
    auto secondPoint = device.submit(std::move(*pass2));
    REQUIRE(secondPoint);
    auto presented1 = device.present(*swapchain, *secondPoint);
    REQUIRE(presented1);
    CHECK(*presented1 == rachis::PresentStatus::presented);

    REQUIRE(device.waitIdle().has_value());
    std::array<std::byte, 4> pixel{};
    REQUIRE(device.readBuffer(*readback, 0, pixel).has_value());

    const auto nearChannel = [](std::byte actual, int expected) {
        const int v = static_cast<int>(std::to_integer<std::uint8_t>(actual));
        return v >= expected - 2 && v <= expected + 2;
    };
    CHECK(nearChannel(pixel[0], 191));
    CHECK(nearChannel(pixel[1], 128));
    CHECK(nearChannel(pixel[2], 64));
    CHECK(nearChannel(pixel[3], 255));

    auto frame2 = device.acquireNextImage(*swapchain);
    REQUIRE(frame2);
    auto batch1 = device.acquireCommandBuffer(*pool);
    auto batch2 = device.acquireCommandBuffer(*pool);
    REQUIRE(batch1);
    REQUIRE(batch2);

    REQUIRE(device
                .textureBarrier(*batch1,
                                {
                                    .texture = frame2->texture,
                                    .stagesBefore = frame2->initialStages,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessBefore = frame2->initialAccess,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutBefore = frame2->initialLayout,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array batchClear{
        rachis::ColorAttachment{
            .view = frame2->renderTargetView,
            .loadOperation = rachis::LoadOperation::clear,
            .storeOperation = rachis::StoreOperation::store,
            .clearValue = {0.1F, 0.2F, 0.3F, 1.0F},
        },
    };
    REQUIRE(device.beginRendering(*batch1, {.colorAttachments = batchClear}).has_value());
    REQUIRE(device.endRendering(*batch1).has_value());

    const std::array batchLoad{
        rachis::ColorAttachment{
            .view = frame2->renderTargetView,
            .loadOperation = rachis::LoadOperation::load,
            .storeOperation = rachis::StoreOperation::store,
        },
    };
    REQUIRE(device.beginRendering(*batch2, {.colorAttachments = batchLoad}).has_value());
    REQUIRE(device.endRendering(*batch2).has_value());
    REQUIRE(device
                .textureBarrier(*batch2,
                                {
                                    .texture = frame2->texture,
                                    .stagesBefore = rachis::PipelineStage::renderTarget,
                                    .stagesAfter = frame2->requiredFinalStages,
                                    .accessBefore = rachis::Access::renderTarget,
                                    .accessAfter = frame2->requiredFinalAccess,
                                    .layoutBefore = rachis::TextureLayout::renderTarget,
                                    .layoutAfter = frame2->requiredFinalLayout,
                                })
                .has_value());

    std::array batch{std::move(*batch1), std::move(*batch2)};
    auto batchPoint = device.submit(batch);
    REQUIRE(batchPoint);
    auto presented2 = device.present(*swapchain, *batchPoint);
    REQUIRE(presented2);
    CHECK(*presented2 == rachis::PresentStatus::presented);

    REQUIRE(device.waitIdle().has_value());
    auto timing = device.querySwapchainFrameTiming(*swapchain);
    REQUIRE(timing);
    CHECK(timing->presentsCompleted >= 2);
    CHECK(timing->inFlightPresents == 0);

    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyCommandPool(*pool).has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal single render pass waits on and renders to multiple swapchain drawables",
          "[size:large][rhi][metal][present][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    LayerHost const hostA(32, 32);
    LayerHost const hostB(32, 32);
    auto surfaceA = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = hostA.layer,
        .displayHandle = hostA.view,
        .label = "multi-swapchain A",
    });
    auto surfaceB = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = hostB.layer,
        .displayHandle = hostB.view,
        .label = "multi-swapchain B",
    });
    REQUIRE(surfaceA);
    REQUIRE(surfaceB);

    auto makeReadableSwapchain = [&](rachis::SurfaceHandle surface, std::string_view label) {
        return device.createSwapchain({
            .surface = surface,
            .width = 32,
            .height = 32,
            .imageCount = 1,
            .format = rachis::Format::bgra8Unorm,
            .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource,
            .presentMode = rachis::PresentMode::fifo,
            .label = label,
        });
    };
    auto swapchainA = makeReadableSwapchain(*surfaceA, "multi-drawable A");
    auto swapchainB = makeReadableSwapchain(*surfaceB, "multi-drawable B");
    REQUIRE(swapchainA);
    REQUIRE(swapchainB);

    auto imageA = device.acquireNextImage(*swapchainA);
    auto imageB = device.acquireNextImage(*swapchainB);
    REQUIRE(imageA);
    REQUIRE(imageB);

    const rachis::BufferTextureCopyRegion singlePixelRegion{
        .bytesPerRow = 256,
        .rowsPerImage = 1,
        .extent = {.width = 1, .height = 1, .depth = 1},
    };
    auto readbackA = device.createBuffer({
        .size = 256,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "multi-drawable readback A",
    });
    auto readbackB = device.createBuffer({
        .size = 256,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "multi-drawable readback B",
    });
    REQUIRE(readbackA);
    REQUIRE(readbackB);

    auto cmd = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(cmd);
    const std::array toRenderTarget{
        rachis::TextureBarrier{
            .texture = imageA->texture,
            .stagesBefore = imageA->initialStages,
            .stagesAfter = rachis::PipelineStage::renderTarget,
            .accessBefore = imageA->initialAccess,
            .accessAfter = rachis::Access::renderTarget,
            .layoutBefore = imageA->initialLayout,
            .layoutAfter = rachis::TextureLayout::renderTarget,
        },
        rachis::TextureBarrier{
            .texture = imageB->texture,
            .stagesBefore = imageB->initialStages,
            .stagesAfter = rachis::PipelineStage::renderTarget,
            .accessBefore = imageB->initialAccess,
            .accessAfter = rachis::Access::renderTarget,
            .layoutBefore = imageB->initialLayout,
            .layoutAfter = rachis::TextureLayout::renderTarget,
        },
    };
    REQUIRE(device.resourceBarriers(*cmd, {.textureBarriers = toRenderTarget}).has_value());

    const std::array attachments{
        rachis::ColorAttachment{
            .view = imageA->renderTargetView,
            .loadOperation = rachis::LoadOperation::clear,
            .storeOperation = rachis::StoreOperation::store,
            .clearValue = {1.0F, 0.0F, 0.0F, 1.0F},
        },
        rachis::ColorAttachment{
            .view = imageB->renderTargetView,
            .loadOperation = rachis::LoadOperation::clear,
            .storeOperation = rachis::StoreOperation::store,
            .clearValue = {0.0F, 1.0F, 0.0F, 1.0F},
        },
    };
    REQUIRE(device.beginRendering(*cmd, {.colorAttachments = attachments}).has_value());
    REQUIRE(device.endRendering(*cmd).has_value());

    const std::array toCopySource{
        rachis::TextureBarrier{
            .texture = imageA->texture,
            .stagesBefore = rachis::PipelineStage::renderTarget,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::renderTarget,
            .accessAfter = rachis::Access::copySource,
            .layoutBefore = rachis::TextureLayout::renderTarget,
            .layoutAfter = rachis::TextureLayout::copySource,
        },
        rachis::TextureBarrier{
            .texture = imageB->texture,
            .stagesBefore = rachis::PipelineStage::renderTarget,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::renderTarget,
            .accessAfter = rachis::Access::copySource,
            .layoutBefore = rachis::TextureLayout::renderTarget,
            .layoutAfter = rachis::TextureLayout::copySource,
        },
    };
    REQUIRE(device.resourceBarriers(*cmd, {.textureBarriers = toCopySource}).has_value());
    REQUIRE(device.copyTextureToBuffer(*cmd, imageA->texture, *readbackA, std::array{singlePixelRegion}).has_value());
    REQUIRE(device.copyTextureToBuffer(*cmd, imageB->texture, *readbackB, std::array{singlePixelRegion}).has_value());

    const std::array toPresent{
        rachis::TextureBarrier{
            .texture = imageA->texture,
            .stagesBefore = rachis::PipelineStage::copy,
            .stagesAfter = imageA->requiredFinalStages,
            .accessBefore = rachis::Access::copySource,
            .accessAfter = imageA->requiredFinalAccess,
            .layoutBefore = rachis::TextureLayout::copySource,
            .layoutAfter = imageA->requiredFinalLayout,
        },
        rachis::TextureBarrier{
            .texture = imageB->texture,
            .stagesBefore = rachis::PipelineStage::copy,
            .stagesAfter = imageB->requiredFinalStages,
            .accessBefore = rachis::Access::copySource,
            .accessAfter = imageB->requiredFinalAccess,
            .layoutBefore = rachis::TextureLayout::copySource,
            .layoutAfter = imageB->requiredFinalLayout,
        },
    };
    REQUIRE(device.resourceBarriers(*cmd, {.textureBarriers = toPresent}).has_value());

    auto submitted = device.submit(std::move(*cmd));
    REQUIRE(submitted);
    auto presentedA = device.present(*swapchainA, *submitted);
    auto presentedB = device.present(*swapchainB, *submitted);
    REQUIRE(presentedA);
    REQUIRE(presentedB);
    CHECK(*presentedA == rachis::PresentStatus::presented);
    CHECK(*presentedB == rachis::PresentStatus::presented);
    REQUIRE(device.waitIdle().has_value());

    std::array<std::uint8_t, 4> pixelA{};
    std::array<std::uint8_t, 4> pixelB{};
    REQUIRE(device.readBuffer(*readbackA, 0, std::as_writable_bytes(std::span{pixelA})).has_value());
    REQUIRE(device.readBuffer(*readbackB, 0, std::as_writable_bytes(std::span{pixelB})).has_value());

    CHECK(pixelA == std::array<std::uint8_t, 4>{0, 0, 255, 255});
    CHECK(pixelB == std::array<std::uint8_t, 4>{0, 255, 0, 255});

    auto timingA = device.querySwapchainFrameTiming(*swapchainA);
    auto timingB = device.querySwapchainFrameTiming(*swapchainB);
    REQUIRE(timingA);
    REQUIRE(timingB);
    CHECK(timingA->presentsCompleted >= 1);
    CHECK(timingB->presentsCompleted >= 1);
    CHECK(timingA->inFlightPresents == 0);
    CHECK(timingB->inFlightPresents == 0);

    REQUIRE(device.destroyBuffer(*readbackA).has_value());
    REQUIRE(device.destroyBuffer(*readbackB).has_value());
    REQUIRE(device.destroySwapchain(*swapchainA).has_value());
    REQUIRE(device.destroySwapchain(*swapchainB).has_value());
    REQUIRE(device.destroySurface(*surfaceA).has_value());
    REQUIRE(device.destroySurface(*surfaceB).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal resolve and copy commands on an acquired drawable wait for it before the commit",
          "[size:large][rhi][metal][present][gpu][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    constexpr std::uint32_t width = 64;
    constexpr std::uint32_t height = 48;
    LayerHost const host(width, height);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = host.layer,
        .displayHandle = host.view,
        .label = "drawable wait surface",
    });
    REQUIRE(surface);
    auto swapchain = device.createSwapchain(readableSwapchainDesc(*surface, width, height));
    REQUIRE(swapchain);
    DrawableUseResources resources;
    REQUIRE(createDrawableUseResources(device, width, height, resources));

    auto use = DrawableUse::renderTarget;
    SECTION("render target")
    {
        use = DrawableUse::renderTarget;
    }
    SECTION("resolve destination")
    {
        use = DrawableUse::resolveDestination;
    }
    SECTION("copy source")
    {
        use = DrawableUse::copySource;
    }

    auto image = device.acquireNextImage(*swapchain);
    REQUIRE(image);
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(recordDrawableUse(device, *command, *image, use, resources));

    const auto waitsBefore = rachis::metal::test::submittedDrawableWaitCount().load();
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    CHECK(rachis::metal::test::submittedDrawableWaitCount().load() == waitsBefore + 1);
    auto presented = device.present(*swapchain, *submitted);
    REQUIRE(presented);
    CHECK(*presented == rachis::PresentStatus::presented);

    REQUIRE(device.waitIdle().has_value());
    REQUIRE(destroyDrawableUseResources(device, resources));
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal rejects submitting a command buffer that recorded a drawable whose acquisition ended",
          "[size:large][rhi][metal][present][gpu][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    constexpr std::uint32_t width = 64;
    constexpr std::uint32_t height = 48;
    LayerHost const host(width, height);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = host.layer,
        .displayHandle = host.view,
        .label = "stale drawable surface",
    });
    REQUIRE(surface);
    auto swapchain = device.createSwapchain(readableSwapchainDesc(*surface, width, height));
    REQUIRE(swapchain);
    DrawableUseResources resources;
    REQUIRE(createDrawableUseResources(device, width, height, resources));

    auto use = DrawableUse::renderTarget;
    bool presentInstead = false;
    SECTION("render target, cancelled")
    {
        use = DrawableUse::renderTarget;
    }
    SECTION("resolve destination, cancelled")
    {
        use = DrawableUse::resolveDestination;
    }
    SECTION("copy source, cancelled")
    {
        use = DrawableUse::copySource;
    }
    SECTION("render target, presented without it")
    {
        use = DrawableUse::renderTarget;
        presentInstead = true;
    }

    auto image = device.acquireNextImage(*swapchain);
    REQUIRE(image);
    auto stale = rachis::test::acquireCommandBuffer(
        device, rachis::QueueType::graphics, rachis::CommandPoolFlag::allowIndividualReset);
    REQUIRE(stale);
    REQUIRE(recordDrawableUse(device, *stale, *image, use, resources));
    if (presentInstead)
    {
        auto presented = device.present(*swapchain, {});
        REQUIRE(presented);
        CHECK(*presented == rachis::PresentStatus::presented);
    }
    else
    {
        REQUIRE(device.cancelSwapchainImage(*swapchain).has_value());
    }

    const auto waitsBefore = rachis::metal::test::submittedDrawableWaitCount().load();
    auto rejected = device.submit(std::move(*stale));
    REQUIRE_FALSE(rejected);
    CHECK(rejected.error().code == rachis::ErrorCode::invalidState);
    CHECK(stale->isValid());
    CHECK(rachis::metal::test::submittedDrawableWaitCount().load() == waitsBefore);
    REQUIRE(device.resetCommandBuffer(*stale).has_value());

    auto next = device.acquireNextImage(*swapchain);
    REQUIRE(next);
    REQUIRE(recordDrawableUse(device, *stale, *next, use, resources));
    auto submitted = device.submit(std::move(*stale));
    REQUIRE(submitted);
    auto presented = device.present(*swapchain, *submitted);
    REQUIRE(presented);
    CHECK(*presented == rachis::PresentStatus::presented);

    REQUIRE(device.waitIdle().has_value());
    REQUIRE(destroyDrawableUseResources(device, resources));
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal submit and present loops keep this process's Mach port count flat",
          "[size:large][rhi][metal][present][gpu][regression]")
{
    constexpr int warmupFrames = 64;
    constexpr int soakFrames = 3000;
    constexpr std::size_t allowedGrowth = 64;

    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    LayerHost const host(64, 48);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::caMetalLayer,
        .windowHandle = host.layer,
        .displayHandle = host.view,
        .label = "soak surface",
    });
    REQUIRE(surface);
    auto swapchain = device.createSwapchain(swapchainDesc(*surface, 64, 48));
    REQUIRE(swapchain);

    const auto frame = [&](bool present) {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command);
        if (!present)
        {
            auto submitted = device.submit(std::move(*command));
            REQUIRE(submitted);
            REQUIRE(device.wait(*submitted).has_value());
            return;
        }
        auto image = device.acquireNextImage(*swapchain);
        REQUIRE(image);
        REQUIRE(recordDrawableUse(device, *command, *image, DrawableUse::renderTarget, DrawableUseResources{}));
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted);
        auto presented = device.present(*swapchain, *submitted);
        REQUIRE(presented);
    };

    for (int index = 0; index < warmupFrames; ++index)
    {
        frame(index % 2 == 0);
    }
    REQUIRE(device.waitIdle().has_value());
    const auto before = currentTaskPortCount();
    REQUIRE(before > 0);

    for (int index = 0; index < soakFrames; ++index)
    {
        frame(index % 2 == 0);
    }
    REQUIRE(device.waitIdle().has_value());
    const auto after = currentTaskPortCount();
    INFO("Mach ports before " << before << ", after " << after << " over " << soakFrames << " frames");
    CHECK(after <= before + allowedGrowth);

    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal swapchain calls leave nothing in a worker thread's autorelease pool",
          "[size:large][rhi][metal][present][gpu][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    LayerHost const host(64, 48);

    struct Measurement final
    {
        std::string operation;
        std::size_t autoreleasedObjects = 0;
        bool succeeded = false;
    };
    std::vector<Measurement> measurements;
    std::thread worker([&] {
        const auto measure = [&](std::string operation, const std::function<bool()>& body) {
            bool succeeded = false;
            const auto count = rachis::metal::autoreleasedObjectCount([&] { succeeded = body(); });
            measurements.push_back(Measurement{
                .operation = std::move(operation),
                .autoreleasedObjects = count,
                .succeeded = succeeded,
            });
            return succeeded;
        };
        rachis::Result<rachis::SurfaceHandle> surface = std::unexpected(rachis::Error{});
        rachis::Result<rachis::SwapchainHandle> swapchain = std::unexpected(rachis::Error{});
        if (!measure("createSurface and createSwapchain", [&] {
                surface = device.createSurface({
                    .kind = rachis::SurfaceKind::caMetalLayer,
                    .windowHandle = host.layer,
                    .displayHandle = host.view,
                    .label = "worker surface",
                });
                if (!surface)
                {
                    return false;
                }
                swapchain = device.createSwapchain(swapchainDesc(*surface, 64, 48));
                return swapchain.has_value();
            }))
        {
            return;
        }
        static_cast<void>(
            measure("querySurfaceCapabilities", [&] { return device.querySurfaceCapabilities(*surface).has_value(); }));
        static_cast<void>(measure("acquireNextImage and cancelSwapchainImage", [&] {
            return device.acquireNextImage(*swapchain).has_value() &&
                   device.cancelSwapchainImage(*swapchain).has_value();
        }));
        rachis::Result<rachis::AcquiredSwapchainImage> image = std::unexpected(rachis::Error{});
        static_cast<void>(measure("acquireNextImage", [&] {
            image = device.acquireNextImage(*swapchain);
            return image.has_value();
        }));
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        const bool recorded =
            image && command &&
            recordDrawableUse(device, *command, *image, DrawableUse::renderTarget, DrawableUseResources{});
        auto submitted = recorded ? device.submit(std::move(*command))
                                  : rachis::Result<rachis::TimelinePoint>{std::unexpected(rachis::Error{})};
        static_cast<void>(measure(
            "present", [&] { return submitted.has_value() && device.present(*swapchain, *submitted).has_value(); }));
        static_cast<void>(device.waitIdle());
        static_cast<void>(
            measure("resizeSwapchain", [&] { return device.resizeSwapchain(*swapchain, 32, 24).has_value(); }));
        static_cast<void>(measure("destroySwapchain and destroySurface", [&] {
            return device.destroySwapchain(*swapchain).has_value() && device.destroySurface(*surface).has_value();
        }));
    });
    worker.join();

    REQUIRE(measurements.size() == 7);
    for (const auto& measurement : measurements)
    {
        CAPTURE(measurement.operation, measurement.autoreleasedObjects);
        CHECK(measurement.succeeded);
        CHECK(measurement.autoreleasedObjects == 0);
    }
    CHECK(device.diagnosticCounts().errors == 0);
}
