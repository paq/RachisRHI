#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include "Internal/CommandBufferAccess.h"
#include "Internal/VulkanCommands.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

    [[nodiscard]] rachis::DeviceDesc vulkanDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::vulkan,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
    }

} // namespace

TEST_CASE("Vulkan command pools are bounded and recycle abandoned recordings",
          "[size:medium][rhi][vulkan][command-pool]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    auto zeroCapacity = device.createCommandPool(rachis::CommandPoolDesc{});
    REQUIRE_FALSE(zeroCapacity);
    CHECK(zeroCapacity.error().code == rachis::ErrorCode::invalidArgument);

    auto invalidQueue = device.createCommandPool(rachis::CommandPoolDesc{
        .queueType = static_cast<rachis::QueueType>(99),
        .maxCommandBuffers = 1,
    });
    REQUIRE_FALSE(invalidQueue);
    CHECK(invalidQueue.error().code == rachis::ErrorCode::invalidArgument);

    const std::string oversizedLabel(rachis::maxDebugLabelBytes + 1, 'x');
    auto oversized = device.createCommandPool(rachis::CommandPoolDesc{
        .maxCommandBuffers = 1,
        .label = oversizedLabel,
    });
    REQUIRE_FALSE(oversized);
    CHECK(oversized.error().code == rachis::ErrorCode::invalidArgument);

    auto unknownFlags = device.createCommandPool(rachis::CommandPoolDesc{
        .maxCommandBuffers = 1,
        .flags = static_cast<rachis::CommandPoolFlag>(1U << 31U),
    });
    REQUIRE_FALSE(unknownFlags);
    CHECK(unknownFlags.error().code == rachis::ErrorCode::invalidArgument);

    auto pool = device.createCommandPool(rachis::CommandPoolDesc{
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .label = "Vulkan bounded pool",
    });
    REQUIRE(pool);

    auto recording = device.acquireCommandBuffer(*pool);
    REQUIRE(recording);
    CHECK(recording->queueType() == rachis::QueueType::graphics);
    auto exhausted = device.acquireCommandBuffer(*pool);
    REQUIRE_FALSE(exhausted);
    CHECK(exhausted.error().code == rachis::ErrorCode::resourceExhausted);

    recording = std::unexpected(rachis::Error{});
    auto recycled = device.acquireCommandBuffer(*pool);
    REQUIRE(recycled);
    recycled = std::unexpected(rachis::Error{});

    REQUIRE(device.destroyCommandPool(*pool).has_value());
    auto staleAcquire = device.acquireCommandBuffer(*pool);
    REQUIRE_FALSE(staleAcquire);
    CHECK(staleAcquire.error().code == rachis::ErrorCode::invalidArgument);
    auto staleDestroy = device.destroyCommandPool(*pool);
    REQUIRE_FALSE(staleDestroy);
    CHECK(staleDestroy.error().code == rachis::ErrorCode::invalidArgument);

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan command-pool reset rejects open scopes and in-flight leases",
          "[size:medium][rhi][vulkan][command-pool][reset]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    auto noIndividualReset = device.createCommandPool({
        .queueType = rachis::QueueType::copy,
        .maxCommandBuffers = 1,
    });
    REQUIRE(noIndividualReset);
    auto noResetCommand = device.acquireCommandBuffer(*noIndividualReset);
    REQUIRE(noResetCommand);
    auto rejectedIndividual = device.resetCommandBuffer(*noResetCommand);
    REQUIRE_FALSE(rejectedIndividual);
    CHECK(rejectedIndividual.error().code == rachis::ErrorCode::invalidState);
    noResetCommand = std::unexpected(rachis::Error{});
    REQUIRE(device.destroyCommandPool(*noIndividualReset).has_value());

    auto pool = device.createCommandPool({
        .queueType = rachis::QueueType::copy,
        .maxCommandBuffers = 2,
        .label = "Vulkan resettable pool",
        .flags = rachis::CommandPoolFlag::allowIndividualReset,
    });
    REQUIRE(pool);
    auto first = device.acquireCommandBuffer(*pool);
    auto second = device.acquireCommandBuffer(*pool);
    REQUIRE(first);
    REQUIRE(second);
    REQUIRE(device.resetCommandBuffer(*first).has_value());
    REQUIRE(first->isValid());

    REQUIRE(device.beginDebugGroup(*first, {.label = "Reset validation"}).has_value());
    auto openScopeReset = device.resetCommandPool(*pool);
    REQUIRE_FALSE(openScopeReset);
    CHECK(openScopeReset.error().code == rachis::ErrorCode::invalidState);
    auto openBufferReset = device.resetCommandBuffer(*first);
    REQUIRE_FALSE(openBufferReset);
    CHECK(openBufferReset.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.endDebugGroup(*first).has_value());
    REQUIRE(device.insertDebugMarker(*first, {.label = "After group"}).has_value());

    REQUIRE(device.resetCommandPool(*pool).has_value());
    REQUIRE(first->isValid());
    REQUIRE(second->isValid());

    REQUIRE(rachis::vulkan::markVulkanCommandBufferInFlight(*first, true).has_value());
    auto inFlightReset = device.resetCommandPool(*pool);
    REQUIRE_FALSE(inFlightReset);
    CHECK(inFlightReset.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(rachis::vulkan::markVulkanCommandBufferInFlight(*first, false).has_value());
    REQUIRE(device.resetCommandPool(*pool).has_value());

    first = std::unexpected(rachis::Error{});
    second = std::unexpected(rachis::Error{});
    REQUIRE(device.destroyCommandPool(*pool).has_value());
    auto staleReset = device.resetCommandPool(*pool);
    REQUIRE_FALSE(staleReset);
    CHECK(staleReset.error().code == rachis::ErrorCode::invalidArgument);

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan command-pool reset waits while a live recording's state lock is held",
          "[size:medium][rhi][vulkan][command-pool][reset]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    auto pool = device.createCommandPool({
        .queueType = rachis::QueueType::copy,
        .maxCommandBuffers = 1,
        .label = "Vulkan locked recording-state pool",
    });
    REQUIRE(pool);
    auto command = device.acquireCommandBuffer(*pool);
    REQUIRE(command);
    const auto* native = rachis::detail::CommandBufferAccess::get(*command);
    REQUIRE(native != nullptr);

    rachis::Status reset;
    std::atomic<bool> resetReturned{false};
    bool resetWaited = false;
    {
        std::jthread resetter;
        const auto held = native->lockRecordingState();
        REQUIRE(held.ownsLock());
        resetter = std::jthread([&] {
            reset = device.resetCommandPool(*pool);
            resetReturned.store(true, std::memory_order_release);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        resetWaited = !resetReturned.load(std::memory_order_acquire);
    }

    CHECK(resetWaited);
    INFO("resetCommandPool: " << (reset.has_value() ? std::string{} : reset.error().message));
    REQUIRE(reset.has_value());
    REQUIRE(command->isValid());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    REQUIRE(device.destroyCommandPool(*pool).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan command-pool resets on another thread race debug groups and copies on a live lease",
          "[size:medium][rhi][vulkan][command-pool][concurrency]")
{
    constexpr std::size_t iterationCount = 256;
    constexpr std::uint64_t bufferBytes = 256;

    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    auto pool = device.createCommandPool({
        .queueType = rachis::QueueType::copy,
        .maxCommandBuffers = 1,
        .label = "Vulkan reset-racing pool",
    });
    REQUIRE(pool);
    auto source = device.createBuffer({
        .size = bufferBytes,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto destination = device.createBuffer({
        .size = bufferBytes,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(source);
    REQUIRE(destination);
    auto command = device.acquireCommandBuffer(*pool);
    REQUIRE(command);

    std::atomic<bool> recording{true};
    bool recorderOk = true;
    bool resetterOk = true;
    std::string resetterError;
    {
        std::jthread resetter([&] {
            while (recording.load(std::memory_order_acquire))
            {
                auto reset = device.resetCommandPool(*pool);
                if (!reset && reset.error().code != rachis::ErrorCode::invalidState)
                {
                    resetterOk = false;
                    resetterError = reset.error().message;
                    return;
                }
            }
        });
        for (std::size_t iteration = 0; iteration < iterationCount && recorderOk; ++iteration)
        {
            if (!device.beginDebugGroup(*command, {.label = "Reset-racing scope"}) ||
                !device.copyBuffer(*command,
                                   {
                                       .source = *source,
                                       .destination = *destination,
                                       .size = bufferBytes,
                                   }) ||
                !device.bufferBarrier(*command,
                                      {
                                          .buffer = *destination,
                                          .stagesBefore = rachis::PipelineStage::copy,
                                          .stagesAfter = rachis::PipelineStage::copy,
                                          .accessBefore = rachis::Access::copyDestination,
                                          .accessAfter = rachis::Access::copyDestination,
                                      }) ||
                !device.insertDebugMarker(*command, {.label = "Reset-racing marker"}) ||
                !device.endDebugGroup(*command))
            {
                recorderOk = false;
                break;
            }
            const auto unmatchedEnd = device.endDebugGroup(*command);
            recorderOk = !unmatchedEnd && unmatchedEnd.error().code == rachis::ErrorCode::invalidState;
        }
        recording.store(false, std::memory_order_release);
    }

    CHECK(recorderOk);
    INFO("resetCommandPool: " << resetterError);
    CHECK(resetterOk);
    REQUIRE(command->isValid());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());

    REQUIRE(device.destroyBuffer(*source).has_value());
    REQUIRE(device.destroyBuffer(*destination).has_value());
    REQUIRE(device.destroyCommandPool(*pool).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan command-pool destruction keeps live leases until they are abandoned",
          "[size:medium][rhi][vulkan][command-pool][lifetime]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    auto firstPool = device.createCommandPool({
        .queueType = rachis::QueueType::copy,
        .maxCommandBuffers = 1,
        .label = "First Vulkan pool",
    });
    auto secondPool = device.createCommandPool({
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .label = "Second Vulkan pool",
    });
    REQUIRE(firstPool);
    REQUIRE(secondPool);

    auto first = device.acquireCommandBuffer(*firstPool);
    auto second = device.acquireCommandBuffer(*secondPool);
    REQUIRE(first);
    REQUIRE(second);
    REQUIRE(device.destroyCommandPool(*firstPool).has_value());
    auto staleAcquire = device.acquireCommandBuffer(*firstPool);
    REQUIRE_FALSE(staleAcquire);
    CHECK(staleAcquire.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(first->isValid());
    first = std::unexpected(rachis::Error{});
    second = std::unexpected(rachis::Error{});

    auto recycled = device.acquireCommandBuffer(*secondPool);
    REQUIRE(recycled);
    recycled = std::unexpected(rachis::Error{});
    REQUIRE(device.destroyCommandPool(*secondPool).has_value());

    auto replacement = device.createCommandPool({
        .queueType = rachis::QueueType::copy,
        .maxCommandBuffers = 1,
        .label = "Replacement Vulkan pool",
    });
    REQUIRE(replacement);
    CHECK(*replacement != *firstPool);
    REQUIRE(device.destroyCommandPool(*replacement).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan shared command pools synchronize concurrent recording, individual reset, and abandon",
          "[size:medium][rhi][vulkan][command-pool][concurrency]")
{
    constexpr std::size_t workerCount = 4;
    constexpr std::size_t iterationCount = 16;
    constexpr std::uint64_t bufferBytes = 256;

    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    auto sharedPool = device.createCommandPool({
        .queueType = rachis::QueueType::copy,
        .maxCommandBuffers = static_cast<std::uint32_t>(workerCount),
        .label = "Vulkan shared concurrent command pool",
        .flags = rachis::CommandPoolFlag::allowIndividualReset,
    });
    REQUIRE(sharedPool);

    std::array<rachis::BufferHandle, workerCount> sources{};
    std::array<rachis::BufferHandle, workerCount> destinations{};
    for (std::size_t workerIndex = 0; workerIndex < workerCount; ++workerIndex)
    {
        auto src = device.createBuffer({
            .size = bufferBytes,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
        });
        auto dst = device.createBuffer({
            .size = bufferBytes,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
        });
        REQUIRE(src);
        REQUIRE(dst);
        sources[workerIndex] = *src;
        destinations[workerIndex] = *dst;
    }

    std::array<bool, workerCount> workerOk{};
    workerOk.fill(true);
    std::barrier roundBarrier(static_cast<std::ptrdiff_t>(workerCount));

    {
        std::vector<std::jthread> workers;
        workers.reserve(workerCount);
        for (std::size_t workerIndex = 0; workerIndex < workerCount; ++workerIndex)
        {
            workers.emplace_back([&, workerIndex] {
                for (std::size_t iteration = 0; iteration < iterationCount; ++iteration)
                {
                    roundBarrier.arrive_and_wait();

                    auto command = device.acquireCommandBuffer(*sharedPool);
                    if (!command)
                    {
                        workerOk[workerIndex] = false;
                        continue;
                    }
                    if (!device.beginDebugGroup(*command, {.label = "Shared pool initial scope"}) ||
                        !device.copyBuffer(*command,
                                           {
                                               .source = sources[workerIndex],
                                               .destination = destinations[workerIndex],
                                               .size = bufferBytes,
                                           }) ||
                        !device.insertDebugMarker(*command, {.label = "Shared pool marker"}) ||
                        !device.endDebugGroup(*command))
                    {
                        workerOk[workerIndex] = false;
                        continue;
                    }

                    if (!device.resetCommandBuffer(*command))
                    {
                        workerOk[workerIndex] = false;
                        continue;
                    }

                    if (!device.beginDebugGroup(*command, {.label = "Abandoned open debug group"}) ||
                        !device.bufferBarrier(*command,
                                              {
                                                  .buffer = destinations[workerIndex],
                                                  .stagesBefore = rachis::PipelineStage::copy,
                                                  .stagesAfter = rachis::PipelineStage::copy,
                                                  .accessBefore = rachis::Access::copyDestination,
                                                  .accessAfter = rachis::Access::copyDestination,
                                              }))
                    {
                        workerOk[workerIndex] = false;
                        continue;
                    }
                    command = std::unexpected(rachis::Error{});
                }
            });
        }
    }

    for (const bool ok : workerOk)
    {
        CHECK(ok);
    }

    std::array<rachis::CommandBuffer, workerCount> leased{};
    for (std::size_t workerIndex = 0; workerIndex < workerCount; ++workerIndex)
    {
        auto command = device.acquireCommandBuffer(*sharedPool);
        REQUIRE(command);
        leased[workerIndex] = std::move(*command);
    }
    leased = {};

    for (std::size_t workerIndex = 0; workerIndex < workerCount; ++workerIndex)
    {
        REQUIRE(device.destroyBuffer(sources[workerIndex]).has_value());
        REQUIRE(device.destroyBuffer(destinations[workerIndex]).has_value());
    }
    REQUIRE(device.destroyCommandPool(*sharedPool).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan unsubmitted command buffers outliving Device release VMA allocations and open scopes cleanly",
          "[size:medium][rhi][vulkan][command-pool][lifetime]")
{
    rachis::CommandBuffer liveGraphicsCommand{};
    rachis::CommandBuffer destroyedPoolCopyCommand{};
    {
        auto created = rachis::createDevice(vulkanDesc());
        if (rachis::test::skipWithoutCreatedDevice(created))
        {
            return;
        }
        auto device = std::move(*created);
        const auto capabilities = device.capabilities();
        REQUIRE(capabilities.has_value());
        const auto priority =
            capabilities->residencyPriorities ? rachis::ResidencyPriority::high : rachis::ResidencyPriority::automatic;

        auto graphicsPool = device.createCommandPool({
            .queueType = rachis::QueueType::graphics,
            .maxCommandBuffers = 1,
            .label = "Vulkan post-device graphics pool",
        });
        auto copyPool = device.createCommandPool({
            .queueType = rachis::QueueType::copy,
            .maxCommandBuffers = 1,
            .label = "Vulkan post-device copy pool",
        });
        REQUIRE(graphicsPool);
        REQUIRE(copyPool);

        auto acquiredGraphics = device.acquireCommandBuffer(*graphicsPool);
        auto acquiredCopy = device.acquireCommandBuffer(*copyPool);
        REQUIRE(acquiredGraphics);
        REQUIRE(acquiredCopy);
        liveGraphicsCommand = std::move(*acquiredGraphics);
        destroyedPoolCopyCommand = std::move(*acquiredCopy);

        auto upload = device.createBuffer({
            .size = 256,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
            .label = "Vulkan post-device upload",
        });
        const rachis::BufferDesc placedDesc{
            .size = 256,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::deviceLocal,
            .label = "Vulkan post-device placed destination",
        };
        auto requirements = device.queryBufferMemoryRequirements(placedDesc);
        REQUIRE(requirements);
        auto heap = device.createResourceHeap({
            .size = requirements->size * 2,
            .memoryType = placedDesc.memoryType,
            .label = "Vulkan post-device heap",
        });
        REQUIRE(heap);
        auto placedDestination = device.createPlacedBuffer({.heap = *heap, .offset = 0}, placedDesc);
        auto target = device.createTexture({
            .width = 16,
            .height = 16,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::renderTarget,
            .residencyPriority = priority,
            .label = "Vulkan post-device render target",
        });
        REQUIRE(upload);
        REQUIRE(placedDestination);
        REQUIRE(target);

        auto targetView = device.createTextureView({
            .texture = *target,
            .usage = rachis::TextureViewUsage::renderTarget,
            .dimension = rachis::TextureViewDimension::texture2D,
            .format = rachis::Format::rgba8Unorm,
        });
        REQUIRE(targetView);

        REQUIRE(device
                    .copyBuffer(destroyedPoolCopyCommand,
                                {.source = *upload, .destination = *placedDestination, .size = 256})
                    .has_value());
        REQUIRE(device.beginDebugGroup(destroyedPoolCopyCommand, {.label = "Abandoned Vulkan copy group"}).has_value());
        REQUIRE(device.destroyCommandPool(*copyPool).has_value());

        REQUIRE(device.beginDebugGroup(liveGraphicsCommand, {.label = "Abandoned Vulkan render group"}).has_value());
        REQUIRE(device
                    .textureBarrier(liveGraphicsCommand,
                                    {
                                        .texture = *target,
                                        .stagesBefore = rachis::PipelineStage::none,
                                        .stagesAfter = rachis::PipelineStage::renderTarget,
                                        .accessBefore = rachis::Access::none,
                                        .accessAfter = rachis::Access::renderTarget,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::renderTarget,
                                    })
                    .has_value());
        const std::array attachments{
            rachis::ColorAttachment{
                .view = *targetView,
                .loadOperation = rachis::LoadOperation::clear,
                .clearValue = {0.3F, 0.5F, 0.7F, 1.0F},
            },
        };
        REQUIRE(device.beginRendering(liveGraphicsCommand, {.colorAttachments = attachments}).has_value());

        REQUIRE(device.destroyTextureView(*targetView).has_value());
        REQUIRE(device.destroyTexture(*target).has_value());
        REQUIRE(device.destroyBuffer(*placedDestination).has_value());
        REQUIRE(device.destroyResourceHeap(*heap).has_value());
        REQUIRE(device.destroyBuffer(*upload).has_value());

        const auto diagnostics = device.diagnosticCounts();
        CHECK(diagnostics.errors == 0);
        CHECK(diagnostics.corruptions == 0);
    }

    CHECK(liveGraphicsCommand.isValid());
    CHECK(liveGraphicsCommand.queueType() == rachis::QueueType::graphics);
    CHECK(destroyedPoolCopyCommand.isValid());
    CHECK(destroyedPoolCopyCommand.queueType() == rachis::QueueType::copy);

    liveGraphicsCommand = {};
    destroyedPoolCopyCommand = {};
}
