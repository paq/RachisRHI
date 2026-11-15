#include "Rachis/Device.h"

#include "Internal/CommandBufferAccess.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <utility>

namespace
{

    [[nodiscard]] rachis::Result<rachis::Device> createWarpDevice()
    {
        return rachis::createDevice(rachis::DeviceDesc{
            .adapterPreference = rachis::AdapterPreference::warp,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = true,
        });
    }

} // namespace

TEST_CASE("explicit command pools are bounded and recycle abandoned recordings",
          "[size:medium][rhi][command-pool][warp]")
{
    auto created = createWarpDevice();
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto zeroCapacity = device.createCommandPool(rachis::CommandPoolDesc{});
    REQUIRE_FALSE(zeroCapacity.has_value());
    CHECK(zeroCapacity.error().code == rachis::ErrorCode::invalidArgument);

    auto invalidQueue = device.createCommandPool(rachis::CommandPoolDesc{
        .queueType = static_cast<rachis::QueueType>(99),
        .maxCommandBuffers = 1,
    });
    REQUIRE_FALSE(invalidQueue.has_value());
    CHECK(invalidQueue.error().code == rachis::ErrorCode::invalidArgument);

    const std::string oversizedLabel(rachis::maxDebugLabelBytes + 1, 'x');
    auto oversized = device.createCommandPool(rachis::CommandPoolDesc{
        .maxCommandBuffers = 1,
        .label = oversizedLabel,
    });
    REQUIRE_FALSE(oversized.has_value());
    CHECK(oversized.error().code == rachis::ErrorCode::invalidArgument);

    auto unknownFlags = device.createCommandPool(rachis::CommandPoolDesc{
        .maxCommandBuffers = 1,
        .flags = static_cast<rachis::CommandPoolFlag>(1U << 31U),
    });
    REQUIRE_FALSE(unknownFlags.has_value());
    CHECK(unknownFlags.error().code == rachis::ErrorCode::invalidArgument);

    auto pool = device.createCommandPool(rachis::CommandPoolDesc{
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .label = "Bounded recording pool",
    });
    REQUIRE(pool.has_value());

    auto recording = device.acquireCommandBuffer(*pool);
    REQUIRE(recording.has_value());
    auto exhausted = device.acquireCommandBuffer(*pool);
    REQUIRE_FALSE(exhausted.has_value());
    CHECK(exhausted.error().code == rachis::ErrorCode::resourceExhausted);

    recording = std::unexpected(rachis::Error{});
    auto recycled = device.acquireCommandBuffer(*pool);
    REQUIRE(recycled.has_value());
    recycled = std::unexpected(rachis::Error{});

    REQUIRE(device.destroyCommandPool(*pool).has_value());
    auto staleAcquire = device.acquireCommandBuffer(*pool);
    REQUIRE_FALSE(staleAcquire.has_value());
    CHECK(staleAcquire.error().code == rachis::ErrorCode::invalidArgument);
    auto staleDestroy = device.destroyCommandPool(*pool);
    REQUIRE_FALSE(staleDestroy.has_value());
    CHECK(staleDestroy.error().code == rachis::ErrorCode::invalidArgument);

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("command-pool resets discard recordings without replacing live leases",
          "[size:medium][rhi][command-pool][reset][warp]")
{
    auto created = createWarpDevice();
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    constexpr std::uint64_t halfSize = 16;
    constexpr std::uint64_t totalSize = halfSize * 2;
    std::array<std::byte, totalSize> firstBytes{};
    std::array<std::byte, totalSize> secondBytes{};
    firstBytes.fill(std::byte{0x11});
    secondBytes.fill(std::byte{0x7A});

    auto firstSource = device.createBuffer({
        .size = totalSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto secondSource = device.createBuffer({
        .size = totalSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto destination = device.createBuffer({
        .size = totalSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(firstSource.has_value());
    REQUIRE(secondSource.has_value());
    REQUIRE(destination.has_value());
    REQUIRE(device.writeBuffer(*firstSource, 0, firstBytes).has_value());
    REQUIRE(device.writeBuffer(*secondSource, 0, secondBytes).has_value());

    auto noIndividualReset = device.createCommandPool({
        .queueType = rachis::QueueType::copy,
        .maxCommandBuffers = 1,
    });
    REQUIRE(noIndividualReset.has_value());
    auto noResetCommand = device.acquireCommandBuffer(*noIndividualReset);
    REQUIRE(noResetCommand.has_value());
    auto rejectedIndividual = device.resetCommandBuffer(*noResetCommand);
    REQUIRE_FALSE(rejectedIndividual.has_value());
    CHECK(rejectedIndividual.error().code == rachis::ErrorCode::invalidState);
    noResetCommand = std::unexpected(rachis::Error{});
    REQUIRE(device.destroyCommandPool(*noIndividualReset).has_value());

    auto pool = device.createCommandPool({
        .queueType = rachis::QueueType::copy,
        .maxCommandBuffers = 2,
        .label = "Resettable copy pool",
        .flags = rachis::CommandPoolFlag::allowIndividualReset,
    });
    REQUIRE(pool.has_value());
    auto first = device.acquireCommandBuffer(*pool);
    auto second = device.acquireCommandBuffer(*pool);
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());

    REQUIRE(device
                .copyBuffer(*first,
                            {
                                .source = *firstSource,
                                .destination = *destination,
                                .size = halfSize,
                            })
                .has_value());
    REQUIRE(device
                .copyBuffer(*second,
                            {
                                .source = *firstSource,
                                .sourceOffset = halfSize,
                                .destination = *destination,
                                .destinationOffset = halfSize,
                                .size = halfSize,
                            })
                .has_value());

    REQUIRE(device.resetCommandBuffer(*first).has_value());
    REQUIRE(first->isValid());
    REQUIRE(device
                .copyBuffer(*first,
                            {
                                .source = *secondSource,
                                .destination = *destination,
                                .size = halfSize,
                            })
                .has_value());

    REQUIRE(device.beginDebugGroup(*first, {.label = "Reset validation"}).has_value());
    auto openScopeReset = device.resetCommandPool(*pool);
    REQUIRE_FALSE(openScopeReset.has_value());
    CHECK(openScopeReset.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.endDebugGroup(*first).has_value());

    REQUIRE(device.resetCommandPool(*pool).has_value());
    REQUIRE(first->isValid());
    REQUIRE(second->isValid());
    REQUIRE(device
                .copyBuffer(*first,
                            {
                                .source = *secondSource,
                                .destination = *destination,
                                .size = halfSize,
                            })
                .has_value());
    REQUIRE(device
                .copyBuffer(*second,
                            {
                                .source = *secondSource,
                                .sourceOffset = halfSize,
                                .destination = *destination,
                                .destinationOffset = halfSize,
                                .size = halfSize,
                            })
                .has_value());

    std::array batch{std::move(*first), std::move(*second)};
    auto submitted = device.submit(batch);
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    std::array<std::byte, totalSize> actual{};
    REQUIRE(device.readBuffer(*destination, 0, actual).has_value());
    CHECK(actual == secondBytes);

    REQUIRE(device.destroyCommandPool(*pool).has_value());
    auto staleReset = device.resetCommandPool(*pool);
    REQUIRE_FALSE(staleReset.has_value());
    CHECK(staleReset.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(device.destroyBuffer(*firstSource).has_value());
    REQUIRE(device.destroyBuffer(*secondSource).has_value());
    REQUIRE(device.destroyBuffer(*destination).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("command-pool resets wait while a live recording's state lock is held",
          "[size:medium][rhi][command-pool][reset][warp]")
{
    auto created = createWarpDevice();
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto pool = device.createCommandPool({
        .queueType = rachis::QueueType::copy,
        .maxCommandBuffers = 1,
        .label = "Locked recording-state pool",
    });
    REQUIRE(pool.has_value());
    auto command = device.acquireCommandBuffer(*pool);
    REQUIRE(command.has_value());
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
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    REQUIRE(device.destroyCommandPool(*pool).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("unretained command pools leave application resource lifetime with the caller",
          "[size:medium][rhi][command-pool][retention][warp]")
{
    auto created = createWarpDevice();
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto unretainedPool = device.createCommandPool({
        .queueType = rachis::QueueType::copy,
        .maxCommandBuffers = 1,
        .label = "Unretained copy pool",
        .flags = rachis::CommandPoolFlag::allowIndividualReset | rachis::CommandPoolFlag::unretainedResources,
    });
    REQUIRE(unretainedPool.has_value());
    auto liveSource = device.createBuffer({
        .size = 64,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto liveDestination = device.createBuffer({
        .size = 64,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(liveSource.has_value());
    REQUIRE(liveDestination.has_value());
    std::array<std::byte, 64> expected{};
    expected.fill(std::byte{0x6D});
    REQUIRE(device.writeBuffer(*liveSource, 0, expected).has_value());
    auto liveCommand = device.acquireCommandBuffer(*unretainedPool);
    REQUIRE(liveCommand.has_value());
    REQUIRE(device
                .copyBuffer(*liveCommand,
                            {
                                .source = *liveSource,
                                .destination = *liveDestination,
                                .size = expected.size(),
                            })
                .has_value());
    REQUIRE(device.resetCommandBuffer(*liveCommand).has_value());
    REQUIRE(device
                .copyBuffer(*liveCommand,
                            {
                                .source = *liveSource,
                                .destination = *liveDestination,
                                .size = expected.size(),
                            })
                .has_value());
    auto submitted = device.submit(std::move(*liveCommand));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();
    std::array<std::byte, 64> actual{};
    REQUIRE(device.readBuffer(*liveDestination, 0, actual).has_value());
    CHECK(actual == expected);
    REQUIRE(device.destroyBuffer(*liveSource).has_value());
    REQUIRE(device.destroyBuffer(*liveDestination).has_value());
    REQUIRE(device.destroyCommandPool(*unretainedPool).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("unsubmitted command buffers outliving Device detach D3D12MA allocations and open scopes cleanly",
          "[size:medium][rhi][command-pool][lifetime][warp]")
{
    rachis::CommandBuffer liveGraphicsCommand{};
    rachis::CommandBuffer destroyedPoolCopyCommand{};
    {
        auto created = createWarpDevice();
        REQUIRE(created.has_value());
        auto device = std::move(*created);
        const auto capabilities = device.capabilities();
        REQUIRE(capabilities.has_value());
        const auto priority =
            capabilities->residencyPriorities ? rachis::ResidencyPriority::high : rachis::ResidencyPriority::normal;

        auto graphicsPool = device.createCommandPool({
            .queueType = rachis::QueueType::graphics,
            .maxCommandBuffers = 1,
            .label = "Post-device D3D12 graphics pool",
        });
        auto copyPool = device.createCommandPool({
            .queueType = rachis::QueueType::copy,
            .maxCommandBuffers = 1,
            .label = "Post-device D3D12 copy pool",
        });
        REQUIRE(graphicsPool.has_value());
        REQUIRE(copyPool.has_value());

        auto acquiredGraphics = device.acquireCommandBuffer(*graphicsPool);
        auto acquiredCopy = device.acquireCommandBuffer(*copyPool);
        REQUIRE(acquiredGraphics.has_value());
        REQUIRE(acquiredCopy.has_value());
        liveGraphicsCommand = std::move(*acquiredGraphics);
        destroyedPoolCopyCommand = std::move(*acquiredCopy);

        auto upload = device.createBuffer({
            .size = 256,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
            .label = "Post-device D3D12 upload",
        });
        auto destination = device.createBuffer({
            .size = 256,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::deviceLocal,
            .residencyPriority = rachis::ResidencyPriority::low,
            .label = "Post-device D3D12 low-priority destination",
        });
        auto target = device.createTexture({
            .width = 16,
            .height = 16,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::renderTarget,
            .residencyPriority = priority,
            .label = "Post-device D3D12 high-priority target",
        });
        REQUIRE(upload.has_value());
        REQUIRE(destination.has_value());
        REQUIRE(target.has_value());

        auto targetView = device.createTextureView({
            .texture = *target,
            .usage = rachis::TextureViewUsage::renderTarget,
            .dimension = rachis::TextureViewDimension::texture2D,
            .format = rachis::Format::rgba8Unorm,
        });
        REQUIRE(targetView.has_value());

        REQUIRE(
            device.copyBuffer(destroyedPoolCopyCommand, {.source = *upload, .destination = *destination, .size = 256})
                .has_value());
        REQUIRE(device.beginDebugGroup(destroyedPoolCopyCommand, {.label = "Abandoned D3D12 copy group"}).has_value());
        REQUIRE(device.destroyCommandPool(*copyPool).has_value());

        REQUIRE(device.beginDebugGroup(liveGraphicsCommand, {.label = "Abandoned D3D12 render group"}).has_value());
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
                .clearValue = {0.2F, 0.4F, 0.6F, 1.0F},
            },
        };
        REQUIRE(device.beginRendering(liveGraphicsCommand, {.colorAttachments = attachments}).has_value());

        REQUIRE(device.destroyTextureView(*targetView).has_value());
        REQUIRE(device.destroyTexture(*target).has_value());
        REQUIRE(device.destroyBuffer(*destination).has_value());
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

TEST_CASE("command-pool resets reset idle entries and preserve clean reacquisition",
          "[size:medium][rhi][command-pool][reset][warp]")
{
    auto created = createWarpDevice();
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    constexpr std::uint64_t halfSize = 16;
    constexpr std::uint64_t totalSize = halfSize * 2;
    std::array<std::byte, totalSize> sourceBytes{};
    sourceBytes.fill(std::byte{0x4E});

    auto source = device.createBuffer({
        .size = totalSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto destination = device.createBuffer({
        .size = totalSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(source.has_value());
    REQUIRE(destination.has_value());
    REQUIRE(device.writeBuffer(*source, 0, sourceBytes).has_value());

    auto pool = device.createCommandPool({
        .queueType = rachis::QueueType::copy,
        .maxCommandBuffers = 2,
        .label = "Idle-reset copy pool",
    });
    REQUIRE(pool.has_value());

    auto submittedEntry = device.acquireCommandBuffer(*pool);
    auto abandonedEntry = device.acquireCommandBuffer(*pool);
    REQUIRE(submittedEntry.has_value());
    REQUIRE(abandonedEntry.has_value());

    REQUIRE(device.copyBuffer(*submittedEntry, {.source = *source, .destination = *destination, .size = halfSize})
                .has_value());
    REQUIRE(device
                .copyBuffer(*abandonedEntry,
                            {
                                .source = *source,
                                .sourceOffset = halfSize,
                                .destination = *destination,
                                .destinationOffset = halfSize,
                                .size = halfSize,
                            })
                .has_value());

    abandonedEntry = std::unexpected(rachis::Error{});
    auto firstSubmit = device.submit(std::move(*submittedEntry));
    REQUIRE(firstSubmit.has_value());
    REQUIRE(device.wait(*firstSubmit).has_value());

    REQUIRE(device.resetCommandPool(*pool).has_value());

    auto reacquiredFirst = device.acquireCommandBuffer(*pool);
    auto reacquiredSecond = device.acquireCommandBuffer(*pool);
    REQUIRE(reacquiredFirst.has_value());
    REQUIRE(reacquiredSecond.has_value());
    REQUIRE(device.copyBuffer(*reacquiredFirst, {.source = *source, .destination = *destination, .size = halfSize})
                .has_value());
    REQUIRE(device
                .copyBuffer(*reacquiredSecond,
                            {
                                .source = *source,
                                .sourceOffset = halfSize,
                                .destination = *destination,
                                .destinationOffset = halfSize,
                                .size = halfSize,
                            })
                .has_value());

    std::array batch{std::move(*reacquiredFirst), std::move(*reacquiredSecond)};
    auto secondSubmit = device.submit(batch);
    REQUIRE(secondSubmit.has_value());
    REQUIRE(device.wait(*secondSubmit).has_value());

    std::array<std::byte, totalSize> actual{};
    REQUIRE(device.readBuffer(*destination, 0, actual).has_value());
    CHECK(actual == sourceBytes);

    REQUIRE(device.destroyCommandPool(*pool).has_value());
    REQUIRE(device.destroyBuffer(*source).has_value());
    REQUIRE(device.destroyBuffer(*destination).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
