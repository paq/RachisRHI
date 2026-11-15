#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include "Internal/CommandPoolRetirementTestHook.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

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

    [[nodiscard]] std::byte batchPattern(std::uint64_t index) noexcept
    {
        return static_cast<std::byte>((index * 29U + 7U) & 0xFFU);
    }

    template<typename T>
    [[nodiscard]] std::string errorMessage(const rachis::Result<T>& result)
    {
        return result ? std::string{} : result.error().message;
    }

} // namespace

TEST_CASE("batched submissions execute ordered command lists behind one timeline point",
          "[size:medium][rhi][queue][batch][warp]")
{
    constexpr std::uint64_t transferSize = 4096;
    auto created = createWarpDevice();
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto upload = device.createBuffer(rachis::BufferDesc{
        .size = transferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Batch submission upload",
    });
    auto intermediate = device.createBuffer(rachis::BufferDesc{
        .size = transferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Batch submission intermediate",
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = transferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Batch submission readback",
    });
    REQUIRE(upload.has_value());
    REQUIRE(intermediate.has_value());
    REQUIRE(readback.has_value());

    auto mappedUpload = device.mapBuffer(*upload);
    REQUIRE(mappedUpload.has_value());
    for (std::uint64_t index = 0; index < transferSize; ++index)
    {
        mappedUpload->data[index] = batchPattern(index);
    }
    REQUIRE(device.unmapBuffer(*upload).has_value());

    auto first = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    auto second = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());
    REQUIRE(device
                .bufferBarrier(*first,
                               rachis::BufferBarrier{
                                   .buffer = *intermediate,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::copyDestination,
                               })
                .has_value());
    REQUIRE(device
                .copyBuffer(*first,
                            rachis::BufferCopy{
                                .source = *upload,
                                .destination = *intermediate,
                                .size = transferSize,
                            })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*second,
                               rachis::BufferBarrier{
                                   .buffer = *intermediate,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device
                .copyBuffer(*second,
                            rachis::BufferCopy{
                                .source = *intermediate,
                                .destination = *readback,
                                .size = transferSize,
                            })
                .has_value());

    std::array<rachis::CommandBuffer, 2> batch{std::move(*first), std::move(*second)};
    auto submitted = device.submit(batch);
    REQUIRE(submitted.has_value());
    CHECK(submitted->value() == 1);
    CHECK_FALSE(batch[0].isValid());
    CHECK_FALSE(batch[1].isValid());

    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*intermediate).has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    auto mappedReadback = device.mapBuffer(*readback);
    REQUIRE(mappedReadback.has_value());
    std::uint64_t firstMismatch = transferSize;
    for (std::uint64_t index = 0; index < transferSize; ++index)
    {
        if (mappedReadback->data[index] != batchPattern(index))
        {
            firstMismatch = index;
            break;
        }
    }
    CAPTURE(firstMismatch);
    CHECK(firstMismatch == transferSize);
    REQUIRE(device.unmapBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    auto freshFirst = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    auto freshSecond = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(freshFirst.has_value());
    REQUIRE(freshSecond.has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("batch validation preserves every unconsumed command buffer",
          "[size:medium][rhi][queue][batch][validation][warp]")
{
    auto created = createWarpDevice();
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    const std::span<rachis::CommandBuffer> empty;
    auto emptySubmission = device.submit(empty);
    REQUIRE_FALSE(emptySubmission.has_value());
    CHECK(emptySubmission.error().code == rachis::ErrorCode::invalidArgument);

    auto graphics = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    auto copy = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(graphics.has_value());
    REQUIRE(copy.has_value());
    std::array mixedQueues{std::move(*graphics), std::move(*copy)};
    auto mixedSubmission = device.submit(mixedQueues);
    REQUIRE_FALSE(mixedSubmission.has_value());
    CHECK(mixedSubmission.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(mixedQueues[0].isValid());
    CHECK(mixedQueues[1].isValid());
    mixedQueues = {};

    auto otherCreated = createWarpDevice();
    REQUIRE(otherCreated.has_value());
    auto local = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    auto foreign = rachis::test::acquireCommandBuffer(*otherCreated, rachis::QueueType::graphics);
    REQUIRE(local.has_value());
    REQUIRE(foreign.has_value());
    std::array foreignBatch{std::move(*local), std::move(*foreign)};
    auto foreignSubmission = device.submit(foreignBatch);
    REQUIRE_FALSE(foreignSubmission.has_value());
    CHECK(foreignSubmission.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(foreignBatch[0].isValid());
    CHECK(foreignBatch[1].isValid());
    foreignBatch = {};

    auto first = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    auto second = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());
    REQUIRE(device.beginDebugGroup(*second, rachis::DebugMarkerDesc{.label = "Unclosed batch group"}).has_value());
    std::array batch{std::move(*first), std::move(*second)};
    auto openGroupSubmission = device.submit(batch);
    REQUIRE_FALSE(openGroupSubmission.has_value());
    CHECK(openGroupSubmission.error().code == rachis::ErrorCode::invalidState);
    CHECK(batch[0].isValid());
    CHECK(batch[1].isValid());
    REQUIRE(device.endDebugGroup(batch[1]).has_value());

    auto dependency = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(dependency.has_value());
    auto dependencyPoint = device.submit(std::move(*dependency));
    REQUIRE(dependencyPoint.has_value());

    const std::array waits{*dependencyPoint};
    auto submitted = device.submit(batch, waits);
    REQUIRE(submitted.has_value());
    CHECK_FALSE(batch[0].isValid());
    CHECK_FALSE(batch[1].isValid());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("queue and device idle waits drain published timeline frontiers", "[size:medium][rhi][queue][idle][warp]")
{
    rachis::Device invalidDevice;
    auto invalidQueueWait = invalidDevice.waitIdle(rachis::QueueType::graphics);
    auto invalidDeviceWait = invalidDevice.waitIdle();
    REQUIRE_FALSE(invalidQueueWait.has_value());
    CHECK(invalidQueueWait.error().code == rachis::ErrorCode::invalidState);
    REQUIRE_FALSE(invalidDeviceWait.has_value());
    CHECK(invalidDeviceWait.error().code == rachis::ErrorCode::invalidState);

    auto created = createWarpDevice();
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    REQUIRE(device.waitIdle(rachis::QueueType::graphics).has_value());
    REQUIRE(device.waitIdle().has_value());

    auto unknownQueueWait = device.waitIdle(static_cast<rachis::QueueType>(99));
    REQUIRE_FALSE(unknownQueueWait.has_value());
    CHECK(unknownQueueWait.error().code == rachis::ErrorCode::invalidArgument);

    auto copy = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(copy.has_value());
    auto copyPoint = device.submit(std::move(*copy));
    REQUIRE(copyPoint.has_value());

    auto compute = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(compute.has_value());
    const std::array copyWait{*copyPoint};
    auto computePoint = device.submit(std::move(*compute), copyWait);
    REQUIRE(computePoint.has_value());

    auto graphics = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(graphics.has_value());
    const std::array computeWait{*computePoint};
    auto graphicsPoint = device.submit(std::move(*graphics), computeWait);
    REQUIRE(graphicsPoint.has_value());

    REQUIRE(device.waitIdle(rachis::QueueType::copy).has_value());
    CHECK(device.completedValue(rachis::QueueType::copy) >= copyPoint->value());
    auto postIdleCopy = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(postIdleCopy.has_value());

    REQUIRE(device.waitIdle().has_value());
    CHECK(device.completedValue(rachis::QueueType::compute) >= computePoint->value());
    CHECK(device.completedValue(rachis::QueueType::graphics) >= graphicsPoint->value());
    auto postIdleCompute = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    auto postIdleGraphics = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(postIdleCompute.has_value());
    REQUIRE(postIdleGraphics.has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("bounded queues use exact timeline retirement", "[size:medium][rhi][queue][warp]")
{
    auto created = createWarpDevice();
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto graphicsPool = device.createCommandPool({
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 2,
        .label = "Bounded graphics queue test pool",
    });
    auto copyPool = device.createCommandPool({
        .queueType = rachis::QueueType::copy,
        .maxCommandBuffers = 1,
        .label = "Bounded copy queue test pool",
    });
    auto computePool = device.createCommandPool({
        .queueType = rachis::QueueType::compute,
        .maxCommandBuffers = 1,
        .label = "Bounded compute queue test pool",
    });
    REQUIRE(graphicsPool.has_value());
    REQUIRE(copyPool.has_value());
    REQUIRE(computePool.has_value());

    auto first = device.acquireCommandBuffer(*graphicsPool);
    auto second = device.acquireCommandBuffer(*graphicsPool);
    auto exhausted = device.acquireCommandBuffer(*graphicsPool);
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());
    REQUIRE_FALSE(exhausted.has_value());
    CHECK(exhausted.error().code == rachis::ErrorCode::resourceExhausted);

    first = std::unexpected(rachis::Error{});
    auto recycledAbandoned = device.acquireCommandBuffer(*graphicsPool);
    REQUIRE(recycledAbandoned.has_value());

    auto graphicsPoint = device.submit(std::move(*recycledAbandoned));
    REQUIRE(graphicsPoint.has_value());
    CHECK(graphicsPoint->isValid());
    CHECK_FALSE(recycledAbandoned->isValid());

    auto copy = device.acquireCommandBuffer(*copyPool);
    REQUIRE(copy.has_value());
    auto copyPoint = device.submit(std::move(*copy));
    REQUIRE(copyPoint.has_value());

    auto compute = device.acquireCommandBuffer(*computePool);
    REQUIRE(compute.has_value());
    const std::array copyWait{*copyPoint};
    auto computePoint = device.submit(std::move(*compute), copyWait);
    REQUIRE(computePoint.has_value());

    const std::array computeWait{*computePoint};
    auto finalPoint = device.submit(std::move(*second), computeWait);
    REQUIRE(finalPoint.has_value());

    auto waited = device.wait(*finalPoint);
    REQUIRE(waited.has_value());
    CHECK(device.completedValue(rachis::QueueType::graphics) >= finalPoint->value());
    device.collectGarbage();

    auto otherCreated = createWarpDevice();
    REQUIRE(otherCreated.has_value());
    auto foreignCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(foreignCommand.has_value());
    auto foreignSubmit = otherCreated->submit(std::move(*foreignCommand));
    auto foreignWait = otherCreated->wait(*finalPoint);
    REQUIRE_FALSE(foreignSubmit.has_value());
    CHECK(foreignSubmit.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE_FALSE(foreignWait.has_value());
    CHECK(foreignWait.error().code == rachis::ErrorCode::invalidArgument);

    auto lifetimeDevice = createWarpDevice();
    REQUIRE(lifetimeDevice.has_value());
    auto lifetimePool = lifetimeDevice->createCommandPool({
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .label = "Device lifetime queue test pool",
    });
    REQUIRE(lifetimePool.has_value());
    auto survivingLease = lifetimeDevice->acquireCommandBuffer(*lifetimePool);
    REQUIRE(survivingLease.has_value());
    *lifetimeDevice = rachis::Device{};
    *survivingLease = rachis::CommandBuffer{};

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
    REQUIRE(device.destroyCommandPool(*computePool).has_value());
    REQUIRE(device.destroyCommandPool(*copyPool).has_value());
    REQUIRE(device.destroyCommandPool(*graphicsPool).has_value());
}

TEST_CASE("same-queue submissions are thread safe and recyclable", "[size:medium][rhi][queue][concurrency][warp]")
{
    constexpr std::size_t concurrentCount = 8;
    auto created = createWarpDevice();
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    auto commandPool = device.createCommandPool({
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = static_cast<std::uint32_t>(concurrentCount),
        .label = "Concurrent graphics queue test pool",
    });
    REQUIRE(commandPool.has_value());

    std::vector<rachis::CommandBuffer> commands;
    commands.reserve(concurrentCount);
    for (std::size_t index = 0; index < concurrentCount; ++index)
    {
        auto command = device.acquireCommandBuffer(*commandPool);
        REQUIRE(command.has_value());
        commands.push_back(std::move(*command));
    }

    std::vector<rachis::Result<rachis::TimelinePoint>> points(concurrentCount);
    {
        std::vector<std::jthread> workers;
        workers.reserve(concurrentCount);
        for (std::size_t index = 0; index < concurrentCount; ++index)
        {
            workers.emplace_back(
                [&device, &commands, &points, index] { points[index] = device.submit(std::move(commands[index])); });
        }
    }

    for (const auto& point : points)
    {
        REQUIRE(point.has_value());
        REQUIRE(device.wait(*point).has_value());
    }
    device.collectGarbage();

    std::vector<rachis::CommandBuffer> recycled;
    recycled.reserve(concurrentCount);
    for (std::size_t index = 0; index < concurrentCount; ++index)
    {
        auto command = device.acquireCommandBuffer(*commandPool);
        REQUIRE(command.has_value());
        recycled.push_back(std::move(*command));
    }

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
    recycled.clear();
    REQUIRE(device.destroyCommandPool(*commandPool).has_value());
}

TEST_CASE("cross-queue submissions cannot deadlock device idle", "[size:medium][rhi][queue][concurrency][idle][warp]")
{
    constexpr std::size_t iterationCount = 16;
    constexpr std::array queueTypes{
        rachis::QueueType::graphics,
        rachis::QueueType::compute,
        rachis::QueueType::copy,
    };

    auto created = createWarpDevice();
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    std::array<rachis::CommandPoolHandle, queueTypes.size()> commandPools;
    for (std::size_t queueIndex = 0; queueIndex < queueTypes.size(); ++queueIndex)
    {
        auto commandPool = device.createCommandPool({
            .queueType = queueTypes[queueIndex],
            .maxCommandBuffers = static_cast<std::uint32_t>(iterationCount),
            .label = "Concurrent cross-queue test pool",
        });
        REQUIRE(commandPool.has_value());
        commandPools[queueIndex] = *commandPool;
    }

    std::array<std::vector<rachis::Result<rachis::TimelinePoint>>, queueTypes.size()> submissions;
    for (auto& queueSubmissions : submissions)
    {
        queueSubmissions.reserve(iterationCount);
    }
    std::vector<rachis::Status> deviceIdleResults;
    deviceIdleResults.reserve(iterationCount);

    std::barrier startBarrier(static_cast<std::ptrdiff_t>(queueTypes.size() + 1));
    {
        std::vector<std::jthread> workers;
        workers.reserve(queueTypes.size() + 1);
        for (std::size_t queueIndex = 0; queueIndex < queueTypes.size(); ++queueIndex)
        {
            workers.emplace_back([&commandPools, &device, &submissions, &startBarrier, queueIndex] {
                startBarrier.arrive_and_wait();
                for (std::size_t iteration = 0; iteration < iterationCount; ++iteration)
                {
                    auto command = device.acquireCommandBuffer(commandPools[queueIndex]);
                    if (!command)
                    {
                        submissions[queueIndex].emplace_back(std::unexpected(command.error()));
                        continue;
                    }
                    submissions[queueIndex].push_back(device.submit(std::move(*command)));
                }
            });
        }
        workers.emplace_back([&device, &deviceIdleResults, &startBarrier] {
            startBarrier.arrive_and_wait();
            for (std::size_t iteration = 0; iteration < iterationCount; ++iteration)
            {
                deviceIdleResults.push_back(device.waitIdle());
            }
        });
    }

    REQUIRE(deviceIdleResults.size() == iterationCount);
    for (const auto& idleResult : deviceIdleResults)
    {
        REQUIRE(idleResult.has_value());
    }
    for (std::size_t queueIndex = 0; queueIndex < queueTypes.size(); ++queueIndex)
    {
        REQUIRE(submissions[queueIndex].size() == iterationCount);
        for (const auto& submission : submissions[queueIndex])
        {
            REQUIRE(submission.has_value());
        }
    }

    REQUIRE(device.waitIdle().has_value());
    for (std::size_t queueIndex = 0; queueIndex < queueTypes.size(); ++queueIndex)
    {
        for (const auto& submission : submissions[queueIndex])
        {
            CHECK(device.completedValue(queueTypes[queueIndex]) >= submission->value());
        }
    }
    device.collectGarbage();

    std::array<std::vector<rachis::CommandBuffer>, queueTypes.size()> recycled;
    for (std::size_t queueIndex = 0; queueIndex < queueTypes.size(); ++queueIndex)
    {
        recycled[queueIndex].reserve(iterationCount);
        for (std::size_t iteration = 0; iteration < iterationCount; ++iteration)
        {
            auto command = device.acquireCommandBuffer(commandPools[queueIndex]);
            REQUIRE(command.has_value());
            recycled[queueIndex].push_back(std::move(*command));
        }
    }

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
    for (const auto commandPool : commandPools)
    {
        REQUIRE(device.destroyCommandPool(commandPool).has_value());
    }
}

TEST_CASE("a collector parked after retiring a submission leaves its command-pool entry reusable",
          "[size:medium][rhi][queue][command-pool][concurrency][warp]")
{
    using rachis::detail::test::CommandPoolRetirementParking;
    constexpr std::uint64_t transferSize = 256;

    auto created = createWarpDevice();
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    auto commandPool = device.createCommandPool({
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .label = "Parked retirement test pool",
    });
    REQUIRE(commandPool.has_value());
    auto upload = device.createBuffer(rachis::BufferDesc{
        .size = transferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Parked retirement upload",
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = transferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Parked retirement readback",
    });
    REQUIRE(upload.has_value());
    REQUIRE(readback.has_value());

    auto command = device.acquireCommandBuffer(*commandPool);
    REQUIRE(command.has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            rachis::BufferCopy{
                                .source = *upload,
                                .destination = *readback,
                                .size = transferSize,
                            })
                .has_value());
    auto point = device.submit(std::move(*command));
    REQUIRE(point.has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    rachis::Status collectorWait;
    rachis::Status ownerWait;
    rachis::Status ownerReset;
    rachis::Result<rachis::CommandBuffer> reacquired = std::unexpected(rachis::Error{});
    bool parked = false;
    {
        const CommandPoolRetirementParking parking;
        std::jthread collector([&device, &collectorWait, &point] {
            CommandPoolRetirementParking::armCurrentThread();
            collectorWait = device.wait(*point);
        });
        parked = CommandPoolRetirementParking::waitUntilParked(std::chrono::seconds(10));
        ownerWait = device.wait(*point);
        ownerReset = device.resetCommandPool(*commandPool);
        reacquired = device.acquireCommandBuffer(*commandPool);
        CommandPoolRetirementParking::release();
    }

    REQUIRE(parked);
    REQUIRE(collectorWait.has_value());
    REQUIRE(ownerWait.has_value());
    INFO("resetCommandPool: " << errorMessage(ownerReset));
    INFO("acquireCommandBuffer: " << errorMessage(reacquired));
    CHECK(ownerReset.has_value());
    REQUIRE(reacquired.has_value());
    auto resubmitted = device.submit(std::move(*reacquired));
    REQUIRE(resubmitted.has_value());
    REQUIRE(device.wait(*resubmitted).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
    REQUIRE(device.destroyCommandPool(*commandPool).has_value());
}

TEST_CASE("one-entry command pools reacquire after their own wait while other threads collect",
          "[size:medium][rhi][queue][command-pool][concurrency][warp]")
{
    constexpr std::size_t workerCount = 8;
    constexpr std::size_t iterationCount = 200;
    constexpr std::uint64_t transferSize = 256;

    auto created = createWarpDevice();
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    std::array<rachis::CommandPoolHandle, workerCount> commandPools{};
    std::array<rachis::BufferHandle, workerCount> uploads{};
    std::array<rachis::BufferHandle, workerCount> readbacks{};
    for (std::size_t workerIndex = 0; workerIndex < workerCount; ++workerIndex)
    {
        auto commandPool = device.createCommandPool({
            .queueType = rachis::QueueType::graphics,
            .maxCommandBuffers = 1,
            .label = "One-entry reacquisition test pool",
        });
        auto upload = device.createBuffer(rachis::BufferDesc{
            .size = transferSize,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
            .label = "One-entry reacquisition upload",
        });
        auto readback = device.createBuffer(rachis::BufferDesc{
            .size = transferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "One-entry reacquisition readback",
        });
        REQUIRE(commandPool.has_value());
        REQUIRE(upload.has_value());
        REQUIRE(readback.has_value());
        commandPools[workerIndex] = *commandPool;
        uploads[workerIndex] = *upload;
        readbacks[workerIndex] = *readback;
    }

    std::atomic<std::size_t> exhaustedAfterOwnWait{0};
    std::atomic<std::size_t> otherFailures{0};
    std::atomic<bool> stopCollector{false};
    std::barrier startBarrier(static_cast<std::ptrdiff_t>(workerCount + 1));
    {
        std::jthread collector([&device, &stopCollector, &startBarrier] {
            startBarrier.arrive_and_wait();
            while (!stopCollector.load(std::memory_order_acquire))
            {
                device.collectGarbage();
                std::this_thread::yield();
            }
        });
        {
            std::vector<std::jthread> workers;
            workers.reserve(workerCount);
            for (std::size_t workerIndex = 0; workerIndex < workerCount; ++workerIndex)
            {
                workers.emplace_back([&, workerIndex] {
                    startBarrier.arrive_and_wait();
                    for (std::size_t iteration = 0; iteration < iterationCount; ++iteration)
                    {
                        auto command = device.acquireCommandBuffer(commandPools[workerIndex]);
                        if (!command)
                        {
                            auto& counter = command.error().code == rachis::ErrorCode::resourceExhausted
                                                ? exhaustedAfterOwnWait
                                                : otherFailures;
                            counter.fetch_add(1, std::memory_order_relaxed);
                            continue;
                        }
                        if (!device.copyBuffer(*command,
                                               rachis::BufferCopy{
                                                   .source = uploads[workerIndex],
                                                   .destination = readbacks[workerIndex],
                                                   .size = transferSize,
                                               }))
                        {
                            otherFailures.fetch_add(1, std::memory_order_relaxed);
                            continue;
                        }
                        auto point = device.submit(std::move(*command));
                        if (!point || !device.wait(*point))
                        {
                            otherFailures.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                });
            }
        }
        stopCollector.store(true, std::memory_order_release);
    }

    CHECK(exhaustedAfterOwnWait.load() == 0);
    CHECK(otherFailures.load() == 0);
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
    for (std::size_t workerIndex = 0; workerIndex < workerCount; ++workerIndex)
    {
        REQUIRE(device.destroyBuffer(readbacks[workerIndex]).has_value());
        REQUIRE(device.destroyBuffer(uploads[workerIndex]).has_value());
        REQUIRE(device.destroyCommandPool(commandPools[workerIndex]).has_value());
    }
}
