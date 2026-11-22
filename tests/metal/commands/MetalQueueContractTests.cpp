#include "Metal/MetalPresentationFeedback.h"
#include "Metal/MetalSubmissionFeedback.h"
#include "Metal/MetalTestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace
{

#if RACHIS_ENABLE_METAL
    [[nodiscard]] rachis::DeviceDesc metalDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
    }

    [[nodiscard]] rachis::Result<rachis::CommandBuffer> acquire(rachis::Device& device, rachis::QueueType queueType)
    {
        return rachis::test::acquireCommandBuffer(device, queueType);
    }
#endif

} // namespace

TEST_CASE("Metal timeline completion waits for feedback in submission order",
          "[size:small][rhi][metal][feedback][regression]")
{
    rachis::metal::MetalSubmissionFeedback feedback;
    feedback.begin(2);
    feedback.begin(4);

    CHECK(feedback.completedValue(3) == 1);
    feedback.complete(4);
    CHECK(feedback.completedValue(4) == 1);
    feedback.wait(4, std::chrono::milliseconds(1));
    CHECK(feedback.completedValue(4) == 1);
    feedback.complete(2);
    CHECK(feedback.completedValue(4) == 4);
    CHECK(feedback.completedValue(3) == 3);
}

TEST_CASE("Metal feedback acknowledgement publishes errors before waking waiters",
          "[size:small][rhi][metal][feedback][regression]")
{
    rachis::metal::MetalSubmissionFeedback feedback;
    feedback.begin(1);
    std::barrier ready(2);
    bool failed = false;
    std::jthread const callback([&] {
        ready.arrive_and_wait();
        failed = true;
        feedback.complete(1);
    });
    ready.arrive_and_wait();
    feedback.wait(1, std::chrono::seconds(5));
    REQUIRE(feedback.completedValue(1) == 1);
    CHECK(failed);
}

TEST_CASE("Metal presentation feedback excludes dropped drawables and stays monotonic",
          "[size:small][rhi][metal][feedback][regression]")
{
    rachis::metal::MetalPresentationFeedback feedback;
    CHECK(feedback.displayedSequence() == 0);
    feedback.presented(1, 0.0);
    CHECK(feedback.displayedSequence() == 0);
    feedback.presented(3, 12.0);
    feedback.presented(2, 11.0);
    feedback.presented(4, 0.0);
    CHECK(feedback.displayedSequence() == 3);
}

TEST_CASE("Metal presentation feedback settles hidden and replaced drawables apart from displayed ones",
          "[size:small][rhi][metal][feedback][regression]")
{
    rachis::metal::MetalPresentationFeedback feedback;
    REQUIRE(feedback.ticksPerSecond() > 0);
    for (std::uint64_t sequence = 1; sequence <= 4; ++sequence)
    {
        feedback.submitted(sequence);
    }
    feedback.presented(1, 2.0);
    feedback.presented(2, 0.0, true);
    feedback.presented(3, 0.0, false);
    feedback.presented(4, 0.0, true);

    std::array<rachis::PresentTiming, 4> records{};
    const auto snapshot = feedback.readTimings(1, records);
    REQUIRE(snapshot.recordCount == 4);
    CHECK(records[0].flags == rachis::PresentTimingFlags::displayed);
    CHECK(records[0].presentTime == 2 * feedback.ticksPerSecond());
    // Replaced by a newer present while visible.
    CHECK(records[1].flags == rachis::PresentTimingFlags::dropped);
    // Not visible: nothing could be shown, so it is not counted as dropped.
    CHECK(records[2].flags == rachis::PresentTimingFlags::hidden);
    // The newest present, visible but not shown: the outcome is unknown.
    CHECK(records[3].flags == rachis::PresentTimingFlags::none);
    CHECK(records[3].presentTime == 0);
    CHECK(feedback.displayedSequence() == 1);
}

#if RACHIS_ENABLE_METAL

TEST_CASE("Metal failed submissions keep command buffers and do not publish points",
          "[size:large][rhi][metal][queue][validation]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    const std::span<rachis::CommandBuffer> empty;
    auto emptySubmission = device.submit(empty);
    REQUIRE_FALSE(emptySubmission);
    CHECK(emptySubmission.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(device.completedValue(rachis::QueueType::graphics) == 0);

    auto graphics = acquire(device, rachis::QueueType::graphics);
    auto copy = acquire(device, rachis::QueueType::copy);
    REQUIRE(graphics);
    REQUIRE(copy);
    std::array mixed{std::move(*graphics), std::move(*copy)};
    auto mixedSubmission = device.submit(mixed);
    REQUIRE_FALSE(mixedSubmission);
    CHECK(mixedSubmission.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(mixed[0].isValid());
    CHECK(mixed[1].isValid());
    mixed = {};

    auto otherCreated = rachis::createDevice(metalDesc());
    REQUIRE(otherCreated);
    auto local = acquire(device, rachis::QueueType::graphics);
    auto foreign = acquire(*otherCreated, rachis::QueueType::graphics);
    REQUIRE(local);
    REQUIRE(foreign);
    std::array foreignBatch{std::move(*local), std::move(*foreign)};
    auto foreignSubmission = device.submit(foreignBatch);
    REQUIRE_FALSE(foreignSubmission);
    CHECK(foreignSubmission.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(foreignBatch[0].isValid());
    CHECK(foreignBatch[1].isValid());
    foreignBatch = {};

    auto first = acquire(device, rachis::QueueType::graphics);
    auto second = acquire(device, rachis::QueueType::graphics);
    REQUIRE(first);
    REQUIRE(second);
    REQUIRE(device.beginDebugGroup(*second, rachis::DebugMarkerDesc{.label = "Unclosed Metal group"}).has_value());
    std::array openBatch{std::move(*first), std::move(*second)};
    auto openSubmission = device.submit(openBatch);
    REQUIRE_FALSE(openSubmission);
    CHECK(openSubmission.error().code == rachis::ErrorCode::invalidState);
    CHECK(openBatch[0].isValid());
    CHECK(openBatch[1].isValid());
    REQUIRE(device.endDebugGroup(openBatch[1]).has_value());
    CHECK(device.completedValue(rachis::QueueType::graphics) == 0);

    auto unpublished = device.queryTimelinePoint({});
    REQUIRE_FALSE(unpublished);
    CHECK(unpublished.error().code == rachis::ErrorCode::invalidArgument);

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal per-queue timelines publish monotonic points and host waits",
          "[size:large][rhi][metal][queue][timeline]")
{
    using namespace std::chrono_literals;

    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto firstCommand = acquire(device, rachis::QueueType::graphics);
    REQUIRE(firstCommand);
    auto first = device.submit(std::move(*firstCommand));
    REQUIRE(first);
    CHECK(first->isValid());
    CHECK(first->queueType() == rachis::QueueType::graphics);
    CHECK(first->value() == 1);
    CHECK_FALSE(firstCommand->isValid());

    auto secondCommand = acquire(device, rachis::QueueType::graphics);
    REQUIRE(secondCommand);
    auto second = device.submit(std::move(*secondCommand));
    REQUIRE(second);
    CHECK(second->value() == 2);

    REQUIRE(device.wait(*second).has_value());
    CHECK(device.completedValue(rachis::QueueType::graphics) >= second->value());
    auto firstStatus = device.queryTimelinePoint(*first);
    auto secondStatus = device.queryTimelinePoint(*second);
    REQUIRE(firstStatus);
    REQUIRE(secondStatus);
    CHECK(*firstStatus == rachis::TimelinePointStatus::complete);
    CHECK(*secondStatus == rachis::TimelinePointStatus::complete);

    auto zeroTimeout = device.wait(*second, 0ns);
    REQUIRE(zeroTimeout);
    CHECK(*zeroTimeout == rachis::TimelinePointStatus::complete);

    auto negativeTimeout = device.wait(*second, -1ns);
    REQUIRE_FALSE(negativeTimeout);
    CHECK(negativeTimeout.error().code == rachis::ErrorCode::invalidArgument);

    auto foreignCreated = rachis::createDevice(metalDesc());
    REQUIRE(foreignCreated);
    auto foreignWait = foreignCreated->wait(*second);
    auto foreignQuery = foreignCreated->queryTimelinePoint(*first);
    REQUIRE_FALSE(foreignWait);
    CHECK(foreignWait.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE_FALSE(foreignQuery);
    CHECK(foreignQuery.error().code == rachis::ErrorCode::invalidArgument);

    auto unsubmittedWait = acquire(device, rachis::QueueType::graphics);
    REQUIRE(unsubmittedWait);
    auto missing = device.submit(std::move(*unsubmittedWait), std::array{rachis::TimelinePoint{}});
    REQUIRE_FALSE(missing);
    CHECK(missing.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(unsubmittedWait->isValid());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal batched submissions consume every lease behind one timeline point",
          "[size:large][rhi][metal][queue][batch]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto first = acquire(device, rachis::QueueType::graphics);
    auto second = acquire(device, rachis::QueueType::graphics);
    REQUIRE(first);
    REQUIRE(second);
    std::array batch{std::move(*first), std::move(*second)};
    auto submitted = device.submit(batch);
    REQUIRE(submitted);
    CHECK(submitted->value() == 1);
    CHECK_FALSE(batch[0].isValid());
    CHECK_FALSE(batch[1].isValid());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    auto recycledFirst = acquire(device, rachis::QueueType::graphics);
    auto recycledSecond = acquire(device, rachis::QueueType::graphics);
    REQUIRE(recycledFirst);
    REQUIRE(recycledSecond);

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal cross-queue waits and idle drain aliased logical timelines", "[size:large][rhi][metal][queue][idle]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    REQUIRE(device.waitIdle(rachis::QueueType::graphics).has_value());
    REQUIRE(device.waitIdle().has_value());
    auto unknown = device.waitIdle(static_cast<rachis::QueueType>(99));
    REQUIRE_FALSE(unknown);
    CHECK(unknown.error().code == rachis::ErrorCode::invalidArgument);

    auto copy = acquire(device, rachis::QueueType::copy);
    REQUIRE(copy);
    auto copyPoint = device.submit(std::move(*copy));
    REQUIRE(copyPoint);

    auto compute = acquire(device, rachis::QueueType::compute);
    REQUIRE(compute);
    auto computePoint = device.submit(std::move(*compute), std::array{*copyPoint});
    REQUIRE(computePoint);

    auto graphics = acquire(device, rachis::QueueType::graphics);
    REQUIRE(graphics);
    auto graphicsPoint = device.submit(std::move(*graphics), std::array{*computePoint});
    REQUIRE(graphicsPoint);

    REQUIRE(device.waitIdle(rachis::QueueType::copy).has_value());
    CHECK(device.completedValue(rachis::QueueType::copy) >= copyPoint->value());
    REQUIRE(device.waitIdle().has_value());
    CHECK(device.completedValue(rachis::QueueType::compute) >= computePoint->value());
    CHECK(device.completedValue(rachis::QueueType::graphics) >= graphicsPoint->value());

    auto postCopy = acquire(device, rachis::QueueType::copy);
    auto postCompute = acquire(device, rachis::QueueType::compute);
    auto postGraphics = acquire(device, rachis::QueueType::graphics);
    REQUIRE(postCopy);
    REQUIRE(postCompute);
    REQUIRE(postGraphics);

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal bounded pools reject in-flight reset and recycle after exact completion",
          "[size:large][rhi][metal][queue][command-pool]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto pool = device.createCommandPool(rachis::CommandPoolDesc{
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .label = "Metal in-flight pool",
    });
    REQUIRE(pool);
    auto command = device.acquireCommandBuffer(*pool);
    REQUIRE(command);
    auto exhausted = device.acquireCommandBuffer(*pool);
    REQUIRE_FALSE(exhausted);
    CHECK(exhausted.error().code == rachis::ErrorCode::resourceExhausted);

    auto feedbackGate = std::make_shared<rachis::metal::MetalCommitFeedbackGate>();
    struct ScopedGateRelease final
    {
        std::shared_ptr<rachis::metal::MetalCommitFeedbackGate> gate;
        ~ScopedGateRelease() noexcept
        {
            gate->release();
        }
    } const releaseOnExit{feedbackGate};
    rachis::metal::holdNextCommitFeedback(feedbackGate);
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(feedbackGate->waitForEntry(std::chrono::seconds(2)));
    auto inFlightReset = device.resetCommandPool(*pool);
    REQUIRE_FALSE(inFlightReset);
    CHECK(inFlightReset.error().code == rachis::ErrorCode::invalidState);

    feedbackGate->release();
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();
    REQUIRE(device.resetCommandPool(*pool).has_value());
    auto recycled = device.acquireCommandBuffer(*pool);
    REQUIRE(recycled);
    REQUIRE(device.destroyCommandPool(*pool).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal pools reset once a wait or query observes completion without collectGarbage",
          "[size:large][rhi][metal][queue][command-pool][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto pool = device.createCommandPool(rachis::CommandPoolDesc{
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .label = "Metal frame-loop pool",
    });
    REQUIRE(pool);
    const auto submitFrame = [&]() -> rachis::Result<rachis::TimelinePoint> {
        auto command = device.acquireCommandBuffer(*pool);
        if (!command)
        {
            return std::unexpected(std::move(command.error()));
        }
        return device.submit(std::move(*command));
    };
    const auto resetMessage = [](const rachis::Status& status) {
        return status ? std::string("ok") : status.error().message;
    };

    auto waitedFrame = submitFrame();
    REQUIRE(waitedFrame);
    REQUIRE(device.wait(*waitedFrame).has_value());
    auto afterWait = device.resetCommandPool(*pool);
    INFO("reset after wait: " << resetMessage(afterWait));
    REQUIRE(afterWait.has_value());

    auto boundedFrame = submitFrame();
    REQUIRE(boundedFrame);
    auto bounded = device.wait(*boundedFrame, std::chrono::seconds(5));
    REQUIRE(bounded);
    REQUIRE(*bounded == rachis::TimelinePointStatus::complete);
    auto afterBoundedWait = device.resetCommandPool(*pool);
    INFO("reset after bounded wait: " << resetMessage(afterBoundedWait));
    REQUIRE(afterBoundedWait.has_value());

    auto queriedFrame = submitFrame();
    REQUIRE(queriedFrame);
    const auto queryDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    auto queried = device.queryTimelinePoint(*queriedFrame);
    while (queried && *queried != rachis::TimelinePointStatus::complete &&
           std::chrono::steady_clock::now() < queryDeadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        queried = device.queryTimelinePoint(*queriedFrame);
    }
    REQUIRE(queried);
    REQUIRE(*queried == rachis::TimelinePointStatus::complete);
    auto afterQuery = device.resetCommandPool(*pool);
    INFO("reset after query: " << resetMessage(afterQuery));
    REQUIRE(afterQuery.has_value());

    auto polledFrame = submitFrame();
    REQUIRE(polledFrame);
    const auto pollDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (device.completedValue(rachis::QueueType::graphics) < polledFrame->value() &&
           std::chrono::steady_clock::now() < pollDeadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(device.completedValue(rachis::QueueType::graphics) >= polledFrame->value());
    auto afterPoll = device.resetCommandPool(*pool);
    INFO("reset after completed value: " << resetMessage(afterPoll));
    REQUIRE(afterPoll.has_value());

    REQUIRE(device.destroyCommandPool(*pool).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal timeline waits release the resources a completed submission retained",
          "[size:large][rhi][metal][queue][lifetime][regression]")
{
    constexpr std::uint64_t bufferSize = 256;
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    const auto submitRetainedCopy = [&](std::weak_ptr<void>& observer,
                                        std::string_view label) -> rachis::Result<rachis::TimelinePoint> {
        auto source = device.createBuffer({
            .size = bufferSize,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
            .label = label,
        });
        if (!source)
        {
            return std::unexpected(std::move(source.error()));
        }
        auto destination = device.createBuffer({
            .size = bufferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::deviceLocal,
            .label = label,
        });
        if (!destination)
        {
            return std::unexpected(std::move(destination.error()));
        }
        auto command = acquire(device, rachis::QueueType::copy);
        if (!command)
        {
            return std::unexpected(std::move(command.error()));
        }
        if (auto copied =
                device.copyBuffer(*command, {.source = *source, .destination = *destination, .size = bufferSize});
            !copied)
        {
            return std::unexpected(std::move(copied.error()));
        }
        observer = rachis::metal::boundBufferOwnerObserver(*destination);
        auto submitted = device.submit(std::move(*command));
        if (!submitted)
        {
            return submitted;
        }
        if (auto destroyed = device.destroyBuffer(*source); !destroyed)
        {
            return std::unexpected(std::move(destroyed.error()));
        }
        if (auto destroyed = device.destroyBuffer(*destination); !destroyed)
        {
            return std::unexpected(std::move(destroyed.error()));
        }
        return submitted;
    };

    std::weak_ptr<void> waitedObserver;
    auto waited = submitRetainedCopy(waitedObserver, "Metal wait-retained copy");
    REQUIRE(waited);
    REQUIRE(device.wait(*waited).has_value());
    CHECK(waitedObserver.expired());

    std::weak_ptr<void> queriedObserver;
    auto queried = submitRetainedCopy(queriedObserver, "Metal query-retained copy");
    REQUIRE(queried);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    auto status = device.queryTimelinePoint(*queried);
    while (status && *status != rachis::TimelinePointStatus::complete && std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        status = device.queryTimelinePoint(*queried);
    }
    REQUIRE(status);
    REQUIRE(*status == rachis::TimelinePointStatus::complete);
    CHECK(queriedObserver.expired());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal same-queue submissions are thread safe and recyclable", "[size:large][rhi][metal][queue][concurrency]")
{
    constexpr std::size_t concurrentCount = 8;
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    auto commandPool = device.createCommandPool(rachis::CommandPoolDesc{
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = static_cast<std::uint32_t>(concurrentCount),
        .label = "Metal concurrent graphics pool",
    });
    REQUIRE(commandPool);

    std::vector<rachis::CommandBuffer> commands;
    commands.reserve(concurrentCount);
    for (std::size_t index = 0; index < concurrentCount; ++index)
    {
        auto command = device.acquireCommandBuffer(*commandPool);
        REQUIRE(command);
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
        REQUIRE(point);
        REQUIRE(device.wait(*point).has_value());
    }
    device.collectGarbage();

    for (std::size_t index = 0; index < concurrentCount; ++index)
    {
        auto command = device.acquireCommandBuffer(*commandPool);
        REQUIRE(command);
    }
    REQUIRE(device.destroyCommandPool(*commandPool).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal cross-queue submissions cannot deadlock device idle",
          "[size:large][rhi][metal][queue][concurrency][idle]")
{
    constexpr std::size_t iterationCount = 16;
    constexpr std::array queueTypes{
        rachis::QueueType::graphics,
        rachis::QueueType::compute,
        rachis::QueueType::copy,
    };

    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    std::array<rachis::CommandPoolHandle, queueTypes.size()> commandPools{};
    for (std::size_t queueIndex = 0; queueIndex < queueTypes.size(); ++queueIndex)
    {
        auto commandPool = device.createCommandPool(rachis::CommandPoolDesc{
            .queueType = queueTypes[queueIndex],
            .maxCommandBuffers = static_cast<std::uint32_t>(iterationCount),
            .label = "Metal concurrent cross-queue pool",
        });
        REQUIRE(commandPool);
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
            REQUIRE(submission);
        }
    }
    REQUIRE(device.waitIdle().has_value());
    device.collectGarbage();
    for (const auto commandPool : commandPools)
    {
        REQUIRE(device.destroyCommandPool(commandPool).has_value());
    }

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

#endif
