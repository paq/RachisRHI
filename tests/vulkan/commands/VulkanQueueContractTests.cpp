#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include "Internal/CommandPoolRetirementTestHook.h"
#include "Internal/VulkanDevice.h"
#include "Internal/VulkanResources.h"
#include "Rachis/TestSupport/PresentInjection.h"
#include "Vulkan/VulkanTestHooks.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <optional>
#include <span>
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

    [[nodiscard]] rachis::Result<rachis::Device> createVulkanDevice()
    {
        return rachis::createDevice(vulkanDesc());
    }

    [[nodiscard]] rachis::Result<rachis::CommandBuffer> acquire(rachis::Device& device, rachis::QueueType queueType)
    {
        return rachis::test::acquireCommandBuffer(device, queueType);
    }

    template<typename T>
    [[nodiscard]] std::string errorMessage(const rachis::Result<T>& result)
    {
        return result ? std::string{} : result.error().message;
    }

    class ForcedRetirementFences final
    {
    public:
        ForcedRetirementFences() noexcept
        {
            rachis::vulkan::test::forceSubmissionRetirementFences().store(true, std::memory_order_release);
        }

        ~ForcedRetirementFences()
        {
            reportStatus(VK_SUCCESS);
            rachis::vulkan::test::forceSubmissionRetirementFences().store(false, std::memory_order_release);
        }

        ForcedRetirementFences(const ForcedRetirementFences&) = delete;
        ForcedRetirementFences& operator=(const ForcedRetirementFences&) = delete;
        ForcedRetirementFences(ForcedRetirementFences&&) = delete;
        ForcedRetirementFences& operator=(ForcedRetirementFences&&) = delete;

        static void hold(bool held) noexcept
        {
            reportStatus(held ? VK_NOT_READY : VK_SUCCESS);
        }

        static void reportStatus(VkResult status) noexcept
        {
            rachis::vulkan::test::retirementFenceStatusOverride().store(static_cast<std::int32_t>(status),
                                                                        std::memory_order_release);
        }
    };

    enum class CompletionObserver
    {
        queryTimelinePoint,
        completedValue,
    };

    enum class QuarantineRecovery
    {
        resetCommandPool,
        acquire,
    };

    [[nodiscard]] rachis::Status pollUntilComplete(rachis::Device& device,
                                                   rachis::TimelinePoint point,
                                                   CompletionObserver observer)
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (std::chrono::steady_clock::now() < deadline)
        {
            if (observer == CompletionObserver::completedValue)
            {
                if (device.completedValue(point.queueType()) >= point.value())
                {
                    return {};
                }
                continue;
            }
            auto status = device.queryTimelinePoint(point);
            if (!status)
            {
                return std::unexpected(std::move(status.error()));
            }
            if (*status == rachis::TimelinePointStatus::complete)
            {
                return {};
            }
        }
        return std::unexpected(rachis::Error{
            rachis::ErrorCode::notReady,
            0,
            "The timeline point did not report complete within 10 seconds.",
        });
    }

} // namespace

TEST_CASE("Vulkan failed submissions keep command buffers and do not publish points",
          "[size:medium][rhi][vulkan][queue][validation]")
{
    auto created = createVulkanDevice();
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
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

    auto otherCreated = createVulkanDevice();
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
    REQUIRE(device.beginDebugGroup(*second, rachis::DebugMarkerDesc{.label = "Unclosed Vulkan group"}).has_value());
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

TEST_CASE("Vulkan per-queue timelines publish monotonic points and host waits",
          "[size:medium][rhi][vulkan][queue][timeline]")
{
    using namespace std::chrono_literals;

    auto created = createVulkanDevice();
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
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

    auto foreignCreated = createVulkanDevice();
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

TEST_CASE("Vulkan batched submissions consume every lease behind one timeline point",
          "[size:medium][rhi][vulkan][queue][batch]")
{
    auto created = createVulkanDevice();
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
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

TEST_CASE("Vulkan cross-queue waits and idle drain shared or dedicated families",
          "[size:medium][rhi][vulkan][queue][idle]")
{
    auto created = createVulkanDevice();
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    const auto graphicsProps = device.queueProperties(rachis::QueueType::graphics);
    const auto computeProps = device.queueProperties(rachis::QueueType::compute);
    const auto copyProps = device.queueProperties(rachis::QueueType::copy);
    CAPTURE(graphicsProps.timestampQueries, computeProps.timestampQueries, copyProps.timestampQueries);

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

TEST_CASE("Vulkan submitted fills complete on graphics, compute, and copy queues",
          "[size:medium][rhi][vulkan][queue][fill]")
{
    constexpr std::uint64_t size = 256;
    auto created = createVulkanDevice();
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    const std::array queueTypes{
        std::pair{rachis::QueueType::copy, 0x11111111U},
        std::pair{rachis::QueueType::compute, 0x22222222U},
        std::pair{rachis::QueueType::graphics, 0x33333333U},
    };
    std::vector<rachis::TimelinePoint> points;
    std::vector<rachis::BufferHandle> buffers;
    points.reserve(queueTypes.size());
    buffers.reserve(queueTypes.size());

    std::optional<rachis::TimelinePoint> previous;
    for (const auto& [queueType, pattern] : queueTypes)
    {
        auto buffer = device.createBuffer(rachis::BufferDesc{
            .size = size,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Vulkan queue fill readback",
        });
        REQUIRE(buffer);
        auto command = acquire(device, queueType);
        REQUIRE(command);
        REQUIRE(rachis::vulkan::recordVulkanFillBuffer(*command, *buffer, pattern).has_value());
        rachis::Result<rachis::TimelinePoint> submitted;
        if (previous)
        {
            const std::array waits{*previous};
            submitted = device.submit(std::move(*command), waits);
        }
        else
        {
            submitted = device.submit(std::move(*command));
        }
        REQUIRE(submitted);
        previous = *submitted;
        points.push_back(*submitted);
        buffers.push_back(*buffer);
    }

    REQUIRE(device.wait(points.back()).has_value());
    device.collectGarbage();

    for (std::size_t index = 0; index < queueTypes.size(); ++index)
    {
        std::vector<std::byte> bytes(size);
        REQUIRE(device.readBuffer(buffers[index], 0, bytes).has_value());
        const auto expected = queueTypes[index].second;
        std::uint32_t observed = 0;
        std::memcpy(&observed, bytes.data(), sizeof(observed));
        CAPTURE(index, expected, observed);
        CHECK(observed == expected);
        REQUIRE(device.destroyBuffer(buffers[index]).has_value());
    }

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan bounded pools reject in-flight reset and recycle after exact completion",
          "[size:medium][rhi][vulkan][queue][command-pool]")
{
    const ForcedRetirementFences fences;
    auto created = createVulkanDevice();
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    auto pool = device.createCommandPool(rachis::CommandPoolDesc{
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .label = "Vulkan in-flight pool",
    });
    REQUIRE(pool);
    auto command = device.acquireCommandBuffer(*pool);
    REQUIRE(command);
    auto exhausted = device.acquireCommandBuffer(*pool);
    REQUIRE_FALSE(exhausted);
    CHECK(exhausted.error().code == rachis::ErrorCode::resourceExhausted);

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    ForcedRetirementFences::hold(true);
    auto inFlightReset = device.resetCommandPool(*pool);
    ForcedRetirementFences::hold(false);
    REQUIRE_FALSE(inFlightReset);
    CHECK(inFlightReset.error().code == rachis::ErrorCode::invalidState);

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

TEST_CASE("Vulkan one-entry command pools reuse the entry a transient submit failure quarantined",
          "[size:medium][rhi][vulkan][queue][command-pool][fault]")
{
    constexpr std::uint64_t size = 256;
    const auto recovery = GENERATE(QuarantineRecovery::resetCommandPool, QuarantineRecovery::acquire);
    CAPTURE(recovery);

    auto created = createVulkanDevice();
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    auto pool = device.createCommandPool(rachis::CommandPoolDesc{
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .label = "Vulkan quarantine recovery pool",
    });
    REQUIRE(pool);
    auto buffer = device.createBuffer(rachis::BufferDesc{
        .size = size,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan quarantine recovery readback",
    });
    REQUIRE(buffer);

    for (std::uint32_t round = 0; round < 2; ++round)
    {
        CAPTURE(round);
        auto failing = device.acquireCommandBuffer(*pool);
        REQUIRE(failing);
        {
            const rachis::test::ScopedAtomicTestHook<std::int32_t> endFails(
                rachis::vulkan::test::nextEndCommandBufferResult(),
                static_cast<std::int32_t>(VK_ERROR_OUT_OF_HOST_MEMORY));
            auto failed = device.submit(std::move(*failing));
            REQUIRE_FALSE(failed);
            CHECK(failed.error().code == rachis::ErrorCode::outOfMemory);
        }
        REQUIRE(device.operationalState() == rachis::DeviceOperationalState::active);

        if (recovery == QuarantineRecovery::resetCommandPool)
        {
            REQUIRE(device.waitIdle().has_value());
            auto reset = device.resetCommandPool(*pool);
            INFO("resetCommandPool: " << errorMessage(reset));
            REQUIRE(reset.has_value());
        }

        auto command = device.acquireCommandBuffer(*pool);
        INFO("acquireCommandBuffer after the failed submit: " << errorMessage(command));
        REQUIRE(command);
        const auto pattern = 0x5A5A0000U + round;
        REQUIRE(rachis::vulkan::recordVulkanFillBuffer(*command, *buffer, pattern).has_value());
        auto point = device.submit(std::move(*command));
        INFO("submit: " << errorMessage(point));
        REQUIRE(point);
        REQUIRE(device.wait(*point).has_value());
        device.collectGarbage();

        std::vector<std::byte> bytes(size);
        REQUIRE(device.readBuffer(*buffer, 0, bytes).has_value());
        std::uint32_t observed = 0;
        std::memcpy(&observed, bytes.data(), sizeof(observed));
        CHECK(observed == pattern);
    }

    REQUIRE(device.destroyBuffer(*buffer).has_value());
    REQUIRE(device.destroyCommandPool(*pool).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan one-entry command pools reset and reacquire once a polled point reports complete",
          "[size:medium][rhi][vulkan][queue][command-pool]")
{
    constexpr int iterationCount = 500;
    const auto forceFences = GENERATE(false, true);
    const auto observer = GENERATE(CompletionObserver::queryTimelinePoint, CompletionObserver::completedValue);
    CAPTURE(forceFences, observer);

    std::optional<ForcedRetirementFences> forced;
    if (forceFences)
    {
        forced.emplace();
    }
    auto created = createVulkanDevice();
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    auto pool = device.createCommandPool(rachis::CommandPoolDesc{
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .label = "Vulkan polled reuse pool",
    });
    REQUIRE(pool);

    std::size_t pollFailures = 0;
    std::size_t resetFailures = 0;
    std::size_t acquireFailures = 0;
    std::string firstFailure;
    for (int iteration = 0; iteration < iterationCount; ++iteration)
    {
        auto command = device.acquireCommandBuffer(*pool);
        if (!command)
        {
            if (acquireFailures++ == 0 && firstFailure.empty())
            {
                firstFailure = "acquireCommandBuffer: " + command.error().message;
            }
            REQUIRE(device.waitIdle().has_value());
            command = device.acquireCommandBuffer(*pool);
            REQUIRE(command);
        }
        auto point = device.submit(std::move(*command));
        REQUIRE(point);
        auto polled = pollUntilComplete(device, *point, observer);
        if (!polled)
        {
            if (pollFailures++ == 0 && firstFailure.empty())
            {
                firstFailure = "poll: " + polled.error().message;
            }
            REQUIRE(device.wait(*point).has_value());
        }
        auto reset = device.resetCommandPool(*pool);
        if (!reset && resetFailures++ == 0 && firstFailure.empty())
        {
            firstFailure = "resetCommandPool: " + reset.error().message;
        }
    }
    REQUIRE(device.waitIdle().has_value());

    INFO("first failure: " << firstFailure);
    CHECK(pollFailures == 0);
    CHECK(resetFailures == 0);
    CHECK(acquireFailures == 0);
    REQUIRE(device.destroyCommandPool(*pool).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan unretained timestamp pools can be destroyed once a polled point reports complete",
          "[size:medium][rhi][vulkan][queue][timeline][lifetime]")
{
    constexpr int iterationCount = 1000;
    constexpr std::uint32_t queryCount = 4;
    const auto observer = GENERATE(CompletionObserver::queryTimelinePoint, CompletionObserver::completedValue);
    CAPTURE(observer);

    auto created = createVulkanDevice();
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    if (!device.queueProperties(rachis::QueueType::graphics).timestampQueries)
    {
        SKIP("The Vulkan graphics queue has no timestamp queries.");
    }
    auto pool = device.createCommandPool(rachis::CommandPoolDesc{
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 2,
        .label = "Vulkan unretained timestamp pool",
        .flags = rachis::CommandPoolFlag::unretainedResources,
    });
    REQUIRE(pool);

    for (int iteration = 0; iteration < iterationCount; ++iteration)
    {
        auto queries = device.createTimestampQueryPool({
            .queueType = rachis::QueueType::graphics,
            .queryCount = queryCount,
            .label = "Vulkan unretained timestamps",
        });
        REQUIRE(queries);
        auto command = device.acquireCommandBuffer(*pool);
        REQUIRE(command);
        for (std::uint32_t index = 0; index < queryCount; ++index)
        {
            REQUIRE(device.writeTimestamp(*command, *queries, index).has_value());
        }
        auto point = device.submit(std::move(*command));
        REQUIRE(point);
        auto polled = pollUntilComplete(device, *point, observer);
        INFO("poll: " << errorMessage(polled));
        REQUIRE(polled.has_value());
        REQUIRE(device.destroyQueryPool(*queries).has_value());
        REQUIRE(device.wait(*point).has_value());
    }
    REQUIRE(device.waitIdle().has_value());
    REQUIRE(device.destroyCommandPool(*pool).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan timeline observers report a point complete only once its retirement fence signals",
          "[size:medium][rhi][vulkan][queue][timeline]")
{
    using namespace std::chrono_literals;

    const ForcedRetirementFences fences;
    auto created = createVulkanDevice();
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    auto pool = device.createCommandPool(rachis::CommandPoolDesc{
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .label = "Vulkan held retirement pool",
    });
    REQUIRE(pool);
    auto command = device.acquireCommandBuffer(*pool);
    REQUIRE(command);
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);

    ForcedRetirementFences::hold(true);
    REQUIRE(device.wait(*submitted).has_value());
    auto heldQuery = device.queryTimelinePoint(*submitted);
    REQUIRE(heldQuery);
    CHECK(*heldQuery == rachis::TimelinePointStatus::pending);
    CHECK(device.completedValue(rachis::QueueType::graphics) == submitted->value() - 1);
    auto heldPoll = device.wait(*submitted, 0ns);
    REQUIRE(heldPoll);
    CHECK(*heldPoll == rachis::TimelinePointStatus::pending);
    auto heldWait = device.wait(*submitted, 10ms);
    REQUIRE(heldWait);
    CHECK(*heldWait == rachis::TimelinePointStatus::pending);
    auto heldReset = device.resetCommandPool(*pool);
    REQUIRE_FALSE(heldReset);
    CHECK(heldReset.error().code == rachis::ErrorCode::invalidState);
    auto heldAcquire = device.acquireCommandBuffer(*pool);
    REQUIRE_FALSE(heldAcquire);
    CHECK(heldAcquire.error().code == rachis::ErrorCode::resourceExhausted);

    ForcedRetirementFences::hold(false);
    auto released = device.resetCommandPool(*pool);
    INFO("resetCommandPool: " << errorMessage(released));
    REQUIRE(released.has_value());
    auto releasedQuery = device.queryTimelinePoint(*submitted);
    REQUIRE(releasedQuery);
    CHECK(*releasedQuery == rachis::TimelinePointStatus::complete);
    CHECK(device.completedValue(rachis::QueueType::graphics) == submitted->value());
    auto releasedPoll = device.wait(*submitted, 0ns);
    REQUIRE(releasedPoll);
    CHECK(*releasedPoll == rachis::TimelinePointStatus::complete);
    auto recycled = device.acquireCommandBuffer(*pool);
    REQUIRE(recycled);
    recycled = std::unexpected(rachis::Error{});
    REQUIRE(device.destroyCommandPool(*pool).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan timeline observers report a lost retirement fence as device loss",
          "[size:medium][rhi][vulkan][queue][timeline][loss]")
{
    using namespace std::chrono_literals;

    const ForcedRetirementFences fences;
    auto created = createVulkanDevice();
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    auto command = acquire(device, rachis::QueueType::graphics);
    REQUIRE(command);
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    ForcedRetirementFences::hold(true);
    REQUIRE(device.wait(*submitted).has_value());

    ForcedRetirementFences::reportStatus(VK_ERROR_DEVICE_LOST);
    rachis::Result<rachis::TimelinePointStatus> observed = rachis::TimelinePointStatus::complete;
    SECTION("queryTimelinePoint")
    {
        observed = device.queryTimelinePoint(*submitted);
    }
    SECTION("wait with a zero timeout")
    {
        observed = device.wait(*submitted, 0ns);
    }
    SECTION("wait with a timeout")
    {
        observed = device.wait(*submitted, 10ms);
    }
    ForcedRetirementFences::hold(false);

    REQUIRE_FALSE(observed);
    CHECK(observed.error().code == rachis::ErrorCode::deviceLost);
    CHECK(device.operationalState() == rachis::DeviceOperationalState::lost);
}

TEST_CASE("Vulkan same-queue submissions are thread safe and recyclable",
          "[size:medium][rhi][vulkan][queue][concurrency]")
{
    constexpr std::size_t concurrentCount = 8;
    auto created = createVulkanDevice();
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    auto commandPool = device.createCommandPool(rachis::CommandPoolDesc{
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = static_cast<std::uint32_t>(concurrentCount),
        .label = "Vulkan concurrent graphics pool",
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

TEST_CASE("Vulkan cross-queue submissions cannot deadlock device idle",
          "[size:medium][rhi][vulkan][queue][concurrency][idle]")
{
    constexpr std::size_t iterationCount = 16;
    constexpr std::array queueTypes{
        rachis::QueueType::graphics,
        rachis::QueueType::compute,
        rachis::QueueType::copy,
    };

    auto created = createVulkanDevice();
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    std::array<rachis::CommandPoolHandle, queueTypes.size()> commandPools{};
    for (std::size_t queueIndex = 0; queueIndex < queueTypes.size(); ++queueIndex)
    {
        auto commandPool = device.createCommandPool(rachis::CommandPoolDesc{
            .queueType = queueTypes[queueIndex],
            .maxCommandBuffers = static_cast<std::uint32_t>(iterationCount),
            .label = "Vulkan concurrent cross-queue pool",
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

TEST_CASE("Vulkan shared command pools synchronize concurrent recording, abandon, submit, and garbage collection",
          "[size:medium][rhi][vulkan][queue][command-pool][concurrency]")
{
    constexpr std::size_t workerCount = 4;
    constexpr std::size_t iterationCount = 12;
    constexpr std::uint64_t fillBytes = 256;
    constexpr std::uint32_t poolCapacity = static_cast<std::uint32_t>(workerCount * 2);

    auto created = createVulkanDevice();
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    auto sharedPool = device.createCommandPool({
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = poolCapacity,
        .label = "Vulkan shared submit/abandon pool",
        .flags = rachis::CommandPoolFlag::allowIndividualReset,
    });
    REQUIRE(sharedPool);

    std::array<rachis::BufferHandle, workerCount> buffers{};
    for (std::size_t workerIndex = 0; workerIndex < workerCount; ++workerIndex)
    {
        auto buffer = device.createBuffer({
            .size = fillBytes,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Vulkan concurrent shared-pool fill readback",
        });
        REQUIRE(buffer);
        buffers[workerIndex] = *buffer;
    }

    std::array<bool, workerCount> workerOk{};
    workerOk.fill(true);
    std::atomic<bool> stopCollector{false};
    std::barrier startBarrier(static_cast<std::ptrdiff_t>(workerCount));

    {
        std::jthread collector([&device, &stopCollector] {
            while (!stopCollector.load(std::memory_order_acquire))
            {
                device.collectGarbage();
                std::this_thread::yield();
            }
        });

        std::vector<std::jthread> workers;
        workers.reserve(workerCount);
        for (std::size_t workerIndex = 0; workerIndex < workerCount; ++workerIndex)
        {
            workers.emplace_back([&, workerIndex] {
                startBarrier.arrive_and_wait();
                for (std::size_t iteration = 0; iteration < iterationCount; ++iteration)
                {
                    auto abandoned = device.acquireCommandBuffer(*sharedPool);
                    auto submittedCmd = device.acquireCommandBuffer(*sharedPool);
                    if (!abandoned || !submittedCmd)
                    {
                        workerOk[workerIndex] = false;
                        break;
                    }

                    if (!device.beginDebugGroup(*abandoned, {.label = "Abandoned while sibling submits"}) ||
                        !rachis::vulkan::recordVulkanFillBuffer(*abandoned, buffers[workerIndex], 0xDEADBEEFU))
                    {
                        workerOk[workerIndex] = false;
                        break;
                    }
                    abandoned = std::unexpected(rachis::Error{});

                    const auto pattern =
                        static_cast<std::uint32_t>(0x10000000U + workerIndex * 0x1000U + iteration + 1U);
                    if (!device.beginDebugGroup(*submittedCmd, {.label = "Submitted shared-pool fill"}) ||
                        !rachis::vulkan::recordVulkanFillBuffer(*submittedCmd, buffers[workerIndex], pattern) ||
                        !device.endDebugGroup(*submittedCmd))
                    {
                        workerOk[workerIndex] = false;
                        break;
                    }

                    auto point = device.submit(std::move(*submittedCmd));
                    if (!point || !device.wait(*point))
                    {
                        workerOk[workerIndex] = false;
                        break;
                    }
                    device.collectGarbage();
                }
            });
        }
        workers.clear();
        stopCollector.store(true, std::memory_order_release);
    }

    for (const bool ok : workerOk)
    {
        REQUIRE(ok);
    }
    device.collectGarbage();

    for (std::size_t workerIndex = 0; workerIndex < workerCount; ++workerIndex)
    {
        std::vector<std::byte> bytes(fillBytes);
        REQUIRE(device.readBuffer(buffers[workerIndex], 0, bytes).has_value());
        std::uint32_t observed = 0;
        std::memcpy(&observed, bytes.data(), sizeof(observed));
        const auto expected = static_cast<std::uint32_t>(0x10000000U + workerIndex * 0x1000U + iterationCount);
        CHECK(observed == expected);
        REQUIRE(device.destroyBuffer(buffers[workerIndex]).has_value());
    }

    std::vector<rachis::CommandBuffer> reacquired;
    reacquired.reserve(poolCapacity);
    for (std::uint32_t index = 0; index < poolCapacity; ++index)
    {
        auto command = device.acquireCommandBuffer(*sharedPool);
        REQUIRE(command);
        reacquired.push_back(std::move(*command));
    }
    reacquired.clear();
    REQUIRE(device.destroyCommandPool(*sharedPool).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan collector parked after retiring a submission leaves its command-pool entry reusable",
          "[size:medium][rhi][vulkan][queue][command-pool][concurrency]")
{
    using rachis::detail::test::CommandPoolRetirementParking;
    constexpr std::uint64_t transferSize = 256;

    auto created = createVulkanDevice();
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    auto commandPool = device.createCommandPool({
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .label = "Vulkan parked retirement test pool",
    });
    REQUIRE(commandPool.has_value());
    auto upload = device.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan parked retirement upload",
    });
    auto readback = device.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan parked retirement readback",
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

TEST_CASE("Vulkan one-entry command pools reacquire after their own wait while other threads collect",
          "[size:medium][rhi][vulkan][queue][command-pool][concurrency]")
{
    constexpr std::size_t workerCount = 8;
    constexpr std::size_t iterationCount = 200;
    constexpr std::uint64_t transferSize = 256;

    auto created = createVulkanDevice();
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    std::array<rachis::CommandPoolHandle, workerCount> commandPools{};
    std::array<rachis::BufferHandle, workerCount> uploads{};
    std::array<rachis::BufferHandle, workerCount> readbacks{};
    for (std::size_t workerIndex = 0; workerIndex < workerCount; ++workerIndex)
    {
        auto commandPool = device.createCommandPool({
            .queueType = rachis::QueueType::graphics,
            .maxCommandBuffers = 1,
            .label = "Vulkan one-entry reacquisition test pool",
        });
        auto upload = device.createBuffer({
            .size = transferSize,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
            .label = "Vulkan one-entry reacquisition upload",
        });
        auto readback = device.createBuffer({
            .size = transferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Vulkan one-entry reacquisition readback",
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

TEST_CASE("Vulkan exported timeline points synchronize concurrent exports into independent handles",
          "[size:medium][rhi][vulkan][queue][timeline][interop][concurrency]")
{
    constexpr std::array queueTypes{
        rachis::QueueType::graphics,
        rachis::QueueType::compute,
        rachis::QueueType::copy,
    };
    constexpr std::size_t exportsPerQueue = 8;

    auto consumerCreated = createVulkanDevice();
    if (rachis::test::skipWithoutCreatedDevice(consumerCreated))
    {
        return;
    }
    auto consumer = std::move(*consumerCreated);

    auto producerCreated = createVulkanDevice();
    REQUIRE(producerCreated);
    auto producer = std::move(*producerCreated);

    const bool sharedTimelinesSupported = rachis::test::requireCapabilities(producer).sharedTimelineHandles &&
                                          rachis::test::requireCapabilities(consumer).sharedTimelineHandles;

    std::array<std::vector<rachis::TimelinePoint>, queueTypes.size()> points{};
    for (std::size_t queueIndex = 0; queueIndex < queueTypes.size(); ++queueIndex)
    {
        points[queueIndex].reserve(exportsPerQueue);
        for (std::size_t index = 0; index < exportsPerQueue; ++index)
        {
            auto command = acquire(producer, queueTypes[queueIndex]);
            REQUIRE(command);
            auto submitted = producer.submit(std::move(*command));
            REQUIRE(submitted);
            points[queueIndex].push_back(*submitted);
        }
    }

    std::array<std::vector<rachis::Result<rachis::SharedTimelinePoint>>, queueTypes.size()> exported{};
    for (auto& queueExports : exported)
    {
        queueExports.reserve(exportsPerQueue);
        for (std::size_t index = 0; index < exportsPerQueue; ++index)
        {
            queueExports.emplace_back(std::unexpect);
        }
    }

    std::barrier startBarrier(static_cast<std::ptrdiff_t>(queueTypes.size()));
    {
        std::vector<std::jthread> workers;
        workers.reserve(queueTypes.size());
        for (std::size_t queueIndex = 0; queueIndex < queueTypes.size(); ++queueIndex)
        {
            workers.emplace_back([&, queueIndex] {
                startBarrier.arrive_and_wait();
                for (std::size_t index = 0; index < exportsPerQueue; ++index)
                {
                    exported[queueIndex][index] = producer.exportTimelinePoint(points[queueIndex][index]);
                }
            });
        }
    }

    if (!sharedTimelinesSupported)
    {
        for (const auto& queueExports : exported)
        {
            for (const auto& item : queueExports)
            {
                REQUIRE_FALSE(item);
                CHECK(item.error().code == rachis::ErrorCode::unsupported);
            }
        }
        REQUIRE(producer.waitIdle().has_value());
        producer.collectGarbage();
    }
    else
    {
        std::vector<std::uintptr_t> handleValues;
        handleValues.reserve(queueTypes.size() * exportsPerQueue);
        for (std::size_t queueIndex = 0; queueIndex < queueTypes.size(); ++queueIndex)
        {
            for (std::size_t index = 0; index < exportsPerQueue; ++index)
            {
                REQUIRE(exported[queueIndex][index].has_value());
                CHECK(exported[queueIndex][index]->value == points[queueIndex][index].value());
                handleValues.push_back(exported[queueIndex][index]->handle.view().value);
            }
        }
        std::ranges::sort(handleValues);
        CHECK(std::ranges::adjacent_find(handleValues) == handleValues.end());

        for (auto& queueExports : exported)
        {
            for (std::size_t index = 0; index < exportsPerQueue; index += 2)
            {
                queueExports[index]->handle.reset();
            }
        }
        for (std::size_t queueIndex = 0; queueIndex < queueTypes.size(); ++queueIndex)
        {
            for (std::size_t index = 1; index < exportsPerQueue; index += 2)
            {
                CAPTURE(queueIndex, index);
                auto importedWait =
                    consumer.enqueueSharedTimelineWait(queueTypes[queueIndex], exported[queueIndex][index]->view());
                REQUIRE(importedWait);
                REQUIRE(consumer.wait(*importedWait).has_value());
            }
        }

        REQUIRE(producer.waitIdle().has_value());
        producer.collectGarbage();

        auto postIdleWait = consumer.enqueueSharedTimelineWait(rachis::QueueType::graphics, exported[0].back()->view());
        REQUIRE(postIdleWait);
        REQUIRE(consumer.wait(*postIdleWait).has_value());

        for (auto& queueExports : exported)
        {
            for (auto& item : queueExports)
            {
                item->handle.reset();
            }
        }

        auto laterCommand = acquire(producer, rachis::QueueType::compute);
        REQUIRE(laterCommand);
        auto laterPoint = producer.submit(std::move(*laterCommand));
        REQUIRE(laterPoint);
        auto laterExport = producer.exportTimelinePoint(*laterPoint);
        REQUIRE(laterExport);
        CHECK(laterExport->value == laterPoint->value());
        auto consumerSyncWait = consumer.enqueueSharedTimelineWait(rachis::QueueType::compute, laterExport->view());
        REQUIRE(consumerSyncWait);
        REQUIRE(consumer.wait(*consumerSyncWait).has_value());
        const auto queryStatus = producer.queryTimelinePoint(*laterPoint);
        REQUIRE(queryStatus.has_value());
        CHECK(*queryStatus == rachis::TimelinePointStatus::complete);
        laterExport->handle.reset();

        consumer.collectGarbage();
    }

    const auto producerDiagnostics = producer.diagnosticCounts();
    CHECK(producerDiagnostics.errors == 0);
    CHECK(producerDiagnostics.corruptions == 0);
    const auto consumerDiagnostics = consumer.diagnosticCounts();
    CHECK(consumerDiagnostics.errors == 0);
    CHECK(consumerDiagnostics.corruptions == 0);
}

#if RACHIS_ENABLE_D3D12
namespace
{

    struct SharedTimelineProducer final
    {
        rachis::Device device;
        rachis::SharedTimelinePoint shared;
        std::uint64_t firstValue = 0;
        std::uint64_t signaledValue = 0;

        [[nodiscard]] rachis::SharedTimelinePointView at(std::uint64_t value) const noexcept
        {
            return rachis::SharedTimelinePointView{
                .handle = shared.handle.view(),
                .value = value,
            };
        }

        void signal(std::uint64_t value)
        {
            while (signaledValue < value)
            {
                auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
                REQUIRE(command);
                auto submitted = device.submit(std::move(*command));
                REQUIRE(submitted);
                signaledValue = submitted->value();
            }
        }
    };

    [[nodiscard]] SharedTimelineProducer createSharedTimelineProducer(const rachis::Device& consumer)
    {
        if (!rachis::test::requireCapabilities(consumer).sharedTimelineHandles)
        {
            SKIP("This Vulkan device cannot import shared timeline handles.");
        }
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .backend = rachis::Backend::d3d12,
            .adapterPreference = rachis::AdapterPreference::highPerformance,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
        });
        REQUIRE(created);
        const auto consumerAdapter = rachis::test::requireAdapterInfo(consumer);
        const auto producerAdapter = rachis::test::requireAdapterInfo(*created);
        if (producerAdapter.vendorId != consumerAdapter.vendorId ||
            producerAdapter.deviceId != consumerAdapter.deviceId)
        {
            SKIP("The Direct3D 12 producer and the Vulkan consumer use different adapters.");
        }

        SharedTimelineProducer producer{.device = std::move(*created)};
        auto command = rachis::test::acquireCommandBuffer(producer.device, rachis::QueueType::graphics);
        REQUIRE(command);
        auto first = producer.device.submit(std::move(*command));
        REQUIRE(first);
        REQUIRE(producer.device.wait(*first).has_value());
        auto shared = producer.device.exportTimelinePoint(*first);
        REQUIRE(shared);
        producer.shared = std::move(*shared);
        producer.firstValue = first->value();
        producer.signaledValue = first->value();
        return producer;
    }

    template<typename Call, typename Rescue>
    [[nodiscard]] auto callWithRescue(Call call, Rescue rescue)
    {
        auto pending = std::async(std::launch::async, std::move(call));
        const bool returnedPromptly = pending.wait_for(std::chrono::seconds{2}) == std::future_status::ready;
        if (!returnedPromptly)
        {
            rescue();
        }
        return std::pair{returnedPromptly, pending.get()};
    }

    [[nodiscard]] rachis::TimelinePointStatus requireStatus(const rachis::Result<rachis::TimelinePointStatus>& status)
    {
        REQUIRE(status);
        return *status;
    }

} // namespace

TEST_CASE("Vulkan shared timeline waits block the queue on the GPU and leave the host thread free",
          "[size:medium][requires:HardwareD3D12][rhi][vulkan][queue][timeline][interop]")
{
    using namespace std::chrono_literals;

    auto consumerCreated = createVulkanDevice();
    if (rachis::test::skipWithoutCreatedDevice(consumerCreated))
    {
        return;
    }
    auto consumer = std::move(*consumerCreated);
    auto producer = createSharedTimelineProducer(consumer);
    const auto first = producer.firstValue;
    const auto release = [&producer, first] { producer.signal(first + 2); };

    auto blocked = consumer.enqueueSharedTimelineWait(rachis::QueueType::graphics, producer.at(first + 1));
    REQUIRE(blocked);

    auto [laterReturned, blockedLater] = callWithRescue(
        [&consumer, &producer, first] {
            return consumer.enqueueSharedTimelineWait(rachis::QueueType::graphics, producer.at(first + 2));
        },
        release);
    REQUIRE(blockedLater);
    REQUIRE(laterReturned);

    auto command = acquire(consumer, rachis::QueueType::graphics);
    REQUIRE(command);
    auto [submitReturned, submitted] = callWithRescue(
        [&consumer, command = std::move(*command)]() mutable { return consumer.submit(std::move(command)); }, release);
    REQUIRE(submitted);
    REQUIRE(submitReturned);

    auto unrelated = acquire(consumer, rachis::QueueType::compute);
    REQUIRE(unrelated);
    auto unrelatedPoint = consumer.submit(std::move(*unrelated));
    REQUIRE(unrelatedPoint);
    CHECK(requireStatus(consumer.wait(*unrelatedPoint, 5s)) == rachis::TimelinePointStatus::complete);

    CHECK(requireStatus(consumer.queryTimelinePoint(*blocked)) == rachis::TimelinePointStatus::pending);
    CHECK(requireStatus(consumer.queryTimelinePoint(*blockedLater)) == rachis::TimelinePointStatus::pending);
    CHECK(requireStatus(consumer.wait(*submitted, 25ms)) == rachis::TimelinePointStatus::pending);

    producer.signal(first + 1);
    CHECK(requireStatus(consumer.wait(*blocked, 5s)) == rachis::TimelinePointStatus::complete);
    CHECK(requireStatus(consumer.wait(*blockedLater, 25ms)) == rachis::TimelinePointStatus::pending);
    CHECK(requireStatus(consumer.queryTimelinePoint(*submitted)) == rachis::TimelinePointStatus::pending);

    producer.signal(first + 2);
    CHECK(requireStatus(consumer.wait(*blockedLater, 5s)) == rachis::TimelinePointStatus::complete);
    CHECK(requireStatus(consumer.wait(*submitted, 5s)) == rachis::TimelinePointStatus::complete);

    auto trailing = acquire(consumer, rachis::QueueType::graphics);
    REQUIRE(trailing);
    auto trailingPoint = consumer.submit(std::move(*trailing));
    REQUIRE(trailingPoint);
    REQUIRE(consumer.wait(*trailingPoint).has_value());
    REQUIRE(consumer.waitIdle().has_value());
    REQUIRE(producer.device.waitIdle().has_value());

    const auto producerDiagnostics = producer.device.diagnosticCounts();
    CHECK(producerDiagnostics.errors == 0);
    CHECK(producerDiagnostics.corruptions == 0);
    const auto consumerDiagnostics = consumer.diagnosticCounts();
    CHECK(consumerDiagnostics.errors == 0);
    CHECK(consumerDiagnostics.corruptions == 0);
}

TEST_CASE("Vulkan shared timeline waits on one queue do not delay work or idle waits on the other queues",
          "[size:medium][requires:HardwareD3D12][rhi][vulkan][queue][timeline][interop][idle]")
{
    using namespace std::chrono_literals;

    auto consumerCreated = createVulkanDevice();
    if (rachis::test::skipWithoutCreatedDevice(consumerCreated))
    {
        return;
    }
    auto consumer = std::move(*consumerCreated);
    auto producer = createSharedTimelineProducer(consumer);

    constexpr std::array queueTypes{
        rachis::QueueType::graphics,
        rachis::QueueType::compute,
        rachis::QueueType::copy,
    };
    auto target = producer.firstValue;
    for (const auto waitQueue : queueTypes)
    {
        for (const auto workQueue : queueTypes)
        {
            if (waitQueue == workQueue)
            {
                continue;
            }
            CAPTURE(static_cast<int>(waitQueue), static_cast<int>(workQueue));
            ++target;
            const auto release = [&producer, target] { producer.signal(target); };

            auto gate = consumer.enqueueSharedTimelineWait(waitQueue, producer.at(target));
            REQUIRE(gate);

            auto command = acquire(consumer, workQueue);
            REQUIRE(command);
            auto work = consumer.submit(std::move(*command));
            REQUIRE(work);
            CHECK(requireStatus(consumer.wait(*work, 2s)) == rachis::TimelinePointStatus::complete);

            auto [idleReturned, idle] =
                callWithRescue([&consumer, workQueue] { return consumer.waitIdle(workQueue); }, release);
            CHECK(idleReturned);
            REQUIRE(idle.has_value());
            if (idleReturned)
            {
                CHECK(requireStatus(consumer.queryTimelinePoint(*gate)) == rachis::TimelinePointStatus::pending);
            }

            producer.signal(target);
            CHECK(requireStatus(consumer.wait(*gate, 5s)) == rachis::TimelinePointStatus::complete);
        }
    }

    REQUIRE(consumer.waitIdle().has_value());
    REQUIRE(producer.device.waitIdle().has_value());

    const auto producerDiagnostics = producer.device.diagnosticCounts();
    CHECK(producerDiagnostics.errors == 0);
    CHECK(producerDiagnostics.corruptions == 0);
    const auto consumerDiagnostics = consumer.diagnosticCounts();
    CHECK(consumerDiagnostics.errors == 0);
    CHECK(consumerDiagnostics.corruptions == 0);
}

TEST_CASE("Vulkan bounded pools reject reset while a submission waits behind a pending shared wait",
          "[size:medium][requires:HardwareD3D12][rhi][vulkan][queue][command-pool][timeline][interop]")
{
    using namespace std::chrono_literals;

    auto consumerCreated = createVulkanDevice();
    if (rachis::test::skipWithoutCreatedDevice(consumerCreated))
    {
        return;
    }
    auto consumer = std::move(*consumerCreated);
    auto producer = createSharedTimelineProducer(consumer);
    const auto target = producer.firstValue + 1;

    auto pool = consumer.createCommandPool(rachis::CommandPoolDesc{
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .label = "Vulkan gated in-flight pool",
    });
    REQUIRE(pool);
    auto gate = consumer.enqueueSharedTimelineWait(rachis::QueueType::graphics, producer.at(target));
    REQUIRE(gate);
    auto command = consumer.acquireCommandBuffer(*pool);
    REQUIRE(command);
    auto submitted = consumer.submit(std::move(*command));
    REQUIRE(submitted);

    const auto submittedStatus = requireStatus(consumer.wait(*submitted, 25ms));
    auto inFlightReset = consumer.resetCommandPool(*pool);
    auto exhausted = consumer.acquireCommandBuffer(*pool);
    producer.signal(target);
    CHECK(submittedStatus == rachis::TimelinePointStatus::pending);
    CHECK_FALSE(inFlightReset);
    if (!inFlightReset)
    {
        CHECK(inFlightReset.error().code == rachis::ErrorCode::invalidState);
    }
    CHECK_FALSE(exhausted);
    if (!exhausted)
    {
        CHECK(exhausted.error().code == rachis::ErrorCode::resourceExhausted);
    }
    exhausted = std::unexpected(rachis::Error{});

    CHECK(requireStatus(consumer.wait(*gate, 5s)) == rachis::TimelinePointStatus::complete);
    CHECK(requireStatus(consumer.wait(*submitted, 5s)) == rachis::TimelinePointStatus::complete);
    REQUIRE(consumer.resetCommandPool(*pool).has_value());
    auto recycled = consumer.acquireCommandBuffer(*pool);
    REQUIRE(recycled);
    auto recycledPoint = consumer.submit(std::move(*recycled));
    REQUIRE(recycledPoint);
    REQUIRE(consumer.wait(*recycledPoint).has_value());
    REQUIRE(consumer.destroyCommandPool(*pool).has_value());
    REQUIRE(consumer.waitIdle().has_value());
    REQUIRE(producer.device.waitIdle().has_value());

    const auto producerDiagnostics = producer.device.diagnosticCounts();
    CHECK(producerDiagnostics.errors == 0);
    CHECK(producerDiagnostics.corruptions == 0);
    const auto consumerDiagnostics = consumer.diagnosticCounts();
    CHECK(consumerDiagnostics.errors == 0);
    CHECK(consumerDiagnostics.corruptions == 0);
}
#endif
