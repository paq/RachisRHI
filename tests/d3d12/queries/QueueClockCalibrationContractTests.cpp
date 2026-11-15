#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstring>

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

TEST_CASE("queue clock calibration correlates resolved GPU ticks with the host monotonic clock",
          "[size:medium][rhi][query][timestamp][calibration][warp]")
{
    const rachis::Device invalidDevice;
    auto invalidDeviceSample = invalidDevice.calibrateQueueClock(rachis::QueueType::graphics);
    REQUIRE_FALSE(invalidDeviceSample.has_value());
    CHECK(invalidDeviceSample.error().code == rachis::ErrorCode::invalidState);

    auto created = createWarpDevice();
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto invalidQueueSample = device.calibrateQueueClock(static_cast<rachis::QueueType>(255));
    REQUIRE_FALSE(invalidQueueSample.has_value());
    CHECK(invalidQueueSample.error().code == rachis::ErrorCode::invalidArgument);

    const auto properties = device.queueProperties(rachis::QueueType::graphics);
    REQUIRE(properties.timestampQueries);
    REQUIRE(properties.timestampFrequency > 0);

    auto before = device.calibrateQueueClock(rachis::QueueType::graphics);
    REQUIRE(before.has_value());
    CHECK(before->gpuFrequency == properties.timestampFrequency);
    CHECK(before->cpuFrequency > 0);
    CHECK(before->cpuTimestamp > 0);

    auto pool = device.createTimestampQueryPool({
        .queueType = rachis::QueueType::graphics,
        .queryCount = 1,
        .label = "Calibrated timestamp pool",
    });
    REQUIRE(pool.has_value());
    auto results = device.createBuffer({
        .size = rachis::timestampQueryResultSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Calibrated timestamp result",
    });
    REQUIRE(results.has_value());
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device.writeTimestamp(*command, *pool, 0).has_value());
    REQUIRE(device
                .resolveTimestamps(*command,
                                   {
                                       .queryPool = *pool,
                                       .queryCount = 1,
                                       .destination = *results,
                                   })
                .has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto mapped = device.mapBuffer(*results);
    REQUIRE(mapped.has_value());
    std::uint64_t resolvedTimestamp = 0;
    std::memcpy(&resolvedTimestamp, mapped->data, sizeof(resolvedTimestamp));
    REQUIRE(device.unmapBuffer(*results).has_value());

    auto after = device.calibrateQueueClock(rachis::QueueType::graphics);
    REQUIRE(after.has_value());
    CAPTURE(before->gpuTimestamp, resolvedTimestamp, after->gpuTimestamp, before->cpuTimestamp, after->cpuTimestamp);
    CHECK(resolvedTimestamp >= before->gpuTimestamp);
    CHECK(resolvedTimestamp <= after->gpuTimestamp);
    CHECK(after->cpuTimestamp >= before->cpuTimestamp);
    CHECK(after->gpuFrequency == before->gpuFrequency);
    CHECK(after->cpuFrequency == before->cpuFrequency);

    REQUIRE(device.destroyQueryPool(*pool).has_value());
    REQUIRE(device.destroyBuffer(*results).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
