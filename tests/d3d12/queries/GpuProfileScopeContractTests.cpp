#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>

namespace
{

    constexpr std::uint64_t resultBufferSize = 4 * rachis::timestampQueryResultSize;

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

TEST_CASE("nested GPU profile scopes pair timestamps with an owned debug hierarchy",
          "[size:medium][rhi][query][gpu-profile][warp]")
{
    auto created = createWarpDevice();
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    const auto properties = device.queueProperties(rachis::QueueType::graphics);
    REQUIRE(properties.inlineTimestampWrites);

    auto pool = device.createTimestampQueryPool({
        .queueType = rachis::QueueType::graphics,
        .queryCount = 4,
        .label = "GPU profile timestamps",
    });
    REQUIRE(pool.has_value());
    auto results = device.createBuffer({
        .size = resultBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "GPU profile results",
    });
    REQUIRE(results.has_value());
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device.beginDebugGroup(*command, {.label = "CPU frame annotation"}).has_value());

    CHECK_FALSE(device
                    .beginGpuProfileScope(*command,
                                          {
                                              .queryPool = *pool,
                                              .beginningQueryIndex = 0,
                                              .endQueryIndex = 0,
                                              .label = "invalid duplicate",
                                          })
                    .has_value());
    CHECK_FALSE(device
                    .beginGpuProfileScope(*command,
                                          {
                                              .queryPool = *pool,
                                              .beginningQueryIndex = 0,
                                              .endQueryIndex = 4,
                                              .label = "invalid range",
                                          })
                    .has_value());
    CHECK_FALSE(device
                    .beginGpuProfileScope(*command,
                                          {
                                              .queryPool = *pool,
                                              .beginningQueryIndex = 0,
                                              .endQueryIndex = 3,
                                          })
                    .has_value());

    REQUIRE(device
                .beginGpuProfileScope(*command,
                                      {
                                          .queryPool = *pool,
                                          .beginningQueryIndex = 0,
                                          .endQueryIndex = 3,
                                          .label = "Frame",
                                          .color =
                                              rachis::DebugColor{
                                                  .red = 0.2F,
                                                  .green = 0.7F,
                                                  .blue = 1.0F,
                                                  .alpha = 1.0F,
                                              },
                                      })
                .has_value());
    auto activeProfileReset = device.resetQueryPool(*command, *pool, 0, 1);
    REQUIRE_FALSE(activeProfileReset.has_value());
    CHECK(activeProfileReset.error().code == rachis::ErrorCode::invalidState);
    CHECK_FALSE(device.endDebugGroup(*command).has_value());
    CHECK_FALSE(device.writeTimestamp(*command, *pool, 3).has_value());
    CHECK_FALSE(device
                    .resolveTimestamps(*command,
                                       {
                                           .queryPool = *pool,
                                           .queryCount = 4,
                                           .destination = *results,
                                       })
                    .has_value());

    REQUIRE(device
                .beginGpuProfileScope(*command,
                                      {
                                          .queryPool = *pool,
                                          .beginningQueryIndex = 1,
                                          .endQueryIndex = 2,
                                          .label = "Lighting",
                                      })
                .has_value());
    REQUIRE(device.endGpuProfileScope(*command).has_value());

    REQUIRE(device.beginDebugGroup(*command, {.label = "profile annotation"}).has_value());
    CHECK_FALSE(device.endGpuProfileScope(*command).has_value());
    REQUIRE(device.endDebugGroup(*command).has_value());

    rachis::Status crossThreadEnd;
    std::jthread otherThread([&] { crossThreadEnd = device.endGpuProfileScope(*command); });
    otherThread.join();
    REQUIRE_FALSE(crossThreadEnd.has_value());
    CHECK(crossThreadEnd.error().code == rachis::ErrorCode::invalidState);

    auto rejectedSubmit = device.submit(std::move(*command));
    REQUIRE_FALSE(rejectedSubmit.has_value());
    CHECK(rejectedSubmit.error().code == rachis::ErrorCode::invalidState);
    CHECK(command->isValid());

    REQUIRE(device.endGpuProfileScope(*command).has_value());
    REQUIRE(device.endDebugGroup(*command).has_value());
    CHECK_FALSE(device.endGpuProfileScope(*command).has_value());
    REQUIRE(device
                .resolveTimestamps(*command,
                                   {
                                       .queryPool = *pool,
                                       .queryCount = 4,
                                       .destination = *results,
                                   })
                .has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    auto mapped = device.mapBuffer(*results);
    REQUIRE(mapped.has_value());
    std::array<std::uint64_t, 4> timestamps{};
    std::memcpy(timestamps.data(), mapped->data, sizeof(timestamps));
    CAPTURE(timestamps[0], timestamps[1], timestamps[2], timestamps[3]);
    CHECK(timestamps[1] >= timestamps[0]);
    CHECK(timestamps[2] >= timestamps[1]);
    CHECK(timestamps[3] >= timestamps[2]);
    REQUIRE(device.unmapBuffer(*results).has_value());

    auto transientPool = device.createTimestampQueryPool({
        .queueType = rachis::QueueType::graphics,
        .queryCount = 2,
        .label = "destroyed active profile pool",
    });
    REQUIRE(transientPool.has_value());
    auto transientCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(transientCommand.has_value());
    REQUIRE(device
                .beginGpuProfileScope(*transientCommand,
                                      {
                                          .queryPool = *transientPool,
                                          .beginningQueryIndex = 0,
                                          .endQueryIndex = 1,
                                          .label = "retained scope",
                                      })
                .has_value());
    REQUIRE(device.destroyQueryPool(*transientPool).has_value());
    REQUIRE(device.endGpuProfileScope(*transientCommand).has_value());
    auto transientSubmitted = device.submit(std::move(*transientCommand));
    REQUIRE(transientSubmitted.has_value());
    REQUIRE(device.wait(*transientSubmitted).has_value());

    REQUIRE(device.destroyQueryPool(*pool).has_value());
    REQUIRE(device.destroyBuffer(*results).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
