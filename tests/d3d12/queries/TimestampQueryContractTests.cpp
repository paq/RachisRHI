#include "D3D12/D3D12TestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace
{

    constexpr std::uint64_t resultBufferSize = 4 * rachis::timestampQueryResultSize;

    class FailResourceRetention final
    {
    public:
        explicit FailResourceRetention(std::int32_t countdown)
            : previous_(rachis::d3d12::test::resourceRetentionFailureCountdown().exchange(countdown,
                                                                                          std::memory_order_relaxed))
        {
        }

        ~FailResourceRetention()
        {
            rachis::d3d12::test::resourceRetentionFailureCountdown().store(previous_, std::memory_order_relaxed);
        }

        FailResourceRetention(const FailResourceRetention&) = delete;
        FailResourceRetention& operator=(const FailResourceRetention&) = delete;
        FailResourceRetention(FailResourceRetention&&) = delete;
        FailResourceRetention& operator=(FailResourceRetention&&) = delete;

    private:
        std::int32_t previous_ = 0;
    };

    void printDiagnostic(const rachis::DiagnosticMessage& message, void*) noexcept
    {
        static_cast<void>(std::fprintf(stderr, "D3D12 [%u]: ", static_cast<unsigned int>(message.nativeId)));
        if (!message.description.empty())
        {
            static_cast<void>(std::fwrite(message.description.data(), 1, message.description.size(), stderr));
        }
        static_cast<void>(std::fputc('\n', stderr));
    }

    [[nodiscard]] rachis::Result<rachis::Device> createWarpDevice()
    {
        return rachis::createDevice(rachis::DeviceDesc{
            .adapterPreference = rachis::AdapterPreference::warp,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = true,
            .diagnosticCallback = &printDiagnostic,
        });
    }

} // namespace

TEST_CASE("failed timestamp resolve publishes neither partial dependencies nor resolved state",
          "[size:medium][rhi][query][timestamp][failure][retention][warp]")
{
    auto created = createWarpDevice();
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    const auto baseline = device.captureResourceMemorySnapshot();
    REQUIRE(baseline.has_value());

    auto pool = device.createTimestampQueryPool({
        .queueType = rachis::QueueType::graphics,
        .queryCount = 1,
        .label = "Transactional timestamp pool",
    });
    auto rejectedDestination = device.createBuffer({
        .size = rachis::timestampQueryResultSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Rejected timestamp destination",
    });
    REQUIRE(pool.has_value());
    REQUIRE(rejectedDestination.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    {
        const FailResourceRetention failWritePublication(1);
        auto failedWrite = device.writeTimestamp(*command, *pool, 0);
        REQUIRE_FALSE(failedWrite.has_value());
        CHECK(failedWrite.error().code == rachis::ErrorCode::outOfMemory);
    }
    REQUIRE(device.writeTimestamp(*command, *pool, 0).has_value());
    {
        const FailResourceRetention failPoolPublication(2);
        auto failed = device.resolveTimestamps(*command,
                                               {
                                                   .queryPool = *pool,
                                                   .queryCount = 1,
                                                   .destination = *rejectedDestination,
                                               });
        REQUIRE_FALSE(failed.has_value());
        CHECK(failed.error().code == rachis::ErrorCode::outOfMemory);
    }

    REQUIRE(device.destroyBuffer(*rejectedDestination).has_value());
    const auto afterRejection = device.captureResourceMemorySnapshot();
    REQUIRE(afterRejection.has_value());
    CHECK(afterRejection->liveAllocationBytes == baseline->liveAllocationBytes);
    CHECK(afterRejection->retainedAllocationBytes == baseline->retainedAllocationBytes);
    CHECK(afterRejection->liveAllocationCount == baseline->liveAllocationCount);
    CHECK(afterRejection->retainedAllocationCount == baseline->retainedAllocationCount);

    auto replacementDestination = device.createBuffer({
        .size = rachis::timestampQueryResultSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Accepted timestamp destination",
    });
    REQUIRE(replacementDestination.has_value());
    REQUIRE(device
                .resolveTimestamps(*command,
                                   {
                                       .queryPool = *pool,
                                       .queryCount = 1,
                                       .destination = *replacementDestination,
                                   })
                .has_value());

    REQUIRE(device.destroyQueryPool(*pool).has_value());
    REQUIRE(device.destroyBuffer(*replacementDestination).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    const auto released = device.captureResourceMemorySnapshot();
    REQUIRE(released.has_value());
    CHECK(released->liveAllocationBytes == baseline->liveAllocationBytes);
    CHECK(released->retainedAllocationBytes == baseline->retainedAllocationBytes);
    CHECK(released->liveAllocationCount == baseline->liveAllocationCount);
    CHECK(released->retainedAllocationCount == baseline->retainedAllocationCount);

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("timestamp queries expose queue clocks, validate resolves, and retain pools",
          "[size:medium][rhi][query][timestamp][warp]")
{
    auto created = createWarpDevice();
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    for (const auto queueType : {rachis::QueueType::graphics, rachis::QueueType::compute})
    {
        const auto properties = device.queueProperties(queueType);
        CHECK(properties.timestampQueries);
        CHECK(properties.timestampFrequency > 0);
    }
    const auto copyProperties = device.queueProperties(rachis::QueueType::copy);

    auto emptyPool = device.createTimestampQueryPool({.queryCount = 0});
    REQUIRE_FALSE(emptyPool.has_value());
    CHECK(emptyPool.error().code == rachis::ErrorCode::invalidArgument);
    auto invalidQueuePool = device.createTimestampQueryPool({
        .queueType = static_cast<rachis::QueueType>(255),
        .queryCount = 1,
    });
    REQUIRE_FALSE(invalidQueuePool.has_value());
    CHECK(invalidQueuePool.error().code == rachis::ErrorCode::invalidArgument);

    auto pool = device.createTimestampQueryPool({
        .queueType = rachis::QueueType::graphics,
        .queryCount = 4,
        .label = "Timestamp query contract pool",
    });
    auto transientPool = device.createTimestampQueryPool({
        .queueType = rachis::QueueType::graphics,
        .queryCount = 2,
        .label = "Destroyed active render-pass timestamp pool",
    });
    auto results = device.createBuffer(rachis::BufferDesc{
        .size = resultBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Timestamp query contract results",
    });
    auto tooSmall = device.createBuffer(rachis::BufferDesc{
        .size = rachis::timestampQueryResultSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    auto wrongUsage = device.createBuffer(rachis::BufferDesc{
        .size = resultBufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto target = device.createTexture({
        .width = 16,
        .height = 16,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .label = "Render-pass timestamp target",
    });
    REQUIRE(target.has_value());
    auto targetView = device.createTextureView({
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    REQUIRE(pool.has_value());
    REQUIRE(transientPool.has_value());
    REQUIRE(results.has_value());
    REQUIRE(tooSmall.has_value());
    REQUIRE(wrongUsage.has_value());
    REQUIRE(targetView.has_value());

    auto copyCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(copyCommand.has_value());
    auto crossQueueWrite = device.writeTimestamp(*copyCommand, *pool, 0);
    REQUIRE_FALSE(crossQueueWrite.has_value());
    CHECK(crossQueueWrite.error().code == rachis::ErrorCode::invalidArgument);
    auto crossQueueReset = device.resetQueryPool(*copyCommand, *pool, 0, 1);
    REQUIRE_FALSE(crossQueueReset.has_value());
    CHECK(crossQueueReset.error().code == rachis::ErrorCode::invalidArgument);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    CHECK_FALSE(device.resetQueryPool(*command, {}, 0, 1).has_value());
    CHECK_FALSE(device.resetQueryPool(*command, *pool, 0, 0).has_value());
    CHECK_FALSE(device.resetQueryPool(*command, *pool, 4, 1).has_value());
    CHECK_FALSE(device.resetQueryPool(*command, *pool, 3, 2).has_value());
    CHECK_FALSE(device.writeTimestamp(*command, {}, 0).has_value());
    CHECK_FALSE(device.writeTimestamp(*command, *pool, 4).has_value());
    CHECK_FALSE(device
                    .resolveTimestamps(*command,
                                       {
                                           .queryPool = *pool,
                                           .queryCount = 1,
                                           .destination = *results,
                                       })
                    .has_value());

    REQUIRE(device.writeTimestamp(*command, *pool, 0).has_value());
    auto duplicateWrite = device.writeTimestamp(*command, *pool, 0);
    REQUIRE_FALSE(duplicateWrite.has_value());
    CHECK(duplicateWrite.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.resetQueryPool(*command, *pool, 0, 1).has_value());
    CHECK_FALSE(device
                    .resolveTimestamps(*command,
                                       {
                                           .queryPool = *pool,
                                           .queryCount = 1,
                                           .destination = *results,
                                       })
                    .has_value());
    REQUIRE(device.writeTimestamp(*command, *pool, 0).has_value());
    REQUIRE(device.writeTimestamp(*command, *pool, 1).has_value());

    CHECK_FALSE(device
                    .beginRendering(*command,
                                    {
                                        .timestampWrites =
                                            rachis::PassTimestampWrites{
                                                .queryPool = *pool,
                                                .beginningOfPassWriteIndex = 2,
                                                .endOfPassWriteIndex = 3,
                                            },
                                    })
                    .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *target,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array attachments{
        rachis::ColorAttachment{
            .view = *targetView,
            .loadOperation = rachis::LoadOperation::clear,
            .storeOperation = rachis::StoreOperation::store,
        },
    };
    CHECK_FALSE(device
                    .beginRendering(*command,
                                    {
                                        .colorAttachments = attachments,
                                        .timestampWrites =
                                            rachis::PassTimestampWrites{
                                                .queryPool = *pool,
                                                .beginningOfPassWriteIndex = 2,
                                                .endOfPassWriteIndex = 2,
                                            },
                                    })
                    .has_value());
    REQUIRE(device
                .beginRendering(*command,
                                {
                                    .colorAttachments = attachments,
                                    .timestampWrites =
                                        rachis::PassTimestampWrites{
                                            .queryPool = *pool,
                                            .beginningOfPassWriteIndex = 2,
                                            .endOfPassWriteIndex = 3,
                                        },
                                })
                .has_value());
    auto activeRenderingReset = device.resetQueryPool(*command, *pool, 0, 1);
    REQUIRE_FALSE(activeRenderingReset.has_value());
    CHECK(activeRenderingReset.error().code == rachis::ErrorCode::invalidState);
    CHECK_FALSE(device.writeTimestamp(*command, *pool, 3).has_value());
    CHECK_FALSE(device
                    .resolveTimestamps(*command,
                                       {
                                           .queryPool = *pool,
                                           .firstQuery = 2,
                                           .queryCount = 1,
                                           .destination = *results,
                                       })
                    .has_value());
    REQUIRE(device.endRendering(*command).has_value());
    CHECK_FALSE(device
                    .beginRendering(*command,
                                    {
                                        .colorAttachments = attachments,
                                        .timestampWrites =
                                            rachis::PassTimestampWrites{
                                                .queryPool = *pool,
                                                .beginningOfPassWriteIndex = 2,
                                                .endOfPassWriteIndex = 3,
                                            },
                                    })
                    .has_value());

    CHECK_FALSE(device
                    .resolveTimestamps(*command,
                                       {
                                           .queryPool = *pool,
                                           .queryCount = 0,
                                           .destination = *results,
                                       })
                    .has_value());
    CHECK_FALSE(device
                    .resolveTimestamps(*command,
                                       {
                                           .queryPool = *pool,
                                           .queryCount = 4,
                                           .destination = *results,
                                           .destinationOffset = 1,
                                       })
                    .has_value());
    CHECK_FALSE(device
                    .resolveTimestamps(*command,
                                       {
                                           .queryPool = *pool,
                                           .queryCount = 4,
                                           .destination = *tooSmall,
                                       })
                    .has_value());
    CHECK_FALSE(device
                    .resolveTimestamps(*command,
                                       {
                                           .queryPool = *pool,
                                           .queryCount = 4,
                                           .destination = *wrongUsage,
                                       })
                    .has_value());
    REQUIRE(device
                .resolveTimestamps(*command,
                                   {
                                       .queryPool = *pool,
                                       .queryCount = 4,
                                       .destination = *results,
                                   })
                .has_value());
    CHECK_FALSE(device
                    .resolveTimestamps(*command,
                                       {
                                           .queryPool = *pool,
                                           .queryCount = 4,
                                           .destination = *results,
                                       })
                    .has_value());

    REQUIRE(device
                .beginRendering(*command,
                                {
                                    .colorAttachments = attachments,
                                    .timestampWrites =
                                        rachis::PassTimestampWrites{
                                            .queryPool = *transientPool,
                                            .endOfPassWriteIndex = 1,
                                        },
                                })
                .has_value());
    const auto staleTransientPool = *transientPool;
    REQUIRE(device.destroyQueryPool(*transientPool).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    CHECK_FALSE(device
                    .beginRendering(*command,
                                    {
                                        .colorAttachments = attachments,
                                        .timestampWrites =
                                            rachis::PassTimestampWrites{
                                                .queryPool = staleTransientPool,
                                                .endOfPassWriteIndex = 0,
                                            },
                                    })
                    .has_value());

    const auto stalePool = *pool;
    REQUIRE(device.destroyQueryPool(*pool).has_value());
    CHECK_FALSE(device.destroyQueryPool(stalePool).has_value());
    CHECK_FALSE(device.writeTimestamp(*command, stalePool, 2).has_value());
    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());

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
    CHECK(timestamps[3] >= timestamps[2]);
    const auto graphicsProperties = device.queueProperties(rachis::QueueType::graphics);
    const auto elapsedSeconds =
        static_cast<double>(timestamps[1] - timestamps[0]) / static_cast<double>(graphicsProperties.timestampFrequency);
    CHECK(elapsedSeconds >= 0.0);
    REQUIRE(device.unmapBuffer(*results).has_value());

    REQUIRE(device.destroyBuffer(*wrongUsage).has_value());
    REQUIRE(device.destroyBuffer(*tooSmall).has_value());
    REQUIRE(device.destroyBuffer(*results).has_value());

    if (copyProperties.timestampQueries)
    {
        auto copyPool = device.createTimestampQueryPool({
            .queueType = rachis::QueueType::copy,
            .queryCount = 1,
        });
        REQUIRE(copyPool.has_value());
        REQUIRE(device.destroyQueryPool(*copyPool).has_value());
    }
    else
    {
        auto unsupportedCopyPool = device.createTimestampQueryPool({
            .queueType = rachis::QueueType::copy,
            .queryCount = 1,
        });
        REQUIRE_FALSE(unsupportedCopyPool.has_value());
        CHECK(unsupportedCopyPool.error().code == rachis::ErrorCode::unsupported);
    }

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("endRendering keeps store discards and pass timestamps transactional across barrier modes",
          "[size:medium][rhi][query][timestamp][warp]")
{
    class ForceLegacyBarriers final
    {
    public:
        explicit ForceLegacyBarriers(bool enabled)
            : previous_(rachis::d3d12::test::forceLegacyBarriers().exchange(enabled, std::memory_order_relaxed))
        {
        }

        ~ForceLegacyBarriers()
        {
            rachis::d3d12::test::forceLegacyBarriers().store(previous_, std::memory_order_relaxed);
        }

        ForceLegacyBarriers(const ForceLegacyBarriers&) = delete;
        ForceLegacyBarriers& operator=(const ForceLegacyBarriers&) = delete;
        ForceLegacyBarriers(ForceLegacyBarriers&&) = delete;
        ForceLegacyBarriers& operator=(ForceLegacyBarriers&&) = delete;

    private:
        bool previous_ = false;
    };

    for (const bool legacyBarriers : {false, true})
    {
        CAPTURE(legacyBarriers);
        const ForceLegacyBarriers guard(legacyBarriers);
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .adapterPreference = rachis::AdapterPreference::warp,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = true,
        });
        auto otherCreated = rachis::createDevice(rachis::DeviceDesc{
            .adapterPreference = rachis::AdapterPreference::warp,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = true,
        });
        if (!created)
        {
            INFO(created.error().message);
        }
        if (!otherCreated)
        {
            INFO(otherCreated.error().message);
        }
        REQUIRE(created.has_value());
        REQUIRE(otherCreated.has_value());
        auto device = std::move(*created);
        auto otherDevice = std::move(*otherCreated);
        if (!device.queueProperties(rachis::QueueType::graphics).timestampQueries)
        {
            continue;
        }

        auto pool = device.createTimestampQueryPool({
            .queueType = rachis::QueueType::graphics,
            .queryCount = 2,
            .label = "Transactional endRendering timestamp pool",
        });
        auto results = device.createBuffer({
            .size = 2 * rachis::timestampQueryResultSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
        });
        auto target = device.createTexture({
            .width = 16,
            .height = 16,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::renderTarget,
        });
        REQUIRE(pool.has_value());
        REQUIRE(results.has_value());
        REQUIRE(target.has_value());
        auto targetView = device.createTextureView({
            .texture = *target,
            .usage = rachis::TextureViewUsage::renderTarget,
        });
        REQUIRE(targetView.has_value());

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *target,
                                        .stagesBefore = rachis::PipelineStage::all,
                                        .stagesAfter = rachis::PipelineStage::renderTarget,
                                        .accessAfter = rachis::Access::renderTarget,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::renderTarget,
                                    })
                    .has_value());
        const std::array attachments{
            rachis::ColorAttachment{
                .view = *targetView,
                .loadOperation = rachis::LoadOperation::clear,
                .storeOperation = rachis::StoreOperation::discard,
                .clearValue = {0.1F, 0.2F, 0.3F, 1.0F},
            },
        };
        REQUIRE(device
                    .beginRendering(*command,
                                    {
                                        .colorAttachments = attachments,
                                        .timestampWrites =
                                            rachis::PassTimestampWrites{
                                                .queryPool = *pool,
                                                .beginningOfPassWriteIndex = 0,
                                                .endOfPassWriteIndex = 1,
                                            },
                                    })
                    .has_value());

        const auto foreignEnd = otherDevice.endRendering(*command);
        REQUIRE_FALSE(foreignEnd.has_value());
        CHECK(foreignEnd.error().code == rachis::ErrorCode::invalidArgument);

        REQUIRE(device.clearColorAttachment(*command, 0, {0.4F, 0.5F, 0.6F, 1.0F}, {}).has_value());
        REQUIRE(device.endRendering(*command).has_value());
        REQUIRE(device
                    .resolveTimestamps(*command,
                                       {
                                           .queryPool = *pool,
                                           .queryCount = 2,
                                           .destination = *results,
                                       })
                    .has_value());

        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());
        device.collectGarbage();

        std::array<std::uint64_t, 2> timestamps{};
        auto mapped = device.mapBuffer(*results);
        REQUIRE(mapped.has_value());
        std::memcpy(timestamps.data(), mapped->data, sizeof(timestamps));
        REQUIRE(device.unmapBuffer(*results).has_value());
        CHECK(timestamps[1] >= timestamps[0]);

        REQUIRE(device.destroyTextureView(*targetView).has_value());
        REQUIRE(device.destroyTexture(*target).has_value());
        REQUIRE(device.destroyBuffer(*results).has_value());
        REQUIRE(device.destroyQueryPool(*pool).has_value());

        const auto diagnostics = device.diagnosticCounts();
        CHECK(diagnostics.warnings == 0);
        CHECK(diagnostics.errors == 0);
        CHECK(diagnostics.corruptions == 0);
        const auto otherDiagnostics = otherDevice.diagnosticCounts();
        CHECK(otherDiagnostics.warnings == 0);
        CHECK(otherDiagnostics.errors == 0);
        CHECK(otherDiagnostics.corruptions == 0);
    }
}
