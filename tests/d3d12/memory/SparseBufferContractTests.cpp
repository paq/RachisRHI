#include "D3D12/D3D12TestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <vector>

namespace
{

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

    [[nodiscard]] std::byte pattern(std::uint64_t index, std::uint32_t salt) noexcept
    {
        return static_cast<std::byte>((index * 29U + salt) & 0xFFU);
    }

} // namespace

TEST_CASE("sparse buffer mappings are atomic queue-ordered immutable versions",
          "[size:medium][rhi][buffer][sparse][mapping][lifetime][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    constexpr std::uint64_t expectedTileSize = 64ULL * 1024ULL;
    const rachis::BufferDesc sparseDesc{
        .size = expectedTileSize * 2,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Sparse buffer contract resource",
    };
    if (!rachis::test::requireCapabilities(device).sparseBuffers)
    {
        const auto unsupported = device.createSparseBuffer(sparseDesc);
        REQUIRE_FALSE(unsupported.has_value());
        CHECK(unsupported.error().code == rachis::ErrorCode::unsupported);
        return;
    }

    auto invalidDesc = sparseDesc;
    invalidDesc.usage = sparseDesc.usage | rachis::BufferUsage::vertex;
    CHECK_FALSE(device.createSparseBuffer(invalidDesc).has_value());
    invalidDesc = sparseDesc;
    invalidDesc.size = std::numeric_limits<std::uint64_t>::max();
    CHECK_FALSE(device.createSparseBuffer(invalidDesc).has_value());

    const auto sparse = device.createSparseBuffer(sparseDesc);
    REQUIRE(sparse.has_value());
    const auto properties = device.querySparseBufferProperties(*sparse);
    REQUIRE(properties.has_value());
    CHECK(properties->tileSize == expectedTileSize);
    CHECK(properties->tileCount == 2);

    const auto ordinary = device.createBuffer({
        .size = expectedTileSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
    });
    REQUIRE(ordinary.has_value());
    CHECK_FALSE(device.querySparseBufferProperties(*ordinary).has_value());

    const auto ordinaryHeap = device.createResourceHeap({
        .size = expectedTileSize,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::buffer,
    });
    REQUIRE(ordinaryHeap.has_value());
    const std::array invalidMappings{
        rachis::SparseBufferTileMapping{
            .buffer = *sparse,
            .bufferTileOffset = 0,
            .tileCount = 1,
            .heap = *ordinaryHeap,
        },
    };
    const auto rejectedMapping = device.enqueueSparseBufferMappings(rachis::QueueType::copy, invalidMappings);
    REQUIRE_FALSE(rejectedMapping.has_value());
    CHECK(rejectedMapping.error().code == rachis::ErrorCode::invalidArgument);

    const auto sparseHeap = device.createResourceHeap({
        .size = expectedTileSize,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::buffer,
        .flags = rachis::ResourceHeapFlag::sparseBacking,
        .label = "Sparse buffer contract heap",
    });
    REQUIRE(sparseHeap.has_value());
    CHECK_FALSE(device.createPlacedBuffer({.heap = *sparseHeap}, sparseDesc).has_value());

    const auto upload = device.createBuffer({
        .size = sparseDesc.size,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    const auto readback = device.createBuffer({
        .size = sparseDesc.size,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(upload.has_value());
    REQUIRE(readback.has_value());

    std::vector<std::byte> expected(sparseDesc.size);
    for (std::uint64_t index = 0; index < expectedTileSize; ++index)
    {
        expected[index] = pattern(index, 7);
        expected[expectedTileSize + index] = pattern(index, 113);
    }
    REQUIRE(device.writeBuffer(*upload, 0, expected).has_value());

    {
        auto unmappedCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(unmappedCommand.has_value());
        const auto unmappedCopy = device.copyBuffer(*unmappedCommand,
                                                    {
                                                        .source = *upload,
                                                        .destination = *sparse,
                                                        .size = expectedTileSize,
                                                    });
        REQUIRE_FALSE(unmappedCopy.has_value());
        CHECK(unmappedCopy.error().code == rachis::ErrorCode::invalidState);
    }

    const auto submitTileCopy =
        [&](std::uint64_t tileOffset, rachis::Access accessBefore, rachis::TimelinePoint mappingPoint) {
            auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
            REQUIRE(command.has_value());
            REQUIRE(device
                        .bufferBarrier(*command,
                                       {
                                           .buffer = *sparse,
                                           .stagesBefore = rachis::PipelineStage::all,
                                           .stagesAfter = rachis::PipelineStage::copy,
                                           .accessBefore = accessBefore,
                                           .accessAfter = rachis::Access::copyDestination,
                                       })
                        .has_value());
            REQUIRE(device
                        .copyBuffer(*command,
                                    {
                                        .source = *upload,
                                        .sourceOffset = tileOffset,
                                        .destination = *sparse,
                                        .destinationOffset = tileOffset,
                                        .size = expectedTileSize,
                                    })
                        .has_value());
            REQUIRE(device
                        .bufferBarrier(*command,
                                       {
                                           .buffer = *sparse,
                                           .stagesBefore = rachis::PipelineStage::copy,
                                           .stagesAfter = rachis::PipelineStage::copy,
                                           .accessBefore = rachis::Access::copyDestination,
                                           .accessAfter = rachis::Access::copySource,
                                       })
                        .has_value());
            REQUIRE(device
                        .copyBuffer(*command,
                                    {
                                        .source = *sparse,
                                        .sourceOffset = tileOffset,
                                        .destination = *readback,
                                        .destinationOffset = tileOffset,
                                        .size = expectedTileSize,
                                    })
                        .has_value());
            rachis::Result<rachis::TimelinePoint> submitted;
            if (mappingPoint.queueType() == rachis::QueueType::graphics)
            {
                submitted = device.submit(std::move(*command));
            }
            else
            {
                const auto missingWait = device.submit(std::move(*command));
                REQUIRE_FALSE(missingWait.has_value());
                CHECK(missingWait.error().code == rachis::ErrorCode::invalidState);
                REQUIRE(command->isValid());
                const std::array waits{mappingPoint};
                submitted = device.submit(std::move(*command), waits);
            }
            REQUIRE(submitted.has_value());
            REQUIRE(device.wait(*submitted).has_value());
            device.collectGarbage();
        };

    const std::array firstMapping{
        rachis::SparseBufferTileMapping{
            .buffer = *sparse,
            .bufferTileOffset = 0,
            .tileCount = 1,
            .heap = *sparseHeap,
            .heapTileOffset = 0,
        },
    };
    const auto firstMappingPoint = device.enqueueSparseBufferMappings(rachis::QueueType::copy, firstMapping);
    REQUIRE(firstMappingPoint.has_value());
    CHECK(firstMappingPoint->queueType() == rachis::QueueType::copy);
    submitTileCopy(0, rachis::Access::common, *firstMappingPoint);

    const std::array unmapFirst{
        rachis::SparseBufferTileMapping{
            .buffer = *sparse,
            .bufferTileOffset = 0,
            .tileCount = 1,
        },
    };
    const std::array sameBatchPhysicalReuse{
        unmapFirst.front(),
        rachis::SparseBufferTileMapping{
            .buffer = *sparse,
            .bufferTileOffset = 1,
            .tileCount = 1,
            .heap = *sparseHeap,
            .heapTileOffset = 0,
        },
    };
    const auto rejectedSameBatchReuse =
        device.enqueueSparseBufferMappings(rachis::QueueType::copy, sameBatchPhysicalReuse);
    REQUIRE_FALSE(rejectedSameBatchReuse.has_value());
    CHECK(rejectedSameBatchReuse.error().code == rachis::ErrorCode::invalidState);

    const auto unmapPoint = device.enqueueSparseBufferMappings(rachis::QueueType::compute, unmapFirst);
    INFO("Cross-queue unmap result: " << (unmapPoint.has_value() ? "accepted" : unmapPoint.error().message));
    REQUIRE(unmapPoint.has_value());
    CHECK(unmapPoint->queueType() == rachis::QueueType::compute);

    const std::array secondMapping{
        rachis::SparseBufferTileMapping{
            .buffer = *sparse,
            .bufferTileOffset = 1,
            .tileCount = 1,
            .heap = *sparseHeap,
            .heapTileOffset = 0,
        },
    };
    const auto secondMappingPoint = device.enqueueSparseBufferMappings(rachis::QueueType::graphics, secondMapping);
    REQUIRE(secondMappingPoint.has_value());
    CHECK(secondMappingPoint->queueType() == rachis::QueueType::graphics);
    REQUIRE(device.destroyResourceHeap(*sparseHeap).has_value());
    submitTileCopy(expectedTileSize, rachis::Access::copySource, *secondMappingPoint);

    std::vector<std::byte> actual(sparseDesc.size);
    REQUIRE(device.readBuffer(*readback, 0, actual).has_value());
    CHECK(actual == expected);

    const rachis::BufferDesc partialDesc{
        .size = expectedTileSize * 4,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Partial sparse-unmap contract resource",
    };
    const auto partialSparse = device.createSparseBuffer(partialDesc);
    const auto partialHeap = device.createResourceHeap({
        .size = partialDesc.size,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::buffer,
        .flags = rachis::ResourceHeapFlag::sparseBacking,
        .label = "Partial sparse-unmap contract heap",
    });
    REQUIRE(partialSparse.has_value());
    REQUIRE(partialHeap.has_value());

    const std::array fullPartialMapping{
        rachis::SparseBufferTileMapping{
            .buffer = *partialSparse,
            .bufferTileOffset = 0,
            .tileCount = 4,
            .heap = *partialHeap,
            .heapTileOffset = 0,
        },
    };
    const auto fullPartialPoint = device.enqueueSparseBufferMappings(rachis::QueueType::copy, fullPartialMapping);
    REQUIRE(fullPartialPoint.has_value());
    REQUIRE(device.wait(*fullPartialPoint).has_value());
    device.collectGarbage();

    const std::array partialUnmap{
        rachis::SparseBufferTileMapping{
            .buffer = *partialSparse,
            .bufferTileOffset = 1,
            .tileCount = 2,
        },
    };
    const std::array overlappingReplacement{
        partialUnmap.front(),
        rachis::SparseBufferTileMapping{
            .buffer = *partialSparse,
            .bufferTileOffset = 1,
            .tileCount = 2,
            .heap = *partialHeap,
            .heapTileOffset = 1,
        },
    };
    const auto rejectedOverlap = device.enqueueSparseBufferMappings(rachis::QueueType::copy, overlappingReplacement);
    REQUIRE_FALSE(rejectedOverlap.has_value());
    CHECK(rejectedOverlap.error().code == rachis::ErrorCode::invalidArgument);

    const std::array inPlaceReplacement{overlappingReplacement.back()};
    const auto inPlaceReplacementPoint =
        device.enqueueSparseBufferMappings(rachis::QueueType::copy, inPlaceReplacement);
    REQUIRE(inPlaceReplacementPoint.has_value());
    REQUIRE(device.wait(*inPlaceReplacementPoint).has_value());
    device.collectGarbage();

    {
        auto rejectedCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
        REQUIRE(rejectedCommand.has_value());
        for (const std::int32_t failureStage : {1, 2})
        {
            const auto beforeRejectedPublication = rachis::d3d12::test::commandDependencySnapshot(*rejectedCommand);
            const auto beforeRejectedRecording =
                rachis::d3d12::test::resourceCommandRecordCount().load(std::memory_order_relaxed);
            REQUIRE(beforeRejectedPublication.has_value());
            {
                const FailResourceRetention failPublication(failureStage);
                auto rejected = device.copyBuffer(*rejectedCommand,
                                                  {
                                                      .source = *upload,
                                                      .destination = *partialSparse,
                                                      .size = expectedTileSize,
                                                  });
                REQUIRE_FALSE(rejected.has_value());
                CHECK(rejected.error().code == rachis::ErrorCode::outOfMemory);
            }
            const auto afterRejectedPublication = rachis::d3d12::test::commandDependencySnapshot(*rejectedCommand);
            REQUIRE(afterRejectedPublication.has_value());
            CHECK(*afterRejectedPublication == *beforeRejectedPublication);
            CHECK(rachis::d3d12::test::resourceCommandRecordCount().load(std::memory_order_relaxed) ==
                  beforeRejectedRecording);
        }

        const auto beforeTwoGenerationPublication = rachis::d3d12::test::commandDependencySnapshot(*rejectedCommand);
        const auto beforeTwoGenerationRecording =
            rachis::d3d12::test::resourceCommandRecordCount().load(std::memory_order_relaxed);
        REQUIRE(beforeTwoGenerationPublication.has_value());
        {
            const FailResourceRetention failSecondSparseGeneration(3);
            auto rejected = device.copyBuffer(*rejectedCommand,
                                              {
                                                  .source = *sparse,
                                                  .sourceOffset = expectedTileSize,
                                                  .destination = *partialSparse,
                                                  .size = expectedTileSize,
                                              });
            REQUIRE_FALSE(rejected.has_value());
            CHECK(rejected.error().code == rachis::ErrorCode::outOfMemory);
        }
        const auto afterTwoGenerationPublication = rachis::d3d12::test::commandDependencySnapshot(*rejectedCommand);
        REQUIRE(afterTwoGenerationPublication.has_value());
        CHECK(*afterTwoGenerationPublication == *beforeTwoGenerationPublication);
        CHECK(rachis::d3d12::test::resourceCommandRecordCount().load(std::memory_order_relaxed) ==
              beforeTwoGenerationRecording);

        const auto beforeBarrierPublication = rachis::d3d12::test::commandDependencySnapshot(*rejectedCommand);
        const auto beforeBarrierRecording =
            rachis::d3d12::test::resourceCommandRecordCount().load(std::memory_order_relaxed);
        REQUIRE(beforeBarrierPublication.has_value());
        {
            const FailResourceRetention failSparseBarrierGeneration(2);
            auto rejected = device.bufferBarrier(*rejectedCommand,
                                                 {
                                                     .buffer = *partialSparse,
                                                     .stagesBefore = rachis::PipelineStage::copy,
                                                     .stagesAfter = rachis::PipelineStage::copy,
                                                     .accessBefore = rachis::Access::copyDestination,
                                                     .accessAfter = rachis::Access::copySource,
                                                 });
            REQUIRE_FALSE(rejected.has_value());
            CHECK(rejected.error().code == rachis::ErrorCode::outOfMemory);
        }
        const auto afterBarrierPublication = rachis::d3d12::test::commandDependencySnapshot(*rejectedCommand);
        REQUIRE(afterBarrierPublication.has_value());
        CHECK(*afterBarrierPublication == *beforeBarrierPublication);
        CHECK(rachis::d3d12::test::resourceCommandRecordCount().load(std::memory_order_relaxed) ==
              beforeBarrierRecording);
    }

    const auto unretainedPool = device.createCommandPool({
        .queueType = rachis::QueueType::copy,
        .maxCommandBuffers = 1,
        .label = "Sparse mapping retention in an unretained pool",
        .flags = rachis::CommandPoolFlag::allowIndividualReset | rachis::CommandPoolFlag::unretainedResources,
    });
    REQUIRE(unretainedPool.has_value());
    {
        auto unretainedCommand = device.acquireCommandBuffer(*unretainedPool);
        REQUIRE(unretainedCommand.has_value());
        REQUIRE(device
                    .copyBuffer(*unretainedCommand,
                                {
                                    .source = *upload,
                                    .destination = *partialSparse,
                                    .size = expectedTileSize,
                                })
                    .has_value());
        const auto retainedByInternalLease =
            device.enqueueSparseBufferMappings(rachis::QueueType::compute, partialUnmap);
        REQUIRE_FALSE(retainedByInternalLease.has_value());
        CHECK(retainedByInternalLease.error().code == rachis::ErrorCode::invalidState);
        REQUIRE(device.resetCommandBuffer(*unretainedCommand).has_value());
    }
    REQUIRE(device.destroyCommandPool(*unretainedPool).has_value());

    {
        auto retainedVersionCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(retainedVersionCommand.has_value());
        REQUIRE(device
                    .copyBuffer(*retainedVersionCommand,
                                {
                                    .source = *upload,
                                    .destination = *partialSparse,
                                    .size = expectedTileSize,
                                })
                    .has_value());
        const auto retainedVersionUnmap = device.enqueueSparseBufferMappings(rachis::QueueType::compute, partialUnmap);
        REQUIRE_FALSE(retainedVersionUnmap.has_value());
        CHECK(retainedVersionUnmap.error().code == rachis::ErrorCode::invalidState);
    }
    const auto partialUnmapPoint = device.enqueueSparseBufferMappings(rachis::QueueType::compute, partialUnmap);
    REQUIRE(partialUnmapPoint.has_value());
    REQUIRE(device.wait(*partialUnmapPoint).has_value());
    device.collectGarbage();

    {
        auto partiallyMappedCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(partiallyMappedCommand.has_value());
        const auto gapCopy = device.copyBuffer(*partiallyMappedCommand,
                                               {
                                                   .source = *upload,
                                                   .destination = *partialSparse,
                                                   .size = expectedTileSize * 2,
                                               });
        REQUIRE_FALSE(gapCopy.has_value());
        CHECK(gapCopy.error().code == rachis::ErrorCode::invalidState);
    }

    const std::array remapFreedTiles{
        rachis::SparseBufferTileMapping{
            .buffer = *partialSparse,
            .bufferTileOffset = 1,
            .tileCount = 2,
            .heap = *partialHeap,
            .heapTileOffset = 1,
        },
    };
    const auto remapFreedPoint = device.enqueueSparseBufferMappings(rachis::QueueType::graphics, remapFreedTiles);
    REQUIRE(remapFreedPoint.has_value());
    REQUIRE(device.wait(*remapFreedPoint).has_value());
    device.collectGarbage();

    REQUIRE(device.destroyBuffer(*partialSparse).has_value());
    REQUIRE(device.destroyResourceHeap(*partialHeap).has_value());

    REQUIRE(device.destroyBuffer(*ordinary).has_value());
    REQUIRE(device.destroyResourceHeap(*ordinaryHeap).has_value());
    REQUIRE(device.destroyBuffer(*sparse).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    device.collectGarbage();

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("sparse buffer remaps slice leases whose retired ancestors are still retained",
          "[size:medium][rhi][buffer][sparse][mapping][lifetime][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    if (!rachis::test::requireCapabilities(device).sparseBuffers)
    {
        return;
    }

    constexpr std::uint64_t tileSize = 64ULL * 1024ULL;
    constexpr std::uint32_t tileCount = 8;
    const auto sparse = device.createSparseBuffer({
        .size = tileSize * tileCount,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Sparse lease ancestry resource",
    });
    const auto heap = device.createResourceHeap({
        .size = tileSize * tileCount,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::buffer,
        .flags = rachis::ResourceHeapFlag::sparseBacking,
        .label = "Sparse lease ancestry heap",
    });
    REQUIRE(sparse.has_value());
    REQUIRE(heap.has_value());

    const auto map = [&](std::uint32_t firstTile, std::uint32_t count) {
        return rachis::SparseBufferTileMapping{
            .buffer = *sparse,
            .bufferTileOffset = firstTile,
            .tileCount = count,
            .heap = *heap,
            .heapTileOffset = firstTile,
        };
    };
    const auto unmap = [&](std::uint32_t firstTile, std::uint32_t count) {
        return rachis::SparseBufferTileMapping{
            .buffer = *sparse,
            .bufferTileOffset = firstTile,
            .tileCount = count,
        };
    };
    rachis::TimelinePoint lastPoint;
    const auto enqueue = [&](std::span<const rachis::SparseBufferTileMapping> batch) {
        const auto point = device.enqueueSparseBufferMappings(rachis::QueueType::graphics, batch);
        INFO("Sparse mapping result: " << (point.has_value() ? "accepted" : point.error().message));
        REQUIRE(point.has_value());
        lastPoint = *point;
    };
    const auto remapFreedTilesAfterRetirement = [&](std::span<const rachis::SparseBufferTileMapping> batch) {
        REQUIRE(device.wait(lastPoint).has_value());
        device.collectGarbage();
        enqueue(batch);
        REQUIRE(device.wait(lastPoint).has_value());
        device.collectGarbage();
        REQUIRE(device.destroyBuffer(*sparse).has_value());
        REQUIRE(device.destroyResourceHeap(*heap).has_value());
        device.collectGarbage();

        const auto diagnostics = device.diagnosticCounts();
        CHECK(diagnostics.errors == 0);
        CHECK(diagnostics.corruptions == 0);
    };

    SECTION("a range coalesced from two retired leases")
    {
        enqueue(std::array{map(0, 4), map(4, 4)});
        enqueue(std::array{unmap(0, 8)});
        enqueue(std::array{map(0, 8)});
        enqueue(std::array{unmap(3, 2)});
        enqueue(std::array{unmap(1, 1), unmap(6, 1)});
        remapFreedTilesAfterRetirement(std::array{map(1, 1), map(3, 2), map(6, 1)});
    }

    SECTION("ranges split from one retired lease in a single batch")
    {
        enqueue(std::array{map(0, 4)});
        enqueue(std::array{unmap(0, 4)});
        enqueue(std::array{map(0, 2), map(2, 2)});
        enqueue(std::array{unmap(1, 2)});
        remapFreedTilesAfterRetirement(std::array{map(1, 2)});
    }
}

TEST_CASE("an unpublished sparse mapping generation cannot be copied or superseded",
          "[size:medium][rhi][buffer][sparse][mapping][failure][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    if (!rachis::test::requireCapabilities(device).sparseBuffers)
    {
        return;
    }

    constexpr std::uint64_t tileSize = 64ULL * 1024ULL;
    const auto sparse = device.createSparseBuffer({
        .size = tileSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
    });
    const auto heap = device.createResourceHeap({
        .size = tileSize,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::buffer,
        .flags = rachis::ResourceHeapFlag::sparseBacking,
    });
    const auto upload = device.createBuffer({
        .size = tileSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(sparse.has_value());
    REQUIRE(heap.has_value());
    REQUIRE(upload.has_value());
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(command.has_value());

    const std::array mapping{rachis::SparseBufferTileMapping{
        .buffer = *sparse,
        .bufferTileOffset = 0,
        .tileCount = 1,
        .heap = *heap,
        .heapTileOffset = 0,
    }};
    rachis::d3d12::test::nextSparseMappingSignalFailure().store(true, std::memory_order_relaxed);
    const auto unpublished = device.enqueueSparseBufferMappings(rachis::QueueType::copy, mapping);
    REQUIRE_FALSE(unpublished.has_value());
    CHECK(unpublished.error().code == rachis::ErrorCode::deviceLost);

    const auto copy = device.copyBuffer(*command,
                                        {
                                            .source = *upload,
                                            .destination = *sparse,
                                            .size = tileSize,
                                        });
    REQUIRE_FALSE(copy.has_value());
    CHECK(copy.error().code == rachis::ErrorCode::invalidState);

    const auto replacement = device.enqueueSparseBufferMappings(rachis::QueueType::copy, mapping);
    REQUIRE_FALSE(replacement.has_value());
    CHECK(replacement.error().code == rachis::ErrorCode::deviceLost);

    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*sparse).has_value());
    REQUIRE(device.destroyResourceHeap(*heap).has_value());
}
