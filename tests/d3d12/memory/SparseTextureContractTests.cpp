#include "D3D12/D3D12TestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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

    [[nodiscard]] std::byte transferByte(std::uint32_t row, std::uint64_t byte) noexcept
    {
        return static_cast<std::byte>((static_cast<std::uint64_t>(row) * 37U + byte * 11U + 23U) & 0xFFU);
    }

} // namespace

TEST_CASE("sparse texture tiles and mip tails publish immutable copy-visible generations",
          "[size:medium][rhi][texture][sparse][mapping][lifetime][warp]")
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

    const rachis::TextureDesc sparseDesc{
        .dimension = rachis::TextureDimension::texture2D,
        .width = 1024,
        .height = 1024,
        .depthOrArrayLayers = 1,
        .mipLevels = 11,
        .sampleCount = 1,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Sparse texture contract resource",
    };
    if (!rachis::test::requireCapabilities(device).sparseTextures)
    {
        const auto unsupported = device.createSparseTexture(sparseDesc);
        REQUIRE_FALSE(unsupported.has_value());
        CHECK(unsupported.error().code == rachis::ErrorCode::unsupported);
        return;
    }

    auto invalidDesc = sparseDesc;
    invalidDesc.usage = sparseDesc.usage | rachis::TextureUsage::renderTarget;
    CHECK_FALSE(device.createSparseTexture(invalidDesc).has_value());
    invalidDesc = sparseDesc;
    invalidDesc.depthOrArrayLayers = 2;
    CHECK_FALSE(device.createSparseTexture(invalidDesc).has_value());
    invalidDesc = sparseDesc;
    invalidDesc.format = rachis::Format::bc1RgbaUnorm;
    CHECK_FALSE(device.createSparseTexture(invalidDesc).has_value());

    const auto sparse = device.createSparseTexture(sparseDesc);
    REQUIRE(sparse.has_value());
    const auto properties = device.querySparseTextureProperties(*sparse);
    REQUIRE(properties.has_value());
    CHECK(properties->tileSize == 64ULL * 1024ULL);
    CHECK(properties->totalTileCount != 0);
    CHECK(properties->standardMipCount + properties->packedMipCount == sparseDesc.mipLevels);
    CHECK(properties->standardMipTileGrids.size() == properties->standardMipCount);
    CHECK(properties->standardTileExtent.width != 0);
    CHECK(properties->standardTileExtent.height != 0);
    CHECK(properties->standardTileExtent.depth != 0);
    REQUIRE(properties->standardMipCount != 0);

    const auto ordinary = device.createTexture({
        .width = 4,
        .height = 4,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copyDestination,
    });
    REQUIRE(ordinary.has_value());
    CHECK_FALSE(device.querySparseTextureProperties(*ordinary).has_value());

    const auto ordinaryHeap = device.createResourceHeap({
        .size = properties->tileSize,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::nonAttachmentTexture,
    });
    REQUIRE(ordinaryHeap.has_value());
    const std::array invalidHeapMapping{
        rachis::SparseTextureTileMapping{
            .texture = *sparse,
            .subresource = {},
            .textureTileOrigin = {},
            .textureTileExtent = {},
            .heap = *ordinaryHeap,
        },
    };
    CHECK_FALSE(device.enqueueSparseTextureMappings(rachis::QueueType::copy, {.standardTiles = invalidHeapMapping})
                    .has_value());

    const auto sparseHeap = device.createResourceHeap({
        .size = properties->tileSize * properties->totalTileCount,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::nonAttachmentTexture,
        .flags = rachis::ResourceHeapFlag::sparseBacking,
        .label = "Sparse texture contract heap",
    });
    REQUIRE(sparseHeap.has_value());
    CHECK_FALSE(device.createPlacedTexture({.heap = *sparseHeap}, sparseDesc).has_value());

    const auto roundTrip = [&](std::uint32_t mipLevel,
                               rachis::Extent3D extent,
                               rachis::TimelinePoint mappingPoint,
                               std::span<const rachis::SparseTextureTileMapping> unmapTiles,
                               std::span<const rachis::SparseTextureMipTailMapping> unmapTails) {
        const rachis::BufferTextureCopyRegion request{
            .textureSubresource = {.mipLevel = mipLevel},
            .extent = extent,
        };
        const auto layout = device.queryTextureTransferLayout(*sparse, request);
        REQUIRE(layout.has_value());
        const auto upload = device.createBuffer({
            .size = layout->requiredBufferSize,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
        });
        const auto readback = device.createBuffer({
            .size = layout->requiredBufferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
        });
        REQUIRE(upload.has_value());
        REQUIRE(readback.has_value());

        auto uploadMemory = device.mapBuffer(*upload);
        REQUIRE(uploadMemory.has_value());
        std::fill_n(uploadMemory->data, uploadMemory->size, std::byte{0});
        for (std::uint32_t row = 0; row < layout->rowCount; ++row)
        {
            const auto rowOffset = static_cast<std::uint64_t>(row) * layout->bytesPerRow;
            for (std::uint64_t byte = 0; byte < layout->rowDataSize; ++byte)
            {
                uploadMemory->data[rowOffset + byte] = transferByte(row, byte);
            }
        }
        REQUIRE(device.unmapBuffer(*upload).has_value());

        const rachis::BufferTextureCopyRegion resolved{
            .bufferOffset = 0,
            .bytesPerRow = layout->bytesPerRow,
            .rowsPerImage = layout->rowsPerImage,
            .textureSubresource = {.mipLevel = mipLevel},
            .extent = extent,
        };
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *sparse,
                                        .stagesBefore = rachis::PipelineStage::all,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::none,
                                        .accessAfter = rachis::Access::copyDestination,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::copyDestination,
                                        .subresources =
                                            {
                                                .firstMipLevel = mipLevel,
                                                .mipLevelCount = 1,
                                                .firstArrayLayer = 0,
                                                .arrayLayerCount = 1,
                                            },
                                    })
                    .has_value());
        const auto beforeUploadPublication = rachis::d3d12::test::commandDependencySnapshot(*command);
        const auto beforeUploadRecording =
            rachis::d3d12::test::resourceCommandRecordCount().load(std::memory_order_relaxed);
        REQUIRE(beforeUploadPublication.has_value());
        {
            const FailResourceRetention failSparseUploadPublication(2);
            const auto rejected = device.copyBufferToTexture(*command, *upload, *sparse, std::span{&resolved, 1});
            REQUIRE_FALSE(rejected.has_value());
            CHECK(rejected.error().code == rachis::ErrorCode::outOfMemory);
        }
        const auto afterUploadFailure = rachis::d3d12::test::commandDependencySnapshot(*command);
        REQUIRE(afterUploadFailure.has_value());
        CHECK(*afterUploadFailure == *beforeUploadPublication);
        CHECK(rachis::d3d12::test::resourceCommandRecordCount().load(std::memory_order_relaxed) ==
              beforeUploadRecording);
        REQUIRE(device.copyBufferToTexture(*command, *upload, *sparse, std::span{&resolved, 1}).has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *sparse,
                                        .stagesBefore = rachis::PipelineStage::copy,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::copyDestination,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::copyDestination,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                        .subresources =
                                            {
                                                .firstMipLevel = mipLevel,
                                                .mipLevelCount = 1,
                                                .firstArrayLayer = 0,
                                                .arrayLayerCount = 1,
                                            },
                                    })
                    .has_value());
        const auto beforeReadbackPublication = rachis::d3d12::test::commandDependencySnapshot(*command);
        const auto beforeReadbackRecording =
            rachis::d3d12::test::resourceCommandRecordCount().load(std::memory_order_relaxed);
        REQUIRE(beforeReadbackPublication.has_value());
        {
            const FailResourceRetention failSparseReadbackPublication(2);
            const auto rejected = device.copyTextureToBuffer(*command, *sparse, *readback, std::span{&resolved, 1});
            REQUIRE_FALSE(rejected.has_value());
            CHECK(rejected.error().code == rachis::ErrorCode::outOfMemory);
        }
        const auto afterReadbackFailure = rachis::d3d12::test::commandDependencySnapshot(*command);
        REQUIRE(afterReadbackFailure.has_value());
        CHECK(*afterReadbackFailure == *beforeReadbackPublication);
        CHECK(rachis::d3d12::test::resourceCommandRecordCount().load(std::memory_order_relaxed) ==
              beforeReadbackRecording);
        REQUIRE(device.copyTextureToBuffer(*command, *sparse, *readback, std::span{&resolved, 1}).has_value());

        const auto retainedMapping = device.enqueueSparseTextureMappings(
            rachis::QueueType::compute, {.standardTiles = unmapTiles, .mipTails = unmapTails});
        REQUIRE_FALSE(retainedMapping.has_value());
        CHECK(retainedMapping.error().code == rachis::ErrorCode::invalidState);

        auto submitted = device.submit(std::move(*command));
        if (mappingPoint.queueType() != rachis::QueueType::graphics)
        {
            REQUIRE_FALSE(submitted.has_value());
            CHECK(submitted.error().code == rachis::ErrorCode::invalidState);
            REQUIRE(command->isValid());
            const std::array waits{mappingPoint};
            submitted = device.submit(std::move(*command), waits);
        }
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());
        device.collectGarbage();

        auto readbackMemory = device.mapBuffer(*readback);
        REQUIRE(readbackMemory.has_value());
        bool exact = true;
        for (std::uint32_t row = 0; row < layout->rowCount; ++row)
        {
            const auto rowOffset = static_cast<std::uint64_t>(row) * layout->bytesPerRow;
            for (std::uint64_t byte = 0; byte < layout->rowDataSize; ++byte)
            {
                exact = exact && readbackMemory->data[rowOffset + byte] == transferByte(row, byte);
            }
        }
        CHECK(exact);
        REQUIRE(device.unmapBuffer(*readback).has_value());
        REQUIRE(device.destroyBuffer(*upload).has_value());
        REQUIRE(device.destroyBuffer(*readback).has_value());
    };

    {
        const rachis::BufferTextureCopyRegion unmappedRegion{
            .textureSubresource = {},
            .extent = {1, 1, 1},
        };
        const auto unmappedLayout = device.queryTextureTransferLayout(*sparse, unmappedRegion);
        REQUIRE(unmappedLayout.has_value());
        const auto unmappedUpload = device.createBuffer({
            .size = unmappedLayout->requiredBufferSize,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
        });
        REQUIRE(unmappedUpload.has_value());
        auto unmappedCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(unmappedCommand.has_value());
        const auto unmappedCopy = device.copyBufferToTexture(*unmappedCommand,
                                                             rachis::BufferTextureCopy{
                                                                 .buffer = *unmappedUpload,
                                                                 .bytesPerRow = unmappedLayout->bytesPerRow,
                                                                 .rowsPerImage = unmappedLayout->rowsPerImage,
                                                                 .texture = *sparse,
                                                                 .extent = {1, 1, 1},
                                                             });
        CHECK_FALSE(unmappedCopy.has_value());
        CHECK(unmappedCopy.error().code == rachis::ErrorCode::invalidState);
        REQUIRE(device.destroyBuffer(*unmappedUpload).has_value());
    }

    const std::array standardMap{
        rachis::SparseTextureTileMapping{
            .texture = *sparse,
            .subresource = {},
            .textureTileOrigin = {},
            .textureTileExtent = {},
            .heap = *sparseHeap,
            .heapTileOffset = 0,
        },
    };
    const std::array standardUnmap{
        rachis::SparseTextureTileMapping{
            .texture = *sparse,
            .subresource = {},
            .textureTileOrigin = {},
            .textureTileExtent = {},
        },
    };
    const auto standardPoint =
        device.enqueueSparseTextureMappings(rachis::QueueType::copy, {.standardTiles = standardMap});
    REQUIRE(standardPoint.has_value());
    {
        auto rejectedCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
        REQUIRE(rejectedCommand.has_value());
        const auto beforePublication = rachis::d3d12::test::commandDependencySnapshot(*rejectedCommand);
        const auto beforeRecording = rachis::d3d12::test::resourceCommandRecordCount().load(std::memory_order_relaxed);
        REQUIRE(beforePublication.has_value());
        {
            const FailResourceRetention failSparseTexturePublication(2);
            const auto rejected = device.copyTexture(*rejectedCommand,
                                                     rachis::TextureCopy{
                                                         .source = *sparse,
                                                         .destination = *ordinary,
                                                         .extent = {1, 1, 1},
                                                     });
            REQUIRE_FALSE(rejected.has_value());
            CHECK(rejected.error().code == rachis::ErrorCode::outOfMemory);
        }
        const auto afterPublication = rachis::d3d12::test::commandDependencySnapshot(*rejectedCommand);
        REQUIRE(afterPublication.has_value());
        CHECK(*afterPublication == *beforePublication);
        CHECK(rachis::d3d12::test::resourceCommandRecordCount().load(std::memory_order_relaxed) == beforeRecording);
    }
    REQUIRE(properties->standardMipTileGrids.front().width >= 2);
    const std::array sameBatchPhysicalReuse{
        standardUnmap.front(),
        rachis::SparseTextureTileMapping{
            .texture = *sparse,
            .subresource = {},
            .textureTileOrigin = {.x = 1},
            .textureTileExtent = {},
            .heap = *sparseHeap,
            .heapTileOffset = 0,
        },
    };
    const auto sameBatchReuse =
        device.enqueueSparseTextureMappings(rachis::QueueType::graphics, {.standardTiles = sameBatchPhysicalReuse});
    REQUIRE_FALSE(sameBatchReuse.has_value());
    CHECK(sameBatchReuse.error().code == rachis::ErrorCode::invalidState);
    const rachis::Extent3D standardCopyExtent{
        .width = (std::min)(sparseDesc.width, properties->standardTileExtent.width),
        .height = (std::min)(sparseDesc.height, properties->standardTileExtent.height),
        .depth = 1,
    };
    roundTrip(0, standardCopyExtent, *standardPoint, standardUnmap, {});
    const auto standardUnmapPoint =
        device.enqueueSparseTextureMappings(rachis::QueueType::compute, {.standardTiles = standardUnmap});
    REQUIRE(standardUnmapPoint.has_value());
    REQUIRE(device.wait(*standardUnmapPoint).has_value());
    device.collectGarbage();

    if (properties->packedMipCount != 0)
    {
        REQUIRE(properties->mipTailTileCount != 0);
        REQUIRE(properties->mipTailTileCount < properties->totalTileCount);
        const auto tailHeapOffset = properties->totalTileCount - properties->mipTailTileCount;
        const std::array tailMap{
            rachis::SparseTextureMipTailMapping{
                .texture = *sparse,
                .heap = *sparseHeap,
                .heapTileOffset = tailHeapOffset,
            },
        };
        const std::array tailUnmap{
            rachis::SparseTextureMipTailMapping{
                .texture = *sparse,
            },
        };
        const auto tailPoint = device.enqueueSparseTextureMappings(rachis::QueueType::copy, {.mipTails = tailMap});
        REQUIRE(tailPoint.has_value());
        const auto tailMip = properties->standardMipCount;
        const rachis::Extent3D tailExtent{
            .width = (std::max)(1U, sparseDesc.width >> tailMip),
            .height = (std::max)(1U, sparseDesc.height >> tailMip),
            .depth = 1,
        };
        roundTrip(tailMip, tailExtent, *tailPoint, {}, tailUnmap);
        const auto tailUnmapPoint =
            device.enqueueSparseTextureMappings(rachis::QueueType::graphics, {.mipTails = tailUnmap});
        REQUIRE(tailUnmapPoint.has_value());
        REQUIRE(device.wait(*tailUnmapPoint).has_value());
        device.collectGarbage();
    }

    REQUIRE(device.destroyTexture(*ordinary).has_value());
    REQUIRE(device.destroyResourceHeap(*ordinaryHeap).has_value());
    REQUIRE(device.destroyTexture(*sparse).has_value());
    REQUIRE(device.destroyResourceHeap(*sparseHeap).has_value());
    device.collectGarbage();

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
