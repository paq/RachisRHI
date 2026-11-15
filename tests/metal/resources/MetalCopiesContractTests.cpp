#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

namespace
{

#if RACHIS_ENABLE_METAL
    constexpr std::uint32_t textureWidth = 5;
    constexpr std::uint32_t textureHeight = 3;
    constexpr std::uint32_t bytesPerPixel = 4;

    [[nodiscard]] rachis::DeviceDesc metalDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
    }

    [[nodiscard]] std::byte texelByte(std::uint32_t x, std::uint32_t y, std::uint32_t channel) noexcept
    {
        return static_cast<std::byte>((x * 31U + y * 47U + channel * 59U + 11U) & 0xFFU);
    }

    void writeTexels(std::byte* destination,
                     const rachis::TextureTransferLayout& layout,
                     std::uint32_t width,
                     std::uint32_t height,
                     std::uint32_t pixelBytes) noexcept
    {
        for (std::uint32_t y = 0; y < height; ++y)
        {
            for (std::uint32_t x = 0; x < width; ++x)
            {
                for (std::uint32_t channel = 0; channel < pixelBytes; ++channel)
                {
                    destination[static_cast<std::size_t>(y) * layout.bytesPerRow +
                                static_cast<std::size_t>(x) * pixelBytes + channel] = texelByte(x, y, channel);
                }
            }
        }
    }

    [[nodiscard]] bool texelsMatch(const std::byte* actual,
                                   const rachis::TextureTransferLayout& layout,
                                   std::uint32_t width,
                                   std::uint32_t height,
                                   std::uint32_t pixelBytes) noexcept
    {
        for (std::uint32_t y = 0; y < height; ++y)
        {
            for (std::uint32_t x = 0; x < width; ++x)
            {
                for (std::uint32_t channel = 0; channel < pixelBytes; ++channel)
                {
                    if (actual[static_cast<std::size_t>(y) * layout.bytesPerRow +
                               static_cast<std::size_t>(x) * pixelBytes + channel] != texelByte(x, y, channel))
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    struct BlockTransferCase final
    {
        rachis::Format format = rachis::Format::unknown;
        rachis::TextureAspect aspect = rachis::TextureAspect::color;
        std::uint32_t bytesPerBlock = 0;
        std::uint32_t blockExtent = 1;
        bool exactContents = true;
    };

    [[nodiscard]] std::vector<std::byte> blockTransferBytes(const BlockTransferCase& transfer, std::size_t size)
    {
        std::vector<std::byte> bytes(size);
        if (transfer.aspect == rachis::TextureAspect::depth)
        {
            for (std::size_t offset = 0; offset + sizeof(float) <= size; offset += sizeof(float))
            {
                const float depth = static_cast<float>((offset / sizeof(float)) % 8U + 1U) * 0.125F;
                std::memcpy(bytes.data() + offset, &depth, sizeof(depth));
            }
            return bytes;
        }
        for (std::size_t index = 0; index < size; ++index)
        {
            bytes[index] = static_cast<std::byte>((index * 37U + 5U) & 0xFFU);
        }
        return bytes;
    }
#endif

} // namespace

#if RACHIS_ENABLE_METAL

TEST_CASE("Metal texture transfer layouts use Metal pitch alignment", "[size:large][rhi][metal][transfer][layout]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto texture = device.createTexture(rachis::TextureDesc{
        .dimension = rachis::TextureDimension::texture3D,
        .width = textureWidth,
        .height = textureHeight,
        .depthOrArrayLayers = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Metal transfer-layout 3D",
    });
    REQUIRE(texture);

    const rachis::BufferTextureCopyRegion request{
        .bufferOffset = 0,
        .extent = {.width = textureWidth, .height = textureHeight, .depth = 2},
    };
    auto queried = device.queryTextureTransferLayout(*texture, request);
    REQUIRE(queried);
    CHECK(queried->rowCount == textureHeight);
    CHECK(queried->rowDataSize == std::uint64_t{textureWidth} * bytesPerPixel);
    CHECK(queried->bytesPerRow >= queried->rowDataSize);
    CHECK(queried->bytesPerRow % queried->rowPitchAlignment == 0);
    CHECK(queried->rowsPerImage == textureHeight);
    CHECK(queried->bytesPerImage == static_cast<std::uint64_t>(queried->bytesPerRow) * queried->rowsPerImage);
    CHECK(queried->requiredBufferSize >= queried->bytesPerImage + queried->rowDataSize);
    CHECK(queried->bufferOffsetAlignment >= 1);

    auto padded = request;
    padded.bytesPerRow = queried->bytesPerRow * 2;
    padded.rowsPerImage = textureHeight + 2;
    auto paddedLayout = device.queryTextureTransferLayout(*texture, padded);
    REQUIRE(paddedLayout);
    CHECK(paddedLayout->bytesPerRow == padded.bytesPerRow);
    CHECK(paddedLayout->rowsPerImage == padded.rowsPerImage);
    CHECK(paddedLayout->rowCount == textureHeight);
    CHECK(paddedLayout->rowDataSize == queried->rowDataSize);

    auto invalidOffset = request;
    invalidOffset.bufferOffset = 1;
    auto invalid = device.queryTextureTransferLayout(*texture, invalidOffset);
    REQUIRE_FALSE(invalid);
    CHECK(invalid.error().code == rachis::ErrorCode::invalidArgument);
}

TEST_CASE("Metal awkward-width uploads preserve sentinels after GPU readback",
          "[size:large][rhi][metal][transfer][gpu]")
{
    const auto [format, pixelBytes] = GENERATE(std::pair{rachis::Format::rgba8Unorm, 4U},
                                               std::pair{rachis::Format::r8Uint, 1U},
                                               std::pair{rachis::Format::r8Sint, 1U},
                                               std::pair{rachis::Format::rg8Uint, 2U},
                                               std::pair{rachis::Format::rg8Sint, 2U},
                                               std::pair{rachis::Format::rgba8Uint, 4U},
                                               std::pair{rachis::Format::rgba8Sint, 4U},
                                               std::pair{rachis::Format::r16Uint, 2U},
                                               std::pair{rachis::Format::r16Sint, 2U},
                                               std::pair{rachis::Format::rg16Uint, 4U},
                                               std::pair{rachis::Format::rg16Sint, 4U},
                                               std::pair{rachis::Format::rgba16Uint, 8U},
                                               std::pair{rachis::Format::rgba16Sint, 8U},
                                               std::pair{rachis::Format::r32Uint, 4U},
                                               std::pair{rachis::Format::r32Sint, 4U},
                                               std::pair{rachis::Format::rg32Uint, 8U},
                                               std::pair{rachis::Format::rg32Sint, 8U},
                                               std::pair{rachis::Format::rgba32Uint, 16U},
                                               std::pair{rachis::Format::rgba32Sint, 16U});
    CAPTURE(format);
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto texture = device.createTexture(rachis::TextureDesc{
        .width = textureWidth,
        .height = textureHeight,
        .format = format,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Metal awkward upload",
    });
    REQUIRE(texture);

    const rachis::BufferTextureCopyRegion region{
        .bufferOffset = 16,
        .extent = {.width = textureWidth, .height = textureHeight, .depth = 1},
    };
    auto layout = device.queryTextureTransferLayout(*texture, region);
    REQUIRE(layout);
    REQUIRE(layout->requiredBufferSize > layout->rowDataSize);

    auto upload = device.createBuffer(rachis::BufferDesc{
        .size = layout->requiredBufferSize + 32,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal transfer upload",
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = layout->requiredBufferSize + 32,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal transfer readback",
    });
    REQUIRE(upload);
    REQUIRE(readback);

    std::vector<std::byte> bytes(static_cast<std::size_t>(layout->requiredBufferSize + 32), std::byte{0xA5});
    writeTexels(
        bytes.data() + static_cast<std::size_t>(region.bufferOffset), *layout, textureWidth, textureHeight, pixelBytes);
    REQUIRE(device.writeBuffer(*upload, 0, bytes).has_value());

    auto initialReadback = device.mapBuffer(*readback);
    REQUIRE(initialReadback);
    std::fill_n(initialReadback->data, bytes.size(), std::byte{0xCD});
    REQUIRE(device.unmapBuffer(*readback).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *texture,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    REQUIRE(device.copyBufferToTexture(*command, *upload, *texture, std::array{region}).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::copyDestination,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device.copyTextureToBuffer(*command, *texture, *readback, std::array{region}).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped);
    CHECK(mapped->data[0] == std::byte{0xCD});
    CHECK(mapped->data[layout->requiredBufferSize + 31] == std::byte{0xCD});
    CHECK(texelsMatch(mapped->data + static_cast<std::size_t>(region.bufferOffset),
                      *layout,
                      textureWidth,
                      textureHeight,
                      pixelBytes));
    REQUIRE(device.unmapBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal compressed and depth transfers use block footprints without linear alignment queries",
          "[size:large][rhi][metal][transfer][format]")
{
    const auto transfer = GENERATE(
        BlockTransferCase{.format = rachis::Format::bc1RgbaUnorm, .bytesPerBlock = 8, .blockExtent = 4},
        BlockTransferCase{.format = rachis::Format::bc1RgbaUnormSrgb, .bytesPerBlock = 8, .blockExtent = 4},
        BlockTransferCase{.format = rachis::Format::bc2RgbaUnorm, .bytesPerBlock = 16, .blockExtent = 4},
        BlockTransferCase{.format = rachis::Format::bc2RgbaUnormSrgb, .bytesPerBlock = 16, .blockExtent = 4},
        BlockTransferCase{.format = rachis::Format::bc3RgbaUnorm, .bytesPerBlock = 16, .blockExtent = 4},
        BlockTransferCase{.format = rachis::Format::bc3RgbaUnormSrgb, .bytesPerBlock = 16, .blockExtent = 4},
        BlockTransferCase{.format = rachis::Format::bc4RUnorm, .bytesPerBlock = 8, .blockExtent = 4},
        BlockTransferCase{.format = rachis::Format::bc4RSnorm, .bytesPerBlock = 8, .blockExtent = 4},
        BlockTransferCase{.format = rachis::Format::bc5RgUnorm, .bytesPerBlock = 16, .blockExtent = 4},
        BlockTransferCase{.format = rachis::Format::bc5RgSnorm, .bytesPerBlock = 16, .blockExtent = 4},
        BlockTransferCase{.format = rachis::Format::bc6hRgbUfloat, .bytesPerBlock = 16, .blockExtent = 4},
        BlockTransferCase{.format = rachis::Format::bc6hRgbSfloat, .bytesPerBlock = 16, .blockExtent = 4},
        BlockTransferCase{.format = rachis::Format::bc7RgbaUnorm, .bytesPerBlock = 16, .blockExtent = 4},
        BlockTransferCase{.format = rachis::Format::bc7RgbaUnormSrgb, .bytesPerBlock = 16, .blockExtent = 4},
        BlockTransferCase{
            .format = rachis::Format::d32Float, .aspect = rachis::TextureAspect::depth, .bytesPerBlock = 4},
        BlockTransferCase{.format = rachis::Format::d24UnormS8Uint,
                          .aspect = rachis::TextureAspect::depth,
                          .bytesPerBlock = 4,
                          .exactContents = false},
        BlockTransferCase{
            .format = rachis::Format::d24UnormS8Uint, .aspect = rachis::TextureAspect::stencil, .bytesPerBlock = 1});
    CAPTURE(transfer.format, transfer.aspect);
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    const auto properties = device.queryFormatProperties(transfer.format);
    if (!rachis::hasAll(properties.features,
                        rachis::FormatFeature::copySource | rachis::FormatFeature::copyDestination))
    {
        return;
    }

    constexpr std::uint32_t extent = 8;
    const bool depthStencil = transfer.aspect != rachis::TextureAspect::color;
    auto texture = device.createTexture(rachis::TextureDesc{
        .width = extent,
        .height = extent,
        .format = transfer.format,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 (depthStencil ? rachis::TextureUsage::depthStencil : rachis::TextureUsage::sampled),
        .label = "Metal block transfer",
    });
    REQUIRE(texture);

    const rachis::BufferTextureCopyRegion region{
        .bufferOffset = transfer.bytesPerBlock,
        .textureSubresource = {.aspect = transfer.aspect},
        .extent = {.width = extent, .height = extent, .depth = 1},
    };
    auto layout = device.queryTextureTransferLayout(*texture, region);
    REQUIRE(layout);
    const auto blocks = extent / transfer.blockExtent;
    CHECK(layout->bufferOffsetAlignment == transfer.bytesPerBlock);
    CHECK(layout->rowPitchAlignment == transfer.bytesPerBlock);
    CHECK(layout->rowCount == blocks);
    CHECK(layout->rowDataSize == std::uint64_t{blocks} * transfer.bytesPerBlock);
    CHECK(layout->bytesPerRow == layout->rowDataSize);
    CHECK(layout->requiredBufferSize == region.bufferOffset + layout->rowDataSize * blocks);

    if (transfer.bytesPerBlock > 1)
    {
        auto misaligned = region;
        misaligned.bufferOffset = transfer.bytesPerBlock + 1U;
        auto rejected = device.queryTextureTransferLayout(*texture, misaligned);
        REQUIRE_FALSE(rejected);
        CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
    }

    auto upload = device.createBuffer(rachis::BufferDesc{
        .size = layout->requiredBufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal block transfer upload",
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = layout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal block transfer readback",
    });
    REQUIRE(upload);
    REQUIRE(readback);
    const auto bytes = blockTransferBytes(transfer, static_cast<std::size_t>(layout->requiredBufferSize));
    REQUIRE(device.writeBuffer(*upload, 0, bytes).has_value());

    const rachis::TextureSubresourceRange subresources{.aspect = transfer.aspect};
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *texture,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                    .subresources = subresources,
                                })
                .has_value());
    REQUIRE(device.copyBufferToTexture(*command, *upload, *texture, std::array{region}).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::copyDestination,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                    .subresources = subresources,
                                })
                .has_value());
    REQUIRE(device.copyTextureToBuffer(*command, *texture, *readback, std::array{region}).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());

    std::vector<std::byte> actual(bytes.size());
    REQUIRE(device.readBuffer(*readback, 0, actual).has_value());
    if (transfer.exactContents)
    {
        const auto offset = static_cast<std::ptrdiff_t>(region.bufferOffset);
        CHECK(std::equal(bytes.begin() + offset, bytes.end(), actual.begin() + offset));
    }

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal buffer-texture batches reject overlap and empty copies atomically",
          "[size:large][rhi][metal][transfer][batch]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto texture = device.createTexture(rachis::TextureDesc{
        .width = 8,
        .height = 8,
        .mipLevels = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Metal overlapping copy dest",
    });
    auto upload = device.createBuffer(rachis::BufferDesc{
        .size = 4096,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(texture);
    REQUIRE(upload);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(command);
    CHECK_FALSE(device.copyBufferToTexture(*command, *upload, *texture, {}).has_value());

    const rachis::BufferTextureCopyRegion first{
        .extent = {.width = 4, .height = 4, .depth = 1},
    };
    auto overlapping = first;
    overlapping.textureOrigin = {.x = 2, .y = 2};
    CHECK_FALSE(device.copyBufferToTexture(*command, *upload, *texture, std::array{first, overlapping}).has_value());

    const rachis::BufferTextureCopyRegion mip{
        .textureSubresource = {.mipLevel = 1},
        .extent = {.width = 4, .height = 4, .depth = 1},
    };
    REQUIRE(device.copyBufferToTexture(*command, *upload, *texture, std::array{first, mip}).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal buffer copies require copy usages and reject overlapping regions",
          "[size:large][rhi][metal][transfer][batch][regression]")
{
    constexpr std::uint64_t bufferSize = 256;
    constexpr std::uint64_t copySize = 64;

    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto scratch = device.createBuffer(rachis::BufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal same-buffer copy scratch",
    });
    auto sourceOnly = device.createBuffer(rachis::BufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal copy-source buffer",
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal copy readback",
    });
    auto storageOnly = device.createBuffer(rachis::BufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::storage,
        .label = "Metal storage-only buffer",
    });
    REQUIRE(scratch);
    REQUIRE(sourceOnly);
    REQUIRE(readback);
    REQUIRE(storageOnly);

    std::vector<std::byte> pattern(static_cast<std::size_t>(bufferSize));
    for (std::size_t index = 0; index < pattern.size(); ++index)
    {
        pattern[index] = static_cast<std::byte>((index * 7U + 3U) & 0xFFU);
    }
    REQUIRE(device.writeBuffer(*scratch, 0, pattern).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(command);

    auto noSourceUsage = device.copyBuffer(*command, {.source = *storageOnly, .destination = *readback, .size = 16});
    REQUIRE_FALSE(noSourceUsage);
    CHECK(noSourceUsage.error().code == rachis::ErrorCode::invalidArgument);
    auto noDestinationUsage =
        device.copyBuffer(*command, {.source = *sourceOnly, .destination = *storageOnly, .size = 16});
    REQUIRE_FALSE(noDestinationUsage);
    CHECK(noDestinationUsage.error().code == rachis::ErrorCode::invalidArgument);

    const std::array overlappingDestinations{
        rachis::BufferCopyRegion{.sourceOffset = 0, .destinationOffset = 0, .size = 32},
        rachis::BufferCopyRegion{.sourceOffset = 64, .destinationOffset = 16, .size = 32},
    };
    auto destinationOverlap = device.copyBuffer(*command, *sourceOnly, *readback, overlappingDestinations);
    REQUIRE_FALSE(destinationOverlap);
    CHECK(destinationOverlap.error().code == rachis::ErrorCode::invalidArgument);

    const std::array selfOverlap{
        rachis::BufferCopyRegion{.sourceOffset = 0, .destinationOffset = 16, .size = 32},
    };
    auto sameRegionOverlap = device.copyBuffer(*command, *scratch, *scratch, selfOverlap);
    REQUIRE_FALSE(sameRegionOverlap);
    CHECK(sameRegionOverlap.error().code == rachis::ErrorCode::invalidArgument);

    const std::array crossOverlap{
        rachis::BufferCopyRegion{.sourceOffset = 0, .destinationOffset = 64, .size = 32},
        rachis::BufferCopyRegion{.sourceOffset = 80, .destinationOffset = 128, .size = 16},
    };
    auto crossRegionOverlap = device.copyBuffer(*command, *scratch, *scratch, crossOverlap);
    REQUIRE_FALSE(crossRegionOverlap);
    CHECK(crossRegionOverlap.error().code == rachis::ErrorCode::invalidArgument);

    const std::array disjoint{
        rachis::BufferCopyRegion{.sourceOffset = 0, .destinationOffset = 128, .size = copySize},
    };
    REQUIRE(device.copyBuffer(*command, *scratch, *scratch, disjoint).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *scratch,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(
        device.copyBuffer(*command, {.source = *scratch, .destination = *readback, .size = bufferSize}).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped);
    const std::span copied{mapped->data, static_cast<std::size_t>(bufferSize)};
    CHECK(std::ranges::equal(copied.subspan(0, 128), std::span{pattern}.subspan(0, 128)));
    CHECK(std::ranges::equal(copied.subspan(128, copySize), std::span{pattern}.subspan(0, copySize)));
    CHECK(std::ranges::equal(copied.subspan(128 + copySize), std::span{pattern}.subspan(128 + copySize)));
    REQUIRE(device.unmapBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

#endif
