#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace
{

    [[nodiscard]] constexpr std::byte transferByte(std::uint32_t slice, std::uint32_t row, std::uint32_t byte) noexcept
    {
        return static_cast<std::byte>((slice * 71U + row * 23U + byte * 11U + 5U) & 0xFFU);
    }

} // namespace

TEST_CASE("texture transfer layouts exactly plan aligned staging storage",
          "[size:medium][rhi][texture][transfer][layout][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .enableDred = true,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto texture = device.createTexture({
        .dimension = rachis::TextureDimension::texture3D,
        .width = 5,
        .height = 3,
        .depthOrArrayLayers = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Transfer-layout 3D texture",
    });
    REQUIRE(texture.has_value());

    const rachis::BufferTextureCopyRegion request{
        .bufferOffset = 512,
        .textureSubresource = {.aspect = rachis::TextureAspect::color},
        .extent = {.width = 5, .height = 3, .depth = 2},
    };
    auto queried = device.queryTextureTransferLayout(*texture, request);
    REQUIRE(queried.has_value());
    CHECK(queried->bytesPerRow == 256);
    CHECK(queried->rowsPerImage == 3);
    CHECK(queried->rowCount == 3);
    CHECK(queried->rowDataSize == 20);
    CHECK(queried->bytesPerImage == 768);
    CHECK(queried->requiredBufferSize == 1812);

    auto paddedRequest = request;
    paddedRequest.bytesPerRow = 512;
    paddedRequest.rowsPerImage = 5;
    auto padded = device.queryTextureTransferLayout(*texture, paddedRequest);
    REQUIRE(padded.has_value());
    CHECK(padded->bytesPerRow == 512);
    CHECK(padded->rowsPerImage == 5);
    CHECK(padded->rowCount == 3);
    CHECK(padded->rowDataSize == 20);
    CHECK(padded->bytesPerImage == 2560);
    CHECK(padded->requiredBufferSize == 4116);

    auto invalidOffset = request;
    ++invalidOffset.bufferOffset;
    auto invalid = device.queryTextureTransferLayout(*texture, invalidOffset);
    REQUIRE_FALSE(invalid.has_value());
    CHECK(invalid.error().code == rachis::ErrorCode::invalidArgument);

    auto upload = device.createBuffer({
        .size = queried->requiredBufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Exact transfer-layout upload",
    });
    auto readback = device.createBuffer({
        .size = queried->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Exact transfer-layout readback",
    });
    REQUIRE(upload.has_value());
    REQUIRE(readback.has_value());
    auto mappedUpload = device.mapBuffer(*upload);
    REQUIRE(mappedUpload.has_value());
    std::fill_n(mappedUpload->data, mappedUpload->size, std::byte{0});
    for (std::uint32_t slice = 0; slice < request.extent.depth; ++slice)
    {
        for (std::uint32_t row = 0; row < queried->rowCount; ++row)
        {
            const auto rowOffset = request.bufferOffset + static_cast<std::uint64_t>(slice) * queried->bytesPerImage +
                                   static_cast<std::uint64_t>(row) * queried->bytesPerRow;
            for (std::uint32_t byte = 0; byte < queried->rowDataSize; ++byte)
            {
                mappedUpload->data[rowOffset + byte] = transferByte(slice, row, byte);
            }
        }
    }
    REQUIRE(device.unmapBuffer(*upload).has_value());

    const auto resolvedRequest = rachis::BufferTextureCopyRegion{
        .bufferOffset = request.bufferOffset,
        .bytesPerRow = queried->bytesPerRow,
        .rowsPerImage = queried->rowsPerImage,
        .textureSubresource = request.textureSubresource,
        .textureOrigin = request.textureOrigin,
        .extent = request.extent,
    };
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    REQUIRE(device.copyBufferToTexture(*command, *upload, *texture, std::span{&resolvedRequest, 1}).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::copyDestination,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device.copyTextureToBuffer(*command, *texture, *readback, std::span{&resolvedRequest, 1}).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto mappedReadback = device.mapBuffer(*readback);
    REQUIRE(mappedReadback.has_value());
    bool exact = true;
    for (std::uint32_t slice = 0; slice < request.extent.depth; ++slice)
    {
        for (std::uint32_t row = 0; row < queried->rowCount; ++row)
        {
            const auto rowOffset = request.bufferOffset + static_cast<std::uint64_t>(slice) * queried->bytesPerImage +
                                   static_cast<std::uint64_t>(row) * queried->bytesPerRow;
            for (std::uint32_t byte = 0; byte < queried->rowDataSize; ++byte)
            {
                exact = exact && mappedReadback->data[rowOffset + byte] == transferByte(slice, row, byte);
            }
        }
    }
    CHECK(exact);
    REQUIRE(device.unmapBuffer(*readback).has_value());

    auto compressed = device.createTexture({
        .width = 7,
        .height = 5,
        .format = rachis::Format::bc1RgbaUnorm,
        .usage = rachis::TextureUsage::copySource,
        .label = "Transfer-layout block texture",
    });
    REQUIRE(compressed.has_value());
    auto blockLayout =
        device.queryTextureTransferLayout(*compressed,
                                          {
                                              .textureSubresource = {.aspect = rachis::TextureAspect::color},
                                              .extent = {.width = 7, .height = 5},
                                          });
    REQUIRE(blockLayout.has_value());
    CHECK(blockLayout->bytesPerRow == 256);
    CHECK(blockLayout->rowsPerImage == 5);
    CHECK(blockLayout->rowCount == 2);
    CHECK(blockLayout->rowDataSize == 16);
    CHECK(blockLayout->bytesPerImage == 512);
    CHECK(blockLayout->requiredBufferSize == 272);

    auto depthStencil = device.createTexture({
        .width = 8,
        .height = 4,
        .format = rachis::Format::d24UnormS8Uint,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::depthStencil,
        .label = "Transfer-layout depth-stencil texture",
    });
    REQUIRE(depthStencil.has_value());
    auto depthLayout =
        device.queryTextureTransferLayout(*depthStencil,
                                          {
                                              .textureSubresource = {.aspect = rachis::TextureAspect::depth},
                                              .extent = {.width = 8, .height = 4},
                                          });
    auto stencilLayout =
        device.queryTextureTransferLayout(*depthStencil,
                                          {
                                              .textureSubresource = {.aspect = rachis::TextureAspect::stencil},
                                              .extent = {.width = 8, .height = 4},
                                          });
    REQUIRE(depthLayout.has_value());
    REQUIRE(stencilLayout.has_value());
    CHECK(depthLayout->rowDataSize == 32);
    CHECK(depthLayout->rowCount == 4);
    CHECK(depthLayout->requiredBufferSize == 800);
    CHECK(stencilLayout->rowDataSize == 8);
    CHECK(stencilLayout->rowCount == 4);
    CHECK(stencilLayout->requiredBufferSize == 776);

    auto multisampled = device.createTexture({
        .width = 8,
        .height = 8,
        .sampleCount = 4,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .label = "Transfer-layout multisampled texture",
    });
    REQUIRE(multisampled.has_value());
    CHECK_FALSE(device
                    .queryTextureTransferLayout(*multisampled,
                                                {
                                                    .textureSubresource = {.aspect = rachis::TextureAspect::color},
                                                    .extent = {.width = 8, .height = 8},
                                                })
                    .has_value());

    const auto staleTexture = *texture;
    REQUIRE(device.destroyTexture(*texture).has_value());
    CHECK_FALSE(device.queryTextureTransferLayout(staleTexture, request).has_value());

    REQUIRE(device.destroyTexture(*multisampled).has_value());
    REQUIRE(device.destroyTexture(*depthStencil).has_value());
    REQUIRE(device.destroyTexture(*compressed).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
