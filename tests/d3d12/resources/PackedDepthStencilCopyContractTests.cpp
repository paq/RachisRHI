#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <utility>

namespace
{

    constexpr std::uint32_t textureWidth = 8;
    constexpr std::uint32_t textureHeight = 4;
    constexpr std::uint32_t rowPitch = 256;
    constexpr std::uint64_t stencilOffset = 1024;
    constexpr std::uint64_t bufferSize = 2048;

    void printDiagnostic(const rachis::DiagnosticMessage& message, void*) noexcept
    {
        static_cast<void>(std::fprintf(stderr, "D3D12 [%u]: ", static_cast<unsigned int>(message.nativeId)));
        if (!message.description.empty())
        {
            static_cast<void>(std::fwrite(message.description.data(), 1, message.description.size(), stderr));
        }
        static_cast<void>(std::fputc('\n', stderr));
    }

    [[nodiscard]] constexpr std::uint32_t depthValue(std::uint32_t x, std::uint32_t y) noexcept
    {
        return ((x + 1U) * 0x010203U + (y + 1U) * 0x111111U) & 0x00FFFFFFU;
    }

    [[nodiscard]] constexpr std::byte stencilValue(std::uint32_t x, std::uint32_t y) noexcept
    {
        return static_cast<std::byte>((x * 29U + y * 53U + 7U) & 0xFFU);
    }

    void writePlanes(std::byte* destination)
    {
        for (std::uint32_t y = 0; y < textureHeight; ++y)
        {
            for (std::uint32_t x = 0; x < textureWidth; ++x)
            {
                const auto depth = depthValue(x, y);
                std::memcpy(
                    destination + static_cast<std::uint64_t>(y) * rowPitch + x * sizeof(depth), &depth, sizeof(depth));
                destination[stencilOffset + static_cast<std::uint64_t>(y) * rowPitch + x] = stencilValue(x, y);
            }
        }
    }

    [[nodiscard]] bool planesMatch(const std::byte* actual) noexcept
    {
        for (std::uint32_t y = 0; y < textureHeight; ++y)
        {
            for (std::uint32_t x = 0; x < textureWidth; ++x)
            {
                std::uint32_t depth = 0;
                std::memcpy(
                    &depth, actual + static_cast<std::uint64_t>(y) * rowPitch + x * sizeof(depth), sizeof(depth));
                if (depth != depthValue(x, y) ||
                    actual[stencilOffset + static_cast<std::uint64_t>(y) * rowPitch + x] != stencilValue(x, y))
                {
                    return false;
                }
            }
        }
        return true;
    }

} // namespace

TEST_CASE("packed depth-stencil planes upload and read back through canonical D3D12 footprints",
          "[size:medium][rhi][texture][transfer][depth-stencil][warp]")
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

    auto upload = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Packed depth-stencil upload",
    });
    auto readback = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Packed depth-stencil readback",
    });
    auto texture = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::d24UnormS8Uint,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::depthStencil,
        .label = "Packed depth-stencil transfer texture",
    });
    REQUIRE(upload.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(texture.has_value());

    auto mappedUpload = device.mapBuffer(*upload);
    REQUIRE(mappedUpload.has_value());
    std::fill_n(mappedUpload->data, mappedUpload->size, std::byte{0});
    writePlanes(mappedUpload->data);
    REQUIRE(device.unmapBuffer(*upload).has_value());

    const std::array regions{
        rachis::BufferTextureCopyRegion{
            .bytesPerRow = rowPitch,
            .rowsPerImage = textureHeight,
            .textureSubresource = {.aspect = rachis::TextureAspect::depth},
            .extent = {.width = textureWidth, .height = textureHeight},
        },
        rachis::BufferTextureCopyRegion{
            .bufferOffset = stencilOffset,
            .bytesPerRow = rowPitch,
            .rowsPerImage = textureHeight,
            .textureSubresource = {.aspect = rachis::TextureAspect::stencil},
            .extent = {.width = textureWidth, .height = textureHeight},
        },
    };

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    auto partialDepth = regions[0];
    --partialDepth.extent.width;
    auto partialResult = device.copyBufferToTexture(*command, *upload, *texture, std::span{&partialDepth, 1});
    REQUIRE_FALSE(partialResult.has_value());
    CHECK(partialResult.error().code == rachis::ErrorCode::unsupported);
    auto paddedDepth = regions[0];
    ++paddedDepth.rowsPerImage;
    auto paddedResult = device.copyBufferToTexture(*command, *upload, *texture, std::span{&paddedDepth, 1});
    REQUIRE_FALSE(paddedResult.has_value());
    CHECK(paddedResult.error().code == rachis::ErrorCode::invalidArgument);
    auto overlappingReadback = regions;
    overlappingReadback[1].bufferOffset = 0;
    CHECK_FALSE(device.copyTextureToBuffer(*command, *texture, *readback, overlappingReadback).has_value());

    for (const auto aspect : {rachis::TextureAspect::depth, rachis::TextureAspect::stencil})
    {
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
                                        .subresources = {.aspect = aspect},
                                    })
                    .has_value());
    }
    REQUIRE(device.copyBufferToTexture(*command, *upload, *texture, regions).has_value());
    for (const auto aspect : {rachis::TextureAspect::depth, rachis::TextureAspect::stencil})
    {
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
                                        .subresources = {.aspect = aspect},
                                    })
                    .has_value());
    }
    REQUIRE(device.copyTextureToBuffer(*command, *texture, *readback, regions).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    auto mappedReadback = device.mapBuffer(*readback);
    REQUIRE(mappedReadback.has_value());
    CHECK(planesMatch(mappedReadback->data));
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyTexture(*texture).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
