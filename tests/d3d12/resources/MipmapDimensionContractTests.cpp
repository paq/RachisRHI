#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace
{

    constexpr std::uint32_t baseSize = 4;
    constexpr std::uint32_t mipLevels = 3;
    constexpr std::uint32_t bytesPerPixel = 4;
    constexpr std::uint32_t rowPitch = 256;
    constexpr std::uint64_t mipTailOffset = 1024;

    void printDiagnostic(const rachis::DiagnosticMessage& message, void*) noexcept
    {
        static_cast<void>(std::fprintf(stderr, "D3D12 [%u]: ", static_cast<unsigned int>(message.nativeId)));
        if (!message.description.empty())
        {
            static_cast<void>(std::fwrite(message.description.data(), 1, message.description.size(), stderr));
        }
        static_cast<void>(std::fputc('\n', stderr));
    }

    [[nodiscard]] rachis::Device createWarpDevice()
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
        return std::move(*created);
    }

    [[nodiscard]] bool nearByte(std::byte actual, std::uint8_t expected) noexcept
    {
        const auto value = std::to_integer<std::uint8_t>(actual);
        const auto lower = expected == 0 ? 0U : static_cast<std::uint32_t>(expected) - 1U;
        const auto upper = (std::min)(static_cast<std::uint32_t>(expected) + 1U, 255U);
        return value >= lower && value <= upper;
    }

    void checkDiagnostics(const rachis::Device& device)
    {
        const auto diagnostics = device.diagnosticCounts();
        CHECK(diagnostics.warnings == 0);
        CHECK(diagnostics.errors == 0);
        CHECK(diagnostics.corruptions == 0);
    }

} // namespace

TEST_CASE("mipmap generation filters a selected 1D array slice on WARP", "[size:medium][rhi][texture][mipmap][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).mipmapGeneration);

    auto upload = device.createBuffer({
        .size = rowPitch,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "1D mipmap seed upload",
    });
    auto readback = device.createBuffer({
        .size = 512 + rowPitch,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "1D generated mip readback",
    });
    const std::array alternateFormats{rachis::Format::rgba8UnormSrgb};
    auto texture = device.createTexture({
        .dimension = rachis::TextureDimension::texture1D,
        .width = baseSize,
        .height = 1,
        .depthOrArrayLayers = 2,
        .mipLevels = mipLevels,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
        .viewFormats = alternateFormats,
        .label = "Generated 1D mip texture array",
    });
    REQUIRE(upload.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(texture.has_value());

    auto mapped = device.mapBuffer(*upload);
    REQUIRE(mapped.has_value());
    std::fill_n(mapped->data, mapped->size, std::byte{0});
    for (std::uint32_t x = 0; x < baseSize; ++x)
    {
        for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
        {
            const auto value = channel == 3 ? 255U : x * 40U;
            mapped->data[static_cast<std::uint64_t>(x) * bytesPerPixel + channel] = static_cast<std::byte>(value);
        }
    }
    REQUIRE(device.unmapBuffer(*upload).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    const rachis::TextureSubresourceRange seedRange{
        .firstMipLevel = 0,
        .mipLevelCount = 1,
        .firstArrayLayer = 1,
        .arrayLayerCount = 1,
        .aspect = rachis::TextureAspect::color,
    };
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
                                    .subresources = seedRange,
                                })
                .has_value());
    REQUIRE(device
                .copyBufferToTexture(*command,
                                     {
                                         .buffer = *upload,
                                         .bytesPerRow = rowPitch,
                                         .rowsPerImage = 1,
                                         .texture = *texture,
                                         .textureSubresource = {.arrayLayer = 1},
                                         .extent = {baseSize, 1, 1},
                                     })
                .has_value());
    REQUIRE(device
                .generateMipmaps(*command,
                                 {
                                     .texture = *texture,
                                     .subresources =
                                         {
                                             .firstMipLevel = 0,
                                             .mipLevelCount = mipLevels,
                                             .firstArrayLayer = 1,
                                             .arrayLayerCount = 1,
                                             .aspect = rachis::TextureAspect::color,
                                         },
                                     .baseLayout = rachis::TextureLayout::copyDestination,
                                     .finalLayout = rachis::TextureLayout::copySource,
                                 })
                .has_value());
    REQUIRE(device
                .copyTextureToBuffer(*command,
                                     {
                                         .buffer = *readback,
                                         .bytesPerRow = rowPitch,
                                         .rowsPerImage = 1,
                                         .texture = *texture,
                                         .textureSubresource = {.mipLevel = 1, .arrayLayer = 1},
                                         .extent = {2, 1, 1},
                                     })
                .has_value());
    REQUIRE(device
                .copyTextureToBuffer(*command,
                                     {
                                         .buffer = *readback,
                                         .bufferOffset = 512,
                                         .bytesPerRow = rowPitch,
                                         .rowsPerImage = 1,
                                         .texture = *texture,
                                         .textureSubresource = {.mipLevel = 2, .arrayLayer = 1},
                                         .extent = {1, 1, 1},
                                     })
                .has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    for (std::uint32_t x = 0; x < 2; ++x)
    {
        const auto expected = static_cast<std::uint8_t>(20U + x * 80U);
        for (std::uint32_t channel = 0; channel < 3; ++channel)
        {
            CAPTURE(x, channel, expected);
            CHECK(nearByte(mapped->data[static_cast<std::uint64_t>(x) * bytesPerPixel + channel], expected));
        }
        CHECK(mapped->data[static_cast<std::uint64_t>(x) * bytesPerPixel + 3] == std::byte{255});
    }
    for (std::uint32_t channel = 0; channel < 3; ++channel)
    {
        CHECK(nearByte(mapped->data[512 + channel], 60));
    }
    CHECK(mapped->data[512 + 3] == std::byte{255});
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    checkDiagnostics(device);
}

TEST_CASE("mipmap generation filters every spatial axis of a 3D texture on WARP",
          "[size:medium][rhi][texture][mipmap][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).mipmapGeneration);

    constexpr auto uploadSize = static_cast<std::uint64_t>(rowPitch) * baseSize * baseSize;
    auto upload = device.createBuffer({
        .size = uploadSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "3D mipmap seed upload",
    });
    auto readback = device.createBuffer({
        .size = mipTailOffset + rowPitch,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "3D generated mip readback",
    });
    auto texture = device.createTexture({
        .dimension = rachis::TextureDimension::texture3D,
        .width = baseSize,
        .height = baseSize,
        .depthOrArrayLayers = baseSize,
        .mipLevels = mipLevels,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
        .label = "Generated 3D mip texture",
    });
    REQUIRE(upload.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(texture.has_value());

    auto mapped = device.mapBuffer(*upload);
    REQUIRE(mapped.has_value());
    std::fill_n(mapped->data, mapped->size, std::byte{0});
    for (std::uint32_t z = 0; z < baseSize; ++z)
    {
        for (std::uint32_t y = 0; y < baseSize; ++y)
        {
            for (std::uint32_t x = 0; x < baseSize; ++x)
            {
                const auto offset = static_cast<std::uint64_t>(z) * rowPitch * baseSize +
                                    static_cast<std::uint64_t>(y) * rowPitch +
                                    static_cast<std::uint64_t>(x) * bytesPerPixel;
                mapped->data[offset] = static_cast<std::byte>(x * 40U);
                mapped->data[offset + 1] = static_cast<std::byte>(y * 40U);
                mapped->data[offset + 2] = static_cast<std::byte>(z * 40U);
                mapped->data[offset + 3] = std::byte{255};
            }
        }
    }
    REQUIRE(device.unmapBuffer(*upload).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    const rachis::TextureMipmapGeneration generation{
        .texture = *texture,
        .subresources =
            {
                .firstMipLevel = 0,
                .mipLevelCount = mipLevels,
                .firstArrayLayer = 0,
                .arrayLayerCount = 1,
                .aspect = rachis::TextureAspect::color,
            },
        .baseLayout = rachis::TextureLayout::copyDestination,
        .finalLayout = rachis::TextureLayout::copySource,
    };
    auto invalidLayer = generation;
    invalidLayer.subresources.firstArrayLayer = 1;
    CHECK_FALSE(device.generateMipmaps(*command, invalidLayer).has_value());
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
                                    .subresources =
                                        {
                                            .firstMipLevel = 0,
                                            .mipLevelCount = 1,
                                            .firstArrayLayer = 0,
                                            .arrayLayerCount = 1,
                                            .aspect = rachis::TextureAspect::color,
                                        },
                                })
                .has_value());
    REQUIRE(device
                .copyBufferToTexture(*command,
                                     {
                                         .buffer = *upload,
                                         .bytesPerRow = rowPitch,
                                         .rowsPerImage = baseSize,
                                         .texture = *texture,
                                         .textureSubresource = {},
                                         .extent = {baseSize, baseSize, baseSize},
                                     })
                .has_value());
    REQUIRE(device.generateMipmaps(*command, generation).has_value());
    REQUIRE(device
                .copyTextureToBuffer(*command,
                                     {
                                         .buffer = *readback,
                                         .bytesPerRow = rowPitch,
                                         .rowsPerImage = 2,
                                         .texture = *texture,
                                         .textureSubresource = {.mipLevel = 1},
                                         .extent = {2, 2, 2},
                                     })
                .has_value());
    REQUIRE(device
                .copyTextureToBuffer(*command,
                                     {
                                         .buffer = *readback,
                                         .bufferOffset = mipTailOffset,
                                         .bytesPerRow = rowPitch,
                                         .rowsPerImage = 1,
                                         .texture = *texture,
                                         .textureSubresource = {.mipLevel = 2},
                                         .extent = {1, 1, 1},
                                     })
                .has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    for (std::uint32_t z = 0; z < 2; ++z)
    {
        for (std::uint32_t y = 0; y < 2; ++y)
        {
            for (std::uint32_t x = 0; x < 2; ++x)
            {
                const auto offset = static_cast<std::uint64_t>(z) * rowPitch * 2 +
                                    static_cast<std::uint64_t>(y) * rowPitch +
                                    static_cast<std::uint64_t>(x) * bytesPerPixel;
                const std::array expected{
                    static_cast<std::uint8_t>(20U + x * 80U),
                    static_cast<std::uint8_t>(20U + y * 80U),
                    static_cast<std::uint8_t>(20U + z * 80U),
                };
                for (std::uint32_t channel = 0; channel < expected.size(); ++channel)
                {
                    CAPTURE(x, y, z, channel, expected[channel]);
                    CHECK(nearByte(mapped->data[offset + channel], expected[channel]));
                }
                CHECK(mapped->data[offset + 3] == std::byte{255});
            }
        }
    }
    for (std::uint32_t channel = 0; channel < 3; ++channel)
    {
        CHECK(nearByte(mapped->data[mipTailOffset + channel], 60));
    }
    CHECK(mapped->data[mipTailOffset + 3] == std::byte{255});
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    checkDiagnostics(device);
}
