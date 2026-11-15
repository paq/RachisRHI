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

    constexpr std::uint32_t baseSize = 2;
    constexpr std::uint32_t rowPitch = 256;
    constexpr std::uint32_t bytesPerPixel = 4;

    void printDiagnostic(const rachis::DiagnosticMessage& message, void*) noexcept
    {
        static_cast<void>(std::fprintf(stderr, "D3D12 [%u]: ", static_cast<unsigned int>(message.nativeId)));
        if (!message.description.empty())
        {
            static_cast<void>(std::fwrite(message.description.data(), 1, message.description.size(), stderr));
        }
        static_cast<void>(std::fputc('\n', stderr));
    }

    [[nodiscard]] bool nearByte(std::byte actual, std::uint8_t expected, std::uint8_t tolerance = 1) noexcept
    {
        const auto value = static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(actual));
        const auto expectedValue = static_cast<std::uint32_t>(expected);
        const auto difference = value > expectedValue ? value - expectedValue : expectedValue - value;
        return difference <= static_cast<std::uint32_t>(tolerance);
    }

} // namespace

TEST_CASE("mipmap generation preserves sRGB encoding while filtering in linear light",
          "[size:medium][rhi][texture][mipmap][srgb][warp]")
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

    REQUIRE(rachis::test::requireCapabilities(device).mipmapGeneration);
    const auto srgbProperties = device.queryFormatProperties(rachis::Format::rgba8UnormSrgb);
    const auto linearProperties = device.queryFormatProperties(rachis::Format::rgba8Unorm);
    REQUIRE(rachis::hasAll(srgbProperties.features,
                           rachis::FormatFeature::filterable | rachis::FormatFeature::viewFormatReinterpretation));
    REQUIRE_FALSE(rachis::hasAny(srgbProperties.features, rachis::FormatFeature::storageWrite));
    REQUIRE(rachis::hasAny(linearProperties.features, rachis::FormatFeature::storageWrite));

    const auto textureUsage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                              rachis::TextureUsage::sampled | rachis::TextureUsage::storage;
    const rachis::TextureDesc missingLinearView{
        .width = baseSize,
        .height = baseSize,
        .mipLevels = 2,
        .format = rachis::Format::rgba8UnormSrgb,
        .usage = textureUsage,
        .label = "Invalid sRGB mip texture without linear view",
    };
    const auto rejected = device.createTexture(missingLinearView);
    REQUIRE_FALSE(rejected.has_value());
    CHECK(rejected.error().code == rachis::ErrorCode::unsupported);

    auto upload = device.createBuffer({
        .size = static_cast<std::uint64_t>(rowPitch) * baseSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "sRGB mip seed upload",
    });
    auto readback = device.createBuffer({
        .size = rowPitch,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "sRGB mip readback",
    });
    const std::array linearViewFormat{rachis::Format::rgba8Unorm};
    auto textureDesc = missingLinearView;
    textureDesc.viewFormats = linearViewFormat;
    textureDesc.label = "Color-space-correct sRGB mip texture";
    auto texture = device.createTexture(textureDesc);
    REQUIRE(upload.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(texture.has_value());

    auto mapped = device.mapBuffer(*upload);
    REQUIRE(mapped.has_value());
    std::fill_n(mapped->data, mapped->size, std::byte{0});
    for (std::uint32_t y = 0; y < baseSize; ++y)
    {
        for (std::uint32_t x = 0; x < baseSize; ++x)
        {
            const auto value = static_cast<std::byte>(((x + y) & 1U) == 0 ? 0 : 255);
            const auto offset =
                static_cast<std::uint64_t>(y) * rowPitch + static_cast<std::uint64_t>(x) * bytesPerPixel;
            mapped->data[offset] = value;
            mapped->data[offset + 1] = value;
            mapped->data[offset + 2] = value;
            mapped->data[offset + 3] = std::byte{255};
        }
    }
    REQUIRE(device.unmapBuffer(*upload).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    const rachis::TextureSubresourceRange seedRange{
        .firstMipLevel = 0,
        .mipLevelCount = 1,
        .firstArrayLayer = 0,
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
                                         .rowsPerImage = baseSize,
                                         .texture = *texture,
                                         .textureSubresource = {},
                                         .extent = {baseSize, baseSize, 1},
                                     })
                .has_value());
    REQUIRE(device
                .generateMipmaps(*command,
                                 {
                                     .texture = *texture,
                                     .subresources =
                                         {
                                             .firstMipLevel = 0,
                                             .mipLevelCount = 2,
                                             .firstArrayLayer = 0,
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
                                         .textureSubresource = {.mipLevel = 1},
                                         .extent = {1, 1, 1},
                                     })
                .has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());

    constexpr std::uint8_t encodedHalfLinear = 188;
    for (std::uint32_t channel = 0; channel < 3; ++channel)
    {
        CAPTURE(channel, std::to_integer<std::uint8_t>(mapped->data[channel]));
        CHECK(nearByte(mapped->data[channel], encodedHalfLinear));
    }
    CHECK(nearByte(mapped->data[3], 255));
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
