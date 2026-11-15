#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

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
    constexpr std::uint8_t encodedHalfLinear = 188;

    struct SrgbMipFormats final
    {
        rachis::Format srgb = rachis::Format::unknown;
        rachis::Format linear = rachis::Format::unknown;
    };

    void printDiagnostic(const rachis::DiagnosticMessage& message, void*) noexcept
    {
        static_cast<void>(std::fprintf(stderr, "Vulkan [%u]: ", static_cast<unsigned int>(message.nativeId)));
        if (!message.description.empty())
        {
            static_cast<void>(std::fwrite(message.description.data(), 1, message.description.size(), stderr));
        }
        static_cast<void>(std::fputc('\n', stderr));
    }

    [[nodiscard]] rachis::DeviceDesc vulkanDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::vulkan,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
            .diagnosticCallback = &printDiagnostic,
        };
    }

    [[nodiscard]] bool nearByte(std::uint8_t actual, std::uint8_t expected, std::uint8_t tolerance = 1) noexcept
    {
        const auto value = static_cast<std::uint32_t>(actual);
        const auto expectedValue = static_cast<std::uint32_t>(expected);
        const auto difference = value > expectedValue ? value - expectedValue : expectedValue - value;
        return difference <= static_cast<std::uint32_t>(tolerance);
    }

    [[nodiscard]] std::array<std::uint8_t, bytesPerPixel> generateCheckerboardMip(rachis::Device& device,
                                                                                  rachis::TextureHandle texture)
    {
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
        REQUIRE(upload.has_value());
        REQUIRE(readback.has_value());

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
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = texture,
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
                                             .texture = texture,
                                             .textureSubresource = {},
                                             .extent = {baseSize, baseSize, 1},
                                         })
                    .has_value());
        REQUIRE(device
                    .generateMipmaps(*command,
                                     {
                                         .texture = texture,
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
                                             .texture = texture,
                                             .textureSubresource = {.mipLevel = 1},
                                             .extent = {1, 1, 1},
                                         })
                    .has_value());

        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());
        mapped = device.mapBuffer(*readback);
        REQUIRE(mapped.has_value());
        std::array<std::uint8_t, bytesPerPixel> texel{};
        for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
        {
            texel[channel] = std::to_integer<std::uint8_t>(mapped->data[channel]);
        }
        REQUIRE(device.unmapBuffer(*readback).has_value());
        REQUIRE(device.destroyBuffer(*upload).has_value());
        REQUIRE(device.destroyBuffer(*readback).has_value());
        return texel;
    }

    void checkLinearLightMip(const std::array<std::uint8_t, bytesPerPixel>& texel)
    {
        for (std::uint32_t channel = 0; channel < 3; ++channel)
        {
            CAPTURE(channel, texel[channel]);
            CHECK(nearByte(texel[channel], encodedHalfLinear));
        }
        CHECK(nearByte(texel[3], 255));
    }

} // namespace

TEST_CASE("Vulkan mipmap generation preserves sRGB encoding while filtering in linear light",
          "[size:medium][rhi][vulkan][texture][mipmap][srgb]")
{
    const auto formats =
        GENERATE(SrgbMipFormats{.srgb = rachis::Format::rgba8UnormSrgb, .linear = rachis::Format::rgba8Unorm},
                 SrgbMipFormats{.srgb = rachis::Format::bgra8UnormSrgb, .linear = rachis::Format::bgra8Unorm});
    CAPTURE(static_cast<int>(formats.srgb));

    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    REQUIRE(rachis::test::requireCapabilities(device).mipmapGeneration);
    const auto srgbProperties = device.queryFormatProperties(formats.srgb);
    const auto linearProperties = device.queryFormatProperties(formats.linear);
    REQUIRE(rachis::hasAll(srgbProperties.features,
                           rachis::FormatFeature::sampled | rachis::FormatFeature::filterable |
                               rachis::FormatFeature::viewFormatReinterpretation));
    const bool srgbStorage = rachis::hasAny(srgbProperties.features, rachis::FormatFeature::storageWrite);
    const bool linearStorage = rachis::hasAny(linearProperties.features, rachis::FormatFeature::storageWrite);

    const auto textureUsage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                              rachis::TextureUsage::sampled | rachis::TextureUsage::storage;
    const rachis::TextureDesc missingLinearView{
        .width = baseSize,
        .height = baseSize,
        .mipLevels = 2,
        .format = formats.srgb,
        .usage = textureUsage,
        .label = "Invalid sRGB mip texture without linear view",
    };
    if (!srgbStorage)
    {
        const auto rejected = device.createTexture(missingLinearView);
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::unsupported);
    }

    const std::array linearViewFormat{formats.linear};
    auto textureDesc = missingLinearView;
    textureDesc.viewFormats = linearViewFormat;
    textureDesc.label = "Color-space-correct sRGB mip texture";
    if (!linearStorage)
    {
        if (!srgbStorage)
        {
            const auto rejected = device.createTexture(textureDesc);
            REQUIRE_FALSE(rejected.has_value());
            CHECK(rejected.error().code == rachis::ErrorCode::unsupported);
        }
        return;
    }

    auto texture = device.createTexture(textureDesc);
    if (!texture)
    {
        UNSCOPED_INFO(texture.error().message);
    }
    REQUIRE(texture.has_value());

    auto linearStorageView = device.createTextureView({
        .texture = *texture,
        .usage = rachis::TextureViewUsage::storage,
        .format = formats.linear,
        .firstMipLevel = 1,
        .mipLevelCount = 1,
        .label = "Linear storage view of an sRGB texture",
    });
    REQUIRE(linearStorageView.has_value());
    REQUIRE(device.destroyTextureView(*linearStorageView).has_value());
    if (!srgbStorage)
    {
        const auto srgbStorageView = device.createTextureView({
            .texture = *texture,
            .usage = rachis::TextureViewUsage::storage,
            .format = formats.srgb,
            .firstMipLevel = 1,
            .mipLevelCount = 1,
            .label = "Invalid sRGB storage view",
        });
        REQUIRE_FALSE(srgbStorageView.has_value());
        CHECK(srgbStorageView.error().code == rachis::ErrorCode::unsupported);
    }

    checkLinearLightMip(generateCheckerboardMip(device, *texture));
    REQUIRE(device.destroyTexture(*texture).has_value());

    auto placedDesc = textureDesc;
    placedDesc.label = "Placed color-space-correct sRGB mip texture";
    const auto requirements = device.queryTextureMemoryRequirements(placedDesc);
    if (!requirements)
    {
        UNSCOPED_INFO(requirements.error().message);
    }
    REQUIRE(requirements.has_value());
    const auto heap = device.createResourceHeap({
        .size = requirements->size,
        .alignment = requirements->alignment,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::nonAttachmentTexture,
        .label = "Placed sRGB mip texture heap",
    });
    REQUIRE(heap.has_value());
    auto placed = device.createPlacedTexture({.heap = *heap, .offset = 0}, placedDesc);
    if (!placed)
    {
        UNSCOPED_INFO(placed.error().message);
    }
    REQUIRE(placed.has_value());
    checkLinearLightMip(generateCheckerboardMip(device, *placed));
    REQUIRE(device.destroyTexture(*placed).has_value());
    REQUIRE(device.destroyResourceHeap(*heap).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
