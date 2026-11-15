#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <span>
#include <string_view>
#include <vector>

namespace
{

    constexpr std::uint32_t blockWidth = 4;
    constexpr std::uint32_t blockHeight = 4;
    constexpr std::uint64_t transferAlignment = 512;
    constexpr std::uint64_t outputOffset = 4'096;
    constexpr std::uint64_t outputStride = 4 * sizeof(float);

    struct BlockCompressedFormatCase final
    {
        rachis::Format format = rachis::Format::unknown;
        std::uint32_t bytesPerBlock = 0;
        std::array<std::byte, 16> block{};
        std::array<float, 4> expected{};
        std::string_view label;
    };

    constexpr std::array<std::byte, 16> bc2Red{
        std::byte{0xFF},
        std::byte{0xFF},
        std::byte{0xFF},
        std::byte{0xFF},
        std::byte{0xFF},
        std::byte{0xFF},
        std::byte{0xFF},
        std::byte{0xFF},
        std::byte{0x00},
        std::byte{0xF8},
        std::byte{0x00},
        std::byte{0x00},
        std::byte{0x00},
        std::byte{0x00},
        std::byte{0x00},
        std::byte{0x00},
    };

    constexpr std::array<std::byte, 16> bc4Positive{
        std::byte{0xFF},
        std::byte{0x00},
        std::byte{0x00},
        std::byte{0x00},
        std::byte{0x00},
        std::byte{0x00},
        std::byte{0x00},
        std::byte{0x00},
    };

    constexpr std::array<std::byte, 16> bc4SignedPositive{
        std::byte{0x7F},
        std::byte{0x81},
        std::byte{0x00},
        std::byte{0x00},
        std::byte{0x00},
        std::byte{0x00},
        std::byte{0x00},
        std::byte{0x00},
    };

    constexpr std::array<std::byte, 16> bc5SignedPositiveNegative{
        std::byte{0x7F},
        std::byte{0x81},
        std::byte{0x00},
        std::byte{0x00},
        std::byte{0x00},
        std::byte{0x00},
        std::byte{0x00},
        std::byte{0x00},
        std::byte{0x81},
        std::byte{0x7F},
        std::byte{0x00},
        std::byte{0x00},
        std::byte{0x00},
        std::byte{0x00},
        std::byte{0x00},
        std::byte{0x00},
    };

    constexpr std::array blockCompressedFormats{
        BlockCompressedFormatCase{rachis::Format::bc2RgbaUnorm, 16, bc2Red, {1.0F, 0.0F, 0.0F, 1.0F}, "BC2 linear"},
        BlockCompressedFormatCase{rachis::Format::bc2RgbaUnormSrgb, 16, bc2Red, {1.0F, 0.0F, 0.0F, 1.0F}, "BC2 sRGB"},
        BlockCompressedFormatCase{rachis::Format::bc4RUnorm, 8, bc4Positive, {1.0F, 0.0F, 0.0F, 1.0F}, "BC4 UNORM"},
        BlockCompressedFormatCase{
            rachis::Format::bc4RSnorm, 8, bc4SignedPositive, {1.0F, 0.0F, 0.0F, 1.0F}, "BC4 SNORM"},
        BlockCompressedFormatCase{
            rachis::Format::bc5RgSnorm, 16, bc5SignedPositiveNegative, {1.0F, -1.0F, 0.0F, 1.0F}, "BC5 SNORM"},
        BlockCompressedFormatCase{
            rachis::Format::bc6hRgbUfloat, 16, {}, {0.0F, 0.0F, 0.0F, 1.0F}, "BC6H unsigned float"},
        BlockCompressedFormatCase{rachis::Format::bc6hRgbSfloat, 16, {}, {0.0F, 0.0F, 0.0F, 1.0F}, "BC6H signed float"},
    };

    constexpr std::uint64_t copySlotSize = 512;
    constexpr std::uint64_t copyRowPitch = 256;

    struct BlockCopyFormat final
    {
        rachis::Format format = rachis::Format::unknown;
        std::uint32_t bytesPerBlock = 0;
        std::string_view label;
    };

    constexpr std::array blockCopyFormats{
        BlockCopyFormat{rachis::Format::bc1RgbaUnorm, 8, "BC1"},
        BlockCopyFormat{rachis::Format::bc2RgbaUnorm, 16, "BC2"},
        BlockCopyFormat{rachis::Format::bc3RgbaUnorm, 16, "BC3"},
        BlockCopyFormat{rachis::Format::bc4RUnorm, 8, "BC4"},
        BlockCopyFormat{rachis::Format::bc5RgUnorm, 16, "BC5"},
        BlockCopyFormat{rachis::Format::bc6hRgbUfloat, 16, "BC6H"},
        BlockCopyFormat{rachis::Format::bc7RgbaUnorm, 16, "BC7"},
    };

    struct BlockCopyFamily final
    {
        rachis::Format linear = rachis::Format::unknown;
        rachis::Format srgb = rachis::Format::unknown;
        std::uint32_t bytesPerBlock = 0;
        std::string_view label;
    };

    constexpr std::array blockCopyFamilies{
        BlockCopyFamily{rachis::Format::bc1RgbaUnorm, rachis::Format::bc1RgbaUnormSrgb, 8, "BC1"},
        BlockCopyFamily{rachis::Format::bc2RgbaUnorm, rachis::Format::bc2RgbaUnormSrgb, 16, "BC2"},
        BlockCopyFamily{rachis::Format::bc3RgbaUnorm, rachis::Format::bc3RgbaUnormSrgb, 16, "BC3"},
        BlockCopyFamily{rachis::Format::bc7RgbaUnorm, rachis::Format::bc7RgbaUnormSrgb, 16, "BC7"},
    };

    struct BlockCopyTexture final
    {
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::uint32_t mipLevels = 0;
    };

    constexpr std::array blockCopyTextures{
        BlockCopyTexture{8, 8, 4},
        BlockCopyTexture{7, 5, 3},
    };

    struct BlockCopyMip final
    {
        std::size_t texture = 0;
        std::uint32_t mipLevel = 0;
        rachis::Extent3D extent;
        std::uint32_t blocksWide = 0;
        std::uint32_t blockRows = 0;
    };

    constexpr std::array blockCopyMips{
        BlockCopyMip{0, 0, {8, 8, 1}, 2, 2},
        BlockCopyMip{0, 1, {4, 4, 1}, 1, 1},
        BlockCopyMip{0, 2, {2, 2, 1}, 1, 1},
        BlockCopyMip{0, 3, {1, 1, 1}, 1, 1},
        BlockCopyMip{1, 0, {7, 5, 1}, 2, 2},
        BlockCopyMip{1, 1, {3, 2, 1}, 1, 1},
        BlockCopyMip{1, 2, {1, 1, 1}, 1, 1},
    };

    constexpr std::uint64_t paddedImageOffset = copySlotSize * blockCopyMips.size();
    constexpr std::uint32_t paddedRowsPerImage = 6;
    constexpr rachis::Extent3D paddedExtent{4, 4, 2};
    constexpr std::uint64_t copyHalfSize = paddedImageOffset + 2 * copySlotSize;

    [[nodiscard]] constexpr std::byte blockCopyByte(std::size_t format, std::uint64_t offset) noexcept
    {
        return static_cast<std::byte>((offset * 13U + (offset >> 8U) * 101U + format * 37U + 7U) & 0xFFU);
    }

    [[nodiscard]] bool rowsMatch(const std::byte* actual,
                                 std::size_t format,
                                 std::uint64_t offset,
                                 std::uint64_t rowCount,
                                 std::uint64_t rowPitch,
                                 std::uint64_t rowDataSize)
    {
        for (std::uint64_t row = 0; row < rowCount; ++row)
        {
            for (std::uint64_t byte = 0; byte < rowDataSize; ++byte)
            {
                const auto at = offset + row * rowPitch + byte;
                if (actual[at] != blockCopyByte(format, at % copyHalfSize))
                {
                    return false;
                }
            }
        }
        return true;
    }

    [[nodiscard]] rachis::TextureBarrier blockCopyBarrier(rachis::TextureHandle texture,
                                                          rachis::TextureLayout before,
                                                          rachis::TextureLayout after)
    {
        const bool fromUndefined = before == rachis::TextureLayout::undefined;
        return rachis::TextureBarrier{
            .texture = texture,
            .stagesBefore = fromUndefined ? rachis::PipelineStage::all : rachis::PipelineStage::copy,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = fromUndefined ? rachis::Access::none : rachis::Access::copyDestination,
            .accessAfter = after == rachis::TextureLayout::copySource ? rachis::Access::copySource
                                                                      : rachis::Access::copyDestination,
            .layoutBefore = before,
            .layoutAfter = after,
        };
    }

    void printDiagnostic(const rachis::DiagnosticMessage& message, void*) noexcept
    {
        static_cast<void>(std::fprintf(stderr, "D3D12 [%u]: ", static_cast<unsigned int>(message.nativeId)));
        if (!message.description.empty())
        {
            static_cast<void>(std::fwrite(message.description.data(), 1, message.description.size(), stderr));
        }
        static_cast<void>(std::fputc('\n', stderr));
    }

    [[nodiscard]] std::vector<std::byte> loadShader()
    {
        std::ifstream stream(RACHIS_TEST_BLOCK_COMPRESSED_TEXTURE_SHADER_PATH, std::ios::binary | std::ios::ate);
        const auto end = stream.tellg();
        if (end <= 0)
        {
            return {};
        }
        std::vector<std::byte> result(static_cast<std::size_t>(end));
        stream.seekg(0, std::ios::beg);
        stream.read(reinterpret_cast<char*>(result.data()), end);
        return stream ? result : std::vector<std::byte>{};
    }

} // namespace

TEST_CASE("the complete desktop BC family preserves block transfers and sampled values",
          "[size:medium][rhi][texture][format][compressed][compute][warp]")
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

    const std::array bc2ViewFormats{rachis::Format::bc2RgbaUnormSrgb};
    auto mutableBc2 = device.createTexture({
        .width = blockWidth,
        .height = blockHeight,
        .format = rachis::Format::bc2RgbaUnorm,
        .usage = rachis::TextureUsage::sampled,
        .viewFormats = bc2ViewFormats,
        .label = "Mutable BC2 texture",
    });
    REQUIRE(mutableBc2.has_value());
    auto srgbView = device.createTextureView({
        .texture = *mutableBc2,
        .usage = rachis::TextureViewUsage::sampled,
        .format = rachis::Format::bc2RgbaUnormSrgb,
        .label = "BC2 sRGB view",
    });
    REQUIRE(srgbView.has_value());
    REQUIRE(device.destroyTextureView(*srgbView).has_value());
    REQUIRE(device.destroyTexture(*mutableBc2).has_value());

    auto invalidStorage = device.createTexture({
        .width = blockWidth,
        .height = blockHeight,
        .format = rachis::Format::bc6hRgbUfloat,
        .usage = rachis::TextureUsage::storage,
    });
    CHECK_FALSE(invalidStorage.has_value());

    const auto bytecode = loadShader();
    REQUIRE_FALSE(bytecode.empty());
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Block-compressed texture shader",
        .shaderInterface = rachis::test::computeShaderInterface(
            {1, 1, 1},
            std::array{
                rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::sampledTexture, 1},
                rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::sampledTexture, 1},
                rachis::ShaderBindingRequirement{0, 2, rachis::ShaderResourceType::sampledTexture, 1},
                rachis::ShaderBindingRequirement{0, 3, rachis::ShaderResourceType::sampledTexture, 1},
                rachis::ShaderBindingRequirement{0, 4, rachis::ShaderResourceType::sampledTexture, 1},
                rachis::ShaderBindingRequirement{0, 5, rachis::ShaderResourceType::sampledTexture, 1},
                rachis::ShaderBindingRequirement{0, 6, rachis::ShaderResourceType::sampledTexture, 1},
                rachis::ShaderBindingRequirement{0, 7, rachis::ShaderResourceType::storageBuffer, 1},
            }),
    });
    REQUIRE(shader.has_value());

    constexpr std::uint64_t uploadSize = transferAlignment * blockCompressedFormats.size();
    constexpr std::uint64_t outputSize = outputStride * blockCompressedFormats.size();
    auto upload = device.createBuffer({
        .size = uploadSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Block-compressed upload",
    });
    auto output = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .label = "Block-compressed sampled output",
    });
    auto readback = device.createBuffer({
        .size = outputOffset + outputSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Block-compressed readback",
    });
    REQUIRE(upload.has_value());
    REQUIRE(output.has_value());
    REQUIRE(readback.has_value());

    std::array<rachis::TextureHandle, blockCompressedFormats.size()> textures{};
    std::array<rachis::TextureViewHandle, blockCompressedFormats.size()> views{};
    std::array<rachis::TextureTransferLayout, blockCompressedFormats.size()> transferLayouts{};
    for (std::size_t index = 0; index < blockCompressedFormats.size(); ++index)
    {
        const auto& formatCase = blockCompressedFormats[index];
        CAPTURE(index, formatCase.label);
        const auto properties = device.queryFormatProperties(formatCase.format);
        CHECK(rachis::hasAll(properties.features,
                             rachis::FormatFeature::texture2D | rachis::FormatFeature::sampled |
                                 rachis::FormatFeature::filterable | rachis::FormatFeature::copySource |
                                 rachis::FormatFeature::copyDestination));
        CHECK_FALSE(rachis::hasAny(properties.features,
                                   rachis::FormatFeature::storageRead | rachis::FormatFeature::storageWrite |
                                       rachis::FormatFeature::colorAttachment |
                                       rachis::FormatFeature::depthStencilAttachment));
        if (formatCase.format == rachis::Format::bc2RgbaUnorm || formatCase.format == rachis::Format::bc2RgbaUnormSrgb)
        {
            CHECK(rachis::hasAny(properties.features, rachis::FormatFeature::viewFormatReinterpretation));
        }

        auto texture = device.createTexture({
            .width = blockWidth,
            .height = blockHeight,
            .format = formatCase.format,
            .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                     rachis::TextureUsage::sampled,
            .label = formatCase.label,
        });
        REQUIRE(texture.has_value());
        textures[index] = *texture;
        auto view = device.createTextureView({
            .texture = *texture,
            .usage = rachis::TextureViewUsage::sampled,
            .label = formatCase.label,
        });
        REQUIRE(view.has_value());
        views[index] = *view;

        const auto bufferOffset = transferAlignment * index;
        auto layout =
            device.queryTextureTransferLayout(*texture,
                                              {
                                                  .bufferOffset = bufferOffset,
                                                  .textureSubresource = {.aspect = rachis::TextureAspect::color},
                                                  .extent = {blockWidth, blockHeight, 1},
                                              });
        REQUIRE(layout.has_value());
        CHECK(layout->rowDataSize == formatCase.bytesPerBlock);
        CHECK(layout->rowCount == 1);
        CHECK(layout->requiredBufferSize == bufferOffset + formatCase.bytesPerBlock);
        transferLayouts[index] = *layout;
        REQUIRE(device.writeBuffer(*upload, bufferOffset, std::span{formatCase.block}.first(formatCase.bytesPerBlock))
                    .has_value());
    }

    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{1, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{2, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{3, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{4, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{5, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{6, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{7, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout =
        device.createBindingLayout({.entries = bindingEntries, .label = "Block-compressed binding layout"});
    REQUIRE(bindingLayout.has_value());
    const std::array writes{
        rachis::BindingWrite{.binding = 0, .resource = views[0]},
        rachis::BindingWrite{.binding = 1, .resource = views[1]},
        rachis::BindingWrite{.binding = 2, .resource = views[2]},
        rachis::BindingWrite{.binding = 3, .resource = views[3]},
        rachis::BindingWrite{.binding = 4, .resource = views[4]},
        rachis::BindingWrite{.binding = 5, .resource = views[5]},
        rachis::BindingWrite{.binding = 6, .resource = views[6]},
        rachis::BindingWrite{
            .binding = 7,
            .resource =
                rachis::BufferBinding{
                    .buffer = *output,
                    .size = outputSize,
                    .structureByteStride = outputStride,
                },
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device,
                                                     {
                                                         .layout = *bindingLayout,
                                                         .writes = writes,
                                                         .label = "Block-compressed binding set",
                                                     });
    REQUIRE(bindingSet.has_value());
    const std::array bindingLayouts{*bindingLayout};
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .label = "Block-compressed pipeline layout",
    });
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Block-compressed sampling pipeline",
    });
    REQUIRE(pipeline.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    for (std::size_t index = 0; index < textures.size(); ++index)
    {
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = textures[index],
                                        .stagesBefore = rachis::PipelineStage::all,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::none,
                                        .accessAfter = rachis::Access::copyDestination,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::copyDestination,
                                    })
                    .has_value());
        const auto& layout = transferLayouts[index];
        REQUIRE(device
                    .copyBufferToTexture(*command,
                                         {
                                             .buffer = *upload,
                                             .bufferOffset = transferAlignment * index,
                                             .bytesPerRow = layout.bytesPerRow,
                                             .rowsPerImage = layout.rowsPerImage,
                                             .texture = textures[index],
                                             .extent = {blockWidth, blockHeight, 1},
                                         })
                    .has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = textures[index],
                                        .stagesBefore = rachis::PipelineStage::copy,
                                        .stagesAfter = rachis::PipelineStage::computeShader,
                                        .accessBefore = rachis::Access::copyDestination,
                                        .accessAfter = rachis::Access::shaderResource,
                                        .layoutBefore = rachis::TextureLayout::copyDestination,
                                        .layoutAfter = rachis::TextureLayout::shaderResource,
                                    })
                    .has_value());
    }
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *output,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *output,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(
        device
            .copyBuffer(
                *command,
                {.source = *output, .destination = *readback, .destinationOffset = outputOffset, .size = outputSize})
            .has_value());
    for (std::size_t index = 0; index < textures.size(); ++index)
    {
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = textures[index],
                                        .stagesBefore = rachis::PipelineStage::computeShader,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::shaderResource,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::shaderResource,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                    })
                    .has_value());
        const auto& layout = transferLayouts[index];
        REQUIRE(device
                    .copyTextureToBuffer(*command,
                                         {
                                             .buffer = *readback,
                                             .bufferOffset = transferAlignment * index,
                                             .bytesPerRow = layout.bytesPerRow,
                                             .rowsPerImage = layout.rowsPerImage,
                                             .texture = textures[index],
                                             .extent = {blockWidth, blockHeight, 1},
                                         })
                    .has_value());
    }

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    for (std::size_t index = 0; index < blockCompressedFormats.size(); ++index)
    {
        const auto& formatCase = blockCompressedFormats[index];
        CAPTURE(index, formatCase.label);
        CHECK(std::memcmp(
                  mapped->data + transferAlignment * index, formatCase.block.data(), formatCase.bytesPerBlock) == 0);
        std::array<float, 4> actual{};
        std::memcpy(actual.data(), mapped->data + outputOffset + outputStride * index, sizeof(actual));
        for (std::size_t component = 0; component < actual.size(); ++component)
        {
            CAPTURE(component);
            CHECK(actual[component] == Catch::Approx(formatCase.expected[component]).margin(0.0001F));
        }
    }
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    for (std::size_t index = 0; index < textures.size(); ++index)
    {
        REQUIRE(device.destroyTextureView(views[index]).has_value());
        REQUIRE(device.destroyTexture(textures[index]).has_value());
    }
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*output).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("block-compressed copies preserve whole mip chains with odd edges and padded images",
          "[size:medium][rhi][texture][format][compressed][transfer][warp]")
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

    constexpr auto copyUsage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination;
    std::vector<std::byte> uploadBytes(static_cast<std::size_t>(copyHalfSize));
    for (std::size_t formatIndex = 0; formatIndex < blockCopyFormats.size(); ++formatIndex)
    {
        const auto& formatCase = blockCopyFormats[formatIndex];
        CAPTURE(formatCase.label);
        CHECK(rachis::hasAll(device.queryFormatProperties(formatCase.format).features,
                             rachis::FormatFeature::texture2D | rachis::FormatFeature::texture3D |
                                 rachis::FormatFeature::copySource | rachis::FormatFeature::copyDestination));

        std::array<rachis::TextureHandle, blockCopyTextures.size()> sources{};
        std::array<rachis::TextureHandle, blockCopyTextures.size()> copies{};
        for (std::size_t textureIndex = 0; textureIndex < blockCopyTextures.size(); ++textureIndex)
        {
            const auto& desc = blockCopyTextures[textureIndex];
            auto source = device.createTexture({
                .width = desc.width,
                .height = desc.height,
                .mipLevels = desc.mipLevels,
                .format = formatCase.format,
                .usage = copyUsage,
                .label = "Block-copy source",
            });
            auto copy = device.createTexture({
                .width = desc.width,
                .height = desc.height,
                .mipLevels = desc.mipLevels,
                .format = formatCase.format,
                .usage = copyUsage,
                .label = "Block-copy destination",
            });
            REQUIRE(source.has_value());
            REQUIRE(copy.has_value());
            sources[textureIndex] = *source;
            copies[textureIndex] = *copy;
        }
        auto padded = device.createTexture({
            .dimension = rachis::TextureDimension::texture3D,
            .width = paddedExtent.width,
            .height = paddedExtent.height,
            .depthOrArrayLayers = paddedExtent.depth,
            .format = formatCase.format,
            .usage = copyUsage,
            .label = "Padded block-copy volume",
        });
        REQUIRE(padded.has_value());

        auto upload = device.createBuffer({
            .size = copyHalfSize,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
            .label = "Block-copy upload",
        });
        auto readback = device.createBuffer({
            .size = 2 * copyHalfSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Block-copy readback",
        });
        REQUIRE(upload.has_value());
        REQUIRE(readback.has_value());
        for (std::size_t offset = 0; offset < uploadBytes.size(); ++offset)
        {
            uploadBytes[offset] = blockCopyByte(formatIndex, offset);
        }
        REQUIRE(device.writeBuffer(*upload, 0, uploadBytes).has_value());

        std::array<std::vector<rachis::BufferTextureCopyRegion>, blockCopyTextures.size()> uploadRegions;
        std::array<std::vector<rachis::BufferTextureCopyRegion>, blockCopyTextures.size()> sourceReadbackRegions;
        std::array<std::vector<rachis::BufferTextureCopyRegion>, blockCopyTextures.size()> copyReadbackRegions;
        std::array<std::vector<rachis::TextureCopyRegion>, blockCopyTextures.size()> textureCopyRegions;
        for (std::size_t mipIndex = 0; mipIndex < blockCopyMips.size(); ++mipIndex)
        {
            const auto& mip = blockCopyMips[mipIndex];
            CAPTURE(mip.texture, mip.mipLevel);
            const auto offset = copySlotSize * mipIndex;
            const rachis::BufferTextureCopyRegion request{
                .bufferOffset = offset,
                .textureSubresource = {.mipLevel = mip.mipLevel},
                .extent = mip.extent,
            };
            auto layout = device.queryTextureTransferLayout(sources[mip.texture], request);
            REQUIRE(layout.has_value());
            const std::uint64_t rowDataSize = std::uint64_t{mip.blocksWide} * formatCase.bytesPerBlock;
            CHECK(layout->bytesPerRow == copyRowPitch);
            CHECK(layout->rowsPerImage == mip.extent.height);
            CHECK(layout->rowCount == mip.blockRows);
            CHECK(layout->rowDataSize == rowDataSize);
            CHECK(layout->bytesPerImage == copyRowPitch * mip.blockRows);
            CHECK(layout->requiredBufferSize == offset + copyRowPitch * (mip.blockRows - 1U) + rowDataSize);

            auto explicitUpload = request;
            explicitUpload.bytesPerRow = layout->bytesPerRow;
            explicitUpload.rowsPerImage = layout->rowsPerImage;
            uploadRegions[mip.texture].push_back(explicitUpload);
            sourceReadbackRegions[mip.texture].push_back(request);

            auto paddedReadback = request;
            paddedReadback.bufferOffset += copyHalfSize;
            paddedReadback.bytesPerRow = static_cast<std::uint32_t>(copyRowPitch);
            paddedReadback.rowsPerImage = mip.extent.height + 2U;
            auto paddedReadbackLayout = device.queryTextureTransferLayout(copies[mip.texture], paddedReadback);
            REQUIRE(paddedReadbackLayout.has_value());
            CHECK(paddedReadbackLayout->rowsPerImage == mip.extent.height + 2U);
            CHECK(paddedReadbackLayout->bytesPerImage ==
                  copyRowPitch * ((mip.extent.height + 2U + blockHeight - 1U) / blockHeight));
            copyReadbackRegions[mip.texture].push_back(paddedReadback);

            textureCopyRegions[mip.texture].push_back(rachis::TextureCopyRegion{
                .sourceSubresource = {.mipLevel = mip.mipLevel},
                .destinationSubresource = {.mipLevel = mip.mipLevel},
                .extent = mip.extent,
            });
        }

        const rachis::BufferTextureCopyRegion paddedRegion{
            .bufferOffset = paddedImageOffset,
            .rowsPerImage = paddedRowsPerImage,
            .extent = paddedExtent,
        };
        auto paddedLayout = device.queryTextureTransferLayout(*padded, paddedRegion);
        REQUIRE(paddedLayout.has_value());
        CHECK(paddedLayout->bytesPerRow == copyRowPitch);
        CHECK(paddedLayout->rowsPerImage == paddedRowsPerImage);
        CHECK(paddedLayout->rowCount == 1);
        CHECK(paddedLayout->rowDataSize == formatCase.bytesPerBlock);
        CHECK(paddedLayout->bytesPerImage == 2 * copyRowPitch);
        CHECK(paddedLayout->requiredBufferSize == paddedImageOffset + 2 * copyRowPitch + formatCase.bytesPerBlock);

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        for (std::size_t textureIndex = 0; textureIndex < blockCopyTextures.size(); ++textureIndex)
        {
            const auto source = sources[textureIndex];
            const auto copy = copies[textureIndex];
            REQUIRE(device
                        .textureBarrier(*command,
                                        blockCopyBarrier(source,
                                                         rachis::TextureLayout::undefined,
                                                         rachis::TextureLayout::copyDestination))
                        .has_value());
            REQUIRE(device
                        .textureBarrier(*command,
                                        blockCopyBarrier(copy,
                                                         rachis::TextureLayout::undefined,
                                                         rachis::TextureLayout::copyDestination))
                        .has_value());
            REQUIRE(device.copyBufferToTexture(*command, *upload, source, uploadRegions[textureIndex]).has_value());
            REQUIRE(device
                        .textureBarrier(*command,
                                        blockCopyBarrier(source,
                                                         rachis::TextureLayout::copyDestination,
                                                         rachis::TextureLayout::copySource))
                        .has_value());
            REQUIRE(device.copyTexture(*command, source, copy, textureCopyRegions[textureIndex]).has_value());
            REQUIRE(device
                        .textureBarrier(*command,
                                        blockCopyBarrier(copy,
                                                         rachis::TextureLayout::copyDestination,
                                                         rachis::TextureLayout::copySource))
                        .has_value());
            REQUIRE(device.copyTextureToBuffer(*command, source, *readback, sourceReadbackRegions[textureIndex])
                        .has_value());
            REQUIRE(
                device.copyTextureToBuffer(*command, copy, *readback, copyReadbackRegions[textureIndex]).has_value());
        }
        REQUIRE(
            device
                .textureBarrier(
                    *command,
                    blockCopyBarrier(*padded, rachis::TextureLayout::undefined, rachis::TextureLayout::copyDestination))
                .has_value());
        REQUIRE(device.copyBufferToTexture(*command, *upload, *padded, std::span{&paddedRegion, 1}).has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    blockCopyBarrier(*padded,
                                                     rachis::TextureLayout::copyDestination,
                                                     rachis::TextureLayout::copySource))
                    .has_value());
        REQUIRE(device.copyTextureToBuffer(*command, *padded, *readback, std::span{&paddedRegion, 1}).has_value());
        auto submitted = device.submit(std::move(*command));
        if (!submitted)
        {
            INFO(submitted.error().message);
        }
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());

        std::vector<std::byte> actual(static_cast<std::size_t>(2 * copyHalfSize));
        REQUIRE(device.readBuffer(*readback, 0, actual).has_value());
        for (std::size_t mipIndex = 0; mipIndex < blockCopyMips.size(); ++mipIndex)
        {
            const auto& mip = blockCopyMips[mipIndex];
            CAPTURE(mip.texture, mip.mipLevel);
            const std::uint64_t rowDataSize = std::uint64_t{mip.blocksWide} * formatCase.bytesPerBlock;
            const auto offset = copySlotSize * mipIndex;
            CHECK(rowsMatch(actual.data(), formatIndex, offset, mip.blockRows, copyRowPitch, rowDataSize));
            CHECK(
                rowsMatch(actual.data(), formatIndex, copyHalfSize + offset, mip.blockRows, copyRowPitch, rowDataSize));
        }
        CHECK(rowsMatch(actual.data(),
                        formatIndex,
                        paddedImageOffset,
                        paddedExtent.depth,
                        paddedLayout->bytesPerImage,
                        formatCase.bytesPerBlock));

        REQUIRE(device.destroyTexture(*padded).has_value());
        for (std::size_t textureIndex = 0; textureIndex < blockCopyTextures.size(); ++textureIndex)
        {
            REQUIRE(device.destroyTexture(sources[textureIndex]).has_value());
            REQUIRE(device.destroyTexture(copies[textureIndex]).has_value());
        }
        REQUIRE(device.destroyBuffer(*upload).has_value());
        REQUIRE(device.destroyBuffer(*readback).has_value());
    }

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("block-compressed copies accept the linear and sRGB formats of one family",
          "[size:medium][rhi][texture][format][compressed][transfer][warp]")
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

    constexpr auto copyUsage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination;
    constexpr rachis::Extent3D blockExtent{blockWidth, blockHeight, 1};
    std::vector<std::byte> uploadBytes(static_cast<std::size_t>(copySlotSize));
    for (std::size_t familyIndex = 0; familyIndex < blockCopyFamilies.size(); ++familyIndex)
    {
        const auto& family = blockCopyFamilies[familyIndex];
        CAPTURE(family.label);
        auto linear = device.createTexture({
            .width = blockWidth,
            .height = blockHeight,
            .format = family.linear,
            .usage = copyUsage,
            .label = "Linear block-copy source",
        });
        auto srgb = device.createTexture({
            .width = blockWidth,
            .height = blockHeight,
            .format = family.srgb,
            .usage = copyUsage,
            .label = "sRGB block-copy destination",
        });
        auto upload = device.createBuffer({
            .size = copySlotSize,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
            .label = "Block-family copy upload",
        });
        auto readback = device.createBuffer({
            .size = 2 * copySlotSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Block-family copy readback",
        });
        REQUIRE(linear.has_value());
        REQUIRE(srgb.has_value());
        REQUIRE(upload.has_value());
        REQUIRE(readback.has_value());
        for (std::size_t offset = 0; offset < uploadBytes.size(); ++offset)
        {
            uploadBytes[offset] = blockCopyByte(familyIndex, offset);
        }
        REQUIRE(device.writeBuffer(*upload, 0, uploadBytes).has_value());

        const rachis::BufferTextureCopyRegion linearRegion{.extent = blockExtent};
        const rachis::BufferTextureCopyRegion srgbRegion{.bufferOffset = copySlotSize, .extent = blockExtent};
        const rachis::TextureCopyRegion familyCopy{.extent = blockExtent};
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        for (const auto texture : {*linear, *srgb})
        {
            REQUIRE(device
                        .textureBarrier(*command,
                                        blockCopyBarrier(texture,
                                                         rachis::TextureLayout::undefined,
                                                         rachis::TextureLayout::copyDestination))
                        .has_value());
        }
        REQUIRE(device.copyBufferToTexture(*command, *upload, *linear, std::span{&linearRegion, 1}).has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    blockCopyBarrier(*linear,
                                                     rachis::TextureLayout::copyDestination,
                                                     rachis::TextureLayout::copySource))
                    .has_value());
        REQUIRE(device.copyTexture(*command, *linear, *srgb, std::span{&familyCopy, 1}).has_value());
        REQUIRE(
            device
                .textureBarrier(
                    *command,
                    blockCopyBarrier(*srgb, rachis::TextureLayout::copyDestination, rachis::TextureLayout::copySource))
                .has_value());
        REQUIRE(device.copyTextureToBuffer(*command, *linear, *readback, std::span{&linearRegion, 1}).has_value());
        REQUIRE(device.copyTextureToBuffer(*command, *srgb, *readback, std::span{&srgbRegion, 1}).has_value());
        auto submitted = device.submit(std::move(*command));
        if (!submitted)
        {
            INFO(submitted.error().message);
        }
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());

        std::vector<std::byte> actual(static_cast<std::size_t>(2 * copySlotSize));
        REQUIRE(device.readBuffer(*readback, 0, actual).has_value());
        const auto expected = std::span{uploadBytes}.first(family.bytesPerBlock);
        CHECK(std::ranges::equal(std::span{actual}.first(family.bytesPerBlock), expected));
        CHECK(std::ranges::equal(std::span{actual}.subspan(copySlotSize, family.bytesPerBlock), expected));

        REQUIRE(device.destroyTexture(*linear).has_value());
        REQUIRE(device.destroyTexture(*srgb).has_value());
        REQUIRE(device.destroyBuffer(*upload).has_value());
        REQUIRE(device.destroyBuffer(*readback).has_value());
    }

    auto bc2 = device.createTexture({
        .width = blockWidth,
        .height = blockHeight,
        .format = rachis::Format::bc2RgbaUnorm,
        .usage = copyUsage,
        .label = "BC2 cross-family copy source",
    });
    auto bc3Srgb = device.createTexture({
        .width = blockWidth,
        .height = blockHeight,
        .format = rachis::Format::bc3RgbaUnormSrgb,
        .usage = copyUsage,
        .label = "BC3 sRGB cross-family copy destination",
    });
    REQUIRE(bc2.has_value());
    REQUIRE(bc3Srgb.has_value());
    {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        const rachis::TextureCopyRegion crossFamilyCopy{.extent = blockExtent};
        const auto rejected = device.copyTexture(*command, *bc2, *bc3Srgb, std::span{&crossFamilyCopy, 1});
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
    }
    REQUIRE(device.destroyTexture(*bc2).has_value());
    REQUIRE(device.destroyTexture(*bc3Srgb).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
