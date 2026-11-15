#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace
{

    struct IntegerFormatCase final
    {
        rachis::Format format = rachis::Format::unknown;
        std::uint32_t bytesPerTexel = 0;
    };

    constexpr std::array integerFormats{
        IntegerFormatCase{rachis::Format::r8Uint, 1},
        IntegerFormatCase{rachis::Format::rg8Uint, 2},
        IntegerFormatCase{rachis::Format::rgba8Uint, 4},
        IntegerFormatCase{rachis::Format::r8Sint, 1},
        IntegerFormatCase{rachis::Format::rg8Sint, 2},
        IntegerFormatCase{rachis::Format::rgba8Sint, 4},
        IntegerFormatCase{rachis::Format::r16Uint, 2},
        IntegerFormatCase{rachis::Format::rg16Uint, 4},
        IntegerFormatCase{rachis::Format::rgba16Uint, 8},
        IntegerFormatCase{rachis::Format::r16Sint, 2},
        IntegerFormatCase{rachis::Format::rg16Sint, 4},
        IntegerFormatCase{rachis::Format::rgba16Sint, 8},
        IntegerFormatCase{rachis::Format::r32Uint, 4},
        IntegerFormatCase{rachis::Format::rg32Uint, 8},
        IntegerFormatCase{rachis::Format::rgba32Uint, 16},
        IntegerFormatCase{rachis::Format::r32Sint, 4},
        IntegerFormatCase{rachis::Format::rg32Sint, 8},
        IntegerFormatCase{rachis::Format::rgba32Sint, 16},
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

    [[nodiscard]] std::vector<std::byte> loadShader(const char* path)
    {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
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

    struct SampledDimensionCase final
    {
        rachis::FormatFeature feature = rachis::FormatFeature::none;
        rachis::TextureDimension dimension = rachis::TextureDimension::texture2D;
        rachis::TextureViewDimension viewDimension = rachis::TextureViewDimension::texture2D;
        std::uint32_t height = 1;
        std::uint32_t depthOrArrayLayers = 1;
    };

    constexpr std::array sampledDimensions{
        SampledDimensionCase{rachis::FormatFeature::texture1D,
                             rachis::TextureDimension::texture1D,
                             rachis::TextureViewDimension::texture1D,
                             1,
                             1},
        SampledDimensionCase{rachis::FormatFeature::texture2D,
                             rachis::TextureDimension::texture2D,
                             rachis::TextureViewDimension::texture2D,
                             8,
                             1},
        SampledDimensionCase{rachis::FormatFeature::texture3D,
                             rachis::TextureDimension::texture3D,
                             rachis::TextureViewDimension::texture3D,
                             8,
                             4},
    };

    void requireSampledIntegerFormat(rachis::Device& device, rachis::Format format)
    {
        CAPTURE(static_cast<unsigned int>(format));
        const auto properties = device.queryFormatProperties(format);
        CHECK_FALSE(rachis::hasAny(properties.features, rachis::FormatFeature::filterable));
        if (!rachis::hasAll(properties.features, rachis::FormatFeature::sampled | rachis::FormatFeature::texture2D))
        {
            return;
        }
        for (const auto& dimensionCase : sampledDimensions)
        {
            if (!rachis::hasAny(properties.features, dimensionCase.feature))
            {
                continue;
            }
            CAPTURE(static_cast<unsigned int>(dimensionCase.dimension));
            auto texture = device.createTexture({
                .dimension = dimensionCase.dimension,
                .width = 8,
                .height = dimensionCase.height,
                .depthOrArrayLayers = dimensionCase.depthOrArrayLayers,
                .format = format,
                .usage = rachis::TextureUsage::sampled,
                .label = "Sampled typed integer texture",
            });
            INFO("createTexture: " << (texture ? std::string{} : texture.error().message));
            REQUIRE(texture.has_value());
            auto view = device.createTextureView({
                .texture = *texture,
                .usage = rachis::TextureViewUsage::sampled,
                .dimension = dimensionCase.viewDimension,
                .label = "Sampled typed integer view",
            });
            INFO("createTextureView: " << (view ? std::string{} : view.error().message));
            REQUIRE(view.has_value());
            REQUIRE(device.destroyTextureView(*view).has_value());
            REQUIRE(device.destroyTexture(*texture).has_value());
        }
    }

    constexpr std::uint32_t sampledLoadExtent = 4;
    constexpr std::uint32_t sampledLoadTexelCount = sampledLoadExtent * sampledLoadExtent;
    constexpr std::uint64_t sampledLoadOutputSize = sampledLoadTexelCount * sizeof(std::uint32_t);

    [[nodiscard]] constexpr std::uint32_t sampledLoadTexel(std::uint32_t index) noexcept
    {
        return 0x89ABCDEFU ^ (index * 0x01010101U);
    }

} // namespace

TEST_CASE("typed integer texture formats preserve width and signed shader writes",
          "[size:medium][rhi][texture][format][integer][compute][warp]")
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

    for (const auto& formatCase : integerFormats)
    {
        const auto properties = device.queryFormatProperties(formatCase.format);
        CHECK(rachis::hasAll(properties.features,
                             rachis::FormatFeature::texture2D | rachis::FormatFeature::copySource |
                                 rachis::FormatFeature::copyDestination));
        CHECK_FALSE(rachis::hasAny(properties.features, rachis::FormatFeature::filterable));
        CHECK_FALSE(rachis::hasAny(properties.features, rachis::FormatFeature::blendable));

        auto texture = device.createTexture({
            .format = formatCase.format,
            .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
            .label = "Typed integer layout probe",
        });
        REQUIRE(texture.has_value());
        auto layout =
            device.queryTextureTransferLayout(*texture,
                                              {
                                                  .textureSubresource = {.aspect = rachis::TextureAspect::color},
                                                  .extent = {1, 1, 1},
                                              });
        REQUIRE(layout.has_value());
        CHECK(layout->rowDataSize == formatCase.bytesPerTexel);
        CHECK(layout->requiredBufferSize == formatCase.bytesPerTexel);
        REQUIRE(device.destroyTexture(*texture).has_value());
    }

    const auto bytecode = loadShader(RACHIS_TEST_TYPED_INTEGER_TEXTURE_SHADER_PATH);
    REQUIRE_FALSE(bytecode.empty());
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Typed integer texture shader",
        .shaderInterface = rachis::test::computeShaderInterface(
            {1, 1, 1},
            std::array{
                rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageTexture, 1},
                rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageTexture, 1},
            }),
    });
    REQUIRE(shader.has_value());

    auto unsignedTexture = device.createTexture({
        .format = rachis::Format::rgba8Uint,
        .usage = rachis::TextureUsage::storage | rachis::TextureUsage::copySource,
        .label = "Unsigned typed integer output",
    });
    auto signedTexture = device.createTexture({
        .format = rachis::Format::rgba8Sint,
        .usage = rachis::TextureUsage::storage | rachis::TextureUsage::copySource,
        .label = "Signed typed integer output",
    });
    REQUIRE(unsignedTexture.has_value());
    REQUIRE(signedTexture.has_value());
    auto unsignedView = device.createTextureView({
        .texture = *unsignedTexture,
        .usage = rachis::TextureViewUsage::storage,
        .label = "Unsigned typed integer UAV",
    });
    auto signedView = device.createTextureView({
        .texture = *signedTexture,
        .usage = rachis::TextureViewUsage::storage,
        .label = "Signed typed integer UAV",
    });
    REQUIRE(unsignedView.has_value());
    REQUIRE(signedView.has_value());

    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageTexture, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageTexture, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = bindingEntries, .label = "Typed integer layout"});
    REQUIRE(bindingLayout.has_value());
    const std::array writes{
        rachis::BindingWrite{.binding = 0, .resource = *unsignedView},
        rachis::BindingWrite{.binding = 1, .resource = *signedView},
    };
    auto bindingSet = rachis::test::createBindingSet(device,
                                                     {
                                                         .layout = *bindingLayout,
                                                         .writes = writes,
                                                         .label = "Typed integer bindings",
                                                     });
    REQUIRE(bindingSet.has_value());
    const std::array bindingLayouts{*bindingLayout};
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .label = "Typed integer pipeline layout",
    });
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Typed integer pipeline",
    });
    REQUIRE(pipeline.has_value());

    constexpr std::uint64_t signedOffset = 512;
    auto readback = device.createBuffer({
        .size = 1024,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Typed integer readback",
    });
    REQUIRE(readback.has_value());
    auto unsignedLayout = device.queryTextureTransferLayout(
        *unsignedTexture, {.textureSubresource = {.aspect = rachis::TextureAspect::color}, .extent = {1, 1, 1}});
    auto signedLayout =
        device.queryTextureTransferLayout(*signedTexture,
                                          {.bufferOffset = signedOffset,
                                           .textureSubresource = {.aspect = rachis::TextureAspect::color},
                                           .extent = {1, 1, 1}});
    REQUIRE(unsignedLayout.has_value());
    REQUIRE(signedLayout.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    for (const auto texture : {*unsignedTexture, *signedTexture})
    {
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = texture,
                                        .stagesBefore = rachis::PipelineStage::all,
                                        .stagesAfter = rachis::PipelineStage::computeShader,
                                        .accessBefore = rachis::Access::none,
                                        .accessAfter = rachis::Access::unorderedAccess,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::unorderedAccess,
                                    })
                    .has_value());
    }
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
    for (const auto texture : {*unsignedTexture, *signedTexture})
    {
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = texture,
                                        .stagesBefore = rachis::PipelineStage::computeShader,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::unorderedAccess,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::unorderedAccess,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                    })
                    .has_value());
    }
    const rachis::BufferTextureCopyRegion unsignedCopy{
        .bytesPerRow = unsignedLayout->bytesPerRow,
        .rowsPerImage = unsignedLayout->rowsPerImage,
        .textureSubresource = {.aspect = rachis::TextureAspect::color},
        .extent = {1, 1, 1},
    };
    const rachis::BufferTextureCopyRegion signedCopy{
        .bufferOffset = signedOffset,
        .bytesPerRow = signedLayout->bytesPerRow,
        .rowsPerImage = signedLayout->rowsPerImage,
        .textureSubresource = {.aspect = rachis::TextureAspect::color},
        .extent = {1, 1, 1},
    };
    REQUIRE(device.copyTextureToBuffer(*command, *unsignedTexture, *readback, std::span{&unsignedCopy, 1}).has_value());
    REQUIRE(device.copyTextureToBuffer(*command, *signedTexture, *readback, std::span{&signedCopy, 1}).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    constexpr std::array unsignedExpected{std::byte{17}, std::byte{34}, std::byte{51}, std::byte{255}};
    constexpr std::array signedExpected{std::byte{0xFF}, std::byte{0xFE}, std::byte{3}, std::byte{4}};
    CHECK(std::equal(unsignedExpected.begin(), unsignedExpected.end(), mapped->data));
    CHECK(std::equal(signedExpected.begin(), signedExpected.end(), mapped->data + signedOffset));
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyTextureView(*unsignedView).has_value());
    REQUIRE(device.destroyTextureView(*signedView).has_value());
    REQUIRE(device.destroyTexture(*unsignedTexture).has_value());
    REQUIRE(device.destroyTexture(*signedTexture).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("typed integer texture formats accept sampled usage and shader loads",
          "[size:medium][rhi][texture][format][integer][compute][warp]")
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

    for (const auto& formatCase : integerFormats)
    {
        requireSampledIntegerFormat(device, formatCase.format);
    }
    requireSampledIntegerFormat(device, rachis::Format::r64Uint);

    const auto bytecode = loadShader(RACHIS_TEST_TYPED_INTEGER_SAMPLED_LOAD_SHADER_PATH);
    REQUIRE_FALSE(bytecode.empty());
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Typed integer sampled load shader",
        .shaderInterface = rachis::test::computeShaderInterface(
            {sampledLoadExtent, sampledLoadExtent, 1},
            std::array{
                rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::sampledTexture, 1},
                rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
            }),
    });
    REQUIRE(shader.has_value());

    auto texture = device.createTexture({
        .width = sampledLoadExtent,
        .height = sampledLoadExtent,
        .format = rachis::Format::r32Uint,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::copyDestination,
        .label = "Typed integer sampled load source",
    });
    REQUIRE(texture.has_value());
    auto mipChain = device.createTexture({
        .width = sampledLoadExtent,
        .height = sampledLoadExtent,
        .mipLevels = 2,
        .format = rachis::Format::r32Uint,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
        .label = "Typed integer unfilterable mip chain",
    });
    REQUIRE(mipChain.has_value());
    auto sampledView = device.createTextureView({
        .texture = *texture,
        .usage = rachis::TextureViewUsage::sampled,
        .label = "Typed integer sampled load SRV",
    });
    REQUIRE(sampledView.has_value());

    const rachis::BufferTextureCopyRegion copyRegion{
        .textureSubresource = {.aspect = rachis::TextureAspect::color},
        .extent = {sampledLoadExtent, sampledLoadExtent, 1},
    };
    auto layout = device.queryTextureTransferLayout(*texture, copyRegion);
    REQUIRE(layout.has_value());
    std::vector<std::byte> uploadBytes(static_cast<std::size_t>(layout->requiredBufferSize), std::byte{0});
    for (std::uint32_t y = 0; y < sampledLoadExtent; ++y)
    {
        for (std::uint32_t x = 0; x < sampledLoadExtent; ++x)
        {
            const auto texel = sampledLoadTexel(y * sampledLoadExtent + x);
            std::memcpy(uploadBytes.data() + static_cast<std::size_t>(y) * layout->bytesPerRow +
                            static_cast<std::size_t>(x) * sizeof(texel),
                        &texel,
                        sizeof(texel));
        }
    }

    auto upload = device.createBuffer({
        .size = layout->requiredBufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Typed integer sampled load upload",
    });
    auto output = device.createBuffer({
        .size = sampledLoadOutputSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .label = "Typed integer sampled load output",
    });
    auto readback = device.createBuffer({
        .size = sampledLoadOutputSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Typed integer sampled load readback",
    });
    REQUIRE(upload.has_value());
    REQUIRE(output.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(device.writeBuffer(*upload, 0, uploadBytes).has_value());

    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout =
        device.createBindingLayout({.entries = bindingEntries, .label = "Typed integer sampled load layout"});
    REQUIRE(bindingLayout.has_value());
    const std::array writes{
        rachis::BindingWrite{.binding = 0, .resource = *sampledView},
        rachis::BindingWrite{
            .binding = 1,
            .resource =
                rachis::BufferBinding{
                    .buffer = *output,
                    .size = sampledLoadOutputSize,
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device,
                                                     {
                                                         .layout = *bindingLayout,
                                                         .writes = writes,
                                                         .label = "Typed integer sampled load bindings",
                                                     });
    REQUIRE(bindingSet.has_value());
    const std::array bindingLayouts{*bindingLayout};
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .label = "Typed integer sampled load pipeline layout",
    });
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Typed integer sampled load pipeline",
    });
    REQUIRE(pipeline.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    const auto rejectedMipmaps = device.generateMipmaps(*command,
                                                        rachis::TextureMipmapGeneration{
                                                            .texture = *mipChain,
                                                            .baseLayout = rachis::TextureLayout::shaderResource,
                                                            .finalLayout = rachis::TextureLayout::shaderResource,
                                                        });
    REQUIRE_FALSE(rejectedMipmaps.has_value());
    CHECK(rejectedMipmaps.error().code == rachis::ErrorCode::unsupported);

    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    REQUIRE(device
                .copyBufferToTexture(*command,
                                     rachis::BufferTextureCopy{
                                         .buffer = *upload,
                                         .bytesPerRow = layout->bytesPerRow,
                                         .rowsPerImage = layout->rowsPerImage,
                                         .texture = *texture,
                                         .extent = {sampledLoadExtent, sampledLoadExtent, 1},
                                     })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::computeShader,
                                    .accessBefore = rachis::Access::copyDestination,
                                    .accessAfter = rachis::Access::shaderResource,
                                    .layoutBefore = rachis::TextureLayout::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::shaderResource,
                                })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
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
                               rachis::BufferBarrier{
                                   .buffer = *output,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device.copyBuffer(*command, {.source = *output, .destination = *readback, .size = sampledLoadOutputSize})
                .has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    std::array<std::uint32_t, sampledLoadTexelCount> actual{};
    std::memcpy(actual.data(), mapped->data, sizeof(actual));
    for (std::uint32_t index = 0; index < sampledLoadTexelCount; ++index)
    {
        CAPTURE(index);
        CHECK(actual[index] == sampledLoadTexel(index));
    }
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyTextureView(*sampledView).has_value());
    REQUIRE(device.destroyTexture(*mipChain).has_value());
    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*output).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
