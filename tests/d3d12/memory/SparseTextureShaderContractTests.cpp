#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <span>
#include <vector>

namespace
{

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
        std::ifstream stream(RACHIS_TEST_SPARSE_TEXTURE_SHADER_PATH, std::ios::binary | std::ios::ate);
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

    [[nodiscard]] std::array<std::uint32_t, 4> readWords(rachis::Device& device, rachis::BufferHandle buffer)
    {
        auto mapped = device.mapBuffer(buffer);
        REQUIRE(mapped.has_value());
        REQUIRE(mapped->size >= 4 * sizeof(std::uint32_t));
        std::array<std::uint32_t, 4> result{};
        std::memcpy(result.data(), mapped->data, sizeof(result));
        REQUIRE(device.unmapBuffer(buffer).has_value());
        return result;
    }

    [[nodiscard]] std::uint32_t readTextureWord(rachis::Device& device, rachis::BufferHandle buffer)
    {
        auto mapped = device.mapBuffer(buffer);
        REQUIRE(mapped.has_value());
        REQUIRE(mapped->size >= sizeof(std::uint32_t));
        std::uint32_t result = 0;
        std::memcpy(&result, mapped->data, sizeof(result));
        REQUIRE(device.unmapBuffer(buffer).has_value());
        return result;
    }

} // namespace

TEST_CASE("shader-visible sparse texture views capture each command's mapping generation",
          "[size:medium][rhi][texture][sparse][binding][compute][lifetime][warp]")
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

    constexpr std::uint32_t textureExtent = 1024;

    constexpr float expectedValue = 0.0F;
    constexpr float nullWriteSentinel = 91.25F;
    constexpr std::uint32_t expectedBits = std::bit_cast<std::uint32_t>(expectedValue);
    constexpr std::uint32_t nullWriteSentinelBits = std::bit_cast<std::uint32_t>(nullWriteSentinel);
    constexpr std::uint64_t constantBufferSize = 256;
    constexpr std::uint64_t resultSize = 4 * sizeof(std::uint32_t);
    const rachis::TextureDesc inputDesc{
        .width = textureExtent,
        .height = textureExtent,
        .format = rachis::Format::r32Float,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::copyDestination,
        .label = "Shader-visible sparse input texture",
    };
    const rachis::TextureDesc outputDesc{
        .width = textureExtent,
        .height = textureExtent,
        .format = rachis::Format::r32Float,
        .usage =
            rachis::TextureUsage::storage | rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Shader-visible sparse output texture",
    };
    if (!rachis::test::requireCapabilities(device).sparseTextures)
    {
        const auto unsupported = device.createSparseTexture(inputDesc);
        REQUIRE_FALSE(unsupported.has_value());
        CHECK(unsupported.error().code == rachis::ErrorCode::unsupported);
        return;
    }

    const auto shaderBytecode = loadShader();
    REQUIRE_FALSE(shaderBytecode.empty());
    const auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = shaderBytecode,
        .entryPoint = "main",
        .label = "Sparse-texture mapping shader",
        .shaderInterface = rachis::test::computeShaderInterface(
            {1, 1, 1},
            std::array{
                rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1},
                rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::sampledTexture, 1},
                rachis::ShaderBindingRequirement{0, 2, rachis::ShaderResourceType::storageTexture, 1},
                rachis::ShaderBindingRequirement{0, 3, rachis::ShaderResourceType::storageBuffer, 1},
            }),
    });
    const auto input = device.createSparseTexture(inputDesc);
    const auto output = device.createSparseTexture(outputDesc);
    REQUIRE(shader.has_value());
    REQUIRE(input.has_value());
    REQUIRE(output.has_value());

    const auto inputProperties = device.querySparseTextureProperties(*input);
    const auto outputProperties = device.querySparseTextureProperties(*output);
    REQUIRE(inputProperties.has_value());
    REQUIRE(outputProperties.has_value());
    REQUIRE(inputProperties->standardMipCount == 1);
    REQUIRE(outputProperties->standardMipCount == 1);
    REQUIRE(inputProperties->standardMipTileGrids.front().width >= 2);
    REQUIRE(outputProperties->standardMipTileGrids.front().width >= 2);
    REQUIRE(inputProperties->standardTileExtent.width < textureExtent);
    CHECK(outputProperties->standardTileExtent.width == inputProperties->standardTileExtent.width);
    CHECK(outputProperties->tileSize == inputProperties->tileSize);

    const auto inputView = device.createTextureView({
        .texture = *input,
        .usage = rachis::TextureViewUsage::sampled,
        .label = "Reusable sparse input SRV",
    });
    const auto outputView = device.createTextureView({
        .texture = *output,
        .usage = rachis::TextureViewUsage::storage,
        .label = "Reusable sparse output UAV",
    });
    REQUIRE(inputView.has_value());
    REQUIRE(outputView.has_value());

    const std::array coordinates{
        0U,
        0U,
        inputProperties->standardTileExtent.width,
        0U,
    };
    const auto parameters = device.createBuffer({
        .size = constantBufferSize,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Sparse-texture shader coordinates",
    });
    const auto results = device.createBuffer({
        .size = resultSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Sparse-texture observed values",
    });
    const auto resultReadback = device.createBuffer({
        .size = resultSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Sparse-texture observed readback",
    });
    REQUIRE(parameters.has_value());
    REQUIRE(results.has_value());
    REQUIRE(resultReadback.has_value());
    REQUIRE(device.writeBuffer(*parameters, 0, std::as_bytes(std::span{coordinates})).has_value());

    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{1, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{2, rachis::BindingType::storageTexture, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{3, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    const auto bindingLayout = device.createBindingLayout({
        .entries = bindingEntries,
        .label = "Sparse-texture shader binding layout",
    });
    REQUIRE(bindingLayout.has_value());
    const std::array bindingWrites{
        rachis::BindingWrite{
            .binding = 0,
            .resource = rachis::BufferBinding{.buffer = *parameters, .size = constantBufferSize},
        },
        rachis::BindingWrite{.binding = 1, .resource = *inputView},
        rachis::BindingWrite{.binding = 2, .resource = *outputView},
        rachis::BindingWrite{
            .binding = 3,
            .resource =
                rachis::BufferBinding{
                    .buffer = *results,
                    .size = resultSize,
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
    };
    const auto bindingSet = rachis::test::createBindingSet(device,
                                                           {
                                                               .layout = *bindingLayout,
                                                               .writes = bindingWrites,
                                                               .label = "Reusable sparse-texture binding set",
                                                           });
    REQUIRE(bindingSet.has_value());
    const std::array pipelineBindingLayouts{*bindingLayout};
    const auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = pipelineBindingLayouts,
        .label = "Sparse-texture pipeline layout",
    });
    REQUIRE(pipelineLayout.has_value());
    const auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Sparse-texture pipeline",
    });
    REQUIRE(pipeline.has_value());

    const auto inputHeap = device.createResourceHeap({
        .size = inputProperties->tileSize,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::nonAttachmentTexture,
        .flags = rachis::ResourceHeapFlag::sparseBacking,
        .label = "Sparse input backing tile",
    });
    const auto outputHeap = device.createResourceHeap({
        .size = 2 * outputProperties->tileSize,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::nonAttachmentTexture,
        .flags = rachis::ResourceHeapFlag::sparseBacking,
        .label = "Sparse output backing tile",
    });
    REQUIRE(inputHeap.has_value());
    REQUIRE(outputHeap.has_value());
    const std::array mapFirstTiles{
        rachis::SparseTextureTileMapping{
            .texture = *input,
            .heap = *inputHeap,
        },
        rachis::SparseTextureTileMapping{
            .texture = *output,
            .heap = *outputHeap,
        },
    };
    const rachis::SparseTextureTileMapping mapSecondOutputTile{
        .texture = *output,
        .textureTileOrigin = {.x = 1},
        .heap = *outputHeap,
        .heapTileOffset = 1,
    };
    const rachis::SparseTextureTileMapping unmapSecondOutputTile{
        .texture = *output,
        .textureTileOrigin = {.x = 1},
    };
    const std::array initialMappings{
        mapFirstTiles[0],
        mapFirstTiles[1],
        mapSecondOutputTile,
    };
    const std::array unmapFirstTiles{
        rachis::SparseTextureTileMapping{.texture = *input},
        rachis::SparseTextureTileMapping{.texture = *output},
    };
    const std::array unmapAllTiles{
        unmapFirstTiles[0],
        unmapFirstTiles[1],
        unmapSecondOutputTile,
    };
    const auto initialMapping =
        device.enqueueSparseTextureMappings(rachis::QueueType::graphics, {.standardTiles = initialMappings});
    REQUIRE(initialMapping.has_value());

    const rachis::BufferTextureCopyRegion inputPixelRequest{
        .extent = {1, 1, 1},
    };
    const auto inputPixelLayout = device.queryTextureTransferLayout(*input, inputPixelRequest);
    const auto outputPixelLayout = device.queryTextureTransferLayout(*output, inputPixelRequest);
    REQUIRE(inputPixelLayout.has_value());
    REQUIRE(outputPixelLayout.has_value());
    const auto inputUpload = device.createBuffer({
        .size = inputPixelLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Sparse input texel upload",
    });
    const auto sentinelUpload = device.createBuffer({
        .size = outputPixelLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Sparse output null-write sentinel upload",
    });
    const auto outputReadback = device.createBuffer({
        .size = outputPixelLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Sparse output texel readback",
    });
    REQUIRE(inputUpload.has_value());
    REQUIRE(sentinelUpload.has_value());
    REQUIRE(outputReadback.has_value());
    REQUIRE(device.writeBuffer(*inputUpload, 0, std::as_bytes(std::span{&expectedValue, 1})).has_value());
    REQUIRE(device.writeBuffer(*sentinelUpload, 0, std::as_bytes(std::span{&nullWriteSentinel, 1})).has_value());
    const rachis::BufferTextureCopyRegion inputPixelCopy{
        .bytesPerRow = inputPixelLayout->bytesPerRow,
        .rowsPerImage = inputPixelLayout->rowsPerImage,
        .extent = {1, 1, 1},
    };

    auto prepare = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(prepare.has_value());
    REQUIRE(device
                .textureBarrier(*prepare,
                                {
                                    .texture = *input,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    REQUIRE(device.copyBufferToTexture(*prepare, *inputUpload, *input, std::span{&inputPixelCopy, 1}).has_value());
    REQUIRE(device
                .textureBarrier(*prepare,
                                {
                                    .texture = *input,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::computeShader,
                                    .accessBefore = rachis::Access::copyDestination,
                                    .accessAfter = rachis::Access::shaderResource,
                                    .layoutBefore = rachis::TextureLayout::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::shaderResource,
                                })
                .has_value());
    REQUIRE(device
                .textureBarrier(*prepare,
                                {
                                    .texture = *output,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    const rachis::BufferTextureCopyRegion sentinelPixelCopy{
        .bytesPerRow = outputPixelLayout->bytesPerRow,
        .rowsPerImage = outputPixelLayout->rowsPerImage,
        .textureOrigin = {.x = inputProperties->standardTileExtent.width},
        .extent = {1, 1, 1},
    };
    REQUIRE(
        device.copyBufferToTexture(*prepare, *sentinelUpload, *output, std::span{&sentinelPixelCopy, 1}).has_value());
    REQUIRE(device
                .textureBarrier(*prepare,
                                {
                                    .texture = *output,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::computeShader,
                                    .accessBefore = rachis::Access::copyDestination,
                                    .accessAfter = rachis::Access::unorderedAccess,
                                    .layoutBefore = rachis::TextureLayout::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::unorderedAccess,
                                })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*prepare,
                               {
                                   .buffer = *results,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    const auto prepared = device.submit(std::move(*prepare));
    REQUIRE(prepared.has_value());
    REQUIRE(device.wait(*prepared).has_value());
    device.collectGarbage();

    const std::array secondOutputUnmap{unmapSecondOutputTile};
    const auto nullOutputMapping =
        device.enqueueSparseTextureMappings(rachis::QueueType::graphics, {.standardTiles = secondOutputUnmap});
    REQUIRE(nullOutputMapping.has_value());
    REQUIRE(device.wait(*nullOutputMapping).has_value());
    device.collectGarbage();

    const auto replacement =
        device.enqueueSparseTextureMappings(rachis::QueueType::graphics, {.standardTiles = mapFirstTiles});
    REQUIRE(replacement.has_value());
    const auto commandPool = device.createCommandPool({
        .queueType = rachis::QueueType::compute,
        .maxCommandBuffers = 1,
        .label = "Unretained sparse-texture shader pool",
        .flags = rachis::CommandPoolFlag::allowIndividualReset | rachis::CommandPoolFlag::unretainedResources,
    });
    REQUIRE(commandPool.has_value());

    auto mappedCommand = device.acquireCommandBuffer(*commandPool);
    REQUIRE(mappedCommand.has_value());
    REQUIRE(device.bindComputePipeline(*mappedCommand, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*mappedCommand, 0, *bindingSet).has_value());
    const auto retainedReplacement =
        device.enqueueSparseTextureMappings(rachis::QueueType::graphics, {.standardTiles = unmapFirstTiles});
    REQUIRE_FALSE(retainedReplacement.has_value());
    CHECK(retainedReplacement.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.dispatch(*mappedCommand, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*mappedCommand,
                               {
                                   .buffer = *results,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device.copyBuffer(*mappedCommand, {*results, 0, *resultReadback, 0, resultSize}).has_value());
    const auto missingWait = device.submit(std::move(*mappedCommand));
    REQUIRE_FALSE(missingWait.has_value());
    CHECK(missingWait.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(mappedCommand->isValid());
    const std::array replacementWaits{*replacement};
    const auto mappedSubmission = device.submit(std::move(*mappedCommand), replacementWaits);
    REQUIRE(mappedSubmission.has_value());
    REQUIRE(device.wait(*mappedSubmission).has_value());
    device.collectGarbage();
    CHECK(readWords(device, *resultReadback) == std::array{expectedBits, 0U, 1U, 0U});

    auto copyOutput = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(copyOutput.has_value());
    REQUIRE(device
                .textureBarrier(*copyOutput,
                                {
                                    .texture = *output,
                                    .stagesBefore = rachis::PipelineStage::computeShader,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::unorderedAccess,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::unorderedAccess,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    const rachis::BufferTextureCopyRegion outputPixelCopy{
        .bytesPerRow = outputPixelLayout->bytesPerRow,
        .rowsPerImage = outputPixelLayout->rowsPerImage,
        .extent = {1, 1, 1},
    };
    REQUIRE(
        device.copyTextureToBuffer(*copyOutput, *output, *outputReadback, std::span{&outputPixelCopy, 1}).has_value());
    REQUIRE(device
                .textureBarrier(*copyOutput,
                                {
                                    .texture = *output,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::computeShader,
                                    .accessBefore = rachis::Access::copySource,
                                    .accessAfter = rachis::Access::unorderedAccess,
                                    .layoutBefore = rachis::TextureLayout::copySource,
                                    .layoutAfter = rachis::TextureLayout::unorderedAccess,
                                })
                .has_value());
    const auto outputCopied = device.submit(std::move(*copyOutput));
    REQUIRE(outputCopied.has_value());
    REQUIRE(device.wait(*outputCopied).has_value());
    device.collectGarbage();
    CHECK(readTextureWord(device, *outputReadback) == std::bit_cast<std::uint32_t>(expectedValue + 1.0F));

    const std::array secondOutputMap{mapSecondOutputTile};
    const auto restoredNullOutputMapping =
        device.enqueueSparseTextureMappings(rachis::QueueType::graphics, {.standardTiles = secondOutputMap});
    REQUIRE(restoredNullOutputMapping.has_value());
    REQUIRE(device.wait(*restoredNullOutputMapping).has_value());
    device.collectGarbage();

    auto copyNullOutput = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(copyNullOutput.has_value());
    REQUIRE(device
                .textureBarrier(*copyNullOutput,
                                {
                                    .texture = *output,
                                    .stagesBefore = rachis::PipelineStage::computeShader,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::unorderedAccess,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::unorderedAccess,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    const rachis::BufferTextureCopyRegion nullOutputPixelCopy{
        .bytesPerRow = outputPixelLayout->bytesPerRow,
        .rowsPerImage = outputPixelLayout->rowsPerImage,
        .textureOrigin = {.x = inputProperties->standardTileExtent.width},
        .extent = {1, 1, 1},
    };
    REQUIRE(device.copyTextureToBuffer(*copyNullOutput, *output, *outputReadback, std::span{&nullOutputPixelCopy, 1})
                .has_value());
    REQUIRE(device
                .textureBarrier(*copyNullOutput,
                                {
                                    .texture = *output,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::computeShader,
                                    .accessBefore = rachis::Access::copySource,
                                    .accessAfter = rachis::Access::unorderedAccess,
                                    .layoutBefore = rachis::TextureLayout::copySource,
                                    .layoutAfter = rachis::TextureLayout::unorderedAccess,
                                })
                .has_value());
    const auto nullOutputCopied = device.submit(std::move(*copyNullOutput));
    REQUIRE(nullOutputCopied.has_value());
    REQUIRE(device.wait(*nullOutputCopied).has_value());
    device.collectGarbage();
    CHECK(readTextureWord(device, *outputReadback) == nullWriteSentinelBits);

    const auto unmappedGeneration =
        device.enqueueSparseTextureMappings(rachis::QueueType::compute, {.standardTiles = unmapAllTiles});
    REQUIRE(unmappedGeneration.has_value());
    auto unmappedCommand = device.acquireCommandBuffer(*commandPool);
    REQUIRE(unmappedCommand.has_value());
    REQUIRE(device
                .bufferBarrier(*unmappedCommand,
                               {
                                   .buffer = *results,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::copySource,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*unmappedCommand, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*unmappedCommand, 0, *bindingSet).has_value());
    const auto retainedUnmappedGeneration =
        device.enqueueSparseTextureMappings(rachis::QueueType::graphics, {.standardTiles = mapFirstTiles});
    REQUIRE_FALSE(retainedUnmappedGeneration.has_value());
    CHECK(retainedUnmappedGeneration.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.dispatch(*unmappedCommand, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*unmappedCommand,
                               {
                                   .buffer = *results,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device.copyBuffer(*unmappedCommand, {*results, 0, *resultReadback, 0, resultSize}).has_value());
    const auto unmappedSubmission = device.submit(std::move(*unmappedCommand));
    REQUIRE(unmappedSubmission.has_value());
    REQUIRE(device.wait(*unmappedSubmission).has_value());
    device.collectGarbage();
    CHECK(readWords(device, *resultReadback) == std::array{0U, 0U, 0U, 0U});

    const auto remappedAfterCollection =
        device.enqueueSparseTextureMappings(rachis::QueueType::graphics, {.standardTiles = mapFirstTiles});
    REQUIRE(remappedAfterCollection.has_value());
    REQUIRE(device.wait(*remappedAfterCollection).has_value());
    device.collectGarbage();

    REQUIRE(device.destroyCommandPool(*commandPool).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyTextureView(*inputView).has_value());
    REQUIRE(device.destroyTextureView(*outputView).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyTexture(*input).has_value());
    REQUIRE(device.destroyTexture(*output).has_value());
    REQUIRE(device.destroyResourceHeap(*inputHeap).has_value());
    REQUIRE(device.destroyResourceHeap(*outputHeap).has_value());
    REQUIRE(device.destroyBuffer(*parameters).has_value());
    REQUIRE(device.destroyBuffer(*results).has_value());
    REQUIRE(device.destroyBuffer(*resultReadback).has_value());
    REQUIRE(device.destroyBuffer(*inputUpload).has_value());
    REQUIRE(device.destroyBuffer(*sentinelUpload).has_value());
    REQUIRE(device.destroyBuffer(*outputReadback).has_value());
    device.collectGarbage();

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
