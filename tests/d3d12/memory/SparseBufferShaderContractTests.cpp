#include "D3D12/D3D12TestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
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
        std::ifstream stream(RACHIS_TEST_SPARSE_BUFFER_SHADER_PATH, std::ios::binary | std::ios::ate);
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

    [[nodiscard]] std::array<std::uint32_t, 2> readResults(rachis::Device& device, rachis::BufferHandle readback)
    {
        auto mapped = device.mapBuffer(readback);
        REQUIRE(mapped.has_value());
        REQUIRE(mapped->size >= 2 * sizeof(std::uint32_t));
        std::array<std::uint32_t, 2> result{};
        std::memcpy(result.data(), mapped->data, sizeof(result));
        REQUIRE(device.unmapBuffer(readback).has_value());
        return result;
    }

} // namespace

TEST_CASE("shader-visible sparse bindings capture the current immutable mapping generation",
          "[size:medium][rhi][buffer][sparse][binding][compute][lifetime][warp]")
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

    constexpr std::uint64_t tileSize = 64ULL * 1024ULL;
    constexpr std::uint64_t resultSize = 2 * sizeof(std::uint32_t);
    constexpr std::uint32_t expectedValue = 0x6D5A1234U;
    const rachis::BufferDesc sparseDesc{
        .size = 2 * tileSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Shader-visible sparse buffer",
    };
    if (!rachis::test::requireCapabilities(device).sparseBuffers)
    {
        const auto unsupported = device.createSparseBuffer(sparseDesc);
        REQUIRE_FALSE(unsupported.has_value());
        CHECK(unsupported.error().code == rachis::ErrorCode::unsupported);
        return;
    }

    const auto bytecode = loadShader();
    REQUIRE_FALSE(bytecode.empty());
    const auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Sparse-buffer mapping shader",
        .shaderInterface = rachis::test::computeShaderInterface(
            {1, 1, 1},
            std::array{
                rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::readOnlyStorageBuffer, 1},
                rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
            }),
    });
    const auto sparse = device.createSparseBuffer(sparseDesc);
    const auto heap = device.createResourceHeap({
        .size = tileSize,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::buffer,
        .flags = rachis::ResourceHeapFlag::sparseBacking,
        .label = "Shader-visible sparse backing",
    });
    const auto upload = device.createBuffer({
        .size = sizeof(expectedValue),
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Sparse shader upload",
    });
    const auto output = device.createBuffer({
        .size = resultSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Sparse shader output",
    });
    const auto readback = device.createBuffer({
        .size = resultSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Sparse shader readback",
    });
    REQUIRE(shader.has_value());
    REQUIRE(sparse.has_value());
    REQUIRE(heap.has_value());
    REQUIRE(upload.has_value());
    REQUIRE(output.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(device.writeBuffer(*upload, 0, std::as_bytes(std::span{&expectedValue, 1})).has_value());

    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::readOnlyStorageBuffer, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    const auto bindingLayout = device.createBindingLayout({
        .entries = bindingEntries,
        .label = "Sparse shader binding layout",
    });
    REQUIRE(bindingLayout.has_value());
    const std::array bindingWrites{
        rachis::BindingWrite{
            .binding = 0,
            .resource =
                rachis::BufferBinding{
                    .buffer = *sparse,
                    .size = sparseDesc.size,
                },
        },
        rachis::BindingWrite{
            .binding = 1,
            .resource =
                rachis::BufferBinding{
                    .buffer = *output,
                    .size = resultSize,
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
    };

    const auto bindingSet = rachis::test::createBindingSet(device,
                                                           {
                                                               .layout = *bindingLayout,
                                                               .writes = bindingWrites,
                                                               .label = "Reusable sparse shader binding set",
                                                           });
    REQUIRE(bindingSet.has_value());
    const std::array bindingLayouts{*bindingLayout};
    const auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .label = "Sparse shader pipeline layout",
    });
    REQUIRE(pipelineLayout.has_value());
    const auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Sparse shader pipeline",
    });
    REQUIRE(pipeline.has_value());

    const std::array mapFirstTile{
        rachis::SparseBufferTileMapping{
            .buffer = *sparse,
            .bufferTileOffset = 0,
            .tileCount = 1,
            .heap = *heap,
            .heapTileOffset = 0,
        },
    };
    const auto initialMapping = device.enqueueSparseBufferMappings(rachis::QueueType::graphics, mapFirstTile);
    REQUIRE(initialMapping.has_value());

    auto uploadCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(uploadCommand.has_value());
    REQUIRE(device
                .bufferBarrier(*uploadCommand,
                               {
                                   .buffer = *sparse,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::copyDestination,
                               })
                .has_value());
    REQUIRE(device.copyBuffer(*uploadCommand, {*upload, 0, *sparse, 0, sizeof(expectedValue)}).has_value());
    REQUIRE(device
                .bufferBarrier(*uploadCommand,
                               {
                                   .buffer = *sparse,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::shaderResource,
                               })
                .has_value());
    const auto uploaded = device.submit(std::move(*uploadCommand));
    REQUIRE(uploaded.has_value());
    REQUIRE(device.wait(*uploaded).has_value());
    device.collectGarbage();

    const auto replacement = device.enqueueSparseBufferMappings(rachis::QueueType::graphics, mapFirstTile);
    REQUIRE(replacement.has_value());

    const auto commandPool = device.createCommandPool({
        .queueType = rachis::QueueType::compute,
        .maxCommandBuffers = 1,
        .label = "Unretained sparse shader pool",
        .flags = rachis::CommandPoolFlag::allowIndividualReset | rachis::CommandPoolFlag::unretainedResources,
    });
    REQUIRE(commandPool.has_value());

    auto mappedCommand = device.acquireCommandBuffer(*commandPool);
    REQUIRE(mappedCommand.has_value());
    REQUIRE(device
                .bufferBarrier(*mappedCommand,
                               {
                                   .buffer = *output,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*mappedCommand, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*mappedCommand, 0, *bindingSet).has_value());
    const std::array unmapFirstTile{
        rachis::SparseBufferTileMapping{
            .buffer = *sparse,
            .bufferTileOffset = 0,
            .tileCount = 1,
        },
    };
    const auto retainedReplacement = device.enqueueSparseBufferMappings(rachis::QueueType::graphics, unmapFirstTile);
    REQUIRE_FALSE(retainedReplacement.has_value());
    CHECK(retainedReplacement.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.dispatch(*mappedCommand, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*mappedCommand,
                               {
                                   .buffer = *output,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device.copyBuffer(*mappedCommand, {*output, 0, *readback, 0, resultSize}).has_value());

    const auto missingWait = device.submit(std::move(*mappedCommand));
    REQUIRE_FALSE(missingWait.has_value());
    CHECK(missingWait.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(mappedCommand->isValid());
    const std::array replacementWaits{*replacement};
    const auto mappedSubmission = device.submit(std::move(*mappedCommand), replacementWaits);
    REQUIRE(mappedSubmission.has_value());
    REQUIRE(device.wait(*mappedSubmission).has_value());
    device.collectGarbage();
    CHECK(readResults(device, *readback) == std::array{expectedValue, 0U});

    const auto unmappedGeneration = device.enqueueSparseBufferMappings(rachis::QueueType::graphics, unmapFirstTile);
    REQUIRE(unmappedGeneration.has_value());

    auto unmappedCommand = device.acquireCommandBuffer(*commandPool);
    REQUIRE(unmappedCommand.has_value());
    REQUIRE(device
                .bufferBarrier(*unmappedCommand,
                               {
                                   .buffer = *output,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::copySource,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*unmappedCommand, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*unmappedCommand, 0, *bindingSet).has_value());
    const auto retainedUnmappedGeneration =
        device.enqueueSparseBufferMappings(rachis::QueueType::graphics, mapFirstTile);
    REQUIRE_FALSE(retainedUnmappedGeneration.has_value());
    CHECK(retainedUnmappedGeneration.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.dispatch(*unmappedCommand, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*unmappedCommand,
                               {
                                   .buffer = *output,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device.copyBuffer(*unmappedCommand, {*output, 0, *readback, 0, resultSize}).has_value());
    const std::array unmappedWaits{*unmappedGeneration};
    const auto unmappedSubmission = device.submit(std::move(*unmappedCommand), unmappedWaits);
    REQUIRE(unmappedSubmission.has_value());
    REQUIRE(device.wait(*unmappedSubmission).has_value());
    device.collectGarbage();
    CHECK(readResults(device, *readback) == std::array{0U, 0U});

    const auto remappedAfterCollection = device.enqueueSparseBufferMappings(rachis::QueueType::graphics, mapFirstTile);
    REQUIRE(remappedAfterCollection.has_value());
    REQUIRE(device.wait(*remappedAfterCollection).has_value());
    device.collectGarbage();

    REQUIRE(device.destroyCommandPool(*commandPool).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBuffer(*sparse).has_value());
    REQUIRE(device.destroyResourceHeap(*heap).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*output).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    device.collectGarbage();

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("shader-visible sparse bindings reject an unpublished mapping generation",
          "[size:medium][rhi][buffer][sparse][binding][failure][warp]")
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
    if (!rachis::test::requireCapabilities(device).sparseBuffers)
    {
        return;
    }

    constexpr std::uint64_t tileSize = 64ULL * 1024ULL;
    constexpr std::uint64_t resultSize = 2 * sizeof(std::uint32_t);
    const auto bytecode = loadShader();
    REQUIRE_FALSE(bytecode.empty());
    const auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Unpublished sparse-buffer binding shader",
        .shaderInterface = rachis::test::computeShaderInterface(
            {1, 1, 1},
            std::array{
                rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::readOnlyStorageBuffer, 1},
                rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
            }),
    });
    const auto sparse = device.createSparseBuffer({
        .size = tileSize,
        .usage = rachis::BufferUsage::storage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Unpublished sparse binding buffer",
    });
    const auto heap = device.createResourceHeap({
        .size = tileSize,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::buffer,
        .flags = rachis::ResourceHeapFlag::sparseBacking,
        .label = "Unpublished sparse binding backing",
    });
    const auto output = device.createBuffer({
        .size = resultSize,
        .usage = rachis::BufferUsage::storage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Unpublished sparse binding output",
    });
    REQUIRE(shader.has_value());
    REQUIRE(sparse.has_value());
    REQUIRE(heap.has_value());
    REQUIRE(output.has_value());

    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::readOnlyStorageBuffer, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    const auto bindingLayout = device.createBindingLayout({
        .entries = bindingEntries,
        .label = "Unpublished sparse binding layout",
    });
    REQUIRE(bindingLayout.has_value());
    const std::array bindingWrites{
        rachis::BindingWrite{
            .binding = 0,
            .resource = rachis::BufferBinding{.buffer = *sparse, .size = tileSize},
        },
        rachis::BindingWrite{
            .binding = 1,
            .resource =
                rachis::BufferBinding{
                    .buffer = *output,
                    .size = resultSize,
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
    };
    const auto bindingSet = rachis::test::createBindingSet(device,
                                                           {
                                                               .layout = *bindingLayout,
                                                               .writes = bindingWrites,
                                                               .label = "Unpublished sparse binding set",
                                                           });
    REQUIRE(bindingSet.has_value());
    const std::array bindingLayouts{*bindingLayout};
    const auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .label = "Unpublished sparse binding pipeline layout",
    });
    REQUIRE(pipelineLayout.has_value());
    const auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Unpublished sparse binding pipeline",
    });
    REQUIRE(pipeline.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());

    const std::array mapping{
        rachis::SparseBufferTileMapping{
            .buffer = *sparse,
            .bufferTileOffset = 0,
            .tileCount = 1,
            .heap = *heap,
            .heapTileOffset = 0,
        },
    };
    rachis::d3d12::test::nextSparseMappingSignalFailure().store(true, std::memory_order_relaxed);
    const auto unpublished = device.enqueueSparseBufferMappings(rachis::QueueType::graphics, mapping);
    REQUIRE_FALSE(unpublished.has_value());
    CHECK(unpublished.error().code == rachis::ErrorCode::deviceLost);

    const auto bound = device.bindComputeBindingSet(*command, 0, *bindingSet);
    REQUIRE_FALSE(bound.has_value());
    CHECK(bound.error().code == rachis::ErrorCode::invalidState);

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBuffer(*sparse).has_value());
    REQUIRE(device.destroyBuffer(*output).has_value());
    REQUIRE(device.destroyResourceHeap(*heap).has_value());
}
