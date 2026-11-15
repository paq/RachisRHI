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
#include <vector>

namespace
{

    constexpr std::uint32_t valueCount = 64;
    constexpr std::uint64_t outputSize = valueCount * sizeof(std::uint32_t);

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

} // namespace

TEST_CASE("shader libraries share immutable staged bytecode through execution",
          "[size:medium][rhi][pipeline][library][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto computeBytecode = loadShader(RACHIS_TEST_COMPUTE_SHADER_PATH);
    const auto vertexBytecode = loadShader(RACHIS_TEST_GRAPHICS_DIRECT_VERTEX_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_GRAPHICS_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(computeBytecode.empty());
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());

    const std::array entries{
        rachis::ShaderLibraryEntryDesc{
            .stage = rachis::ShaderStage::compute,
            .bytecode = computeBytecode,
            .entryPoint = "main",
        },
        rachis::ShaderLibraryEntryDesc{
            .stage = rachis::ShaderStage::vertex,
            .bytecode = vertexBytecode,
            .entryPoint = "directVertex",
        },
        rachis::ShaderLibraryEntryDesc{
            .stage = rachis::ShaderStage::pixel,
            .bytecode = pixelBytecode,
            .entryPoint = "pixelMain",
        },
    };
    auto library = device.createShaderLibrary({
        .entries = entries,
        .label = "Multi-stage DXIL library",
    });
    REQUIRE(library.has_value());

    CHECK_FALSE(device.createShaderLibrary({}).has_value());
    const std::array invalidStageEntry{
        rachis::ShaderLibraryEntryDesc{
            .stage = rachis::ShaderStage::all,
            .bytecode = vertexBytecode,
        },
    };
    CHECK_FALSE(device.createShaderLibrary({.entries = invalidStageEntry}).has_value());
    const std::array emptyBytecodeEntry{
        rachis::ShaderLibraryEntryDesc{
            .stage = rachis::ShaderStage::vertex,
            .entryPoint = "directVertex",
        },
    };
    CHECK_FALSE(device.createShaderLibrary({.entries = emptyBytecodeEntry}).has_value());
    const std::array emptyNameEntry{
        rachis::ShaderLibraryEntryDesc{
            .stage = rachis::ShaderStage::vertex,
            .bytecode = vertexBytecode,
            .entryPoint = {},
        },
    };
    CHECK_FALSE(device.createShaderLibrary({.entries = emptyNameEntry}).has_value());
    const std::array duplicateEntries{
        rachis::ShaderLibraryEntryDesc{
            .stage = rachis::ShaderStage::vertex,
            .bytecode = vertexBytecode,
            .entryPoint = "directVertex",
        },
        rachis::ShaderLibraryEntryDesc{
            .stage = rachis::ShaderStage::vertex,
            .bytecode = pixelBytecode,
            .entryPoint = "directVertex",
        },
    };
    CHECK_FALSE(device.createShaderLibrary({.entries = duplicateEntries}).has_value());

    computeBytecode.assign(computeBytecode.size(), std::byte{0});
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .entryPoint = "main",
        .label = "Library compute module",
        .library = *library,
        .shaderInterface = rachis::test::computeWriteShaderInterface(),
    });
    REQUIRE(shader.has_value());
    auto vertexShader = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .entryPoint = "directVertex",
        .library = *library,
        .shaderInterface = {},
    });
    REQUIRE(vertexShader.has_value());
    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::compute,
                        .bytecode = vertexBytecode,
                        .entryPoint = "main",
                        .library = *library,
                        .shaderInterface = rachis::test::computeShaderInterface({64, 1, 1}),
                    })
                    .has_value());
    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::compute,
                        .entryPoint = "missing",
                        .library = *library,
                        .shaderInterface = rachis::test::computeShaderInterface({64, 1, 1}),
                    })
                    .has_value());
    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::compute,
                        .entryPoint = "directVertex",
                        .library = *library,
                        .shaderInterface = rachis::test::computeShaderInterface({64, 1, 1}),
                    })
                    .has_value());
    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::compute,
                        .entryPoint = "main",
                        .shaderInterface = rachis::test::computeShaderInterface({64, 1, 1}),
                    })
                    .has_value());

    const auto staleLibrary = *library;
    REQUIRE(device.destroyShaderLibrary(*library).has_value());
    CHECK_FALSE(device.destroyShaderLibrary(staleLibrary).has_value());
    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::compute,
                        .entryPoint = "main",
                        .library = staleLibrary,
                        .shaderInterface = rachis::test::computeShaderInterface({64, 1, 1}),
                    })
                    .has_value());

    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = bindingEntries});
    REQUIRE(bindingLayout.has_value());
    auto storage = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
    });
    auto readback = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(storage.has_value());
    REQUIRE(readback.has_value());
    const std::array writes{
        rachis::BindingWrite{
            .binding = 0,
            .resource =
                rachis::BufferBinding{
                    .buffer = *storage,
                    .size = outputSize,
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device,
                                                     {
                                                         .layout = *bindingLayout,
                                                         .writes = writes,
                                                     });
    REQUIRE(bindingSet.has_value());
    const std::array bindingLayouts{*bindingLayout};
    const std::array pushConstants{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)},
    };
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = pushConstants,
    });
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Library-backed compute pipeline",
    });
    REQUIRE(pipeline.has_value());
    CHECK_FALSE(device
                    .createComputePipeline({
                        .layout = *pipelineLayout,
                        .computeShader = *vertexShader,
                    })
                    .has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *storage,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
    const std::array constants{7U, 13U};
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(constants))).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *storage,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            {
                                .source = *storage,
                                .destination = *readback,
                                .size = outputSize,
                            })
                .has_value());

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBuffer(*storage).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    for (std::uint32_t index = 0; index < valueCount; ++index)
    {
        std::uint32_t value = 0;
        std::memcpy(&value, mapped->data + index * sizeof(value), sizeof(value));
        CAPTURE(index, value);
        REQUIRE(value == index * constants[0] + constants[1]);
    }
    REQUIRE(device.unmapBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
