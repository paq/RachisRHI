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

    [[nodiscard]] std::vector<std::byte> loadShader()
    {
        std::ifstream stream(RACHIS_TEST_COMPUTE_SHADER_PATH, std::ios::binary | std::ios::ate);
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

TEST_CASE("compute dispatch writes an immutable storage binding and survives logical destruction",
          "[size:medium][rhi][compute][pipeline][warp]")
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
    REQUIRE(rachis::test::requireCapabilities(device).indirectDispatch);

    const auto bytecode = loadShader();
    REQUIRE_FALSE(bytecode.empty());
    auto shader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Compute write shader",
        .shaderInterface = rachis::test::computeWriteShaderInterface(),
    });
    REQUIRE(shader.has_value());

    auto storage = device.createBuffer(rachis::BufferDesc{
        .size = outputSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Compute output",
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = outputSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Compute readback",
    });
    REQUIRE(storage.has_value());
    REQUIRE(readback.has_value());

    constexpr std::uint64_t indirectArgumentOffset = sizeof(std::uint32_t);
    const rachis::DispatchIndirectArguments indirectArguments{1, 1, 1};
    auto argumentBuffer = device.createBuffer(rachis::BufferDesc{
        .size = indirectArgumentOffset + sizeof(indirectArguments),
        .usage = rachis::BufferUsage::indirect,
        .memoryType = rachis::MemoryType::upload,
        .label = "Indirect compute arguments",
    });
    REQUIRE(argumentBuffer.has_value());
    REQUIRE(device.writeBuffer(*argumentBuffer, indirectArgumentOffset, std::as_bytes(std::span(&indirectArguments, 1)))
                .has_value());
    auto staleArgumentBuffer = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(indirectArguments),
        .usage = rachis::BufferUsage::indirect,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(staleArgumentBuffer.has_value());
    const auto staleArguments = *staleArgumentBuffer;
    REQUIRE(device.destroyBuffer(*staleArgumentBuffer).has_value());

    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout(rachis::BindingLayoutDesc{
        .entries = bindingEntries,
        .label = "Compute output layout",
    });
    REQUIRE(bindingLayout.has_value());
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
                                                     rachis::BindingSetDesc{
                                                         .layout = *bindingLayout,
                                                         .writes = writes,
                                                         .label = "Compute output set",
                                                     });
    REQUIRE(bindingSet.has_value());

    const std::array bindingLayouts{*bindingLayout};
    const std::array pushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)},
    };
    auto pipelineLayout = device.createPipelineLayout(rachis::PipelineLayoutDesc{
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = pushConstantRanges,
        .label = "Compute write pipeline layout",
    });
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline(rachis::ComputePipelineDesc{
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Compute write pipeline",
    });
    REQUIRE(pipeline.has_value());

    auto vertexShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = bytecode,
        .shaderInterface = {},
    });
    REQUIRE(vertexShader.has_value());
    CHECK_FALSE(device
                    .createComputePipeline(rachis::ComputePipelineDesc{
                        .layout = *pipelineLayout,
                        .computeShader = *vertexShader,
                    })
                    .has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());

    auto emptyLayout = device.createBindingLayout({});
    REQUIRE(emptyLayout.has_value());
    auto emptySet = rachis::test::createBindingSet(device, rachis::BindingSetDesc{.layout = *emptyLayout});
    REQUIRE(emptySet.has_value());

    auto copyCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(copyCommand.has_value());
    CHECK_FALSE(device.bindComputePipeline(*copyCommand, *pipeline).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    CHECK_FALSE(device.dispatch(*command, 1).has_value());
    CHECK_FALSE(device.dispatchIndirect(*command, *argumentBuffer, indirectArgumentOffset).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *storage,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    CHECK_FALSE(device.dispatch(*command, 1).has_value());
    CHECK_FALSE(device.dispatchIndirect(*command, *argumentBuffer, indirectArgumentOffset).has_value());
    CHECK_FALSE(device.bindComputeBindingSet(*command, 0, *emptySet).has_value());
    CHECK_FALSE(device.bindComputeBindingSet(*command, 1, *bindingSet).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());

    const std::array directConstants{3U, 7U};
    CHECK_FALSE(device.setComputePushConstants(*command, 4, std::as_bytes(std::span(directConstants))).has_value());
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(directConstants))).has_value());
    CHECK_FALSE(device.dispatch(*command, 0).has_value());
    CHECK_FALSE(device.dispatchIndirect(*command, {}, 0).has_value());
    CHECK_FALSE(device.dispatchIndirect(*command, *storage, 0).has_value());
    CHECK_FALSE(device.dispatchIndirect(*command, *argumentBuffer, 2).has_value());
    CHECK_FALSE(device.dispatchIndirect(*command, *argumentBuffer, 8).has_value());
    CHECK_FALSE(device.dispatchIndirect(*command, staleArguments, 0).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *storage,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    const std::array indirectConstants{5U, 11U};
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(indirectConstants))).has_value());
    REQUIRE(device.dispatchIndirect(*command, *argumentBuffer, indirectArgumentOffset).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *storage,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            rachis::BufferCopy{
                                .source = *storage,
                                .destination = *readback,
                                .size = outputSize,
                            })
                .has_value());

    const auto stalePipeline = *pipeline;
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyBuffer(*storage).has_value());
    REQUIRE(device.destroyBuffer(*argumentBuffer).has_value());
    CHECK_FALSE(device.destroyPipeline(stalePipeline).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    REQUIRE(mapped->size == outputSize);
    std::uint32_t firstMismatch = valueCount;
    for (std::uint32_t index = 0; index < valueCount; ++index)
    {
        std::uint32_t value = 0;
        std::memcpy(&value, mapped->data + index * sizeof(value), sizeof(value));
        if (value != index * indirectConstants[0] + indirectConstants[1])
        {
            firstMismatch = index;
            break;
        }
    }
    CAPTURE(firstMismatch);
    CHECK(firstMismatch == valueCount);
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyBindingSet(*emptySet).has_value());
    REQUIRE(device.destroyBindingLayout(*emptyLayout).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
