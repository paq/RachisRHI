#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace
{

    constexpr std::uint64_t bindingSize = 256;

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
        std::ifstream stream(RACHIS_TEST_PARTIALLY_BOUND_BINDING_SHADER_PATH, std::ios::binary | std::ios::ate);
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

    [[nodiscard]] std::uint32_t readValue(const std::byte* data, std::uint64_t offset)
    {
        std::uint32_t value = 0;
        std::memcpy(&value, data + offset, sizeof(value));
        return value;
    }

} // namespace

TEST_CASE("partially-bound arrays publish sparse copy-on-write descriptor versions",
          "[size:medium][rhi][binding][bindless][compute][warp]")
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
    REQUIRE(rachis::test::requireCapabilities(device).partiallyBoundBindingArrays);

    const auto bytecode = loadShader();
    REQUIRE_FALSE(bytecode.empty());
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Partially-bound array shader",
        .shaderInterface = rachis::test::computeShaderInterface(
            {2, 1, 1},
            std::array{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 4}}),
    });
    auto outputOne = device.createBuffer({
        .size = bindingSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .label = "Sparse output one",
    });
    auto outputThree = device.createBuffer({
        .size = bindingSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .label = "Sparse output three",
    });
    auto readback = device.createBuffer({
        .size = 2 * bindingSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Sparse output readback",
    });
    REQUIRE(shader.has_value());
    REQUIRE(outputOne.has_value());
    REQUIRE(outputThree.has_value());
    REQUIRE(readback.has_value());

    const std::array sparseEntries{
        rachis::BindingLayoutEntry{
            0,
            rachis::BindingType::storageBuffer,
            rachis::ShaderStage::compute,
            4,
            rachis::BindingFlag::partiallyBound,
        },
    };
    auto bindingLayout = device.createBindingLayout({
        .entries = sparseEntries,
        .label = "Sparse array layout",
    });
    REQUIRE(bindingLayout.has_value());
    const std::array sparseWrites{
        rachis::BindingWrite{
            0,
            1,
            rachis::BufferBinding{*outputOne, 0, bindingSize, sizeof(std::uint32_t)},
        },
        rachis::BindingWrite{
            0,
            3,
            rachis::BufferBinding{*outputThree, 0, bindingSize, sizeof(std::uint32_t)},
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device,
                                                     {
                                                         .layout = *bindingLayout,
                                                         .writes = sparseWrites,
                                                         .label = "Sparse array set",
                                                     });
    REQUIRE(bindingSet.has_value());
    const std::array layouts{*bindingLayout};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Sparse array pipeline",
    });
    REQUIRE(pipeline.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    for (const auto output : {*outputOne, *outputThree})
    {
        REQUIRE(device
                    .bufferBarrier(*command,
                                   {
                                       .buffer = output,
                                       .stagesBefore = rachis::PipelineStage::all,
                                       .stagesAfter = rachis::PipelineStage::computeShader,
                                       .accessBefore = rachis::Access::common,
                                       .accessAfter = rachis::Access::unorderedAccess,
                                   })
                    .has_value());
    }
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
    for (const auto output : {*outputOne, *outputThree})
    {
        REQUIRE(device
                    .bufferBarrier(*command,
                                   {
                                       .buffer = output,
                                       .stagesBefore = rachis::PipelineStage::computeShader,
                                       .stagesAfter = rachis::PipelineStage::copy,
                                       .accessBefore = rachis::Access::unorderedAccess,
                                       .accessAfter = rachis::Access::copySource,
                                   })
                    .has_value());
    }
    REQUIRE(device.copyBuffer(*command, {*outputOne, 0, *readback, 0, bindingSize}).has_value());
    REQUIRE(device.copyBuffer(*command, {*outputThree, 0, *readback, bindingSize, bindingSize}).has_value());

    const std::array clearWrite{
        rachis::BindingWrite{.binding = 0, .arrayElement = 1, .resource = std::monostate{}},
    };
    REQUIRE(device.updateBindingSet(*bindingSet, clearWrite).has_value());

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyBuffer(*outputOne).has_value());
    REQUIRE(device.destroyBuffer(*outputThree).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    CHECK(readValue(mapped->data, 0) == 101);
    CHECK(readValue(mapped->data, bindingSize) == 103);
    REQUIRE(device.unmapBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("partially-bound arrays materialize placeholders for every descriptor class",
          "[size:medium][rhi][binding][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    constexpr auto partial = rachis::BindingFlag::partiallyBound;
    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::all, 1, partial},
        rachis::BindingLayoutEntry{2, rachis::BindingType::readOnlyStorageBuffer, rachis::ShaderStage::all, 1, partial},
        rachis::BindingLayoutEntry{4, rachis::BindingType::storageBuffer, rachis::ShaderStage::all, 1, partial},
        rachis::BindingLayoutEntry{6, rachis::BindingType::sampledTexture, rachis::ShaderStage::all, 2, partial},
        rachis::BindingLayoutEntry{8, rachis::BindingType::storageTexture, rachis::ShaderStage::all, 2, partial},
        rachis::BindingLayoutEntry{10, rachis::BindingType::sampler, rachis::ShaderStage::all, 2, partial},
    };
    auto layout = device.createBindingLayout({.entries = entries, .label = "All sparse descriptor classes"});
    REQUIRE(layout.has_value());
    auto set = rachis::test::createBindingSet(device, {.layout = *layout, .label = "All placeholder descriptors"});
    REQUIRE(set.has_value());

    const std::array requiredEntry{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    auto requiredLayout = device.createBindingLayout({.entries = requiredEntry});
    REQUIRE(requiredLayout.has_value());
    CHECK_FALSE(rachis::test::createBindingSet(device, {.layout = *requiredLayout}).has_value());
    const std::array invalidClear{
        rachis::BindingWrite{.binding = 0, .resource = std::monostate{}},
    };
    CHECK_FALSE(
        rachis::test::createBindingSet(device, {.layout = *requiredLayout, .writes = invalidClear}).has_value());

    REQUIRE(device.destroyBindingSet(*set).has_value());
    REQUIRE(device.destroyBindingLayout(*layout).has_value());
    REQUIRE(device.destroyBindingLayout(*requiredLayout).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
