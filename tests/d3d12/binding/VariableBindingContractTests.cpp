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
        std::ifstream stream(RACHIS_TEST_VARIABLE_BINDING_SHADER_PATH, std::ios::binary | std::ios::ate);
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

    [[nodiscard]] rachis::BindingFlag variableFlags() noexcept
    {
        return rachis::BindingFlag::partiallyBound | rachis::BindingFlag::variableDescriptorCount;
    }

} // namespace

TEST_CASE("variable-count tails execute dynamically indexed sparse descriptors",
          "[size:medium][rhi][binding][bindless][compute][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).variableDescriptorCountBindingArrays);

    const auto bytecode = loadShader();
    REQUIRE_FALSE(bytecode.empty());
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Variable descriptor array shader",
        .shaderInterface = rachis::test::computeShaderInterface(
            {2, 1, 1},
            std::array{rachis::ShaderBindingRequirement{
                0, 0, rachis::ShaderResourceType::storageBuffer, rachis::unboundedBindingCount}}),
    });
    auto outputOne = device.createBuffer({
        .size = bindingSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .label = "Variable output one",
    });
    auto outputThree = device.createBuffer({
        .size = bindingSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .label = "Variable output three",
    });
    auto readback = device.createBuffer({
        .size = 2 * bindingSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Variable output readback",
    });
    REQUIRE(shader.has_value());
    REQUIRE(outputOne.has_value());
    REQUIRE(outputThree.has_value());
    REQUIRE(readback.has_value());

    const std::array entries{
        rachis::BindingLayoutEntry{
            0,
            rachis::BindingType::storageBuffer,
            rachis::ShaderStage::compute,
            rachis::unboundedBindingCount,
            variableFlags(),
        },
    };
    auto bindingLayout = device.createBindingLayout({.entries = entries, .label = "Variable array layout"});
    REQUIRE(bindingLayout.has_value());
    const std::array writes{
        rachis::BindingWrite{0, 1, rachis::BufferBinding{*outputOne, 0, bindingSize, sizeof(std::uint32_t)}},
        rachis::BindingWrite{0, 3, rachis::BufferBinding{*outputThree, 0, bindingSize, sizeof(std::uint32_t)}},
    };
    auto bindingSet = rachis::test::createBindingSet(device,
                                                     {
                                                         .layout = *bindingLayout,
                                                         .writes = writes,
                                                         .variableDescriptorCount = 4,
                                                         .label = "Variable array set",
                                                     });
    REQUIRE(bindingSet.has_value());
    const std::array layouts{*bindingLayout};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Variable array pipeline",
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
    CHECK(readValue(mapped->data, 0) == 201);
    CHECK(readValue(mapped->data, bindingSize) == 203);
    REQUIRE(device.unmapBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("variable-count tails enforce layout and set boundaries", "[size:medium][rhi][binding][bindless][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    const auto capabilities = *device.capabilities();
    REQUIRE(capabilities.variableDescriptorCountBindingArrays);

    const std::array validEntry{
        rachis::BindingLayoutEntry{7,
                                   rachis::BindingType::storageBuffer,
                                   rachis::ShaderStage::compute,
                                   rachis::unboundedBindingCount,
                                   variableFlags()},
    };
    auto layout = device.createBindingLayout({.entries = validEntry});
    REQUIRE(layout.has_value());
    auto empty = rachis::test::createBindingSet(device, {.layout = *layout, .variableDescriptorCount = 0});
    REQUIRE(empty.has_value());
    const std::array firstWrite{
        rachis::BindingWrite{.binding = 7, .arrayElement = 0, .resource = std::monostate{}},
    };
    CHECK_FALSE(device.updateBindingSet(*empty, firstWrite).has_value());
    CHECK_FALSE(rachis::test::createBindingSet(
                    device,
                    {
                        .layout = *layout,
                        .variableDescriptorCount = capabilities.limits.maxVariableResourceDescriptors + 1,
                    })
                    .has_value());

    auto two = rachis::test::createBindingSet(device, {.layout = *layout, .variableDescriptorCount = 2});
    REQUIRE(two.has_value());
    const std::array onePast{
        rachis::BindingWrite{.binding = 7, .arrayElement = 2, .resource = std::monostate{}},
    };
    CHECK_FALSE(device.updateBindingSet(*two, onePast).has_value());

    const std::array prefixedEntries{
        rachis::BindingLayoutEntry{0,
                                   rachis::BindingType::storageBuffer,
                                   rachis::ShaderStage::compute,
                                   2,
                                   rachis::BindingFlag::partiallyBound},
        rachis::BindingLayoutEntry{4,
                                   rachis::BindingType::storageBuffer,
                                   rachis::ShaderStage::compute,
                                   rachis::unboundedBindingCount,
                                   variableFlags()},
    };
    auto prefixedLayout = device.createBindingLayout({.entries = prefixedEntries});
    REQUIRE(prefixedLayout.has_value());
    auto prefixedSet =
        rachis::test::createBindingSet(device, {.layout = *prefixedLayout, .variableDescriptorCount = 3});
    REQUIRE(prefixedSet.has_value());
    const std::array prefixedLayouts{*prefixedLayout};
    auto prefixedPipelineLayout = device.createPipelineLayout({.bindingLayouts = prefixedLayouts});
    REQUIRE(prefixedPipelineLayout.has_value());

    const std::array fixedEntry{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    auto fixedLayout = device.createBindingLayout({.entries = fixedEntry});
    REQUIRE(fixedLayout.has_value());
    CHECK_FALSE(
        rachis::test::createBindingSet(device, {.layout = *fixedLayout, .variableDescriptorCount = 1}).has_value());

    const auto variable = rachis::BindingFlag::variableDescriptorCount;
    const std::array sentinelWithoutFlag{
        rachis::BindingLayoutEntry{
            0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, rachis::unboundedBindingCount},
    };
    const std::array finiteWithFlag{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 4, variable},
    };
    const std::array notLargest{
        rachis::BindingLayoutEntry{0,
                                   rachis::BindingType::storageBuffer,
                                   rachis::ShaderStage::compute,
                                   rachis::unboundedBindingCount,
                                   variable},
        rachis::BindingLayoutEntry{1, rachis::BindingType::sampler, rachis::ShaderStage::compute, 1},
    };
    const std::array multiple{
        rachis::BindingLayoutEntry{0,
                                   rachis::BindingType::storageBuffer,
                                   rachis::ShaderStage::compute,
                                   rachis::unboundedBindingCount,
                                   variable},
        rachis::BindingLayoutEntry{
            1, rachis::BindingType::sampler, rachis::ShaderStage::compute, rachis::unboundedBindingCount, variable},
    };
    const std::array dynamicSparse{
        rachis::BindingLayoutEntry{0,
                                   rachis::BindingType::dynamicStorageBuffer,
                                   rachis::ShaderStage::compute,
                                   1,
                                   rachis::BindingFlag::partiallyBound},
    };
    CHECK_FALSE(device.createBindingLayout({.entries = sentinelWithoutFlag}).has_value());
    CHECK_FALSE(device.createBindingLayout({.entries = finiteWithFlag}).has_value());
    CHECK_FALSE(device.createBindingLayout({.entries = notLargest}).has_value());
    CHECK_FALSE(device.createBindingLayout({.entries = multiple}).has_value());
    CHECK_FALSE(device.createBindingLayout({.entries = dynamicSparse}).has_value());

    REQUIRE(device.destroyBindingSet(*empty).has_value());
    REQUIRE(device.destroyBindingSet(*two).has_value());
    REQUIRE(device.destroyPipelineLayout(*prefixedPipelineLayout).has_value());
    REQUIRE(device.destroyBindingSet(*prefixedSet).has_value());
    REQUIRE(device.destroyBindingLayout(*prefixedLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*layout).has_value());
    REQUIRE(device.destroyBindingLayout(*fixedLayout).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
