#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace
{

    constexpr std::uint64_t bindingSize = 256;

    [[nodiscard]] rachis::DeviceDesc vulkanDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::vulkan,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
    }

    [[nodiscard]] std::vector<std::byte> loadBytes(const std::filesystem::path& path)
    {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        const auto end = stream.tellg();
        if (end <= 0)
        {
            return {};
        }
        std::vector<std::byte> bytes(static_cast<std::size_t>(end));
        stream.seekg(0, std::ios::beg);
        stream.read(reinterpret_cast<char*>(bytes.data()), end);
        return stream ? bytes : std::vector<std::byte>{};
    }

    [[nodiscard]] std::uint32_t readValue(const std::byte* data, std::uint64_t offset)
    {
        std::uint32_t value = 0;
        std::memcpy(&value, data + offset, sizeof(value));
        return value;
    }

} // namespace

TEST_CASE("Vulkan advertises descriptor indexing only as a complete implemented contract",
          "[size:medium][rhi][vulkan][binding][bindless]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = device.capabilities();
    REQUIRE(capabilities.has_value());
    REQUIRE(capabilities->partiallyBoundBindingArrays);
    REQUIRE(capabilities->variableDescriptorCountBindingArrays);
    REQUIRE(capabilities->limits.maxVariableResourceDescriptors >= 4);
    CHECK(capabilities->limits.maxVariableResourceDescriptors <= 65'536);

    const std::array partial{
        rachis::BindingLayoutEntry{
            0,
            rachis::BindingType::storageBuffer,
            rachis::ShaderStage::compute,
            4,
            rachis::BindingFlag::partiallyBound,
        },
    };
    auto partialLayout = device.createBindingLayout({.entries = partial, .label = "Vulkan partial layout"});
    REQUIRE(partialLayout);

    const std::array variable{
        rachis::BindingLayoutEntry{
            0,
            rachis::BindingType::storageBuffer,
            rachis::ShaderStage::compute,
            rachis::unboundedBindingCount,
            rachis::BindingFlag::partiallyBound | rachis::BindingFlag::variableDescriptorCount,
        },
    };
    auto variableLayout = device.createBindingLayout({.entries = variable, .label = "Vulkan variable layout"});
    REQUIRE(variableLayout);

    const std::array missingFlag{
        rachis::BindingLayoutEntry{
            0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, rachis::unboundedBindingCount},
    };
    auto missing = device.createBindingLayout({.entries = missingFlag});
    REQUIRE_FALSE(missing);
    CHECK(missing.error().code == rachis::ErrorCode::invalidArgument);

    const std::array twoTails{
        rachis::BindingLayoutEntry{
            0,
            rachis::BindingType::storageBuffer,
            rachis::ShaderStage::compute,
            rachis::unboundedBindingCount,
            rachis::BindingFlag::variableDescriptorCount,
        },
        rachis::BindingLayoutEntry{
            1,
            rachis::BindingType::sampledTexture,
            rachis::ShaderStage::compute,
            rachis::unboundedBindingCount,
            rachis::BindingFlag::variableDescriptorCount,
        },
    };
    auto two = device.createBindingLayout({.entries = twoTails});
    REQUIRE_FALSE(two);
    CHECK(two.error().code == rachis::ErrorCode::invalidArgument);

    REQUIRE(device.destroyBindingLayout(*partialLayout));
    REQUIRE(device.destroyBindingLayout(*variableLayout));
}

#ifdef RACHIS_SPIRV_DIR
TEST_CASE("Vulkan sparse and variable descriptor arrays produce GPU results without update-after-bind",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][binding][bindless][compute]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).partiallyBoundBindingArrays);
    REQUIRE(rachis::test::requireCapabilities(device).variableDescriptorCountBindingArrays);

    const auto sparseBytecode = loadBytes(std::filesystem::path(RACHIS_SPIRV_DIR) / "PartiallyBoundBinding.spv");
    const auto variableBytecode = loadBytes(std::filesystem::path(RACHIS_SPIRV_DIR) / "VariableBinding.spv");
    REQUIRE_FALSE(sparseBytecode.empty());
    REQUIRE_FALSE(variableBytecode.empty());

    auto sparseShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = sparseBytecode,
        .entryPoint = "main",
        .label = "Vulkan sparse array shader",
        .shaderInterface = rachis::test::computeShaderInterface(
            {2, 1, 1},
            std::array{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 4}}),
    });
    auto variableShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = variableBytecode,
        .entryPoint = "main",
        .label = "Vulkan variable array shader",
        .shaderInterface = rachis::test::computeShaderInterface(
            {2, 1, 1},
            std::array{rachis::ShaderBindingRequirement{
                0, 0, rachis::ShaderResourceType::storageBuffer, rachis::unboundedBindingCount}}),
    });
    REQUIRE(sparseShader);
    REQUIRE(variableShader);

    auto outputOne = device.createBuffer({
        .size = bindingSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .label = "Vulkan sparse output one",
    });
    auto outputThree = device.createBuffer({
        .size = bindingSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .label = "Vulkan sparse output three",
    });
    auto readback = device.createBuffer({
        .size = 2 * bindingSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan sparse readback",
    });
    REQUIRE(outputOne);
    REQUIRE(outputThree);
    REQUIRE(readback);

    const std::array sparseEntries{
        rachis::BindingLayoutEntry{
            0,
            rachis::BindingType::storageBuffer,
            rachis::ShaderStage::compute,
            4,
            rachis::BindingFlag::partiallyBound,
        },
    };
    auto sparseLayout = device.createBindingLayout({.entries = sparseEntries, .label = "Vulkan sparse layout"});
    REQUIRE(sparseLayout);
    const std::array sparseWrites{
        rachis::BindingWrite{0, 1, rachis::BufferBinding{*outputOne, 0, bindingSize, sizeof(std::uint32_t)}},
        rachis::BindingWrite{0, 3, rachis::BufferBinding{*outputThree, 0, bindingSize, sizeof(std::uint32_t)}},
    };
    auto sparseSet = rachis::test::createBindingSet(device,
                                                    {
                                                        .layout = *sparseLayout,
                                                        .writes = sparseWrites,
                                                        .label = "Vulkan sparse set",
                                                    });
    REQUIRE(sparseSet);

    const std::array layouts{*sparseLayout};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts});
    REQUIRE(pipelineLayout);
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *sparseShader,
        .label = "Vulkan sparse pipeline",
    });
    REQUIRE(pipeline);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command);
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
    REQUIRE(device.bindComputePipeline(*command, *pipeline));
    REQUIRE(device.bindComputeBindingSet(*command, 0, *sparseSet));
    REQUIRE(device.dispatch(*command, 1));
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
    REQUIRE(device.copyBuffer(*command, {*outputOne, 0, *readback, 0, bindingSize}));
    REQUIRE(device.copyBuffer(*command, {*outputThree, 0, *readback, bindingSize, bindingSize}));
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted));

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped);
    CHECK(readValue(mapped->data, 0) == 101);
    CHECK(readValue(mapped->data, bindingSize) == 103);
    REQUIRE(device.unmapBuffer(*readback));

    const std::array clearWrite{
        rachis::BindingWrite{.binding = 0, .arrayElement = 1, .resource = std::monostate{}},
    };
    REQUIRE(device.updateBindingSet(*sparseSet, clearWrite));

    const std::array variableEntries{
        rachis::BindingLayoutEntry{
            0,
            rachis::BindingType::storageBuffer,
            rachis::ShaderStage::compute,
            rachis::unboundedBindingCount,
            rachis::BindingFlag::partiallyBound | rachis::BindingFlag::variableDescriptorCount,
        },
    };
    auto variableLayout =
        device.createBindingLayout({.entries = variableEntries, .label = "Vulkan variable GPU layout"});
    REQUIRE(variableLayout);
    auto emptyTail = rachis::test::createBindingSet(device, {.layout = *variableLayout, .variableDescriptorCount = 0});
    REQUIRE(emptyTail);
    auto overflow = rachis::test::createBindingSet(
        device,
        {
            .layout = *variableLayout,
            .variableDescriptorCount =
                rachis::test::requireCapabilities(device).limits.maxVariableResourceDescriptors + 1,
        });
    REQUIRE_FALSE(overflow);
    CHECK(overflow.error().code == rachis::ErrorCode::invalidArgument);
    auto variableSet = rachis::test::createBindingSet(device,
                                                      {
                                                          .layout = *variableLayout,
                                                          .writes = sparseWrites,
                                                          .variableDescriptorCount = 4,
                                                          .label = "Vulkan variable set",
                                                      });
    REQUIRE(variableSet);
    const std::array variableLayouts{*variableLayout};
    auto variablePipelineLayout = device.createPipelineLayout({.bindingLayouts = variableLayouts});
    REQUIRE(variablePipelineLayout);
    auto variablePipeline = device.createComputePipeline({
        .layout = *variablePipelineLayout,
        .computeShader = *variableShader,
        .label = "Vulkan variable pipeline",
    });
    REQUIRE(variablePipeline);
    auto variableCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(variableCommand);
    for (const auto output : {*outputOne, *outputThree})
    {
        REQUIRE(device
                    .bufferBarrier(*variableCommand,
                                   {
                                       .buffer = output,
                                       .stagesBefore = rachis::PipelineStage::all,
                                       .stagesAfter = rachis::PipelineStage::computeShader,
                                       .accessBefore = rachis::Access::common,
                                       .accessAfter = rachis::Access::unorderedAccess,
                                   })
                    .has_value());
    }
    REQUIRE(device.bindComputePipeline(*variableCommand, *variablePipeline));
    REQUIRE(device.bindComputeBindingSet(*variableCommand, 0, *variableSet));
    REQUIRE(device.dispatch(*variableCommand, 1));
    for (const auto output : {*outputOne, *outputThree})
    {
        REQUIRE(device
                    .bufferBarrier(*variableCommand,
                                   {
                                       .buffer = output,
                                       .stagesBefore = rachis::PipelineStage::computeShader,
                                       .stagesAfter = rachis::PipelineStage::copy,
                                       .accessBefore = rachis::Access::unorderedAccess,
                                       .accessAfter = rachis::Access::copySource,
                                   })
                    .has_value());
    }
    REQUIRE(device.copyBuffer(*variableCommand, {*outputOne, 0, *readback, 0, bindingSize}));
    REQUIRE(device.copyBuffer(*variableCommand, {*outputThree, 0, *readback, bindingSize, bindingSize}));
    auto variableSubmitted = device.submit(std::move(*variableCommand));
    REQUIRE(variableSubmitted);
    REQUIRE(device.wait(*variableSubmitted));

    auto variableMapped = device.mapBuffer(*readback);
    REQUIRE(variableMapped);
    CHECK(readValue(variableMapped->data, 0) == 201);
    CHECK(readValue(variableMapped->data, bindingSize) == 203);
    REQUIRE(device.unmapBuffer(*readback));

    REQUIRE(device.destroyPipeline(*variablePipeline));
    REQUIRE(device.destroyPipelineLayout(*variablePipelineLayout));
    REQUIRE(device.destroyBindingSet(*variableSet));
    REQUIRE(device.destroyBindingSet(*emptyTail));
    REQUIRE(device.destroyBindingLayout(*variableLayout));
    REQUIRE(device.destroyPipeline(*pipeline));
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout));
    REQUIRE(device.destroyBindingSet(*sparseSet));
    REQUIRE(device.destroyBindingLayout(*sparseLayout));
    REQUIRE(device.destroyShaderModule(*sparseShader));
    REQUIRE(device.destroyShaderModule(*variableShader));
    REQUIRE(device.destroyBuffer(*outputOne));
    REQUIRE(device.destroyBuffer(*outputThree));
    REQUIRE(device.destroyBuffer(*readback));

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
#endif
