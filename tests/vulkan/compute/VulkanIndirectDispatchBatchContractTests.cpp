#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <vector>

namespace
{

    constexpr std::uint32_t valueCount = 64;
    constexpr std::uint64_t outputSize = valueCount * sizeof(std::uint32_t);
    constexpr std::uint64_t argumentOffset = sizeof(std::uint32_t);
    constexpr std::uint32_t argumentStride = 16;
    constexpr std::uint32_t dispatchCount = 3;
    constexpr std::uint64_t argumentBufferSize = argumentOffset +
                                                 static_cast<std::uint64_t>(dispatchCount - 1) * argumentStride +
                                                 sizeof(rachis::DispatchIndirectArguments);
    constexpr std::uint64_t countOffset = sizeof(std::uint32_t);
    constexpr std::uint32_t gpuCount = 2;

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

    [[nodiscard]] rachis::BufferBarrier outputBarrier(rachis::BufferHandle output,
                                                      rachis::PipelineStage stagesAfter,
                                                      rachis::Access accessAfter) noexcept
    {
        return rachis::BufferBarrier{
            .buffer = output,
            .stagesBefore = rachis::PipelineStage::computeShader,
            .stagesAfter = stagesAfter,
            .accessBefore = rachis::Access::unorderedAccess,
            .accessAfter = accessAfter,
        };
    }

} // namespace

TEST_CASE("Vulkan indirect compute batches execute fixed and GPU-counted records and keep the bound state",
          "[size:medium][rhi][vulkan][compute][indirect][batch]")
{
    auto created = rachis::createDevice(rachis::test::vulkanDeviceDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    REQUIRE(capabilities.indirectDispatch);
    REQUIRE(capabilities.multiDispatchIndirect);
    if (rachis::test::skipUnlessReferenceDesktopAdvertises(capabilities.dispatchIndirectCount, "dispatchIndirectCount"))
    {
        return;
    }
    const bool conditional = capabilities.conditionalExecution;

    const auto countBytecode =
        loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanComputeWrite.countDispatch.spv");
    const auto clearBytecode = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanComputeWrite.spv");
    REQUIRE_FALSE(countBytecode.empty());
    REQUIRE_FALSE(clearBytecode.empty());
    auto shader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::compute,
        .bytecode = countBytecode,
        .entryPoint = "countDispatch",
        .label = "Vulkan indirect dispatch batch counter",
        .shaderInterface = rachis::test::computeWriteShaderInterface(),
    });
    auto clearShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::compute,
        .bytecode = clearBytecode,
        .entryPoint = "main",
        .label = "Vulkan indirect dispatch batch clear",
        .shaderInterface = rachis::test::computeWriteShaderInterface(),
    });
    REQUIRE(shader.has_value());
    REQUIRE(clearShader.has_value());

    auto output = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan indirect dispatch batch output",
    });
    auto readback = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan indirect dispatch batch readback",
    });
    REQUIRE(output.has_value());
    REQUIRE(readback.has_value());

    std::array<std::byte, argumentBufferSize> argumentBytes{};
    const rachis::DispatchIndirectArguments arguments{1, 1, 1};
    for (std::uint32_t index = 0; index < dispatchCount; ++index)
    {
        std::memcpy(argumentBytes.data() + argumentOffset + static_cast<std::uint64_t>(index) * argumentStride,
                    &arguments,
                    sizeof(arguments));
    }
    auto argumentBuffer = device.createBuffer({
        .size = argumentBytes.size(),
        .usage = rachis::BufferUsage::indirect,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan padded indirect dispatch records",
    });
    REQUIRE(argumentBuffer.has_value());
    REQUIRE(device.writeBuffer(*argumentBuffer, 0, argumentBytes).has_value());

    std::array<std::byte, countOffset + sizeof(std::uint32_t)> countBytes{};
    std::memcpy(countBytes.data() + countOffset, &gpuCount, sizeof(gpuCount));
    auto countBuffer = device.createBuffer({
        .size = countBytes.size(),
        .usage = rachis::BufferUsage::indirect,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan indirect dispatch GPU count",
    });
    REQUIRE(countBuffer.has_value());
    REQUIRE(device.writeBuffer(*countBuffer, 0, countBytes).has_value());

    auto staleCountBuffer = device.createBuffer({
        .size = countBytes.size(),
        .usage = rachis::BufferUsage::indirect,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(staleCountBuffer.has_value());
    const auto staleCount = *staleCountBuffer;
    REQUIRE(device.destroyBuffer(*staleCountBuffer).has_value());

    const std::array<std::uint64_t, 2> predicateValues{0, 1};
    auto predicate = device.createBuffer({
        .size = sizeof(predicateValues),
        .usage = rachis::BufferUsage::predicate,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan indirect dispatch batch predicate",
    });
    REQUIRE(predicate.has_value());
    REQUIRE(device.writeBuffer(*predicate, 0, std::as_bytes(std::span(predicateValues))).has_value());

    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({
        .entries = bindingEntries,
        .label = "Vulkan indirect dispatch batch layout",
    });
    REQUIRE(bindingLayout.has_value());
    const std::array writes{
        rachis::BindingWrite{
            .binding = 0,
            .resource =
                rachis::BufferBinding{
                    .buffer = *output,
                    .size = outputSize,
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device,
                                                     {
                                                         .layout = *bindingLayout,
                                                         .writes = writes,
                                                         .label = "Vulkan indirect dispatch batch set",
                                                     });
    REQUIRE(bindingSet.has_value());
    const std::array bindingLayouts{*bindingLayout};
    const std::array pushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)},
    };
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = pushConstantRanges,
        .label = "Vulkan indirect dispatch batch pipeline layout",
    });
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Vulkan indirect dispatch batch pipeline",
    });
    auto clearPipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *clearShader,
        .label = "Vulkan indirect dispatch batch clear pipeline",
    });
    REQUIRE(pipeline.has_value());
    REQUIRE(clearPipeline.has_value());

    const rachis::IndirectDispatchDesc fixedBatch{
        .argumentBuffer = *argumentBuffer,
        .argumentOffset = argumentOffset,
        .maxDispatchCount = dispatchCount,
        .stride = argumentStride,
    };
    const rachis::IndirectDispatchDesc countedBatch{
        .argumentBuffer = *argumentBuffer,
        .argumentOffset = argumentOffset,
        .maxDispatchCount = dispatchCount,
        .stride = argumentStride,
        .countBuffer = *countBuffer,
        .countOffset = countOffset,
    };
    const std::array countConstants{0U, 1U};

    auto unretainedPool = device.createCommandPool({
        .queueType = rachis::QueueType::compute,
        .maxCommandBuffers = 1,
        .label = "Vulkan unretained indirect dispatch pool",
        .flags = rachis::CommandPoolFlag::unretainedResources,
    });
    REQUIRE(unretainedPool.has_value());
    auto unretainedCommand = device.acquireCommandBuffer(*unretainedPool);
    REQUIRE(unretainedCommand.has_value());
    REQUIRE(device
                .bufferBarrier(*unretainedCommand,
                               {
                                   .buffer = *output,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*unretainedCommand, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*unretainedCommand, 0, *bindingSet).has_value());
    REQUIRE(
        device.setComputePushConstants(*unretainedCommand, 0, std::as_bytes(std::span(countConstants))).has_value());
    REQUIRE(device.dispatchIndirect(*unretainedCommand, countedBatch).has_value());
    REQUIRE(device
                .bufferBarrier(*unretainedCommand,
                               outputBarrier(*output, rachis::PipelineStage::all, rachis::Access::common))
                .has_value());
    auto unretainedSubmission = device.submit(std::move(*unretainedCommand));
    REQUIRE(unretainedSubmission.has_value());
    REQUIRE(device.wait(*unretainedSubmission).has_value());
    device.collectGarbage();
    REQUIRE(device.destroyCommandPool(*unretainedPool).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
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
    REQUIRE(device.bindComputePipeline(*command, *clearPipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
    const std::array clearConstants{0U, 0U};
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(clearConstants))).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
    REQUIRE(
        device
            .bufferBarrier(
                *command, outputBarrier(*output, rachis::PipelineStage::computeShader, rachis::Access::unorderedAccess))
            .has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(countConstants))).has_value());

    CHECK_FALSE(device.dispatchIndirect(*command, rachis::IndirectDispatchDesc{}).has_value());
    CHECK_FALSE(
        device.dispatchIndirect(*command, {.argumentBuffer = *argumentBuffer, .argumentOffset = 2}).has_value());
    CHECK_FALSE(device
                    .dispatchIndirect(
                        *command, {.argumentBuffer = *argumentBuffer, .argumentOffset = argumentOffset, .stride = 8})
                    .has_value());
    CHECK_FALSE(device
                    .dispatchIndirect(
                        *command, {.argumentBuffer = *argumentBuffer, .argumentOffset = argumentOffset, .stride = 14})
                    .has_value());
    CHECK_FALSE(device
                    .dispatchIndirect(*command,
                                      {
                                          .argumentBuffer = *argumentBuffer,
                                          .argumentOffset = argumentOffset,
                                          .countOffset = countOffset,
                                      })
                    .has_value());
    CHECK_FALSE(device.dispatchIndirect(*command, {.argumentBuffer = *output}).has_value());
    CHECK_FALSE(device
                    .dispatchIndirect(*command,
                                      {
                                          .argumentBuffer = *argumentBuffer,
                                          .argumentOffset = argumentBufferSize - sizeof(std::uint32_t),
                                      })
                    .has_value());
    CHECK_FALSE(device
                    .dispatchIndirect(*command,
                                      {
                                          .argumentBuffer = *argumentBuffer,
                                          .argumentOffset = argumentOffset,
                                          .maxDispatchCount = dispatchCount,
                                          .stride = argumentStride + 4,
                                      })
                    .has_value());
    CHECK_FALSE(device
                    .dispatchIndirect(*command,
                                      {
                                          .argumentBuffer = *argumentBuffer,
                                          .argumentOffset = argumentOffset,
                                          .countBuffer = *output,
                                      })
                    .has_value());
    CHECK_FALSE(device
                    .dispatchIndirect(*command,
                                      {
                                          .argumentBuffer = *argumentBuffer,
                                          .argumentOffset = argumentOffset,
                                          .countBuffer = *countBuffer,
                                          .countOffset = countBytes.size(),
                                      })
                    .has_value());
    CHECK_FALSE(device
                    .dispatchIndirect(*command,
                                      {
                                          .argumentBuffer = *argumentBuffer,
                                          .argumentOffset = argumentOffset,
                                          .countBuffer = staleCount,
                                      })
                    .has_value());
    REQUIRE(device
                .dispatchIndirect(*command,
                                  {
                                      .argumentBuffer = *argumentBuffer,
                                      .argumentOffset = argumentBufferSize,
                                      .maxDispatchCount = 0,
                                      .stride = argumentStride,
                                      .countBuffer = *countBuffer,
                                      .countOffset = countOffset,
                                  })
                .has_value());

    std::uint32_t expected = 0;
    REQUIRE(device.dispatchIndirect(*command, fixedBatch).has_value());
    expected += dispatchCount;
    REQUIRE(
        device
            .bufferBarrier(
                *command, outputBarrier(*output, rachis::PipelineStage::computeShader, rachis::Access::unorderedAccess))
            .has_value());
    REQUIRE(device.dispatchIndirect(*command, countedBatch).has_value());
    expected += gpuCount;
    REQUIRE(
        device
            .bufferBarrier(
                *command, outputBarrier(*output, rachis::PipelineStage::computeShader, rachis::Access::unorderedAccess))
            .has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
    expected += 1;
    REQUIRE(
        device
            .bufferBarrier(
                *command, outputBarrier(*output, rachis::PipelineStage::computeShader, rachis::Access::unorderedAccess))
            .has_value());

    if (conditional)
    {
        for (const std::uint64_t predicateOffset : {std::uint64_t{0}, std::uint64_t{sizeof(std::uint64_t)}})
        {
            REQUIRE(device
                        .beginConditionalExecution(*command,
                                                   {
                                                       .buffer = *predicate,
                                                       .offset = predicateOffset,
                                                       .operation = rachis::PredicateOperation::notEqualZero,
                                                   })
                        .has_value());
            REQUIRE(device.dispatchIndirect(*command, countedBatch).has_value());
            REQUIRE(device.dispatch(*command, 1).has_value());
            REQUIRE(device.endConditionalExecution(*command).has_value());
            REQUIRE(
                device
                    .bufferBarrier(
                        *command,
                        outputBarrier(*output, rachis::PipelineStage::computeShader, rachis::Access::unorderedAccess))
                    .has_value());
        }
        expected += gpuCount + 1;
    }

    REQUIRE(
        device.bufferBarrier(*command, outputBarrier(*output, rachis::PipelineStage::copy, rachis::Access::copySource))
            .has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            {
                                .source = *output,
                                .destination = *readback,
                                .size = outputSize,
                            })
                .has_value());

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipeline(*clearPipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyShaderModule(*clearShader).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyBuffer(*output).has_value());
    REQUIRE(device.destroyBuffer(*argumentBuffer).has_value());
    REQUIRE(device.destroyBuffer(*countBuffer).has_value());
    REQUIRE(device.destroyBuffer(*predicate).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    REQUIRE(mapped->size == outputSize);
    std::uint32_t firstMismatch = valueCount;
    std::uint32_t mismatchValue = 0;
    for (std::uint32_t index = 0; index < valueCount; ++index)
    {
        std::uint32_t value = 0;
        std::memcpy(&value, mapped->data + index * sizeof(value), sizeof(value));
        if (value != expected)
        {
            firstMismatch = index;
            mismatchValue = value;
            break;
        }
    }
    CAPTURE(expected, firstMismatch, mismatchValue);
    CHECK(firstMismatch == valueCount);
    REQUIRE(device.unmapBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan GPU-counted indirect dispatch clamps the count to maxDispatchCount and reads it on the GPU",
          "[size:medium][rhi][vulkan][compute][indirect][batch]")
{
    auto created = rachis::createDevice(rachis::test::vulkanDeviceDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    if (rachis::test::skipUnlessReferenceDesktopAdvertises(
            rachis::test::requireCapabilities(device).dispatchIndirectCount, "dispatchIndirectCount"))
    {
        return;
    }

    const auto countBytecode =
        loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanComputeWrite.countDispatch.spv");
    REQUIRE_FALSE(countBytecode.empty());
    auto shader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::compute,
        .bytecode = countBytecode,
        .entryPoint = "countDispatch",
        .shaderInterface = rachis::test::computeWriteShaderInterface(),
    });
    REQUIRE(shader.has_value());

    constexpr std::uint32_t recordCount = 4;
    std::array<rachis::DispatchIndirectArguments, recordCount> records{};
    records.fill({1, 1, 1});
    auto argumentBuffer = device.createBuffer({
        .size = sizeof(records),
        .usage = rachis::BufferUsage::indirect,
        .memoryType = rachis::MemoryType::upload,
    });
    auto countBuffer = device.createBuffer({
        .size = sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::indirect | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
    });
    auto countSeed = device.createBuffer({
        .size = sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto output = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
    });
    auto outputSeed = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto readback = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(argumentBuffer.has_value());
    REQUIRE(countBuffer.has_value());
    REQUIRE(countSeed.has_value());
    REQUIRE(output.has_value());
    REQUIRE(outputSeed.has_value());
    REQUIRE(readback.has_value());
    const std::array<std::uint32_t, valueCount> zeros{};
    REQUIRE(device.writeBuffer(*outputSeed, 0, std::as_bytes(std::span(zeros))).has_value());
    REQUIRE(device.writeBuffer(*argumentBuffer, 0, std::as_bytes(std::span(records))).has_value());
    constexpr std::uint32_t largeCount = 100;
    REQUIRE(device.writeBuffer(*countSeed, 0, std::as_bytes(std::span(&largeCount, 1))).has_value());

    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = bindingEntries});
    REQUIRE(bindingLayout.has_value());
    const std::array writes{
        rachis::BindingWrite{
            .binding = 0,
            .resource =
                rachis::BufferBinding{
                    .buffer = *output,
                    .size = outputSize,
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device, {.layout = *bindingLayout, .writes = writes});
    REQUIRE(bindingSet.has_value());
    const std::array bindingLayouts{*bindingLayout};
    const std::array pushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)},
    };
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = pushConstantRanges,
    });
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *shader});
    REQUIRE(pipeline.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *countBuffer,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::copyDestination,
                               })
                .has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            {
                                .source = *countSeed,
                                .destination = *countBuffer,
                                .size = sizeof(std::uint32_t),
                            })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *countBuffer,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::indirect,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::indirectArgument,
                               })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *output,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::copyDestination,
                               })
                .has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            {
                                .source = *outputSeed,
                                .destination = *output,
                                .size = outputSize,
                            })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *output,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
    const std::array constants{0U, 1U};
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(constants))).has_value());
    REQUIRE(device
                .dispatchIndirect(*command,
                                  {
                                      .argumentBuffer = *argumentBuffer,
                                      .maxDispatchCount = 3,
                                      .countBuffer = *countBuffer,
                                  })
                .has_value());
    REQUIRE(
        device.bufferBarrier(*command, outputBarrier(*output, rachis::PipelineStage::copy, rachis::Access::copySource))
            .has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            {
                                .source = *output,
                                .destination = *readback,
                                .size = outputSize,
                            })
                .has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    std::array<std::uint32_t, valueCount> values{};
    std::memcpy(values.data(), mapped->data, sizeof(values));
    CHECK(values.front() == 3);
    CHECK(values.back() == 3);
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyBuffer(*argumentBuffer).has_value());
    REQUIRE(device.destroyBuffer(*countBuffer).has_value());
    REQUIRE(device.destroyBuffer(*countSeed).has_value());
    REQUIRE(device.destroyBuffer(*output).has_value());
    REQUIRE(device.destroyBuffer(*outputSeed).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
