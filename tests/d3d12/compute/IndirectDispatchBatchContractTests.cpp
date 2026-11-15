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

    constexpr std::uint32_t valueCount = 64;
    constexpr std::uint64_t outputSize = valueCount * sizeof(std::uint32_t);
    constexpr std::uint64_t argumentOffset = sizeof(std::uint32_t);
    constexpr std::uint32_t argumentStride = 16;
    constexpr std::uint32_t dispatchCount = 3;
    constexpr std::uint64_t argumentBufferSize = argumentOffset +
                                                 static_cast<std::uint64_t>(dispatchCount - 1) * argumentStride +
                                                 sizeof(rachis::DispatchIndirectArguments);
    constexpr std::uint64_t countOffset = sizeof(std::uint32_t);

    class FailResourceRetention final
    {
    public:
        explicit FailResourceRetention(std::int32_t countdown)
            : previous_(rachis::d3d12::test::resourceRetentionFailureCountdown().exchange(countdown,
                                                                                          std::memory_order_relaxed))
        {
        }

        ~FailResourceRetention()
        {
            rachis::d3d12::test::resourceRetentionFailureCountdown().store(previous_, std::memory_order_relaxed);
        }

        FailResourceRetention(const FailResourceRetention&) = delete;
        FailResourceRetention& operator=(const FailResourceRetention&) = delete;
        FailResourceRetention(FailResourceRetention&&) = delete;
        FailResourceRetention& operator=(FailResourceRetention&&) = delete;

    private:
        std::int32_t previous_;
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

} // namespace

TEST_CASE("indirect compute batches execute fixed and GPU-counted records atomically",
          "[size:medium][rhi][compute][indirect][batch][warp]")
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
    REQUIRE(rachis::test::requireCapabilities(device).multiDispatchIndirect);
    REQUIRE(rachis::test::requireCapabilities(device).dispatchIndirectCount);

    const auto bytecode = loadShader(RACHIS_TEST_INDIRECT_DISPATCH_BATCH_SHADER_PATH);
    const auto clearBytecode = loadShader(RACHIS_TEST_COMPUTE_SHADER_PATH);
    REQUIRE_FALSE(bytecode.empty());
    REQUIRE_FALSE(clearBytecode.empty());
    auto shader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "countDispatch",
        .label = "Indirect dispatch batch counter",
        .shaderInterface = rachis::test::computeWriteShaderInterface(),
    });
    REQUIRE(shader.has_value());
    auto clearShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::compute,
        .bytecode = clearBytecode,
        .entryPoint = "main",
        .label = "Indirect dispatch batch clear",
        .shaderInterface = rachis::test::computeWriteShaderInterface(),
    });
    REQUIRE(clearShader.has_value());

    auto output = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Indirect dispatch batch output",
    });
    auto readback = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Indirect dispatch batch readback",
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
        .label = "Padded indirect dispatch records",
    });
    REQUIRE(argumentBuffer.has_value());
    REQUIRE(device.writeBuffer(*argumentBuffer, 0, argumentBytes).has_value());

    std::array<std::byte, countOffset + sizeof(std::uint32_t)> countBytes{};
    constexpr std::uint32_t gpuCount = 2;
    std::memcpy(countBytes.data() + countOffset, &gpuCount, sizeof(gpuCount));
    auto countBuffer = device.createBuffer({
        .size = countBytes.size(),
        .usage = rachis::BufferUsage::indirect,
        .memoryType = rachis::MemoryType::upload,
        .label = "Indirect dispatch GPU count",
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

    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({
        .entries = bindingEntries,
        .label = "Indirect dispatch batch layout",
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
                                                         .label = "Indirect dispatch batch set",
                                                     });
    REQUIRE(bindingSet.has_value());
    const std::array bindingLayouts{*bindingLayout};
    const std::array pushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)},
    };
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = pushConstantRanges,
        .label = "Indirect dispatch batch pipeline layout",
    });
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Indirect dispatch batch pipeline",
    });
    REQUIRE(pipeline.has_value());
    auto clearPipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *clearShader,
        .label = "Indirect dispatch batch clear pipeline",
    });
    REQUIRE(clearPipeline.has_value());

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
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *output,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
    const std::array countConstants{0U, 1U};
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(countConstants))).has_value());

    const rachis::IndirectDispatchDesc fixedBatch{
        .argumentBuffer = *argumentBuffer,
        .argumentOffset = argumentOffset,
        .maxDispatchCount = dispatchCount,
        .stride = argumentStride,
    };
    CHECK_FALSE(device.dispatchIndirect(*command, rachis::IndirectDispatchDesc{}).has_value());
    CHECK_FALSE(device
                    .dispatchIndirect(*command,
                                      {
                                          .argumentBuffer = *argumentBuffer,
                                          .argumentOffset = 2,
                                      })
                    .has_value());
    CHECK_FALSE(device
                    .dispatchIndirect(*command,
                                      {
                                          .argumentBuffer = *argumentBuffer,
                                          .argumentOffset = argumentOffset,
                                          .stride = 8,
                                      })
                    .has_value());
    CHECK_FALSE(device
                    .dispatchIndirect(*command,
                                      {
                                          .argumentBuffer = *argumentBuffer,
                                          .argumentOffset = argumentOffset,
                                          .stride = 14,
                                      })
                    .has_value());
    CHECK_FALSE(device
                    .dispatchIndirect(*command,
                                      {
                                          .argumentBuffer = *argumentBuffer,
                                          .argumentOffset = argumentOffset,
                                          .countOffset = countOffset,
                                      })
                    .has_value());
    CHECK_FALSE(device
                    .dispatchIndirect(*command,
                                      {
                                          .argumentBuffer = *output,
                                      })
                    .has_value());
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

    const auto beforeZeroCount = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(beforeZeroCount.has_value());
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
    const auto afterZeroCount = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(afterZeroCount.has_value());
    CHECK(*afterZeroCount == *beforeZeroCount);

    const rachis::IndirectDispatchDesc countedBatch{
        .argumentBuffer = *argumentBuffer,
        .argumentOffset = argumentOffset,
        .maxDispatchCount = dispatchCount,
        .stride = argumentStride,
        .countBuffer = *countBuffer,
        .countOffset = countOffset,
    };
    auto unretainedPool = device.createCommandPool({
        .queueType = rachis::QueueType::compute,
        .maxCommandBuffers = 1,
        .label = "Unretained indirect dispatch pool",
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
    const auto beforeUnretainedPublication = rachis::d3d12::test::commandDependencySnapshot(*unretainedCommand);
    REQUIRE(beforeUnretainedPublication.has_value());
    const auto beforeUnretainedDependencies =
        beforeUnretainedPublication.value_or(rachis::d3d12::test::CommandDependencySnapshot{});
    REQUIRE(device.dispatchIndirect(*unretainedCommand, countedBatch).has_value());
    const auto afterUnretainedPublication = rachis::d3d12::test::commandDependencySnapshot(*unretainedCommand);
    REQUIRE(afterUnretainedPublication.has_value());
    const auto afterUnretainedDependencies =
        afterUnretainedPublication.value_or(rachis::d3d12::test::CommandDependencySnapshot{});
    CHECK(afterUnretainedDependencies.resourceOwners == beforeUnretainedDependencies.resourceOwners);
    CHECK(afterUnretainedDependencies.residencyLeases == beforeUnretainedDependencies.residencyLeases);
    CHECK(afterUnretainedDependencies.swapchainReferences == beforeUnretainedDependencies.swapchainReferences);
    CHECK(afterUnretainedDependencies.sparseMappingDependencies ==
          beforeUnretainedDependencies.sparseMappingDependencies);
    REQUIRE(device
                .bufferBarrier(*unretainedCommand,
                               {
                                   .buffer = *output,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::all,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::common,
                               })
                .has_value());
    auto unretainedSubmission = device.submit(std::move(*unretainedCommand));
    REQUIRE(unretainedSubmission.has_value());
    REQUIRE(device.wait(*unretainedSubmission).has_value());
    device.collectGarbage();
    REQUIRE(device.destroyCommandPool(*unretainedPool).has_value());

    const auto beforeRejectedArgumentPublication = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(beforeRejectedArgumentPublication.has_value());
    {
        const FailResourceRetention failArgumentPublication(1);
        const auto rejected = device.dispatchIndirect(*command, countedBatch);
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::outOfMemory);
    }
    const auto afterRejectedArgumentPublication = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(afterRejectedArgumentPublication.has_value());
    CHECK(*afterRejectedArgumentPublication == *beforeRejectedArgumentPublication);

    const auto beforeRejectedPublication = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(beforeRejectedPublication.has_value());
    {
        const FailResourceRetention failCountPublication(2);
        const auto rejected = device.dispatchIndirect(*command, countedBatch);
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::outOfMemory);
    }
    const auto afterRejectedPublication = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(afterRejectedPublication.has_value());
    CHECK(*afterRejectedPublication == *beforeRejectedPublication);

    REQUIRE(device.dispatchIndirect(*command, fixedBatch).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *output,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.dispatchIndirect(*command, countedBatch).has_value());
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
        if (value != dispatchCount + gpuCount)
        {
            firstMismatch = index;
            break;
        }
    }
    CAPTURE(firstMismatch);
    CHECK(firstMismatch == valueCount);
    REQUIRE(device.unmapBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
