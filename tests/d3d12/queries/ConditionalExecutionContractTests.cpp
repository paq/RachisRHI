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
        std::ifstream stream(RACHIS_TEST_CONDITIONAL_EXECUTION_SHADER_PATH, std::ios::binary | std::ios::ate);
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

TEST_CASE("conditional execution gates dispatches, rejects copies, and retains its predicate resource",
          "[size:medium][rhi][query][predicate][compute][warp]")
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
    REQUIRE(rachis::test::requireCapabilities(device).conditionalExecution);

    const std::array<std::uint64_t, 2> predicateValues{0, 1};
    auto predicateSeed = device.createBuffer({
        .size = sizeof(predicateValues),
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::predicate,
        .memoryType = rachis::MemoryType::upload,
        .label = "Conditional-execution predicate seed",
    });
    auto predicate = device.createBuffer({
        .size = sizeof(predicateValues),
        .usage = rachis::BufferUsage::copyDestination | rachis::BufferUsage::predicate,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Conditional-execution predicate",
    });
    auto badUsage = device.createBuffer({
        .size = sizeof(std::uint64_t),
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto stalePredicate = device.createBuffer({
        .size = sizeof(std::uint64_t),
        .usage = rachis::BufferUsage::predicate,
        .memoryType = rachis::MemoryType::upload,
    });
    auto transientPredicate = device.createBuffer({
        .size = sizeof(std::uint64_t),
        .usage = rachis::BufferUsage::predicate,
        .memoryType = rachis::MemoryType::upload,
        .label = "Destroyed active predicate",
    });
    REQUIRE(predicateSeed.has_value());
    REQUIRE(predicate.has_value());
    REQUIRE(badUsage.has_value());
    REQUIRE(stalePredicate.has_value());
    REQUIRE(transientPredicate.has_value());
    REQUIRE(device.writeBuffer(*predicateSeed, 0, std::as_bytes(std::span(predicateValues))).has_value());
    REQUIRE(
        device.writeBuffer(*transientPredicate, 0, std::as_bytes(std::span(predicateValues).first<1>())).has_value());
    const auto stale = *stalePredicate;
    REQUIRE(device.destroyBuffer(*stalePredicate).has_value());

    auto output = device.createBuffer({
        .size = sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::copyDestination | rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Conditional-execution output",
    });
    auto readback = device.createBuffer({
        .size = sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Conditional-execution readback",
    });
    REQUIRE(output.has_value());
    REQUIRE(readback.has_value());

    auto copyReadback = device.createBuffer({
        .size = sizeof(predicateValues),
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Conditional-execution copy readback",
    });
    REQUIRE(copyReadback.has_value());

    const auto bytecode = loadShader();
    REQUIRE_FALSE(bytecode.empty());
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Conditional-execution increment shader",
        .shaderInterface = rachis::test::computeShaderInterface(
            {1, 1, 1},
            std::array{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1}}),
    });
    REQUIRE(shader.has_value());
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
                    .size = sizeof(std::uint32_t),
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device, {.layout = *bindingLayout, .writes = writes});
    REQUIRE(bindingSet.has_value());
    const std::array bindingLayouts{*bindingLayout};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = bindingLayouts});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *shader});
    REQUIRE(pipeline.has_value());

    auto copyCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(copyCommand.has_value());
    CHECK_FALSE(device
                    .beginConditionalExecution(*copyCommand,
                                               {
                                                   .buffer = *predicateSeed,
                                                   .operation = rachis::PredicateOperation::notEqualZero,
                                               })
                    .has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    CHECK_FALSE(device.endConditionalExecution(*command).has_value());
    CHECK_FALSE(device
                    .beginConditionalExecution(*command,
                                               {
                                                   .buffer = *predicate,
                                                   .operation = static_cast<rachis::PredicateOperation>(255),
                                               })
                    .has_value());
    CHECK_FALSE(device.beginConditionalExecution(*command, {.buffer = *badUsage}).has_value());
    CHECK_FALSE(device.beginConditionalExecution(*command, {.buffer = stale}).has_value());
    CHECK_FALSE(device.beginConditionalExecution(*command, {.buffer = *predicate, .offset = 1}).has_value());
    CHECK_FALSE(device.beginConditionalExecution(*command, {.buffer = *predicate, .offset = sizeof(predicateValues)})
                    .has_value());

    REQUIRE(device
                .beginConditionalExecution(*command,
                                           {
                                               .buffer = *transientPredicate,
                                               .operation = rachis::PredicateOperation::notEqualZero,
                                           })
                .has_value());
    CHECK_FALSE(device.beginConditionalExecution(*command, {.buffer = *transientPredicate}).has_value());
    REQUIRE(device.destroyBuffer(*transientPredicate).has_value());
    auto rejectedSubmission = device.submit(std::move(*command));
    REQUIRE_FALSE(rejectedSubmission.has_value());
    CHECK(rejectedSubmission.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(command->isValid());
    REQUIRE(device.endConditionalExecution(*command).has_value());
    CHECK_FALSE(device.endConditionalExecution(*command).has_value());

    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *predicate,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::copyDestination,
                               })
                .has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            {
                                .source = *predicateSeed,
                                .destination = *predicate,
                                .size = sizeof(predicateValues),
                            })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *predicate,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::conditionalExecution,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::predicate,
                               })
                .has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            {
                                .source = *predicateSeed,
                                .destination = *copyReadback,
                                .size = sizeof(predicateValues),
                            })
                .has_value());
    REQUIRE(device
                .beginConditionalExecution(*command,
                                           {
                                               .buffer = *predicate,
                                               .operation = rachis::PredicateOperation::notEqualZero,
                                           })
                .has_value());
    const auto conditionalCopy = device.copyBuffer(*command,
                                                   {
                                                       .source = *predicateSeed,
                                                       .sourceOffset = sizeof(std::uint64_t),
                                                       .destination = *copyReadback,
                                                       .size = sizeof(std::uint64_t),
                                                   });
    REQUIRE_FALSE(conditionalCopy.has_value());
    CHECK(conditionalCopy.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.endConditionalExecution(*command).has_value());
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
                                .source = *predicateSeed,
                                .destination = *output,
                                .size = sizeof(std::uint32_t),
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

    const auto conditionalDispatch = [&](std::uint64_t offset, rachis::PredicateOperation operation) {
        REQUIRE(
            device.beginConditionalExecution(*command, {.buffer = *predicate, .offset = offset, .operation = operation})
                .has_value());
        REQUIRE(device.dispatch(*command, 1).has_value());
        REQUIRE(device.endConditionalExecution(*command).has_value());
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
    };
    conditionalDispatch(0, rachis::PredicateOperation::notEqualZero);
    conditionalDispatch(0, rachis::PredicateOperation::equalZero);
    conditionalDispatch(sizeof(std::uint64_t), rachis::PredicateOperation::equalZero);
    conditionalDispatch(sizeof(std::uint64_t), rachis::PredicateOperation::notEqualZero);

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
                                .size = sizeof(std::uint32_t),
                            })
                .has_value());

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyBuffer(*predicate).has_value());
    REQUIRE(device.destroyBuffer(*predicateSeed).has_value());
    REQUIRE(device.destroyBuffer(*badUsage).has_value());
    REQUIRE(device.destroyBuffer(*output).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    std::uint32_t dispatchCount = 0;
    std::memcpy(&dispatchCount, mapped->data, sizeof(dispatchCount));
    CHECK(dispatchCount == 2);
    REQUIRE(device.unmapBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    auto copyMapped = device.mapBuffer(*copyReadback);
    REQUIRE(copyMapped.has_value());
    std::array<std::uint64_t, 2> copyResults{};
    std::memcpy(copyResults.data(), copyMapped->data, sizeof(copyResults));
    CHECK(copyResults == predicateValues);
    REQUIRE(device.unmapBuffer(*copyReadback).has_value());
    REQUIRE(device.destroyBuffer(*copyReadback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
