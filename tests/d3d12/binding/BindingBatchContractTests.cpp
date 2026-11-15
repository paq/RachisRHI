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

    constexpr std::uint64_t bindingRangeSize = 256;
    constexpr std::uint64_t bufferSize = 2 * bindingRangeSize;
    constexpr std::uint32_t valueCount = 64;

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

    [[nodiscard]] std::vector<std::byte> loadShader()
    {
        std::ifstream stream(RACHIS_TEST_BINDING_BATCH_SHADER_PATH, std::ios::binary | std::ios::ate);
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

    void writeRanges(rachis::Device& device,
                     rachis::BufferHandle buffer,
                     std::span<const std::uint32_t> first,
                     std::span<const std::uint32_t> second)
    {
        auto mapped = device.mapBuffer(buffer);
        REQUIRE(mapped.has_value());
        std::memcpy(mapped->data, first.data(), first.size_bytes());
        std::memcpy(mapped->data + bindingRangeSize, second.data(), second.size_bytes());
        REQUIRE(device.unmapBuffer(buffer).has_value());
    }

    void verifyRange(const std::byte* values, std::uint32_t range, std::uint32_t multiplier, std::uint32_t addend)
    {
        std::uint32_t firstMismatch = valueCount;
        for (std::uint32_t index = 0; index < valueCount; ++index)
        {
            std::uint32_t value = 0;
            const auto byteOffset = static_cast<std::uint64_t>(range) * bindingRangeSize + index * sizeof(value);
            std::memcpy(&value, values + byteOffset, sizeof(value));
            if (value != index * multiplier + addend)
            {
                firstMismatch = index;
                break;
            }
        }
        CAPTURE(range, firstMismatch);
        CHECK(firstMismatch == valueCount);
    }

    void verifyZeroRange(const std::byte* values, std::uint32_t range)
    {
        verifyRange(values, range, 0, 0);
    }

} // namespace

TEST_CASE("binding-set batches validate atomically and retain every dynamic descriptor lease",
          "[size:medium][rhi][binding][compute][warp]")
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

    const auto bytecode = loadShader();
    REQUIRE_FALSE(bytecode.empty());
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .label = "Binding batch shader",
        .shaderInterface = rachis::test::computeShaderInterface(
            {64, 1, 1},
            std::array{
                rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1},
                rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
                rachis::ShaderBindingRequirement{1, 0, rachis::ShaderResourceType::uniformBuffer, 1},
            }),
    });
    auto coefficients = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Binding batch coefficients",
    });
    auto biases = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Binding batch biases",
    });
    auto storage = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .label = "Binding batch storage",
    });
    auto zeroUpload = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Binding batch zero upload",
    });
    auto atomicReadback = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Failed binding batch readback",
    });
    auto finalReadback = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Successful binding batch readback",
    });
    REQUIRE(shader.has_value());
    REQUIRE(coefficients.has_value());
    REQUIRE(biases.has_value());
    REQUIRE(storage.has_value());
    REQUIRE(zeroUpload.has_value());
    REQUIRE(atomicReadback.has_value());
    REQUIRE(finalReadback.has_value());

    const std::array firstCoefficients{3U, 7U};
    const std::array secondCoefficients{5U, 11U};
    const std::array firstBias{13U};
    const std::array secondBias{17U};
    writeRanges(device, *coefficients, firstCoefficients, secondCoefficients);
    writeRanges(device, *biases, firstBias, secondBias);
    auto zeroMapped = device.mapBuffer(*zeroUpload);
    REQUIRE(zeroMapped.has_value());
    std::memset(zeroMapped->data, 0, static_cast<std::size_t>(bufferSize));
    REQUIRE(device.unmapBuffer(*zeroUpload).has_value());

    const std::array firstLayoutEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::dynamicUniformBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::dynamicStorageBuffer, rachis::ShaderStage::compute, 1},
    };
    const std::array secondLayoutEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::dynamicUniformBuffer, rachis::ShaderStage::compute, 1},
    };
    auto firstLayout = device.createBindingLayout({.entries = firstLayoutEntries, .label = "Batch set zero layout"});
    auto secondLayout = device.createBindingLayout({.entries = secondLayoutEntries, .label = "Batch set one layout"});
    REQUIRE(firstLayout.has_value());
    REQUIRE(secondLayout.has_value());

    const std::array firstWrites{
        rachis::BindingWrite{
            .binding = 0,
            .resource = rachis::BufferBinding{.buffer = *coefficients, .size = bindingRangeSize},
        },
        rachis::BindingWrite{
            .binding = 1,
            .resource =
                rachis::BufferBinding{
                    .buffer = *storage,
                    .size = bindingRangeSize,
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
    };
    const std::array secondWrites{
        rachis::BindingWrite{
            .binding = 0,
            .resource = rachis::BufferBinding{.buffer = *biases, .size = bindingRangeSize},
        },
    };
    auto firstSet = rachis::test::createBindingSet(
        device, {.layout = *firstLayout, .writes = firstWrites, .label = "Batch set zero"});
    auto secondSet = rachis::test::createBindingSet(
        device, {.layout = *secondLayout, .writes = secondWrites, .label = "Batch set one"});
    REQUIRE(firstSet.has_value());
    REQUIRE(secondSet.has_value());

    const std::array layouts{*firstLayout, *secondLayout};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts, .label = "Binding batch layout"});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Binding batch pipeline",
    });
    REQUIRE(pipeline.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *storage,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::copyDestination,
                               })
                .has_value());
    REQUIRE(
        device.copyBuffer(*command, {.source = *zeroUpload, .destination = *storage, .size = bufferSize}).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *storage,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());

    const std::array sets{*firstSet, *secondSet};
    const std::array firstOffsets{0U, 0U, 0U};
    REQUIRE(device.bindComputeBindingSets(*command, 0, sets, firstOffsets).has_value());
    CHECK_FALSE(device.bindComputeBindingSets(*command, 0, {}).has_value());
    CHECK_FALSE(device.bindComputeBindingSets(*command, 1, sets, firstOffsets).has_value());
    CHECK_FALSE(device.bindComputeBindingSets(*command, 0, sets, std::span{firstOffsets}.first(2)).has_value());

    const std::array mismatchedSets{*firstSet, *firstSet};
    const std::array secondOffsets{256U, 256U, 256U};
    CHECK_FALSE(device.bindComputeBindingSets(*command, 0, mismatchedSets, secondOffsets).has_value());
    const auto beforeRejectedPublication = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(beforeRejectedPublication.has_value());
    {
        const FailResourceRetention failInternalOwnerPublication(2);
        auto failedPublication = device.bindComputeBindingSets(*command, 0, sets, secondOffsets);
        REQUIRE_FALSE(failedPublication.has_value());
        CHECK(failedPublication.error().code == rachis::ErrorCode::outOfMemory);
    }
    const auto afterRejectedPublication = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(afterRejectedPublication.has_value());
    CHECK(*afterRejectedPublication == *beforeRejectedPublication);
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
    REQUIRE(device.copyBuffer(*command, {.source = *storage, .destination = *atomicReadback, .size = bufferSize})
                .has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *storage,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::copySource,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());

    REQUIRE(device.bindComputeBindingSets(*command, 0, sets, secondOffsets).has_value());
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
    REQUIRE(device.copyBuffer(*command, {.source = *storage, .destination = *finalReadback, .size = bufferSize})
                .has_value());

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBindingSet(*firstSet).has_value());
    REQUIRE(device.destroyBindingSet(*secondSet).has_value());
    REQUIRE(device.destroyBindingLayout(*firstLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*secondLayout).has_value());
    REQUIRE(device.destroyBuffer(*coefficients).has_value());
    REQUIRE(device.destroyBuffer(*biases).has_value());
    REQUIRE(device.destroyBuffer(*storage).has_value());
    REQUIRE(device.destroyBuffer(*zeroUpload).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    auto atomicMapped = device.mapBuffer(*atomicReadback);
    REQUIRE(atomicMapped.has_value());
    verifyRange(atomicMapped->data, 0, 3, 20);
    verifyZeroRange(atomicMapped->data, 1);
    REQUIRE(device.unmapBuffer(*atomicReadback).has_value());

    auto finalMapped = device.mapBuffer(*finalReadback);
    REQUIRE(finalMapped.has_value());
    verifyRange(finalMapped->data, 0, 3, 20);
    verifyRange(finalMapped->data, 1, 5, 28);
    REQUIRE(device.unmapBuffer(*finalReadback).has_value());
    REQUIRE(device.destroyBuffer(*atomicReadback).has_value());
    REQUIRE(device.destroyBuffer(*finalReadback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
