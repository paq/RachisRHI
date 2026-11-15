#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace
{

    constexpr std::uint32_t invocationCount = 64;
    constexpr std::uint32_t valuesPerInvocation = 4;
    constexpr std::size_t outputValueCount = std::size_t{invocationCount} * valuesPerInvocation;
    constexpr std::uint64_t outputSize = static_cast<std::uint64_t>(outputValueCount) * sizeof(std::uint32_t);

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
        std::ifstream stream(RACHIS_TEST_SUBGROUP_OPERATIONS_SHADER_PATH, std::ios::binary | std::ios::ate);
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

TEST_CASE("reported shader subgroup operations execute wave queries and reductions",
          "[size:medium][rhi][compute][subgroup][d3d12]")
{
    auto adapters = rachis::enumerateAdapters();
    if (!adapters)
    {
        INFO(adapters.error().message);
    }
    REQUIRE(adapters.has_value());

    std::optional<rachis::Device> selectedDevice;
    std::string selectedAdapter;
    for (const auto& adapter : *adapters)
    {
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .adapterPreference = adapter.type == rachis::AdapterType::software
                                     ? rachis::AdapterPreference::warp
                                     : rachis::AdapterPreference::highPerformance,
            .adapterId = adapter.id,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = true,
            .diagnosticCallback = &printDiagnostic,
        });
        if (created && rachis::test::requireCapabilities(*created).subgroupOperations)
        {
            selectedAdapter = adapter.name;
            selectedDevice.emplace(std::move(*created));
            break;
        }
    }
    if (!selectedDevice)
    {
        SKIP("No D3D12 adapter reports Shader Model 6.0 subgroup operations.");
    }
    INFO("Selected adapter: " << selectedAdapter);
    auto& device = *selectedDevice;
    const auto subgroupCapabilities = rachis::test::requireCapabilities(device).subgroup;
    REQUIRE(rachis::test::requireCapabilities(device).shaderModel >= rachis::ApiVersion{6, 0});
    REQUIRE(subgroupCapabilities.minSize > 0);
    REQUIRE(subgroupCapabilities.minSize <= subgroupCapabilities.maxSize);

    const auto bytecode = loadShader();
    REQUIRE_FALSE(bytecode.empty());
    auto shader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Subgroup operations shader",
        .shaderInterface = rachis::test::computeShaderInterface(
            {64, 1, 1},
            std::array{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1}}),
    });
    REQUIRE(shader.has_value());

    auto storage = device.createBuffer(rachis::BufferDesc{
        .size = outputSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Subgroup output",
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = outputSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Subgroup readback",
    });
    REQUIRE(storage.has_value());
    REQUIRE(readback.has_value());

    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout(rachis::BindingLayoutDesc{
        .entries = bindingEntries,
        .label = "Subgroup binding layout",
    });
    REQUIRE(bindingLayout.has_value());
    const std::array writes{
        rachis::BindingWrite{
            .binding = 0,
            .resource =
                rachis::BufferBinding{
                    .buffer = *storage,
                    .size = outputSize,
                    .structureByteStride = valuesPerInvocation * sizeof(std::uint32_t),
                },
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device,
                                                     rachis::BindingSetDesc{
                                                         .layout = *bindingLayout,
                                                         .writes = writes,
                                                         .label = "Subgroup binding set",
                                                     });
    REQUIRE(bindingSet.has_value());

    const std::array bindingLayouts{*bindingLayout};
    auto pipelineLayout = device.createPipelineLayout(rachis::PipelineLayoutDesc{
        .bindingLayouts = bindingLayouts,
        .label = "Subgroup pipeline layout",
    });
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline(rachis::ComputePipelineDesc{
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Subgroup compute pipeline",
    });
    REQUIRE(pipeline.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
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
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
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

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    REQUIRE(mapped->size == outputSize);
    std::array<std::uint32_t, outputValueCount> output{};
    std::memcpy(output.data(), mapped->data, outputSize);
    for (std::uint32_t invocation = 0; invocation < invocationCount; ++invocation)
    {
        const auto offset = std::size_t{invocation} * valuesPerInvocation;
        const auto laneCount = output[offset];
        const auto laneIndex = output[offset + 1];
        const auto activeCount = output[offset + 2];
        const auto activeIndexSum = output[offset + 3];
        CAPTURE(invocation, laneCount, laneIndex, activeCount, activeIndexSum);
        CHECK(laneCount >= subgroupCapabilities.minSize);
        CHECK(laneCount <= subgroupCapabilities.maxSize);
        CHECK(laneIndex < laneCount);
        CHECK(activeCount > 0);
        CHECK(activeCount <= laneCount);
        CHECK(activeIndexSum == activeCount * (activeCount - 1) / 2);
    }
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyBuffer(*storage).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
