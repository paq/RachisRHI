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
#include <string_view>
#include <vector>

namespace
{

    constexpr std::uint64_t atomicSum = (64U * 65U) / 2U;

    void printDiagnostic(const rachis::DiagnosticMessage& message, void*) noexcept
    {
        static_cast<void>(std::fprintf(stderr, "D3D12 [%u]: ", static_cast<unsigned int>(message.nativeId)));
        if (!message.description.empty())
        {
            static_cast<void>(std::fwrite(message.description.data(), 1, message.description.size(), stderr));
        }
        static_cast<void>(std::fputc('\n', stderr));
    }

    [[nodiscard]] std::vector<std::byte> loadShader(std::string_view path)
    {
        std::ifstream stream(std::string{path}, std::ios::binary | std::ios::ate);
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

    struct SelectedDevice final
    {
        rachis::Device device;
        std::string adapterName;
    };

    using CapabilityMember = bool rachis::DeviceCapabilities::*;

    [[nodiscard]] std::optional<SelectedDevice> selectDevice(CapabilityMember capability)
    {
        auto adapters = rachis::enumerateAdapters();
        if (!adapters)
        {
            INFO(adapters.error().message);
            return std::nullopt;
        }
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
            if (created)
            {
                const auto capabilities = created->capabilities();
                if (capabilities.has_value() && capabilities.value().*capability)
                {
                    return SelectedDevice{std::move(*created), adapter.name};
                }
            }
        }
        return std::nullopt;
    }

    void verifyDiagnostics(const rachis::Device& device)
    {
        const auto diagnostics = device.diagnosticCounts();
        CHECK(diagnostics.warnings == 0);
        CHECK(diagnostics.errors == 0);
        CHECK(diagnostics.corruptions == 0);
    }

} // namespace

TEST_CASE("reported group-shared int64 atomics accumulate every compute lane",
          "[size:medium][rhi][compute][shader64][atomic][group-shared][hardware]")
{
    auto selected = selectDevice(&rachis::DeviceCapabilities::shaderInt64AtomicGroupShared);
    if (!selected)
    {
        SKIP("No D3D12 adapter reports group-shared 64-bit integer atomics.");
    }
    INFO("Selected adapter: " << selected->adapterName);
    auto& device = selected->device;
    REQUIRE(rachis::test::requireCapabilities(device).shaderInt64);
    REQUIRE(rachis::test::requireCapabilities(device).shaderModel >= rachis::ApiVersion{6, 6});

    const auto bytecode = loadShader(RACHIS_TEST_SHADER_INT64_ATOMIC_GROUP_SHARED_PATH);
    REQUIRE_FALSE(bytecode.empty());
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Group-shared int64 atomic shader",
        .shaderInterface = rachis::test::computeShaderInterface(
            {64, 1, 1},
            std::array{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1}}),
    });
    REQUIRE(shader.has_value());

    constexpr std::uint64_t outputSize = 2 * sizeof(std::uint32_t);
    auto output = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Group-shared int64 atomic output",
    });
    auto readback = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Group-shared int64 atomic readback",
    });
    REQUIRE(output.has_value());
    REQUIRE(readback.has_value());

    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = entries, .label = "Group-shared atomic layout"});
    REQUIRE(bindingLayout.has_value());
    const std::array writes{
        rachis::BindingWrite{
            .binding = 0,
            .resource = rachis::BufferBinding{.buffer = *output, .size = outputSize, .structureByteStride = 4},
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device,
                                                     {
                                                         .layout = *bindingLayout,
                                                         .writes = writes,
                                                         .label = "Group-shared atomic bindings",
                                                     });
    REQUIRE(bindingSet.has_value());
    const std::array layouts{*bindingLayout};
    auto pipelineLayout =
        device.createPipelineLayout({.bindingLayouts = layouts, .label = "Group-shared atomic pipeline layout"});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Group-shared atomic pipeline",
    });
    REQUIRE(pipeline.has_value());

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
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
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
    REQUIRE(device.copyBuffer(*command, {.source = *output, .destination = *readback, .size = outputSize}).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    std::array<std::uint32_t, 2> result{};
    std::memcpy(result.data(), mapped->data, outputSize);
    CHECK(result == std::array<std::uint32_t, 2>{static_cast<std::uint32_t>(atomicSum), 0});
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyBuffer(*output).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    verifyDiagnostics(device);
}

TEST_CASE("reported typed-resource int64 atomics accumulate every compute lane",
          "[size:medium][rhi][compute][shader64][atomic][typed-resource][hardware]")
{
    auto selected = selectDevice(&rachis::DeviceCapabilities::shaderInt64AtomicTypedResource);
    if (!selected)
    {
        SKIP("No D3D12 adapter reports typed-resource 64-bit integer atomics.");
    }
    INFO("Selected adapter: " << selected->adapterName);
    auto& device = selected->device;
    REQUIRE(rachis::test::requireCapabilities(device).shaderInt64);
    REQUIRE(rachis::test::requireCapabilities(device).shaderModel >= rachis::ApiVersion{6, 6});

    const auto bytecode = loadShader(RACHIS_TEST_SHADER_INT64_ATOMIC_TYPED_RESOURCE_PATH);
    REQUIRE_FALSE(bytecode.empty());
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Typed-resource int64 atomic shader",
        .shaderInterface = rachis::test::computeShaderInterface(
            {64, 1, 1},
            std::array{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageTexture, 1}}),
    });
    REQUIRE(shader.has_value());

    auto output = device.createTexture({
        .format = rachis::Format::rg32Uint,
        .usage = rachis::TextureUsage::storage | rachis::TextureUsage::copySource,
        .label = "Typed-resource int64 atomic output",
    });
    REQUIRE(output.has_value());
    auto outputView = device.createTextureView({
        .texture = *output,
        .usage = rachis::TextureViewUsage::storage,
        .label = "Typed-resource int64 atomic UAV",
    });
    REQUIRE(outputView.has_value());
    auto transfer = device.queryTextureTransferLayout(
        *output, {.textureSubresource = {.aspect = rachis::TextureAspect::color}, .extent = {1, 1, 1}});
    REQUIRE(transfer.has_value());
    auto readback = device.createBuffer({
        .size = transfer->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Typed-resource int64 atomic readback",
    });
    REQUIRE(readback.has_value());

    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageTexture, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = entries, .label = "Typed-resource atomic layout"});
    REQUIRE(bindingLayout.has_value());
    const std::array writes{rachis::BindingWrite{.binding = 0, .resource = *outputView}};
    auto bindingSet = rachis::test::createBindingSet(device,
                                                     {
                                                         .layout = *bindingLayout,
                                                         .writes = writes,
                                                         .label = "Typed-resource atomic bindings",
                                                     });
    REQUIRE(bindingSet.has_value());
    const std::array layouts{*bindingLayout};
    auto pipelineLayout =
        device.createPipelineLayout({.bindingLayouts = layouts, .label = "Typed-resource atomic pipeline layout"});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Typed-resource atomic pipeline",
    });
    REQUIRE(pipeline.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *output,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::computeShader,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::unorderedAccess,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::unorderedAccess,
                                })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *output,
                                    .stagesBefore = rachis::PipelineStage::computeShader,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::unorderedAccess,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::unorderedAccess,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    const rachis::BufferTextureCopyRegion copy{
        .bytesPerRow = transfer->bytesPerRow,
        .rowsPerImage = transfer->rowsPerImage,
        .textureSubresource = {.aspect = rachis::TextureAspect::color},
        .extent = {1, 1, 1},
    };
    REQUIRE(device.copyTextureToBuffer(*command, *output, *readback, std::span{&copy, 1}).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    std::uint64_t result = 0;
    std::memcpy(&result, mapped->data, sizeof(result));
    CHECK(result == atomicSum);
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyTextureView(*outputView).has_value());
    REQUIRE(device.destroyTexture(*output).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    verifyDiagnostics(device);
}
