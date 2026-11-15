#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

    constexpr std::uint64_t bindingRangeSize = 256;
    constexpr std::uint32_t valueCount = 64;

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
        std::ifstream stream(RACHIS_TEST_DYNAMIC_BINDING_SHADER_PATH, std::ios::binary | std::ios::ate);
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

    void writeConstants(rachis::Device& device,
                        rachis::BufferHandle buffer,
                        std::uint32_t multiplier,
                        std::uint32_t addend)
    {
        auto mapped = device.mapBuffer(buffer);
        REQUIRE(mapped.has_value());
        const std::array values{multiplier, addend};
        std::memcpy(mapped->data, values.data(), sizeof(values));
        REQUIRE(device.unmapBuffer(buffer).has_value());
    }

    void verifyRange(const std::byte* values, std::uint64_t baseOffset, std::uint32_t multiplier, std::uint32_t addend)
    {
        std::uint32_t firstMismatch = valueCount;
        for (std::uint32_t index = 0; index < valueCount; ++index)
        {
            std::uint32_t value = 0;
            std::memcpy(&value, values + baseOffset + index * sizeof(value), sizeof(value));
            if (value != index * multiplier + addend)
            {
                firstMismatch = index;
                break;
            }
        }
        CAPTURE(baseOffset, firstMismatch);
        CHECK(firstMismatch == valueCount);
    }

    [[nodiscard]] rachis::Device createWarpDevice()
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
        return std::move(*created);
    }

    struct SampledView final
    {
        rachis::TextureHandle texture = {};
        rachis::TextureViewHandle view = {};
    };

    [[nodiscard]] SampledView createSampledView(rachis::Device& device, std::string_view label)
    {
        auto texture = device.createTexture({
            .width = 4,
            .height = 4,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::sampled,
            .label = label,
        });
        REQUIRE(texture.has_value());
        auto view = device.createTextureView({
            .texture = *texture,
            .usage = rachis::TextureViewUsage::sampled,
            .dimension = rachis::TextureViewDimension::texture2D,
            .label = label,
        });
        REQUIRE(view.has_value());
        return SampledView{.texture = *texture, .view = *view};
    }

    void destroySampledView(rachis::Device& device, const SampledView& sampled)
    {
        REQUIRE(device.destroyTextureView(sampled.view).has_value());
        REQUIRE(device.destroyTexture(sampled.texture).has_value());
    }

    [[nodiscard]] std::vector<rachis::BindingWrite> repeatedWrites(std::uint32_t count,
                                                                   const rachis::BindingResource& resource)
    {
        std::vector<rachis::BindingWrite> writes;
        writes.reserve(count);
        for (std::uint32_t element = 0; element < count; ++element)
        {
            writes.push_back(rachis::BindingWrite{0, element, resource});
        }
        return writes;
    }

    [[nodiscard]] rachis::BindingPoolHandle createSingleTypePool(rachis::Device& device,
                                                                 std::uint32_t maxBindingSets,
                                                                 rachis::BindingType type,
                                                                 std::uint32_t descriptorCount)
    {
        const std::array sizes{rachis::BindingPoolSize{type, descriptorCount}};
        auto pool = device.createBindingPool({
            .maxBindingSets = maxBindingSets,
            .sizes = sizes,
            .label = "Single-type binding pool",
        });
        const std::string failure = pool ? std::string{} : pool.error().message;
        CAPTURE(failure);
        REQUIRE(pool.has_value());
        return *pool;
    }

} // namespace

TEST_CASE("binding set updates publish copy-on-write descriptor versions", "[size:medium][rhi][binding][compute][warp]")
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

    const auto bytecode = loadShader();
    REQUIRE_FALSE(bytecode.empty());
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Binding update shader",
        .shaderInterface = rachis::test::computeShaderInterface(
            {64, 1, 1},
            std::array{
                rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1},
                rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
            }),
    });
    auto constantsA = device.createBuffer({
        .size = bindingRangeSize,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Binding update constants A",
    });
    auto constantsB = device.createBuffer({
        .size = bindingRangeSize,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Binding update constants B",
    });
    auto outputA = device.createBuffer({
        .size = bindingRangeSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .label = "Binding update output A",
    });
    auto outputB = device.createBuffer({
        .size = bindingRangeSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .label = "Binding update output B",
    });
    auto readback = device.createBuffer({
        .size = 2 * bindingRangeSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Binding update readback",
    });
    auto texture = device.createTexture({
        .width = 1,
        .height = 1,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled,
        .label = "Binding update retained texture",
    });
    REQUIRE(shader.has_value());
    REQUIRE(constantsA.has_value());
    REQUIRE(constantsB.has_value());
    REQUIRE(outputA.has_value());
    REQUIRE(outputB.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(texture.has_value());
    auto textureView = device.createTextureView({
        .texture = *texture,
        .usage = rachis::TextureViewUsage::sampled,
        .label = "Binding update retained view",
    });
    REQUIRE(textureView.has_value());
    writeConstants(device, *constantsA, 3, 7);
    writeConstants(device, *constantsB, 5, 11);

    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{2, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute, 1},
    };
    auto bindingLayout = device.createBindingLayout({.entries = entries, .label = "Binding update layout"});
    REQUIRE(bindingLayout.has_value());
    const std::array initialWrites{
        rachis::BindingWrite{0, 0, rachis::BufferBinding{*constantsA, 0, bindingRangeSize}},
        rachis::BindingWrite{
            1,
            0,
            rachis::BufferBinding{*outputA, 0, bindingRangeSize, sizeof(std::uint32_t)},
        },
        rachis::BindingWrite{2, 0, *textureView},
    };
    auto bindingSet = rachis::test::createBindingSet(device,
                                                     {
                                                         .layout = *bindingLayout,
                                                         .writes = initialWrites,
                                                         .label = "Binding update set",
                                                     });
    REQUIRE(bindingSet.has_value());
    const std::array layouts{*bindingLayout};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *shader});
    REQUIRE(pipeline.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    for (const auto output : {*outputA, *outputB})
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

    CHECK_FALSE(device.updateBindingSet(*bindingSet, {}).has_value());
    const std::array constantsUpdate{
        rachis::BindingWrite{0, 0, rachis::BufferBinding{*constantsB, 0, bindingRangeSize}},
    };
    REQUIRE(device.updateBindingSet(*bindingSet, constantsUpdate).has_value());
    const std::array duplicateUpdate{
        rachis::BindingWrite{0, 0, rachis::BufferBinding{*constantsA, 0, bindingRangeSize}},
        rachis::BindingWrite{0, 0, rachis::BufferBinding{*constantsB, 0, bindingRangeSize}},
    };
    CHECK_FALSE(device.updateBindingSet(*bindingSet, duplicateUpdate).has_value());
    const std::array outputUpdate{
        rachis::BindingWrite{
            1,
            0,
            rachis::BufferBinding{*outputB, 0, bindingRangeSize, sizeof(std::uint32_t)},
        },
    };
    REQUIRE(device.updateBindingSet(*bindingSet, outputUpdate).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());

    for (const auto output : {*outputA, *outputB})
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
    REQUIRE(device.copyBuffer(*command, {*outputA, 0, *readback, 0, bindingRangeSize}).has_value());
    REQUIRE(device.copyBuffer(*command, {*outputB, 0, *readback, bindingRangeSize, bindingRangeSize}).has_value());

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    CHECK_FALSE(device.updateBindingSet(*bindingSet, constantsUpdate).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyBuffer(*constantsA).has_value());
    REQUIRE(device.destroyBuffer(*constantsB).has_value());
    REQUIRE(device.destroyBuffer(*outputA).has_value());
    REQUIRE(device.destroyBuffer(*outputB).has_value());
    REQUIRE(device.destroyTextureView(*textureView).has_value());
    REQUIRE(device.destroyTexture(*texture).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    verifyRange(mapped->data, 0, 3, 7);
    verifyRange(mapped->data, bindingRangeSize, 5, 11);
    REQUIRE(device.unmapBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("binding set updates share the shader-visible descriptor table they do not write",
          "[size:medium][rhi][binding][compute][warp]")
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

    const auto bytecode = loadShader();
    REQUIRE_FALSE(bytecode.empty());
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Binding table sharing shader",
        .shaderInterface = rachis::test::computeShaderInterface(
            {64, 1, 1},
            std::array{
                rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1},
                rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
            }),
    });
    auto constants = device.createBuffer({
        .size = bindingRangeSize,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Binding table sharing constants",
    });
    auto outputA = device.createBuffer({
        .size = bindingRangeSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .label = "Binding table sharing output A",
    });
    auto outputB = device.createBuffer({
        .size = bindingRangeSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .label = "Binding table sharing output B",
    });
    auto readback = device.createBuffer({
        .size = bindingRangeSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Binding table sharing readback",
    });
    auto samplerA = device.createSampler({.label = "Binding table sharing sampler A"});
    auto samplerB = device.createSampler({
        .minFilter = rachis::FilterMode::linear,
        .magFilter = rachis::FilterMode::linear,
        .label = "Binding table sharing sampler B",
    });
    REQUIRE(shader.has_value());
    REQUIRE(constants.has_value());
    REQUIRE(outputA.has_value());
    REQUIRE(outputB.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(samplerA.has_value());
    REQUIRE(samplerB.has_value());
    writeConstants(device, *constants, 3, 7);

    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{2, rachis::BindingType::sampler, rachis::ShaderStage::compute, 1},
    };
    auto bindingLayout = device.createBindingLayout({.entries = entries, .label = "Binding table sharing layout"});
    REQUIRE(bindingLayout.has_value());
    const std::array initialWrites{
        rachis::BindingWrite{0, 0, rachis::BufferBinding{*constants, 0, bindingRangeSize}},
        rachis::BindingWrite{
            1,
            0,
            rachis::BufferBinding{*outputA, 0, bindingRangeSize, sizeof(std::uint32_t)},
        },
        rachis::BindingWrite{2, 0, *samplerA},
    };
    std::array<rachis::BindingSetHandle, 2> sets{};
    for (auto& set : sets)
    {
        auto createdSet = rachis::test::createBindingSet(device,
                                                         {
                                                             .layout = *bindingLayout,
                                                             .writes = initialWrites,
                                                             .label = "Binding table sharing set",
                                                         });
        REQUIRE(createdSet.has_value());
        set = *createdSet;
    }
    const std::array layouts{*bindingLayout};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *shader});
    REQUIRE(pipeline.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    for (const auto set : sets)
    {
        REQUIRE(device.bindComputeBindingSet(*command, 0, set).has_value());
    }

    constexpr std::uint32_t shaderVisibleSamplerHeapCapacity = 2048;
    constexpr std::uint32_t resourceUpdatesPerSet = shaderVisibleSamplerHeapCapacity / 2 + 32;
    const std::array outputs{*outputA, *outputB};
    const auto beforeResourceUpdates = device.bindingUpdateStatistics();
    for (std::uint32_t update = 0; update < resourceUpdatesPerSet; ++update)
    {
        const std::array outputUpdate{
            rachis::BindingWrite{
                1,
                0,
                rachis::BufferBinding{
                    outputs[(update + 1) % outputs.size()], 0, bindingRangeSize, sizeof(std::uint32_t)},
            },
        };
        for (const auto set : sets)
        {
            auto updated = device.updateBindingSet(set, outputUpdate);
            const std::string failure = updated ? std::string{} : updated.error().message;
            CAPTURE(update, failure);
            REQUIRE(updated.has_value());
            REQUIRE(device.bindComputeBindingSet(*command, 0, set).has_value());
        }
    }
    const auto afterResourceUpdates = device.bindingUpdateStatistics();
    CHECK(afterResourceUpdates.resourceTablePublishCount - beforeResourceUpdates.resourceTablePublishCount ==
          sets.size() * resourceUpdatesPerSet);
    CHECK(afterResourceUpdates.resourceDescriptorsPrepared - beforeResourceUpdates.resourceDescriptorsPrepared ==
          sets.size() * resourceUpdatesPerSet);
    CHECK(afterResourceUpdates.samplerTablePublishCount == beforeResourceUpdates.samplerTablePublishCount);
    CHECK(afterResourceUpdates.samplerDescriptorsPrepared == beforeResourceUpdates.samplerDescriptorsPrepared);
    CHECK(afterResourceUpdates.samplerDescriptorsPublished == beforeResourceUpdates.samplerDescriptorsPublished);

    const std::array samplerUpdate{
        rachis::BindingWrite{2, 0, *samplerB},
    };
    REQUIRE(device.updateBindingSet(sets[0], samplerUpdate).has_value());
    const auto afterSamplerUpdate = device.bindingUpdateStatistics();
    CHECK(afterSamplerUpdate.samplerTablePublishCount - afterResourceUpdates.samplerTablePublishCount == 1);
    CHECK(afterSamplerUpdate.samplerDescriptorsPrepared - afterResourceUpdates.samplerDescriptorsPrepared == 1);
    CHECK(afterSamplerUpdate.samplerDescriptorsPublished - afterResourceUpdates.samplerDescriptorsPublished == 1);
    CHECK(afterSamplerUpdate.resourceTablePublishCount == afterResourceUpdates.resourceTablePublishCount);
    CHECK(afterSamplerUpdate.resourceDescriptorsPrepared == afterResourceUpdates.resourceDescriptorsPrepared);
    CHECK(afterSamplerUpdate.resourceDescriptorsPublished == afterResourceUpdates.resourceDescriptorsPublished);

    const auto finalOutput = outputs[resourceUpdatesPerSet % outputs.size()];
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = finalOutput,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, sets[0]).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = finalOutput,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device.copyBuffer(*command, {finalOutput, 0, *readback, 0, bindingRangeSize}).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    verifyRange(mapped->data, 0, 3, 7);
    REQUIRE(device.unmapBuffer(*readback).has_value());

    for (const auto set : sets)
    {
        REQUIRE(device.destroyBindingSet(set).has_value());
    }
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroySampler(*samplerA).has_value());
    REQUIRE(device.destroySampler(*samplerB).has_value());
    REQUIRE(device.destroyBuffer(*constants).has_value());
    REQUIRE(device.destroyBuffer(*outputA).has_value());
    REQUIRE(device.destroyBuffer(*outputB).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("bounded binding pools retire exact copy-on-write versions",
          "[size:medium][rhi][binding-pool][compute][warp]")
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

    const auto bytecode = loadShader();
    REQUIRE_FALSE(bytecode.empty());
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Binding pool shader",
        .shaderInterface = rachis::test::computeShaderInterface(
            {64, 1, 1},
            std::array{
                rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1},
                rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
            }),
    });
    auto constantsA = device.createBuffer({
        .size = bindingRangeSize,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Binding pool constants A",
    });
    auto constantsB = device.createBuffer({
        .size = bindingRangeSize,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Binding pool constants B",
    });
    auto outputA = device.createBuffer({
        .size = bindingRangeSize,
        .usage = rachis::BufferUsage::storage,
        .label = "Binding pool output A",
    });
    auto outputB = device.createBuffer({
        .size = bindingRangeSize,
        .usage = rachis::BufferUsage::storage,
        .label = "Binding pool output B",
    });
    REQUIRE(shader.has_value());
    REQUIRE(constantsA.has_value());
    REQUIRE(constantsB.has_value());
    REQUIRE(outputA.has_value());
    REQUIRE(outputB.has_value());

    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    auto layout = device.createBindingLayout({.entries = entries, .label = "Binding pool layout"});
    REQUIRE(layout.has_value());
    const std::array poolSizes{
        rachis::BindingPoolSize{rachis::BindingType::uniformBuffer, 2},
        rachis::BindingPoolSize{rachis::BindingType::storageBuffer, 2},
    };
    auto pool = device.createBindingPool({
        .maxBindingSets = 1,
        .sizes = poolSizes,
        .label = "One-set two-version binding pool",
    });
    REQUIRE(pool.has_value());

    const std::array initialWrites{
        rachis::BindingWrite{0, 0, rachis::BufferBinding{*constantsA, 0, bindingRangeSize}},
        rachis::BindingWrite{
            1,
            0,
            rachis::BufferBinding{*outputA, 0, bindingRangeSize, sizeof(std::uint32_t)},
        },
    };
    const rachis::BindingSetDesc setDesc{
        .layout = *layout,
        .writes = initialWrites,
        .label = "Pooled binding set",
        .pool = *pool,
    };
    auto set = rachis::test::createBindingSet(device, setDesc);
    REQUIRE(set.has_value());
    auto exhaustedSet = rachis::test::createBindingSet(device, setDesc);
    REQUIRE_FALSE(exhaustedSet.has_value());
    CHECK(exhaustedSet.error().code == rachis::ErrorCode::resourceExhausted);

    const std::array layouts{*layout};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *shader});
    REQUIRE(pipeline.has_value());
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *set).has_value());

    const std::array updatedWrites{
        rachis::BindingWrite{0, 0, rachis::BufferBinding{*constantsB, 0, bindingRangeSize}},
        rachis::BindingWrite{
            1,
            0,
            rachis::BufferBinding{*outputB, 0, bindingRangeSize, sizeof(std::uint32_t)},
        },
    };
    REQUIRE(device.updateBindingSet(*set, updatedWrites).has_value());
    auto exhaustedUpdate = device.updateBindingSet(*set, initialWrites);
    REQUIRE_FALSE(exhaustedUpdate.has_value());
    CHECK(exhaustedUpdate.error().code == rachis::ErrorCode::resourceExhausted);

    REQUIRE(device.destroyBindingSet(*set).has_value());
    exhaustedSet = rachis::test::createBindingSet(device, setDesc);
    REQUIRE_FALSE(exhaustedSet.has_value());
    CHECK(exhaustedSet.error().code == rachis::ErrorCode::resourceExhausted);

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto replacement = rachis::test::createBindingSet(device, setDesc);
    REQUIRE(replacement.has_value());
    auto unretainedPool = device.createCommandPool({
        .queueType = rachis::QueueType::compute,
        .maxCommandBuffers = 1,
        .label = "Binding-pool unretained command pool",
        .flags = rachis::CommandPoolFlag::unretainedResources,
    });
    REQUIRE(unretainedPool.has_value());
    auto unretainedCommand = device.acquireCommandBuffer(*unretainedPool);
    REQUIRE(unretainedCommand.has_value());
    REQUIRE(device.bindComputePipeline(*unretainedCommand, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*unretainedCommand, 0, *replacement).has_value());
    REQUIRE(device.updateBindingSet(*replacement, updatedWrites).has_value());
    exhaustedUpdate = device.updateBindingSet(*replacement, initialWrites);
    REQUIRE_FALSE(exhaustedUpdate.has_value());
    CHECK(exhaustedUpdate.error().code == rachis::ErrorCode::resourceExhausted);
    auto unretainedSubmitted = device.submit(std::move(*unretainedCommand));
    REQUIRE(unretainedSubmitted.has_value());
    REQUIRE(device.wait(*unretainedSubmitted).has_value());
    REQUIRE(device.updateBindingSet(*replacement, initialWrites).has_value());

    REQUIRE(device.destroyBindingSet(*replacement).has_value());
    REQUIRE(device.destroyCommandPool(*unretainedPool).has_value());
    REQUIRE(device.destroyBindingPool(*pool).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*layout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBuffer(*constantsA).has_value());
    REQUIRE(device.destroyBuffer(*constantsB).has_value());
    REQUIRE(device.destroyBuffer(*outputA).has_value());
    REQUIRE(device.destroyBuffer(*outputB).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("one-set single-version binding pools allow idle updates while rejecting bound updates until timeline wait",
          "[size:medium][rhi][binding-pool][compute][warp]")
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

    const auto bytecode = loadShader();
    REQUIRE_FALSE(bytecode.empty());
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Single-version binding pool shader",
        .shaderInterface = rachis::test::computeShaderInterface(
            {64, 1, 1},
            std::array{
                rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1},
                rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
            }),
    });
    auto constantsA = device.createBuffer({
        .size = bindingRangeSize,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Single-version constants A",
    });
    auto constantsB = device.createBuffer({
        .size = bindingRangeSize,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Single-version constants B",
    });
    auto outputA = device.createBuffer({
        .size = bindingRangeSize,
        .usage = rachis::BufferUsage::storage,
        .label = "Single-version output A",
    });
    auto outputB = device.createBuffer({
        .size = bindingRangeSize,
        .usage = rachis::BufferUsage::storage,
        .label = "Single-version output B",
    });
    REQUIRE(shader.has_value());
    REQUIRE(constantsA.has_value());
    REQUIRE(constantsB.has_value());
    REQUIRE(outputA.has_value());
    REQUIRE(outputB.has_value());

    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    auto layout = device.createBindingLayout({.entries = entries, .label = "Single-version binding pool layout"});
    REQUIRE(layout.has_value());

    const std::array poolSizes{
        rachis::BindingPoolSize{rachis::BindingType::uniformBuffer, 1},
        rachis::BindingPoolSize{rachis::BindingType::storageBuffer, 1},
    };
    auto pool = device.createBindingPool({
        .maxBindingSets = 1,
        .sizes = poolSizes,
        .label = "One-set single-version binding pool",
    });
    REQUIRE(pool.has_value());

    const std::array initialWrites{
        rachis::BindingWrite{0, 0, rachis::BufferBinding{*constantsA, 0, bindingRangeSize}},
        rachis::BindingWrite{
            1,
            0,
            rachis::BufferBinding{*outputA, 0, bindingRangeSize, sizeof(std::uint32_t)},
        },
    };
    const std::array updatedWrites{
        rachis::BindingWrite{0, 0, rachis::BufferBinding{*constantsB, 0, bindingRangeSize}},
        rachis::BindingWrite{
            1,
            0,
            rachis::BufferBinding{*outputB, 0, bindingRangeSize, sizeof(std::uint32_t)},
        },
    };
    auto set = rachis::test::createBindingSet(device,
                                              {
                                                  .layout = *layout,
                                                  .writes = initialWrites,
                                                  .label = "Single-version pooled set",
                                                  .pool = *pool,
                                              });
    REQUIRE(set.has_value());

    REQUIRE(device.updateBindingSet(*set, updatedWrites).has_value());
    REQUIRE(device.updateBindingSet(*set, initialWrites).has_value());

    const std::array layouts{*layout};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *shader});
    REQUIRE(pipeline.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *set).has_value());

    auto boundUpdate = device.updateBindingSet(*set, updatedWrites);
    REQUIRE_FALSE(boundUpdate.has_value());
    CHECK(boundUpdate.error().code == rachis::ErrorCode::resourceExhausted);

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    REQUIRE(device.updateBindingSet(*set, updatedWrites).has_value());

    REQUIRE(device.destroyBindingSet(*set).has_value());
    REQUIRE(device.destroyBindingPool(*pool).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*layout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBuffer(*constantsA).has_value());
    REQUIRE(device.destroyBuffer(*constantsB).has_value());
    REQUIRE(device.destroyBuffer(*outputA).has_value());
    REQUIRE(device.destroyBuffer(*outputB).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("binding sets in different pools share one shader-visible table for identical samplers",
          "[size:medium][rhi][binding-pool][warp]")
{
    auto device = createWarpDevice();
    auto samplerA = device.createSampler({
        .minFilter = rachis::FilterMode::linear,
        .magFilter = rachis::FilterMode::linear,
        .label = "Shared sampler A",
    });
    auto samplerB = device.createSampler({.label = "Shared sampler B"});
    REQUIRE(samplerA.has_value());
    REQUIRE(samplerB.has_value());

    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::sampler, rachis::ShaderStage::compute, 1},
    };
    auto layout = device.createBindingLayout({.entries = entries, .label = "Shared sampler layout"});
    REQUIRE(layout.has_value());

    constexpr std::uint32_t setsPerPool = 1024;
    constexpr std::uint32_t poolCount = 3;
    std::array<rachis::BindingPoolHandle, poolCount> pools{};
    for (auto& pool : pools)
    {
        pool = createSingleTypePool(device, setsPerPool, rachis::BindingType::sampler, setsPerPool);
    }

    const std::array writesA{rachis::BindingWrite{0, 0, *samplerA}};
    const auto beforeSets = device.bindingUpdateStatistics();
    std::vector<rachis::BindingSetHandle> sets;
    sets.reserve(setsPerPool * poolCount);
    for (std::uint32_t index = 0; index < setsPerPool * poolCount; ++index)
    {
        auto set = device.createBindingSet({
            .layout = *layout,
            .writes = writesA,
            .label = "Shared sampler set",
            .pool = pools[index % poolCount],
        });
        const std::string failure = set ? std::string{} : set.error().message;
        CAPTURE(index, failure);
        REQUIRE(set.has_value());
        sets.push_back(*set);
    }
    const auto afterSets = device.bindingUpdateStatistics();
    CHECK(afterSets.samplerTablePublishCount - beforeSets.samplerTablePublishCount == 1);
    CHECK(afterSets.samplerDescriptorsPublished - beforeSets.samplerDescriptorsPublished == 1);

    const std::array writesB{rachis::BindingWrite{0, 0, *samplerB}};
    REQUIRE(device.updateBindingSet(sets[0], writesB).has_value());
    REQUIRE(device.updateBindingSet(sets[1], writesB).has_value());
    REQUIRE(device.updateBindingSet(sets[0], writesA).has_value());
    REQUIRE(device.updateBindingSet(sets[setsPerPool * poolCount - 1], writesA).has_value());
    const auto afterUpdates = device.bindingUpdateStatistics();
    CHECK(afterUpdates.samplerTablePublishCount - afterSets.samplerTablePublishCount == 1);
    CHECK(afterUpdates.samplerDescriptorsPublished - afterSets.samplerDescriptorsPublished == 1);

    for (const auto set : sets)
    {
        REQUIRE(device.destroyBindingSet(set).has_value());
    }
    for (const auto pool : pools)
    {
        REQUIRE(device.destroyBindingPool(pool).has_value());
    }

    constexpr std::uint32_t wideSamplerCount = 1024;
    const std::array wideEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::sampler, rachis::ShaderStage::compute, wideSamplerCount},
    };
    auto wideLayout = device.createBindingLayout({.entries = wideEntries, .label = "Wide sampler layout"});
    REQUIRE(wideLayout.has_value());
    const auto wideA = repeatedWrites(wideSamplerCount, *samplerA);
    const auto wideB = repeatedWrites(wideSamplerCount, *samplerB);
    auto mixed = wideA;
    for (std::uint32_t element = 1; element < wideSamplerCount; element += 2)
    {
        mixed[element].resource = *samplerB;
    }
    std::array<rachis::BindingPoolHandle, 4> widePools{};
    for (auto& pool : widePools)
    {
        pool = createSingleTypePool(device, 1, rachis::BindingType::sampler, wideSamplerCount);
    }
    auto wideSetA = device.createBindingSet({.layout = *wideLayout, .writes = wideA, .pool = widePools[0]});
    auto wideSetB = device.createBindingSet({.layout = *wideLayout, .writes = wideB, .pool = widePools[1]});
    REQUIRE(wideSetA.has_value());
    REQUIRE(wideSetB.has_value());
    CHECK(rachis::test::requireCapabilities(device).limits.maxBindingSamplerDescriptors == 2 * wideSamplerCount);
    auto distinctSet = device.createBindingSet({.layout = *wideLayout, .writes = mixed, .pool = widePools[2]});
    REQUIRE_FALSE(distinctSet.has_value());
    CHECK(distinctSet.error().code == rachis::ErrorCode::resourceExhausted);
    auto repeatedSet = device.createBindingSet({.layout = *wideLayout, .writes = wideA, .pool = widePools[3]});
    REQUIRE(repeatedSet.has_value());

    REQUIRE(device.destroyBindingSet(*repeatedSet).has_value());
    REQUIRE(device.destroyBindingSet(*wideSetB).has_value());
    REQUIRE(device.destroyBindingSet(*wideSetA).has_value());
    for (const auto pool : widePools)
    {
        REQUIRE(device.destroyBindingPool(pool).has_value());
    }
    REQUIRE(device.destroyBindingLayout(*wideLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*layout).has_value());
    REQUIRE(device.destroySampler(*samplerA).has_value());
    REQUIRE(device.destroySampler(*samplerB).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("idle binding set updates rewrite full exact-size descriptor tables in place",
          "[size:medium][rhi][binding-pool][warp]")
{
    auto device = createWarpDevice();
    auto samplerA = device.createSampler({
        .minFilter = rachis::FilterMode::linear,
        .magFilter = rachis::FilterMode::linear,
        .label = "In-place sampler A",
    });
    auto samplerB = device.createSampler({.label = "In-place sampler B"});
    REQUIRE(samplerA.has_value());
    REQUIRE(samplerB.has_value());
    const auto viewA = createSampledView(device, "In-place texture A");
    const auto viewB = createSampledView(device, "In-place texture B");

    constexpr std::uint32_t samplerCount = 1100;
    const std::array samplerEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::sampler, rachis::ShaderStage::compute, samplerCount},
    };
    auto samplerLayout = device.createBindingLayout({.entries = samplerEntries, .label = "Full sampler layout"});
    REQUIRE(samplerLayout.has_value());
    const auto samplerPool = createSingleTypePool(device, 1, rachis::BindingType::sampler, samplerCount);
    const auto samplerWrites = repeatedWrites(samplerCount, *samplerA);
    auto samplerSet = device.createBindingSet({
        .layout = *samplerLayout,
        .writes = samplerWrites,
        .label = "Full sampler set",
        .pool = samplerPool,
    });
    REQUIRE(samplerSet.has_value());

    const auto beforeSamplerUpdates = device.bindingUpdateStatistics();
    const std::array samplerUpdate{rachis::BindingWrite{0, samplerCount - 1, *samplerB}};
    auto updated = device.updateBindingSet(*samplerSet, samplerUpdate);
    std::string failure = updated ? std::string{} : updated.error().message;
    CAPTURE(failure);
    REQUIRE(updated.has_value());
    const std::array samplerRestore{rachis::BindingWrite{0, samplerCount - 1, *samplerA}};
    REQUIRE(device.updateBindingSet(*samplerSet, samplerRestore).has_value());
    const auto afterSamplerUpdates = device.bindingUpdateStatistics();
    CHECK(afterSamplerUpdates.samplerTablePublishCount - beforeSamplerUpdates.samplerTablePublishCount == 2);
    CHECK(afterSamplerUpdates.samplerDescriptorsPublished - beforeSamplerUpdates.samplerDescriptorsPublished == 2);

    constexpr std::uint32_t textureCount = 40'000;
    const std::array textureEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute, textureCount},
    };
    auto textureLayout = device.createBindingLayout({.entries = textureEntries, .label = "Full texture layout"});
    REQUIRE(textureLayout.has_value());
    const auto texturePool = createSingleTypePool(device, 1, rachis::BindingType::sampledTexture, textureCount);
    const auto textureWrites = repeatedWrites(textureCount, viewA.view);
    auto textureSet = device.createBindingSet({
        .layout = *textureLayout,
        .writes = textureWrites,
        .label = "Full texture set",
        .pool = texturePool,
    });
    REQUIRE(textureSet.has_value());

    const auto beforeTextureUpdate = device.bindingUpdateStatistics();
    const std::array textureUpdate{rachis::BindingWrite{0, textureCount / 2, viewB.view}};
    updated = device.updateBindingSet(*textureSet, textureUpdate);
    failure = updated ? std::string{} : updated.error().message;
    CAPTURE(failure);
    REQUIRE(updated.has_value());
    const auto afterTextureUpdate = device.bindingUpdateStatistics();
    CHECK(afterTextureUpdate.resourceTablePublishCount - beforeTextureUpdate.resourceTablePublishCount == 1);
    CHECK(afterTextureUpdate.resourceDescriptorsPublished - beforeTextureUpdate.resourceDescriptorsPublished == 1);
    CHECK(afterTextureUpdate.resourceDescriptorsPrepared - beforeTextureUpdate.resourceDescriptorsPrepared == 1);

    REQUIRE(device.destroyBindingSet(*textureSet).has_value());
    REQUIRE(device.destroyBindingSet(*samplerSet).has_value());
    REQUIRE(device.destroyBindingPool(texturePool).has_value());
    REQUIRE(device.destroyBindingPool(samplerPool).has_value());
    REQUIRE(device.destroyBindingLayout(*textureLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*samplerLayout).has_value());
    destroySampledView(device, viewA);
    destroySampledView(device, viewB);
    REQUIRE(device.destroySampler(*samplerA).has_value());
    REQUIRE(device.destroySampler(*samplerB).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("binding pools reserve the reported shader-visible descriptor budget apart from internal descriptors",
          "[size:medium][rhi][binding-pool][mipmap][warp]")
{
    auto device = createWarpDevice();
    const auto capabilities = rachis::test::requireCapabilities(device);
    REQUIRE(capabilities.mipmapGeneration);
    const auto budget = capabilities.limits.maxBindingPoolResourceDescriptors;
    CHECK(budget == 1'000'000U - 65'536U - 65'536U);
    CHECK(capabilities.limits.maxBindingSamplerDescriptors == 2048);
    if (capabilities.variableDescriptorCountBindingArrays)
    {
        CHECK(capabilities.limits.maxVariableResourceDescriptors == budget);
        CHECK(capabilities.limits.maxVariableBufferDescriptors == budget);
    }

    const std::array oversizedSizes{rachis::BindingPoolSize{rachis::BindingType::uniformBuffer, budget + 1}};
    auto oversizedPool = device.createBindingPool({.maxBindingSets = 1, .sizes = oversizedSizes});
    REQUIRE_FALSE(oversizedPool.has_value());
    CHECK(oversizedPool.error().code == rachis::ErrorCode::invalidArgument);

    const auto viewA = createSampledView(device, "Budget texture A");
    const auto viewB = createSampledView(device, "Budget texture B");
    constexpr std::uint32_t smallSetCount = 40'000;
    const auto largeSetCount = budget - smallSetCount;
    const auto smallPool = createSingleTypePool(device, 1, rachis::BindingType::sampledTexture, smallSetCount);
    auto largePool = createSingleTypePool(device, 1, rachis::BindingType::sampledTexture, largeSetCount);
    const std::array extraSizes{rachis::BindingPoolSize{rachis::BindingType::uniformBuffer, 1}};
    auto extraPool = device.createBindingPool({.maxBindingSets = 1, .sizes = extraSizes});
    REQUIRE_FALSE(extraPool.has_value());
    CHECK(extraPool.error().code == rachis::ErrorCode::resourceExhausted);

    const std::array smallEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute, smallSetCount},
    };
    const std::array largeEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute, largeSetCount},
    };
    auto smallLayout = device.createBindingLayout({.entries = smallEntries, .label = "Budget small layout"});
    auto largeLayout = device.createBindingLayout({.entries = largeEntries, .label = "Budget large layout"});
    REQUIRE(smallLayout.has_value());
    REQUIRE(largeLayout.has_value());
    const auto smallWrites = repeatedWrites(smallSetCount, viewA.view);
    auto smallSet = device.createBindingSet({
        .layout = *smallLayout,
        .writes = smallWrites,
        .label = "Budget small set",
        .pool = smallPool,
    });
    REQUIRE(smallSet.has_value());
    std::optional<rachis::BindingSetHandle> largeSet;
    {
        const auto largeWrites = repeatedWrites(largeSetCount, viewA.view);
        auto created = device.createBindingSet({
            .layout = *largeLayout,
            .writes = largeWrites,
            .label = "Budget large set",
            .pool = largePool,
        });
        const std::string failure = created ? std::string{} : created.error().message;
        CAPTURE(failure);
        REQUIRE(created.has_value());
        largeSet = *created;
    }

    auto mipTexture = device.createTexture({
        .dimension = rachis::TextureDimension::texture2D,
        .width = 32,
        .height = 32,
        .mipLevels = 6,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::storage | rachis::TextureUsage::copySource |
                 rachis::TextureUsage::copyDestination,
        .label = "Budget mipmap texture",
    });
    REQUIRE(mipTexture.has_value());
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *mipTexture,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                    .subresources = {.mipLevelCount = 1, .arrayLayerCount = 1},
                                })
                .has_value());
    auto generated = device.generateMipmaps(*command,
                                            {
                                                .texture = *mipTexture,
                                                .subresources = {.mipLevelCount = 6, .arrayLayerCount = 1},
                                                .baseLayout = rachis::TextureLayout::copyDestination,
                                                .finalLayout = rachis::TextureLayout::shaderResource,
                                            });
    const std::string mipmapFailure = generated ? std::string{} : generated.error().message;
    CAPTURE(mipmapFailure);
    REQUIRE(generated.has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    const std::array smallUpdate{rachis::BindingWrite{0, 0, viewB.view}};
    auto updated = device.updateBindingSet(*smallSet, smallUpdate);
    const std::string updateFailure = updated ? std::string{} : updated.error().message;
    CAPTURE(updateFailure);
    REQUIRE(updated.has_value());

    REQUIRE(device.destroyBindingPool(largePool).has_value());
    extraPool = device.createBindingPool({.maxBindingSets = 1, .sizes = extraSizes});
    REQUIRE_FALSE(extraPool.has_value());
    CHECK(extraPool.error().code == rachis::ErrorCode::resourceExhausted);
    REQUIRE(device.destroyBindingSet(*largeSet).has_value());
    extraPool = device.createBindingPool({.maxBindingSets = 1, .sizes = extraSizes});
    REQUIRE(extraPool.has_value());

    REQUIRE(device.destroyBindingPool(*extraPool).has_value());
    REQUIRE(device.destroyBindingSet(*smallSet).has_value());
    REQUIRE(device.destroyBindingPool(smallPool).has_value());
    REQUIRE(device.destroyBindingLayout(*largeLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*smallLayout).has_value());
    REQUIRE(device.destroyTexture(*mipTexture).has_value());
    destroySampledView(device, viewA);
    destroySampledView(device, viewB);

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("one command buffer generates mipmaps for 1,000 textures of 12 levels in the internal descriptor range",
          "[size:medium][rhi][binding-pool][mipmap][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).mipmapGeneration);
    constexpr std::uint32_t textureCount = 1'000;
    constexpr std::uint32_t mipLevels = 12;
    std::vector<rachis::TextureHandle> textures;
    textures.reserve(textureCount);
    for (std::uint32_t index = 0; index < textureCount; ++index)
    {
        auto texture = device.createTexture({
            .dimension = rachis::TextureDimension::texture2D,
            .width = 1U << (mipLevels - 1),
            .height = 1,
            .mipLevels = mipLevels,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::storage | rachis::TextureUsage::copySource |
                     rachis::TextureUsage::copyDestination,
            .label = "Internal-range mipmap texture",
        });
        REQUIRE(texture.has_value());
        textures.push_back(*texture);
    }
    const auto beforeMipmaps = device.captureBackendAllocationSnapshot();

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    std::uint32_t failedIndex = textureCount;
    std::string failure;
    for (std::uint32_t index = 0; index < textureCount && failedIndex == textureCount; ++index)
    {
        auto barrier = device.textureBarrier(*command,
                                             {
                                                 .texture = textures[index],
                                                 .stagesBefore = rachis::PipelineStage::all,
                                                 .stagesAfter = rachis::PipelineStage::copy,
                                                 .accessBefore = rachis::Access::none,
                                                 .accessAfter = rachis::Access::copyDestination,
                                                 .layoutBefore = rachis::TextureLayout::undefined,
                                                 .layoutAfter = rachis::TextureLayout::copyDestination,
                                                 .subresources = {.mipLevelCount = 1, .arrayLayerCount = 1},
                                             });
        REQUIRE(barrier.has_value());
        auto generated = device.generateMipmaps(*command,
                                                {
                                                    .texture = textures[index],
                                                    .subresources = {.mipLevelCount = mipLevels, .arrayLayerCount = 1},
                                                    .baseLayout = rachis::TextureLayout::copyDestination,
                                                    .finalLayout = rachis::TextureLayout::shaderResource,
                                                });
        if (!generated)
        {
            failedIndex = index;
            failure = generated.error().message;
        }
    }
    CAPTURE(failedIndex, failure);
    REQUIRE(failedIndex == textureCount);
    CHECK(device.captureBackendAllocationSnapshot().liveDescriptorCount - beforeMipmaps.liveDescriptorCount >=
          2ULL * (mipLevels - 1) * textureCount);

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();
    CHECK(device.captureBackendAllocationSnapshot().liveDescriptorCount == beforeMipmaps.liveDescriptorCount);

    for (const auto texture : textures)
    {
        REQUIRE(device.destroyTexture(texture).has_value());
    }
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
