#include "Rachis/Device.h"
#include "RhiTestSupport.h"
#include "SubgroupSize128.h"
#include "SubgroupSize16.h"
#include "SubgroupSize32.h"
#include "SubgroupSize4.h"
#include "SubgroupSize64.h"
#include "SubgroupSize8.h"
#include "SubgroupSizeAny.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

namespace
{

    constexpr std::array<std::uint32_t, 6> subgroupSizes{4, 8, 16, 32, 64, 128};
    constexpr std::uint32_t anySizeGroupWidth = 128;
    constexpr std::uint32_t laneCountGroupCount = 2;

    using LaneCounts = std::vector<std::array<std::uint32_t, 2>>;

    [[nodiscard]] rachis::DeviceDesc metalDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
    }

    [[nodiscard]] std::span<const std::byte> subgroupSizeShader(std::uint32_t size)
    {
        constexpr auto backend = rachis::Backend::metal;
        switch (size)
        {
        case 4:
            return rachis_shaders::SubgroupSize4(backend);
        case 8:
            return rachis_shaders::SubgroupSize8(backend);
        case 16:
            return rachis_shaders::SubgroupSize16(backend);
        case 32:
            return rachis_shaders::SubgroupSize32(backend);
        case 64:
            return rachis_shaders::SubgroupSize64(backend);
        case 128:
            return rachis_shaders::SubgroupSize128(backend);
        default:
            return rachis_shaders::SubgroupSizeAny(backend);
        }
    }

    [[nodiscard]] rachis::Result<LaneCounts> runLaneCountKernel(rachis::Device& device,
                                                                std::span<const std::byte> bytecode,
                                                                std::uint32_t requiredSubgroupSize,
                                                                std::uint32_t groupWidth)
    {
        REQUIRE_FALSE(bytecode.empty());
        const std::uint32_t laneCountInvocations = groupWidth * laneCountGroupCount;
        const std::uint64_t laneCountBytes = std::uint64_t{laneCountInvocations} * 2U * sizeof(std::uint32_t);
        auto shader = device.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = bytecode,
            .entryPoint = "main",
            .label = "Metal subgroup size shader",
            .shaderInterface = rachis::test::computeShaderInterface(
                {groupWidth, 1, 1},
                std::array{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1}}),
        });
        if (!shader)
        {
            UNSCOPED_INFO(shader.error().message);
        }
        REQUIRE(shader);
        const std::array entries{
            rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
        };
        auto layout = device.createBindingLayout({.entries = entries});
        REQUIRE(layout);
        const std::array layouts{*layout};
        auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts});
        REQUIRE(pipelineLayout);
        auto pipeline = device.createComputePipeline({
            .layout = *pipelineLayout,
            .computeShader = *shader,
            .label = "Metal subgroup size pipeline",
            .requiredSubgroupSize = requiredSubgroupSize,
        });
        if (!pipeline)
        {
            CHECK(device.destroyPipelineLayout(*pipelineLayout));
            CHECK(device.destroyBindingLayout(*layout));
            CHECK(device.destroyShaderModule(*shader));
            return std::unexpected(std::move(pipeline.error()));
        }

        auto storage = device.createBuffer({
            .size = laneCountBytes,
            .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
            .label = "Metal subgroup size output",
        });
        auto readback = device.createBuffer({
            .size = laneCountBytes,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Metal subgroup size readback",
        });
        REQUIRE(storage);
        REQUIRE(readback);
        const std::array writes{
            rachis::BindingWrite{0, 0, rachis::BufferBinding{*storage, 0, laneCountBytes, 2U * sizeof(std::uint32_t)}},
        };
        auto set = rachis::test::createBindingSet(device, {.layout = *layout, .writes = writes});
        REQUIRE(set);

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command);
        REQUIRE(device.bindComputePipeline(*command, *pipeline));
        REQUIRE(device.bindComputeBindingSet(*command, 0, *set));
        REQUIRE(device.dispatch(*command, laneCountGroupCount));
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
        REQUIRE(device.copyBuffer(*command, {*storage, 0, *readback, 0, laneCountBytes}));
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted);
        REQUIRE(device.wait(*submitted));
        auto mapped = device.mapBuffer(*readback);
        REQUIRE(mapped);
        LaneCounts lanes(laneCountInvocations);
        std::memcpy(lanes.data(), mapped->data, laneCountBytes);
        REQUIRE(device.unmapBuffer(*readback));

        REQUIRE(device.destroyPipeline(*pipeline));
        REQUIRE(device.destroyPipelineLayout(*pipelineLayout));
        REQUIRE(device.destroyBindingSet(*set));
        REQUIRE(device.destroyBindingLayout(*layout));
        REQUIRE(device.destroyShaderModule(*shader));
        REQUIRE(device.destroyBuffer(*storage));
        REQUIRE(device.destroyBuffer(*readback));
        return lanes;
    }

} // namespace

TEST_CASE("Metal reports no 64-bit integer atomics", "[size:large][rhi][metal][shader64][atomic]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    const auto capabilities = rachis::test::requireCapabilities(*created);
    CHECK(capabilities.shaderInt64);
    CHECK_FALSE(capabilities.shaderInt64AtomicStorageBuffer);
    CHECK_FALSE(capabilities.shaderInt64AtomicGroupShared);
    CHECK_FALSE(capabilities.shaderInt64AtomicTypedResource);
}

TEST_CASE("Metal compute pipelines accept only a required subgroup size of 32 and run full SIMD groups",
          "[size:large][rhi][metal][shader][subgroup][compute]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto subgroup = rachis::test::requireCapabilities(device).subgroup;
    CHECK(subgroup.minSize == 32);
    CHECK(subgroup.maxSize == 32);
    CHECK(subgroup.requiredSizeStages == rachis::ShaderStage::compute);

    for (const auto size : subgroupSizes)
    {
        CAPTURE(size);
        auto lanes = runLaneCountKernel(device, subgroupSizeShader(size), size, size);
        if (size != 32)
        {
            REQUIRE_FALSE(lanes.has_value());
            CHECK(lanes.error().code == rachis::ErrorCode::unsupported);
            continue;
        }
        if (!lanes)
        {
            UNSCOPED_INFO(lanes.error().message);
        }
        REQUIRE(lanes.has_value());
        for (const auto& lane : *lanes)
        {
            CHECK(lane == std::array<std::uint32_t, 2>{32, 32});
        }
    }

    auto anyShader =
        runLaneCountKernel(device, rachis_shaders::SubgroupSizeAny(rachis::Backend::metal), 32, anySizeGroupWidth);
    if (!anyShader)
    {
        UNSCOPED_INFO(anyShader.error().message);
    }
    REQUIRE(anyShader.has_value());
    for (const auto& lane : *anyShader)
    {
        CHECK(lane == std::array<std::uint32_t, 2>{32, 32});
    }
    CHECK(device.diagnosticCounts().errors == 0);
}
