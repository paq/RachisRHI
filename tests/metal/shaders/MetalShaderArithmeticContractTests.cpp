#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string_view>
#include <vector>

namespace
{

    constexpr std::uint32_t storageBinding = 1;
    constexpr std::uint32_t invocationCount = 64;
    constexpr std::uint32_t valuesPerInvocation = 4;

    [[nodiscard]] rachis::DeviceDesc metalDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
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

    [[nodiscard]] std::filesystem::path pnmsPath(std::string_view metallibName)
    {
        return std::filesystem::path(RACHIS_METALLIB_DIR) /
               std::filesystem::path(metallibName).replace_extension(".pnms");
    }

    struct ComputeHarness final
    {
        rachis::Device* device = nullptr;
        rachis::BufferHandle storage{};
        rachis::BufferHandle readback{};
        rachis::BindingLayoutHandle layout{};
        rachis::BindingSetHandle set{};
        rachis::PipelineLayoutHandle pipelineLayout{};
        rachis::ShaderModuleHandle shader{};
        rachis::PipelineHandle pipeline{};
        std::uint64_t bytes = 0;
    };

    [[nodiscard]] ComputeHarness createCompute(rachis::Device& device,
                                               std::string_view metallibName,
                                               std::string_view entry,
                                               rachis::ShaderWorkgroupSize workgroup,
                                               std::uint64_t bytes,
                                               std::uint32_t pushBytes)
    {
        ComputeHarness harness{.device = &device, .bytes = bytes};
        const auto bytecode = loadBytes(pnmsPath(metallibName));
        REQUIRE_FALSE(bytecode.empty());
        const std::array shaderBindings{
            rachis::ShaderBindingRequirement{0, storageBinding, rachis::ShaderResourceType::storageBuffer, 1},
        };
        std::array<rachis::ShaderPushConstantRequirement, 1> shaderPush{
            rachis::ShaderPushConstantRequirement{0, pushBytes},
        };
        auto shader = device.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = bytecode,
            .entryPoint = entry,
            .shaderInterface = pushBytes == 0
                                   ? rachis::test::computeShaderInterface(workgroup, shaderBindings)
                                   : rachis::test::computeShaderInterface(workgroup, shaderBindings, shaderPush),
        });
        INFO((shader ? "" : shader.error().message));
        REQUIRE(shader.has_value());
        harness.shader = *shader;
        harness.storage = *device.createBuffer(rachis::BufferDesc{
            .size = bytes,
            .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
            .label = "Metal storage",
        });
        harness.readback = *device.createBuffer(rachis::BufferDesc{
            .size = bytes,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Metal readback",
        });
        const std::array layoutEntries{
            rachis::BindingLayoutEntry{
                storageBinding, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
        };
        harness.layout = *device.createBindingLayout({.entries = layoutEntries});
        const std::array writes{rachis::BindingWrite{
            .binding = storageBinding,
            .resource = rachis::BufferBinding{.buffer = harness.storage, .structureByteStride = sizeof(std::uint32_t)},
        }};
        harness.set = *rachis::test::createBindingSet(device, {.layout = harness.layout, .writes = writes});
        const std::array layouts{harness.layout};
        if (pushBytes == 0)
        {
            harness.pipelineLayout = *device.createPipelineLayout({.bindingLayouts = layouts});
        }
        else
        {
            const std::array push{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, pushBytes}};
            harness.pipelineLayout =
                *device.createPipelineLayout({.bindingLayouts = layouts, .pushConstantRanges = push});
        }
        harness.pipeline =
            *device.createComputePipeline({.layout = harness.pipelineLayout, .computeShader = harness.shader});
        return harness;
    }

    [[nodiscard]] std::vector<std::byte> dispatchAndRead(ComputeHarness& harness,
                                                         std::span<const std::byte> pushConstants = {})
    {
        auto command = rachis::test::acquireCommandBuffer(*harness.device, rachis::QueueType::compute);
        REQUIRE(command.has_value());
        REQUIRE(harness.device
                    ->bufferBarrier(*command,
                                    {.buffer = harness.storage,
                                     .stagesAfter = rachis::PipelineStage::computeShader,
                                     .accessAfter = rachis::Access::unorderedAccess})
                    .has_value());
        REQUIRE(harness.device->bindComputePipeline(*command, harness.pipeline).has_value());
        REQUIRE(harness.device->bindComputeBindingSet(*command, 0, harness.set).has_value());
        if (!pushConstants.empty())
        {
            REQUIRE(harness.device->setComputePushConstants(*command, 0, pushConstants).has_value());
        }
        REQUIRE(harness.device->dispatch(*command, 1).has_value());
        REQUIRE(harness.device
                    ->bufferBarrier(*command,
                                    {.buffer = harness.storage,
                                     .stagesBefore = rachis::PipelineStage::computeShader,
                                     .stagesAfter = rachis::PipelineStage::copy,
                                     .accessBefore = rachis::Access::unorderedAccess,
                                     .accessAfter = rachis::Access::copySource})
                    .has_value());
        REQUIRE(harness.device
                    ->copyBuffer(*command,
                                 {.source = harness.storage, .destination = harness.readback, .size = harness.bytes})
                    .has_value());
        auto submitted = harness.device->submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(harness.device->wait(*submitted).has_value());
        auto mapped = harness.device->mapBuffer(harness.readback);
        REQUIRE(mapped.has_value());
        std::vector<std::byte> bytes(mapped->data, mapped->data + static_cast<std::size_t>(harness.bytes));
        REQUIRE(harness.device->unmapBuffer(harness.readback).has_value());
        return bytes;
    }

} // namespace

TEST_CASE("Metal advertises 16-bit arithmetic, 64-bit integer, and SIMD subgroups only as complete GPU contracts",
          "[size:large][rhi][metal][shader]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto capabilities = device.capabilities();
    REQUIRE(capabilities.has_value());
    REQUIRE(capabilities->shaderFloat16);
    REQUIRE(capabilities->shaderInt16);
    REQUIRE(capabilities->shaderInt64);
    REQUIRE(capabilities->subgroupOperations);

    auto arithmetic = createCompute(device,
                                    "MetalShaderArithmetic.arithmetic16Main.metallib",
                                    "arithmetic16Main",
                                    {1, 1, 1},
                                    2 * sizeof(std::uint32_t),
                                    16);
    const std::array arithmeticPush{
        std::bit_cast<std::uint32_t>(4.0F),
        std::uint32_t{123},
        0U,
        0U,
    };
    const auto arithmeticBytes = dispatchAndRead(arithmetic, std::as_bytes(std::span(arithmeticPush)));
    std::array<std::uint32_t, 2> arithmeticOut{};
    std::memcpy(arithmeticOut.data(), arithmeticBytes.data(), sizeof(arithmeticOut));
    CHECK(arithmeticOut == std::array<std::uint32_t, 2>{8, 364});

    auto int64 = createCompute(
        device, "MetalShaderArithmetic.int64Main.metallib", "int64Main", {1, 1, 1}, 2 * sizeof(std::uint32_t), 0);
    const auto int64Bytes = dispatchAndRead(int64);
    std::array<std::uint32_t, 2> int64Out{};
    std::memcpy(int64Out.data(), int64Bytes.data(), sizeof(int64Out));
    CHECK(int64Out[0] == 7U);
    CHECK(int64Out[1] == 256U);

    auto subgroup = createCompute(device,
                                  "MetalShaderArithmetic.subgroupMain.metallib",
                                  "subgroupMain",
                                  {64, 1, 1},
                                  std::size_t{invocationCount} * valuesPerInvocation * sizeof(std::uint32_t),
                                  0);
    const auto subgroupBytes = dispatchAndRead(subgroup);
    for (std::uint32_t invocation = 0; invocation < invocationCount; ++invocation)
    {
        std::array<std::uint32_t, 4> lane{};
        std::memcpy(lane.data(),
                    subgroupBytes.data() + std::size_t{invocation} * valuesPerInvocation * sizeof(std::uint32_t),
                    sizeof(lane));
        CAPTURE(invocation, lane[0], lane[1], lane[2], lane[3]);
        CHECK(lane[0] >= capabilities->subgroup.minSize);
        CHECK(lane[0] <= capabilities->subgroup.maxSize);
        CHECK(lane[1] < lane[0]);
        CHECK(lane[2] > 0);
        CHECK(lane[2] <= lane[0]);
        CHECK(lane[3] == lane[2] * (lane[2] - 1) / 2);
    }
}
