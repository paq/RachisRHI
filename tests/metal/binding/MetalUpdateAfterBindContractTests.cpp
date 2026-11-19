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
#include <string_view>
#include <vector>

namespace
{

    constexpr std::uint32_t arrayLength = 1000;

    [[nodiscard]] rachis::DeviceDesc metalDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
    }

    [[nodiscard]] std::vector<std::byte> loadPnms(std::string_view name)
    {
        std::ifstream stream(std::filesystem::path(RACHIS_METALLIB_DIR) / std::filesystem::path(name),
                             std::ios::binary | std::ios::ate);
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

    [[nodiscard]] rachis::BufferHandle valueBuffer(rachis::Device& device, std::uint32_t value)
    {
        auto buffer = device.createBuffer({
            .size = sizeof(std::uint32_t),
            .usage = rachis::BufferUsage::storage,
            .memoryType = rachis::MemoryType::upload,
        });
        REQUIRE(buffer.has_value());
        auto mapped = device.mapBuffer(*buffer);
        REQUIRE(mapped.has_value());
        std::memcpy(mapped->data, &value, sizeof(value));
        REQUIRE(device.unmapBuffer(*buffer).has_value());
        return *buffer;
    }

} // namespace

TEST_CASE("Metal update-after-bind array elements change in place while other elements are in flight",
          "[size:large][rhi][metal][binding]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    REQUIRE(capabilities.updateAfterBindBindingArrays);

    const auto fixedEntries = std::array{rachis::BindingLayoutEntry{
        .binding = 0,
        .type = rachis::BindingType::readOnlyStorageBuffer,
        .visibility = rachis::ShaderStage::compute,
        .count = 4,
        .flags = rachis::BindingFlag::updateAfterBind,
    }};
    auto fixedLayout = device.createBindingLayout({.entries = fixedEntries});
    REQUIRE_FALSE(fixedLayout.has_value());
    CHECK(fixedLayout.error().code == rachis::ErrorCode::unsupported);

    const auto shaderBytes = loadPnms("MetalUpdateAfterBind.cs.pnms");
    REQUIRE_FALSE(shaderBytes.empty());
    const std::array requirements{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1},
        rachis::ShaderBindingRequirement{
            0, 1, rachis::ShaderResourceType::readOnlyStorageBuffer, rachis::unboundedBindingCount},
    };
    const std::array push{rachis::ShaderPushConstantRequirement{0, 2 * sizeof(std::uint32_t)}};
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = shaderBytes,
        .entryPoint = "updateAfterBindMain",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, requirements, push),
    });
    REQUIRE(shader.has_value());

    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{
            .binding = 1,
            .type = rachis::BindingType::readOnlyStorageBuffer,
            .visibility = rachis::ShaderStage::compute,
            .count = rachis::unboundedBindingCount,
            .flags = rachis::BindingFlag::variableDescriptorCount | rachis::BindingFlag::partiallyBound |
                     rachis::BindingFlag::updateAfterBind,
        },
    };
    auto layout = device.createBindingLayout({.entries = entries});
    REQUIRE(layout.has_value());
    const std::array<std::uint32_t, 4> zeros{};
    auto results = device.createBuffer({
        .size = sizeof(zeros),
        .usage = rachis::BufferUsage::storage,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(results.has_value());
    auto mappedResults = device.mapBuffer(*results);
    REQUIRE(mappedResults.has_value());
    std::memcpy(mappedResults->data, zeros.data(), sizeof(zeros));
    REQUIRE(device.unmapBuffer(*results).has_value());

    std::vector<rachis::BufferHandle> values;
    std::vector<rachis::BindingWrite> writes{rachis::BindingWrite{
        .binding = 0,
        .resource = rachis::BufferBinding{.buffer = *results, .structureByteStride = sizeof(std::uint32_t)},
    }};
    for (std::uint32_t element = 0; element < arrayLength; ++element)
    {
        values.push_back(valueBuffer(device, 100 + element));
        writes.push_back({
            .binding = 1,
            .arrayElement = element,
            .resource = rachis::BufferBinding{.buffer = values.back(), .structureByteStride = sizeof(std::uint32_t)},
        });
    }
    auto set = rachis::test::createBindingSet(
        device, {.layout = *layout, .writes = writes, .variableDescriptorCount = arrayLength});
    REQUIRE(set.has_value());
    const std::array layouts{*layout};
    const std::array pushRanges{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)}};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts, .pushConstantRanges = pushRanges});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *shader});
    REQUIRE(pipeline.has_value());

    const auto record = [&](std::uint32_t element, std::uint32_t slot) {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command.has_value());
        REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
        REQUIRE(device.bindComputeBindingSet(*command, 0, *set).has_value());
        const std::array<std::uint32_t, 2> read{element, slot};
        REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(read))).has_value());
        REQUIRE(device.dispatch(*command, 1).has_value());
        return std::move(*command);
    };

    auto inFlight = device.submit(record(0, 0));
    REQUIRE(inFlight.has_value());
    auto recordedBefore = record(1, 1);

    const auto replacement = valueBuffer(device, 777);
    const auto before = device.bindingUpdateStatistics();
    const std::array update{rachis::BindingWrite{
        .binding = 1,
        .arrayElement = 1,
        .resource = rachis::BufferBinding{.buffer = replacement, .structureByteStride = sizeof(std::uint32_t)},
    }};
    REQUIRE(device.updateBindingSet(*set, update).has_value());
    const auto after = device.bindingUpdateStatistics();
    CHECK(after.resourceDescriptorsPublished - before.resourceDescriptorsPublished == 1);
    CHECK(after.resourceDescriptorsPrepared - before.resourceDescriptorsPrepared == 1);

    auto submittedAfter = device.submit(std::move(recordedBefore));
    REQUIRE(submittedAfter.has_value());
    auto recordedAfter = device.submit(record(1, 2));
    REQUIRE(recordedAfter.has_value());
    REQUIRE(device.wait(*inFlight).has_value());
    REQUIRE(device.wait(*submittedAfter).has_value());
    REQUIRE(device.wait(*recordedAfter).has_value());

    auto mapped = device.mapBuffer(*results);
    REQUIRE(mapped.has_value());
    std::array<std::uint32_t, 4> observed{};
    std::memcpy(observed.data(), mapped->data, sizeof(observed));
    REQUIRE(device.unmapBuffer(*results).has_value());
    CHECK(observed[0] == 100);
    CHECK(observed[1] == 777);
    CHECK(observed[2] == 777);

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingSet(*set).has_value());
    REQUIRE(device.destroyBindingLayout(*layout).has_value());
    REQUIRE(device.destroyBuffer(replacement).has_value());
    for (const auto buffer : values)
    {
        REQUIRE(device.destroyBuffer(buffer).has_value());
    }
    REQUIRE(device.destroyBuffer(*results).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
}

TEST_CASE("Metal commands recorded before an update-after-bind write keep the written resource alive",
          "[size:large][rhi][metal][binding][lifetime]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).updateAfterBindBindingArrays);

    const auto shaderBytes = loadPnms("MetalUpdateAfterBind.cs.pnms");
    REQUIRE_FALSE(shaderBytes.empty());
    const std::array requirements{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1},
        rachis::ShaderBindingRequirement{
            0, 1, rachis::ShaderResourceType::readOnlyStorageBuffer, rachis::unboundedBindingCount},
    };
    const std::array push{rachis::ShaderPushConstantRequirement{0, 2 * sizeof(std::uint32_t)}};
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = shaderBytes,
        .entryPoint = "updateAfterBindMain",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, requirements, push),
    });
    REQUIRE(shader.has_value());
    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{
            .binding = 1,
            .type = rachis::BindingType::readOnlyStorageBuffer,
            .visibility = rachis::ShaderStage::compute,
            .count = rachis::unboundedBindingCount,
            .flags = rachis::BindingFlag::variableDescriptorCount | rachis::BindingFlag::partiallyBound |
                     rachis::BindingFlag::updateAfterBind,
        },
    };
    auto layout = device.createBindingLayout({.entries = entries});
    REQUIRE(layout.has_value());
    const std::array<std::uint32_t, 4> zeros{};
    auto results = device.createBuffer({
        .size = sizeof(zeros),
        .usage = rachis::BufferUsage::storage,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(results.has_value());
    auto mappedResults = device.mapBuffer(*results);
    REQUIRE(mappedResults.has_value());
    std::memcpy(mappedResults->data, zeros.data(), sizeof(zeros));
    REQUIRE(device.unmapBuffer(*results).has_value());

    const auto original = valueBuffer(device, 100);
    const std::array writes{
        rachis::BindingWrite{
            .binding = 0,
            .resource = rachis::BufferBinding{.buffer = *results, .structureByteStride = sizeof(std::uint32_t)},
        },
        rachis::BindingWrite{
            .binding = 1,
            .arrayElement = 0,
            .resource = rachis::BufferBinding{.buffer = original, .structureByteStride = sizeof(std::uint32_t)},
        },
    };
    auto set = rachis::test::createBindingSet(device, {.layout = *layout, .writes = writes, .variableDescriptorCount = 4});
    REQUIRE(set.has_value());
    const std::array layouts{*layout};
    const std::array pushRanges{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)}};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts, .pushConstantRanges = pushRanges});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *shader});
    REQUIRE(pipeline.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *set).has_value());
    const std::array<std::uint32_t, 2> read{0, 0};
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(read))).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());

    const auto replacement = valueBuffer(device, 888);
    const std::array update{rachis::BindingWrite{
        .binding = 1,
        .arrayElement = 0,
        .resource = rachis::BufferBinding{.buffer = replacement, .structureByteStride = sizeof(std::uint32_t)},
    }};
    REQUIRE(device.updateBindingSet(*set, update).has_value());
    REQUIRE(device.destroyBindingSet(*set).has_value());
    REQUIRE(device.destroyBuffer(replacement).has_value());
    REQUIRE(device.destroyBuffer(original).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto mapped = device.mapBuffer(*results);
    REQUIRE(mapped.has_value());
    std::uint32_t observed = 0;
    std::memcpy(&observed, mapped->data, sizeof(observed));
    REQUIRE(device.unmapBuffer(*results).has_value());
    CHECK(observed == 888);
    CHECK(device.diagnosticCounts().errors == 0);

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*layout).has_value());
    REQUIRE(device.destroyBuffer(*results).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
}
