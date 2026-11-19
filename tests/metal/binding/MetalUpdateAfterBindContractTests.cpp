#include "Metal/MetalTestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
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

    [[nodiscard]] rachis::BufferHandle valueBuffer(rachis::Device& device,
                                                   std::uint32_t value,
                                                   std::string_view label = {})
    {
        auto buffer = device.createBuffer({
            .size = sizeof(std::uint32_t),
            .usage = rachis::BufferUsage::storage,
            .memoryType = rachis::MemoryType::upload,
            .label = label,
        });
        REQUIRE(buffer.has_value());
        auto mapped = device.mapBuffer(*buffer);
        REQUIRE(mapped.has_value());
        std::memcpy(mapped->data, &value, sizeof(value));
        REQUIRE(device.unmapBuffer(*buffer).has_value());
        return *buffer;
    }

    struct ScopedGateRelease final
    {
        std::shared_ptr<rachis::metal::MetalCommitFeedbackGate> gate;

        ~ScopedGateRelease()
        {
            gate->release();
        }
    };

    [[nodiscard]] rachis::ShaderModuleHandle updateAfterBindShader(rachis::Device& device)
    {
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
        return *shader;
    }

    [[nodiscard]] std::array<rachis::BindingLayoutEntry, 2> updateAfterBindEntries() noexcept
    {
        return {
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
    }

    [[nodiscard]] rachis::BindingWrite elementWrite(std::uint32_t element, rachis::BufferHandle buffer) noexcept
    {
        return rachis::BindingWrite{
            .binding = 1,
            .arrayElement = element,
            .resource = rachis::BufferBinding{.buffer = buffer, .structureByteStride = sizeof(std::uint32_t)},
        };
    }

    [[nodiscard]] std::optional<rachis::ResourceMemoryEntry> memoryEntry(const rachis::Device& device,
                                                                         std::string_view label)
    {
        const auto snapshot = device.captureResourceMemorySnapshot();
        REQUIRE(snapshot.has_value());
        const auto found = std::ranges::find(snapshot->entries, label, &rachis::ResourceMemoryEntry::label);
        if (found == snapshot->entries.end())
        {
            return std::nullopt;
        }
        return *found;
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

    struct ScopedGpuHold final
    {
        ScopedGpuHold() noexcept
        {
            rachis::metal::holdNextSubmitOnGpu();
        }
        ~ScopedGpuHold()
        {
            rachis::metal::releaseSubmitHeldOnGpu();
        }
        ScopedGpuHold(const ScopedGpuHold&) = delete;
        ScopedGpuHold& operator=(const ScopedGpuHold&) = delete;
    };
    std::optional<ScopedGpuHold> heldGpu;
    heldGpu.emplace();
    auto inFlight = device.submit(record(0, 0));
    REQUIRE(inFlight.has_value());
    auto recordedBefore = record(1, 1);

    const auto replacement = valueBuffer(device, 777);
    const auto before = device.bindingUpdateStatistics();
    const auto storageBefore = rachis::metal::boundBindingSetStorage(*set);
    REQUIRE(storageBefore.elements != nullptr);
    const std::array update{rachis::BindingWrite{
        .binding = 1,
        .arrayElement = 1,
        .resource = rachis::BufferBinding{.buffer = replacement, .structureByteStride = sizeof(std::uint32_t)},
    }};
    CHECK(device.queryTimelinePoint(*inFlight) == rachis::TimelinePointStatus::pending);
    REQUIRE(device.updateBindingSet(*set, update).has_value());
    CHECK(device.queryTimelinePoint(*inFlight) == rachis::TimelinePointStatus::pending);
    const auto after = device.bindingUpdateStatistics();
    CHECK(after.resourceDescriptorsPublished - before.resourceDescriptorsPublished == 1);
    CHECK(after.resourceDescriptorsPrepared - before.resourceDescriptorsPrepared == 1);
    const auto storageAfter = rachis::metal::boundBindingSetStorage(*set);
    CHECK(storageAfter.version == storageBefore.version);
    CHECK(storageAfter.elements == storageBefore.elements);
    heldGpu.reset();

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

    constexpr std::string_view originalLabel = "Metal in-place original";
    constexpr std::string_view replacementLabel = "Metal in-place replacement";
    const auto original = valueBuffer(device, 100, originalLabel);
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
    auto set =
        rachis::test::createBindingSet(device, {.layout = *layout, .writes = writes, .variableDescriptorCount = 4});
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

    const auto replacement = valueBuffer(device, 888, replacementLabel);
    const std::array update{rachis::BindingWrite{
        .binding = 1,
        .arrayElement = 0,
        .resource = rachis::BufferBinding{.buffer = replacement, .structureByteStride = sizeof(std::uint32_t)},
    }};
    REQUIRE(device.updateBindingSet(*set, update).has_value());
    REQUIRE(device.destroyBindingSet(*set).has_value());
    REQUIRE(device.destroyBuffer(replacement).has_value());
    REQUIRE(device.destroyBuffer(original).has_value());
    for (const auto label : {replacementLabel, originalLabel})
    {
        CAPTURE(label);
        const auto entry = memoryEntry(device, label);
        REQUIRE(entry.has_value());
        CHECK_FALSE(entry->logicalAlive);
    }

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    CHECK_FALSE(memoryEntry(device, replacementLabel).has_value());
    CHECK_FALSE(memoryEntry(device, originalLabel).has_value());

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

TEST_CASE("Metal update-after-bind writes leave the binding array in place without copying it",
          "[size:large][rhi][metal][binding]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).updateAfterBindBindingArrays);
    constexpr std::uint32_t largeArray = 100'000;

    const auto entries = updateAfterBindEntries();
    auto layout = device.createBindingLayout({.entries = entries});
    REQUIRE(layout.has_value());
    const auto value = valueBuffer(device, 5);
    const std::array outputWrite{rachis::BindingWrite{
        .binding = 0,
        .resource = rachis::BufferBinding{.buffer = value, .structureByteStride = sizeof(std::uint32_t)},
    }};
    const std::array sizes{
        rachis::BindingPoolSize{.type = rachis::BindingType::storageBuffer, .descriptorCount = 1},
        rachis::BindingPoolSize{.type = rachis::BindingType::readOnlyStorageBuffer, .descriptorCount = largeArray},
    };
    auto pool = device.createBindingPool({.maxBindingSets = 1, .sizes = sizes});
    REQUIRE(pool.has_value());
    auto set = device.createBindingSet(
        {.layout = *layout, .writes = outputWrite, .variableDescriptorCount = largeArray, .pool = *pool});
    REQUIRE(set.has_value());

    for (const std::uint32_t element : {0U, largeArray / 2U, largeArray - 1U})
    {
        CAPTURE(element);
        const auto storageBefore = rachis::metal::boundBindingSetStorage(*set);
        REQUIRE(storageBefore.version != nullptr);
        REQUIRE(storageBefore.elements != nullptr);
        const auto before = device.bindingUpdateStatistics();
        const std::array update{elementWrite(element, value)};
        REQUIRE(device.updateBindingSet(*set, update).has_value());
        const auto after = device.bindingUpdateStatistics();
        CHECK(after.resourceDescriptorsPrepared - before.resourceDescriptorsPrepared == 1);
        CHECK(after.resourceDescriptorsPublished - before.resourceDescriptorsPublished == 1);
        const auto storageAfter = rachis::metal::boundBindingSetStorage(*set);
        CHECK(storageAfter.version == storageBefore.version);
        CHECK(storageAfter.elements == storageBefore.elements);
    }

    const std::array outOfRange{elementWrite(largeArray, value)};
    auto rejected = device.updateBindingSet(*set, outOfRange);
    REQUIRE_FALSE(rejected.has_value());
    CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
    const std::array repeated{elementWrite(7, value), elementWrite(7, value)};
    rejected = device.updateBindingSet(*set, repeated);
    REQUIRE_FALSE(rejected.has_value());
    CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);

    REQUIRE(device.destroyBindingSet(*set).has_value());
    REQUIRE(device.destroyBindingPool(*pool).has_value());
    REQUIRE(device.destroyBindingLayout(*layout).has_value());
    REQUIRE(device.destroyBuffer(value).has_value());
}

TEST_CASE("Metal update-after-bind writes fail once the binding pool is destroyed", "[size:large][rhi][metal][binding]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).updateAfterBindBindingArrays);

    const auto entries = updateAfterBindEntries();
    auto layout = device.createBindingLayout({.entries = entries});
    REQUIRE(layout.has_value());
    const std::array sizes{
        rachis::BindingPoolSize{.type = rachis::BindingType::storageBuffer, .descriptorCount = 4},
        rachis::BindingPoolSize{.type = rachis::BindingType::readOnlyStorageBuffer, .descriptorCount = 16},
    };
    auto pool = device.createBindingPool({.maxBindingSets = 1, .sizes = sizes});
    REQUIRE(pool.has_value());
    const auto value = valueBuffer(device, 9);
    const std::array outputWrite{rachis::BindingWrite{
        .binding = 0,
        .resource = rachis::BufferBinding{.buffer = value, .structureByteStride = sizeof(std::uint32_t)},
    }};
    auto set = device.createBindingSet(
        {.layout = *layout, .writes = outputWrite, .variableDescriptorCount = 4, .pool = *pool});
    REQUIRE(set.has_value());
    const std::array update{elementWrite(1, value)};
    REQUIRE(device.updateBindingSet(*set, update).has_value());

    REQUIRE(device.destroyBindingPool(*pool).has_value());
    const auto afterDestroy = device.updateBindingSet(*set, update);
    REQUIRE_FALSE(afterDestroy.has_value());
    CHECK(afterDestroy.error().code == rachis::ErrorCode::invalidState);

    REQUIRE(device.destroyBindingSet(*set).has_value());
    REQUIRE(device.destroyBindingLayout(*layout).has_value());
    REQUIRE(device.destroyBuffer(value).has_value());
}

TEST_CASE("Metal update-after-bind writes swap, repeat, and clear elements in one update",
          "[size:large][rhi][metal][binding]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).updateAfterBindBindingArrays);

    const auto shader = updateAfterBindShader(device);
    const auto entries = updateAfterBindEntries();
    auto layout = device.createBindingLayout({.entries = entries});
    REQUIRE(layout.has_value());
    const std::array<std::uint32_t, 4> zeros{};
    auto results = device.createBuffer({
        .size = sizeof(zeros),
        .usage = rachis::BufferUsage::storage,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(results.has_value());
    const auto first = valueBuffer(device, 100);
    const auto second = valueBuffer(device, 200);
    const std::array writes{
        rachis::BindingWrite{
            .binding = 0,
            .resource = rachis::BufferBinding{.buffer = *results, .structureByteStride = sizeof(std::uint32_t)},
        },
        elementWrite(0, first),
        elementWrite(1, second),
    };
    auto set =
        rachis::test::createBindingSet(device, {.layout = *layout, .writes = writes, .variableDescriptorCount = 2});
    REQUIRE(set.has_value());
    const std::array layouts{*layout};
    const std::array pushRanges{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)}};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts, .pushConstantRanges = pushRanges});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = shader});
    REQUIRE(pipeline.has_value());

    const auto readElements = [&](std::span<const std::array<std::uint32_t, 2>> reads) {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command.has_value());
        REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
        REQUIRE(device.bindComputeBindingSet(*command, 0, *set).has_value());
        for (const auto& read : reads)
        {
            REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(read))).has_value());
            REQUIRE(device.dispatch(*command, 1).has_value());
        }
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());
        auto mapped = device.mapBuffer(*results);
        REQUIRE(mapped.has_value());
        std::array<std::uint32_t, 4> observed{};
        std::memcpy(observed.data(), mapped->data, sizeof(observed));
        REQUIRE(device.unmapBuffer(*results).has_value());
        return observed;
    };
    const std::array<std::array<std::uint32_t, 2>, 2> bothElements{{{0, 0}, {1, 1}}};

    const std::array swap{elementWrite(0, second), elementWrite(1, first)};
    REQUIRE(device.updateBindingSet(*set, swap).has_value());
    auto observed = readElements(bothElements);
    CHECK(observed[0] == 200);
    CHECK(observed[1] == 100);

    const std::array repeat{elementWrite(1, second)};
    REQUIRE(device.updateBindingSet(*set, repeat).has_value());
    observed = readElements(bothElements);
    CHECK(observed[0] == 200);
    CHECK(observed[1] == 200);

    const std::array clearAndRestore{
        rachis::BindingWrite{.binding = 1, .arrayElement = 0},
        elementWrite(1, first),
    };
    REQUIRE(device.updateBindingSet(*set, clearAndRestore).has_value());
    const std::array<std::array<std::uint32_t, 2>, 1> secondElement{{{1, 2}}};
    observed = readElements(secondElement);
    CHECK(observed[2] == 100);
    CHECK(device.diagnosticCounts().errors == 0);

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingSet(*set).has_value());
    REQUIRE(device.destroyBindingLayout(*layout).has_value());
    REQUIRE(device.destroyBuffer(second).has_value());
    REQUIRE(device.destroyBuffer(first).has_value());
    REQUIRE(device.destroyBuffer(*results).has_value());
    REQUIRE(device.destroyShaderModule(shader).has_value());
}

TEST_CASE("Metal unretained command pools keep the argument buffer a recorded command addresses",
          "[size:large][rhi][metal][binding][lifetime]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    const auto shader = updateAfterBindShader(device);
    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{
            .binding = 1,
            .type = rachis::BindingType::readOnlyStorageBuffer,
            .visibility = rachis::ShaderStage::compute,
            .count = rachis::unboundedBindingCount,
            .flags = rachis::BindingFlag::variableDescriptorCount | rachis::BindingFlag::partiallyBound,
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
    const auto firstValue = valueBuffer(device, 300);
    const auto secondValue = valueBuffer(device, 400);
    const auto replacement = valueBuffer(device, 500);
    const auto makeSet = [&](rachis::BufferHandle value) {
        const std::array writes{
            rachis::BindingWrite{
                .binding = 0,
                .resource = rachis::BufferBinding{.buffer = *results, .structureByteStride = sizeof(std::uint32_t)},
            },
            elementWrite(0, value),
        };
        auto set =
            rachis::test::createBindingSet(device, {.layout = *layout, .writes = writes, .variableDescriptorCount = 2});
        REQUIRE(set.has_value());
        return *set;
    };
    const auto first = makeSet(firstValue);
    const auto second = makeSet(secondValue);
    const std::array layouts{*layout};
    const std::array pushRanges{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)}};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts, .pushConstantRanges = pushRanges});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = shader});
    REQUIRE(pipeline.has_value());

    auto pool = device.createCommandPool({
        .queueType = rachis::QueueType::compute,
        .maxCommandBuffers = 1,
        .label = "Metal unretained argument-buffer pool",
        .flags = rachis::CommandPoolFlag::unretainedResources,
    });
    REQUIRE(pool.has_value());
    auto command = device.acquireCommandBuffer(*pool);
    REQUIRE(command.has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, first).has_value());
    const std::array<std::uint32_t, 2> readFirst{0, 0};
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(readFirst))).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, second).has_value());
    const std::array<std::uint32_t, 2> readSecond{0, 1};
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(readSecond))).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());

    const auto recordedArgumentBuffer = rachis::metal::boundBindingSetStorage(first).argumentBuffer;
    REQUIRE_FALSE(recordedArgumentBuffer.expired());
    const std::array update{elementWrite(0, replacement)};
    REQUIRE(device.updateBindingSet(first, update).has_value());
    CHECK_FALSE(recordedArgumentBuffer.expired());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    CHECK(recordedArgumentBuffer.expired());

    auto mapped = device.mapBuffer(*results);
    REQUIRE(mapped.has_value());
    std::array<std::uint32_t, 4> observed{};
    std::memcpy(observed.data(), mapped->data, sizeof(observed));
    REQUIRE(device.unmapBuffer(*results).has_value());
    CHECK(observed[0] == 300);
    CHECK(observed[1] == 400);
    CHECK(device.diagnosticCounts().errors == 0);

    REQUIRE(device.destroyCommandPool(*pool).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingSet(second).has_value());
    REQUIRE(device.destroyBindingSet(first).has_value());
    REQUIRE(device.destroyBindingLayout(*layout).has_value());
    REQUIRE(device.destroyBuffer(replacement).has_value());
    REQUIRE(device.destroyBuffer(secondValue).has_value());
    REQUIRE(device.destroyBuffer(firstValue).has_value());
    REQUIRE(device.destroyBuffer(*results).has_value());
    REQUIRE(device.destroyShaderModule(shader).has_value());
}

TEST_CASE("Metal commands keep elements written in place alive when a later update replaces them",
          "[size:large][rhi][metal][binding][lifetime]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).updateAfterBindBindingArrays);

    const auto shader = updateAfterBindShader(device);
    const auto entries = updateAfterBindEntries();
    auto layout = device.createBindingLayout({.entries = entries});
    REQUIRE(layout.has_value());
    const std::array layouts{*layout};
    const std::array pushRanges{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)}};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts, .pushConstantRanges = pushRanges});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = shader});
    REQUIRE(pipeline.has_value());
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

    constexpr std::string_view firstLabel = "Metal in-place first element";
    constexpr std::string_view secondLabel = "Metal in-place second element";
    constexpr std::string_view thirdLabel = "Metal in-place third element";
    const auto reader = valueBuffer(device, 777);
    const auto first = valueBuffer(device, 100, firstLabel);
    const std::array writes{
        rachis::BindingWrite{
            .binding = 0,
            .resource = rachis::BufferBinding{.buffer = *results, .structureByteStride = sizeof(std::uint32_t)},
        },
        elementWrite(0, first),
        elementWrite(1, reader),
    };
    auto set =
        rachis::test::createBindingSet(device, {.layout = *layout, .writes = writes, .variableDescriptorCount = 2});
    REQUIRE(set.has_value());
    const auto recordReadOfSecondElement = [&](rachis::CommandBuffer& command) {
        REQUIRE(device.bindComputePipeline(command, *pipeline).has_value());
        REQUIRE(device.bindComputeBindingSet(command, 0, *set).has_value());
        const std::array<std::uint32_t, 2> read{1, 0};
        REQUIRE(device.setComputePushConstants(command, 0, std::as_bytes(std::span(read))).has_value());
        REQUIRE(device.dispatch(command, 1).has_value());
    };
    const auto requireRetained = [&](std::string_view label) {
        CAPTURE(label);
        const auto entry = memoryEntry(device, label);
        REQUIRE(entry.has_value());
        CHECK_FALSE(entry->logicalAlive);
    };

    SECTION("A command keeps every element it could see until it completes")
    {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command.has_value());
        recordReadOfSecondElement(*command);

        const auto second = valueBuffer(device, 200, secondLabel);
        const std::array beforeSubmit{elementWrite(0, second)};
        REQUIRE(device.updateBindingSet(*set, beforeSubmit).has_value());

        const ScopedGateRelease heldFeedback{std::make_shared<rachis::metal::MetalCommitFeedbackGate>()};
        rachis::metal::holdNextCommitFeedback(heldFeedback.gate);
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        const auto third = valueBuffer(device, 300, thirdLabel);
        const std::array whilePending{elementWrite(0, third)};
        REQUIRE(device.updateBindingSet(*set, whilePending).has_value());
        CHECK(device.queryTimelinePoint(*submitted) == rachis::TimelinePointStatus::pending);

        REQUIRE(device.destroyBindingSet(*set).has_value());
        for (const auto buffer : {first, second, third})
        {
            REQUIRE(device.destroyBuffer(buffer).has_value());
        }
        device.collectGarbage();
        for (const auto label : {firstLabel, secondLabel, thirdLabel})
        {
            requireRetained(label);
        }

        heldFeedback.gate->release();
        REQUIRE(device.wait(*submitted).has_value());
        for (const auto label : {firstLabel, secondLabel, thirdLabel})
        {
            CAPTURE(label);
            CHECK_FALSE(memoryEntry(device, label).has_value());
        }
        auto mapped = device.mapBuffer(*results);
        REQUIRE(mapped.has_value());
        std::uint32_t observed = 0;
        std::memcpy(&observed, mapped->data, sizeof(observed));
        REQUIRE(device.unmapBuffer(*results).has_value());
        CHECK(observed == 777);
    }

    SECTION("Commands bound after an update do not keep the element it replaced")
    {
        const auto second = valueBuffer(device, 200, secondLabel);
        const std::array replace{elementWrite(0, second)};
        REQUIRE(device.updateBindingSet(*set, replace).has_value());
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command.has_value());
        recordReadOfSecondElement(*command);
        REQUIRE(device.destroyBuffer(first).has_value());
        CHECK_FALSE(memoryEntry(device, firstLabel).has_value());

        const auto third = valueBuffer(device, 300, thirdLabel);
        const std::array replaceAgain{elementWrite(0, third)};
        REQUIRE(device.updateBindingSet(*set, replaceAgain).has_value());
        REQUIRE(device.destroyBuffer(second).has_value());
        requireRetained(secondLabel);

        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());
        CHECK_FALSE(memoryEntry(device, secondLabel).has_value());
        REQUIRE(device.destroyBindingSet(*set).has_value());
        REQUIRE(device.destroyBuffer(third).has_value());
    }

    SECTION("Commands from unretained pools do not keep replaced elements alive")
    {
        auto pool = device.createCommandPool({
            .queueType = rachis::QueueType::compute,
            .maxCommandBuffers = 1,
            .label = "Metal unretained in-place pool",
            .flags = rachis::CommandPoolFlag::unretainedResources,
        });
        REQUIRE(pool.has_value());
        {
            auto command = device.acquireCommandBuffer(*pool);
            REQUIRE(command.has_value());
            recordReadOfSecondElement(*command);
            const auto second = valueBuffer(device, 200, secondLabel);
            const std::array replace{elementWrite(0, second)};
            REQUIRE(device.updateBindingSet(*set, replace).has_value());
            REQUIRE(device.destroyBuffer(first).has_value());
            CHECK_FALSE(memoryEntry(device, firstLabel).has_value());
            REQUIRE(device.destroyBindingSet(*set).has_value());
            REQUIRE(device.destroyBuffer(second).has_value());
        }
        REQUIRE(device.destroyCommandPool(*pool).has_value());
    }

    CHECK(device.diagnosticCounts().errors == 0);
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*layout).has_value());
    REQUIRE(device.destroyBuffer(reader).has_value());
    REQUIRE(device.destroyBuffer(*results).has_value());
    REQUIRE(device.destroyShaderModule(shader).has_value());
}

TEST_CASE("Metal binding-set binds that fail keep none of the set's resources alive",
          "[size:large][rhi][metal][binding][lifetime]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    const auto shader = updateAfterBindShader(device);
    const auto entries = updateAfterBindEntries();
    auto layout = device.createBindingLayout({.entries = entries});
    REQUIRE(layout.has_value());
    const std::array layouts{*layout};
    const std::array pushRanges{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)}};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts, .pushConstantRanges = pushRanges});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = shader});
    REQUIRE(pipeline.has_value());

    const rachis::BufferDesc placedDesc{
        .size = 256,
        .usage = rachis::BufferUsage::storage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Metal failed-bind placed buffer",
    };
    const auto requirements = device.queryBufferMemoryRequirements(placedDesc);
    REQUIRE(requirements.has_value());
    auto heap = device.createResourceHeap({
        .size = requirements->size,
        .alignment = requirements->alignment,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::buffer,
    });
    REQUIRE(heap.has_value());
    auto placed = device.createPlacedBuffer({.heap = *heap, .offset = 0}, placedDesc);
    REQUIRE(placed.has_value());
    const auto element = valueBuffer(device, 5);
    const std::array writes{
        rachis::BindingWrite{
            .binding = 0,
            .resource = rachis::BufferBinding{.buffer = *placed, .structureByteStride = sizeof(std::uint32_t)},
        },
        elementWrite(0, element),
    };
    auto set = rachis::test::createBindingSet(device, {.layout = *layout, .writes = writes, .variableDescriptorCount = 1});
    REQUIRE(set.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    rachis::metal::setFailNextBindingResidencyReserve(true);
    const auto bound = device.bindComputeBindingSet(*command, 0, *set);
    rachis::metal::setFailNextBindingResidencyReserve(false);
    REQUIRE_FALSE(bound.has_value());
    CHECK(bound.error().code == rachis::ErrorCode::outOfMemory);

    REQUIRE(device.destroyBindingSet(*set).has_value());
    REQUIRE(device.destroyBuffer(*placed).has_value());
    auto replacement = device.createPlacedBuffer({.heap = *heap, .offset = 0}, placedDesc);
    REQUIRE(replacement.has_value());

    REQUIRE(device.destroyBuffer(*replacement).has_value());
    REQUIRE(device.destroyResourceHeap(*heap).has_value());
    REQUIRE(device.destroyBuffer(element).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*layout).has_value());
    REQUIRE(device.destroyShaderModule(shader).has_value());
}

TEST_CASE("Metal update-after-bind writes grow the replaced elements of a recorded command geometrically",
          "[size:large][rhi][metal][binding][lifetime]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).updateAfterBindBindingArrays);
    constexpr std::uint32_t updates = 1000;

    const auto shader = updateAfterBindShader(device);
    const auto entries = updateAfterBindEntries();
    auto layout = device.createBindingLayout({.entries = entries});
    REQUIRE(layout.has_value());
    const std::array layouts{*layout};
    const std::array pushRanges{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)}};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts, .pushConstantRanges = pushRanges});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = shader});
    REQUIRE(pipeline.has_value());
    const auto results = valueBuffer(device, 0);
    const std::array values{valueBuffer(device, 1), valueBuffer(device, 2)};
    const std::array writes{
        rachis::BindingWrite{
            .binding = 0,
            .resource = rachis::BufferBinding{.buffer = results, .structureByteStride = sizeof(std::uint32_t)},
        },
        elementWrite(0, values[0]),
    };
    auto set = rachis::test::createBindingSet(
        device, {.layout = *layout, .writes = writes, .variableDescriptorCount = 1});
    REQUIRE(set.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *set).has_value());
    const std::array<std::uint32_t, 2> read{0, 0};
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(read))).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());

    std::size_t capacity = 0;
    std::uint32_t growths = 0;
    for (std::uint32_t update = 0; update < updates; ++update)
    {
        const std::array write{elementWrite(0, values[(update + 1) % values.size()])};
        REQUIRE(device.updateBindingSet(*set, write).has_value());
        const auto storage = rachis::metal::boundBindingSetStorage(*set);
        REQUIRE(storage.replacedCount == update + 1);
        if (storage.replacedCapacity != capacity)
        {
            capacity = storage.replacedCapacity;
            ++growths;
        }
    }
    CHECK(growths <= 16);

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingSet(*set).has_value());
    REQUIRE(device.destroyBindingLayout(*layout).has_value());
    for (const auto buffer : values)
    {
        REQUIRE(device.destroyBuffer(buffer).has_value());
    }
    REQUIRE(device.destroyBuffer(results).has_value());
    REQUIRE(device.destroyShaderModule(shader).has_value());
}
