#include "Internal/MetalShaderManifest.h"
#include "Metal/MetalTestHooks.h"
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
#include <string>
#include <string_view>
#include <vector>

namespace
{

    constexpr std::uint32_t textureCount = 200;
    constexpr std::uint32_t bufferCount = 40;
    constexpr std::uint32_t resultCount = 256;
    constexpr std::uint32_t uploadPitch = 256;

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

    [[nodiscard]] rachis::BufferHandle createFilledBuffer(rachis::Device& device,
                                                          std::span<const std::byte> bytes,
                                                          rachis::BufferUsage usage)
    {
        auto buffer = device.createBuffer({
            .size = bytes.size(),
            .usage = usage,
            .memoryType = rachis::MemoryType::upload,
        });
        REQUIRE(buffer.has_value());
        auto mapped = device.mapBuffer(*buffer);
        REQUIRE(mapped.has_value());
        std::memcpy(mapped->data, bytes.data(), bytes.size());
        REQUIRE(device.unmapBuffer(*buffer).has_value());
        return *buffer;
    }

} // namespace

TEST_CASE("Metal binding arrays exceed the argument-table slots through argument buffers",
          "[size:large][rhi][metal][binding]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    REQUIRE(capabilities.variableDescriptorCountBindingArrays);
    CHECK(capabilities.limits.maxVariableResourceDescriptors == 500000);
    CHECK(capabilities.limits.maxVariableBufferDescriptors == 500000);
    const auto family = rachis::metal::boundAppleGpuFamily();
    REQUIRE(family >= 7);
    CHECK(capabilities.limits.maxVariableSamplerDescriptors == (family >= 9 ? 1024U : 996U));

    const auto shaderBytes = loadPnms("MetalBindingArrays.cs.pnms");
    REQUIRE_FALSE(shaderBytes.empty());
    const std::array requirements{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1},
        rachis::ShaderBindingRequirement{
            1, 0, rachis::ShaderResourceType::sampledTexture, rachis::unboundedBindingCount},
        rachis::ShaderBindingRequirement{
            2, 0, rachis::ShaderResourceType::readOnlyStorageBuffer, rachis::unboundedBindingCount},
    };
    const std::array push{rachis::ShaderPushConstantRequirement{0, 2 * sizeof(std::uint32_t)}};
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = shaderBytes,
        .entryPoint = "bindingArraysMain",
        .shaderInterface = rachis::test::computeShaderInterface({64, 1, 1}, requirements, push),
    });
    REQUIRE(shader.has_value());

    std::vector<rachis::TextureHandle> textures;
    std::vector<rachis::TextureViewHandle> views;
    for (std::uint32_t index = 0; index < textureCount; ++index)
    {
        auto texture = device.createTexture({
            .width = 1,
            .height = 1,
            .format = rachis::Format::r32Float,
            .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::copyDestination,
        });
        REQUIRE(texture.has_value());
        auto view = device.createTextureView({.texture = *texture, .usage = rachis::TextureViewUsage::sampled});
        REQUIRE(view.has_value());
        textures.push_back(*texture);
        views.push_back(*view);
    }
    std::vector<std::byte> uploadBytes(static_cast<std::size_t>(uploadPitch) * textureCount);
    for (std::uint32_t index = 0; index < textureCount; ++index)
    {
        const auto value = static_cast<float>(index);
        std::memcpy(uploadBytes.data() + static_cast<std::size_t>(index) * uploadPitch, &value, sizeof(value));
    }
    const auto upload = createFilledBuffer(device, uploadBytes, rachis::BufferUsage::copySource);

    std::vector<rachis::BufferHandle> buffers;
    for (std::uint32_t index = 0; index < bufferCount; ++index)
    {
        const std::array<std::uint32_t, 1> value{index};
        buffers.push_back(createFilledBuffer(device, std::as_bytes(std::span(value)), rachis::BufferUsage::storage));
    }
    const std::vector<std::byte> zeros(static_cast<std::size_t>(resultCount) * sizeof(std::uint32_t));
    const auto results = createFilledBuffer(device, zeros, rachis::BufferUsage::storage);

    const std::array outputEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    const std::array textureEntries{rachis::BindingLayoutEntry{
        .binding = 0,
        .type = rachis::BindingType::sampledTexture,
        .visibility = rachis::ShaderStage::compute,
        .count = rachis::unboundedBindingCount,
        .flags = rachis::BindingFlag::variableDescriptorCount | rachis::BindingFlag::partiallyBound,
    }};
    const std::array bufferEntries{rachis::BindingLayoutEntry{
        .binding = 0,
        .type = rachis::BindingType::readOnlyStorageBuffer,
        .visibility = rachis::ShaderStage::compute,
        .count = rachis::unboundedBindingCount,
        .flags = rachis::BindingFlag::variableDescriptorCount | rachis::BindingFlag::partiallyBound,
    }};
    auto outputLayout = device.createBindingLayout({.entries = outputEntries});
    auto textureLayout = device.createBindingLayout({.entries = textureEntries});
    auto bufferLayout = device.createBindingLayout({.entries = bufferEntries});
    REQUIRE(outputLayout.has_value());
    REQUIRE(textureLayout.has_value());
    REQUIRE(bufferLayout.has_value());

    const std::array outputWrites{rachis::BindingWrite{
        .binding = 0,
        .resource = rachis::BufferBinding{.buffer = results, .structureByteStride = sizeof(std::uint32_t)},
    }};
    std::vector<rachis::BindingWrite> textureWrites;
    for (std::uint32_t index = 0; index < textureCount; ++index)
    {
        textureWrites.push_back({.binding = 0, .arrayElement = index, .resource = views[index]});
    }
    std::vector<rachis::BindingWrite> bufferWrites;
    for (std::uint32_t index = 0; index < bufferCount; ++index)
    {
        bufferWrites.push_back({
            .binding = 0,
            .arrayElement = index,
            .resource = rachis::BufferBinding{.buffer = buffers[index], .structureByteStride = sizeof(std::uint32_t)},
        });
    }
    auto outputSet = rachis::test::createBindingSet(device, {.layout = *outputLayout, .writes = outputWrites});
    auto textureSet = rachis::test::createBindingSet(
        device, {.layout = *textureLayout, .writes = textureWrites, .variableDescriptorCount = textureCount});
    auto bufferSet = rachis::test::createBindingSet(
        device, {.layout = *bufferLayout, .writes = bufferWrites, .variableDescriptorCount = bufferCount});
    REQUIRE(outputSet.has_value());
    REQUIRE(textureSet.has_value());
    REQUIRE(bufferSet.has_value());

    const std::array layouts{*outputLayout, *textureLayout, *bufferLayout};
    const std::array pushRanges{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)}};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts, .pushConstantRanges = pushRanges});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *shader});
    REQUIRE(pipeline.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    for (std::uint32_t index = 0; index < textureCount; ++index)
    {
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = textures[index],
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessAfter = rachis::Access::copyDestination,
                                        .layoutAfter = rachis::TextureLayout::copyDestination,
                                    })
                    .has_value());
        REQUIRE(device
                    .copyBufferToTexture(*command,
                                         {
                                             .buffer = upload,
                                             .bufferOffset = static_cast<std::uint64_t>(index) * uploadPitch,
                                             .bytesPerRow = uploadPitch,
                                             .rowsPerImage = 1,
                                             .texture = textures[index],
                                             .extent = {1, 1, 1},
                                         })
                    .has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = textures[index],
                                        .stagesBefore = rachis::PipelineStage::copy,
                                        .stagesAfter = rachis::PipelineStage::computeShader,
                                        .accessBefore = rachis::Access::copyDestination,
                                        .accessAfter = rachis::Access::shaderResource,
                                        .layoutBefore = rachis::TextureLayout::copyDestination,
                                        .layoutAfter = rachis::TextureLayout::shaderResource,
                                    })
                    .has_value());
    }
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *outputSet).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 1, *textureSet).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 2, *bufferSet).has_value());
    const std::array<std::uint32_t, 2> counts{textureCount, bufferCount};
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(counts))).has_value());
    REQUIRE(device.dispatch(*command, resultCount / 64).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto mapped = device.mapBuffer(results);
    REQUIRE(mapped.has_value());
    std::vector<std::uint32_t> values(resultCount);
    std::memcpy(values.data(), mapped->data, values.size() * sizeof(std::uint32_t));
    REQUIRE(device.unmapBuffer(results).has_value());
    for (std::uint32_t index = 0; index < resultCount; ++index)
    {
        const std::uint32_t expected = (index < textureCount ? index : 0U) + (index < bufferCount ? 1000U * index : 0U);
        CAPTURE(index);
        CHECK(values[index] == expected);
    }

    auto overLimit = rachis::test::createBindingSet(
        device,
        {.layout = *textureLayout, .variableDescriptorCount = capabilities.limits.maxVariableResourceDescriptors + 1});
    REQUIRE_FALSE(overLimit.has_value());
    CHECK(overLimit.error().code == rachis::ErrorCode::invalidArgument);

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingSet(*bufferSet).has_value());
    REQUIRE(device.destroyBindingSet(*textureSet).has_value());
    REQUIRE(device.destroyBindingSet(*outputSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bufferLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*textureLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*outputLayout).has_value());
    REQUIRE(device.destroyBuffer(results).has_value());
    for (const auto buffer : buffers)
    {
        REQUIRE(device.destroyBuffer(buffer).has_value());
    }
    REQUIRE(device.destroyBuffer(upload).has_value());
    for (std::uint32_t index = 0; index < textureCount; ++index)
    {
        REQUIRE(device.destroyTextureView(views[index]).has_value());
        REQUIRE(device.destroyTexture(textures[index]).has_value());
    }
    REQUIRE(device.destroyShaderModule(*shader).has_value());
}

TEST_CASE("Metal sampler binding arrays stay within the per-stage argument-buffer sampler limit",
          "[size:large][rhi][metal][binding]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    const auto family = rachis::metal::boundAppleGpuFamily();
    REQUIRE(family >= 7);
    const auto samplerLimit = capabilities.limits.maxVariableSamplerDescriptors;
    REQUIRE(samplerLimit == (family >= 9 ? 1024U : 996U));
    constexpr std::uint32_t fittingCount = 400;
    constexpr std::uint32_t overflowingCount = 600;

    const auto shaderBytes = loadPnms("MetalSamplerArrays.cs.pnms");
    REQUIRE_FALSE(shaderBytes.empty());
    const std::array requirements{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1},
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::sampledTexture, 1},
        rachis::ShaderBindingRequirement{1, 0, rachis::ShaderResourceType::sampler, rachis::unboundedBindingCount},
        rachis::ShaderBindingRequirement{2, 0, rachis::ShaderResourceType::sampler, rachis::unboundedBindingCount},
    };
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = shaderBytes,
        .entryPoint = "samplerArraysMain",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, requirements),
    });
    REQUIRE(shader.has_value());

    auto texture = device.createTexture({
        .width = 1,
        .height = 1,
        .format = rachis::Format::r32Float,
        .usage = rachis::TextureUsage::sampled,
    });
    REQUIRE(texture.has_value());
    auto view = device.createTextureView({.texture = *texture, .usage = rachis::TextureViewUsage::sampled});
    REQUIRE(view.has_value());
    auto sampler = device.createSampler({});
    REQUIRE(sampler.has_value());
    const std::vector<std::byte> zeros(sizeof(float));
    const auto results = createFilledBuffer(device, zeros, rachis::BufferUsage::storage);

    const std::array imageEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute, 1},
    };
    const std::array samplerEntries{rachis::BindingLayoutEntry{
        .binding = 0,
        .type = rachis::BindingType::sampler,
        .visibility = rachis::ShaderStage::compute,
        .count = rachis::unboundedBindingCount,
        .flags = rachis::BindingFlag::variableDescriptorCount | rachis::BindingFlag::partiallyBound,
    }};
    auto imageLayout = device.createBindingLayout({.entries = imageEntries});
    auto samplerLayout = device.createBindingLayout({.entries = samplerEntries});
    REQUIRE(imageLayout.has_value());
    REQUIRE(samplerLayout.has_value());

    const std::array imageWrites{
        rachis::BindingWrite{
            .binding = 0,
            .resource = rachis::BufferBinding{.buffer = results, .structureByteStride = sizeof(float)},
        },
        rachis::BindingWrite{.binding = 1, .resource = *view},
    };
    auto imageSet = rachis::test::createBindingSet(device, {.layout = *imageLayout, .writes = imageWrites});
    REQUIRE(imageSet.has_value());
    const std::array samplerWrites{rachis::BindingWrite{.binding = 0, .resource = *sampler}};
    const auto samplerSet = [&](std::uint32_t count) {
        auto set = rachis::test::createBindingSet(
            device, {.layout = *samplerLayout, .writes = samplerWrites, .variableDescriptorCount = count});
        REQUIRE(set.has_value());
        return *set;
    };

    auto overLimit = rachis::test::createBindingSet(
        device, {.layout = *samplerLayout, .writes = samplerWrites, .variableDescriptorCount = samplerLimit + 1});
    REQUIRE_FALSE(overLimit.has_value());
    CHECK(overLimit.error().code == rachis::ErrorCode::invalidArgument);
    const auto atLimit = samplerSet(samplerLimit);

    const std::array layouts{*imageLayout, *samplerLayout, *samplerLayout};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *shader});
    REQUIRE(pipeline.has_value());

    const auto dispatchWith = [&](rachis::BindingSetHandle first, rachis::BindingSetHandle second) {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command.has_value());
        REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
        const std::array sets{*imageSet, first, second};
        REQUIRE(device.bindComputeBindingSets(*command, 0, sets).has_value());
        auto dispatched = device.dispatch(*command, 1);
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());
        return dispatched;
    };

    const auto overflowingFirst = samplerSet(overflowingCount);
    const auto overflowingSecond = samplerSet(overflowingCount);
    const auto fittingFirst = samplerSet(fittingCount);
    const auto fittingSecond = samplerSet(fittingCount);
    const auto overflowing = dispatchWith(overflowingFirst, overflowingSecond);
    REQUIRE_FALSE(overflowing.has_value());
    CHECK(overflowing.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(dispatchWith(fittingFirst, fittingSecond).has_value());
    CHECK(device.diagnosticCounts().errors == 0);

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    for (const auto set : {fittingSecond, fittingFirst, overflowingSecond, overflowingFirst, atLimit, *imageSet})
    {
        REQUIRE(device.destroyBindingSet(set).has_value());
    }
    REQUIRE(device.destroyBindingLayout(*samplerLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*imageLayout).has_value());
    REQUIRE(device.destroyBuffer(results).has_value());
    REQUIRE(device.destroySampler(*sampler).has_value());
    REQUIRE(device.destroyTextureView(*view).has_value());
    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
}

TEST_CASE("Metal rejects a bound binding array larger than the shader's compiled argument array",
          "[size:large][rhi][metal][binding]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    constexpr std::uint32_t compiledCapacity = 4;

    const auto cooked = loadPnms("MetalUpdateAfterBind.cs.pnms");
    REQUIRE_FALSE(cooked.empty());
    const auto payload = rachis::metal::unpackShaderManifest(cooked);
    REQUIRE(payload.has_value());
    std::string json(reinterpret_cast<const char*>(payload->json.data()), payload->json.size());
    const std::string cookedCapacity = "\"capacity\": 500000";
    const auto found = json.find(cookedCapacity);
    REQUIRE(found != std::string::npos);
    json.replace(found, cookedCapacity.size(), "\"capacity\": " + std::to_string(compiledCapacity));
    const auto shaderBytes = rachis::metal::packShaderManifest(std::as_bytes(std::span(json)), payload->metallib);

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
            .flags = rachis::BindingFlag::variableDescriptorCount | rachis::BindingFlag::partiallyBound,
        },
    };
    auto layout = device.createBindingLayout({.entries = entries});
    REQUIRE(layout.has_value());
    const std::array layouts{*layout};
    const std::array pushRanges{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)}};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts, .pushConstantRanges = pushRanges});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *shader});
    REQUIRE(pipeline.has_value());

    const std::array<std::uint32_t, 1> value{42};
    const auto source = createFilledBuffer(device, std::as_bytes(std::span(value)), rachis::BufferUsage::storage);
    const std::vector<std::byte> zeros(sizeof(std::uint32_t));
    const auto results = createFilledBuffer(device, zeros, rachis::BufferUsage::storage);
    const std::array writes{
        rachis::BindingWrite{
            .binding = 0,
            .resource = rachis::BufferBinding{.buffer = results, .structureByteStride = sizeof(std::uint32_t)},
        },
        rachis::BindingWrite{
            .binding = 1,
            .resource = rachis::BufferBinding{.buffer = source, .structureByteStride = sizeof(std::uint32_t)},
        },
    };
    const auto dispatchWith = [&](std::uint32_t count) {
        auto set = rachis::test::createBindingSet(
            device, {.layout = *layout, .writes = writes, .variableDescriptorCount = count});
        REQUIRE(set.has_value());
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command.has_value());
        REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
        REQUIRE(device.bindComputeBindingSet(*command, 0, *set).has_value());
        const std::array<std::uint32_t, 2> read{0, 0};
        REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(read))).has_value());
        auto dispatched = device.dispatch(*command, 1);
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());
        REQUIRE(device.destroyBindingSet(*set).has_value());
        return dispatched;
    };

    const auto oversized = dispatchWith(compiledCapacity * 2);
    REQUIRE_FALSE(oversized.has_value());
    CHECK(oversized.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(dispatchWith(compiledCapacity).has_value());
    auto mapped = device.mapBuffer(results);
    REQUIRE(mapped.has_value());
    std::uint32_t observed = 0;
    std::memcpy(&observed, mapped->data, sizeof(observed));
    REQUIRE(device.unmapBuffer(results).has_value());
    CHECK(observed == 42);

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*layout).has_value());
    REQUIRE(device.destroyBuffer(results).has_value());
    REQUIRE(device.destroyBuffer(source).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
}
