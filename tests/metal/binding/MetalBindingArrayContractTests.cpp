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
    CHECK(capabilities.limits.maxVariableSamplerDescriptors == 1024);

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
