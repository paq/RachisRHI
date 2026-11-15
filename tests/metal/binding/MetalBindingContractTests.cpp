#include "Internal/MetalShaderBinding.h"
#include "Metal/MetalTestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string_view>
#include <vector>

namespace
{

    constexpr std::uint64_t capturedBytes = 256;

    [[nodiscard]] rachis::DeviceDesc metalDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
    }

    [[nodiscard]] std::byte capturedPattern(std::uint64_t index) noexcept
    {
        return static_cast<std::byte>((index * 19U + 11U) & 0xFFU);
    }

    [[nodiscard]] std::vector<std::byte> loadPnmsBytes(std::string_view metallibName)
    {
        const auto path =
            std::filesystem::path(RACHIS_METALLIB_DIR) / std::filesystem::path(metallibName).replace_extension(".pnms");
        INFO("pnms=" << path.string());
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        REQUIRE(stream);
        const auto end = stream.tellg();
        REQUIRE(end > 0);
        std::vector<std::byte> bytes(static_cast<std::size_t>(end));
        stream.seekg(0, std::ios::beg);
        stream.read(reinterpret_cast<char*>(bytes.data()), end);
        REQUIRE(stream);
        return bytes;
    }

    template<typename T>
    [[nodiscard]] std::span<const std::byte> asByteSpan(const T& value) noexcept
    {
        return {reinterpret_cast<const std::byte*>(&value), sizeof(T)};
    }

    [[nodiscard]] std::uint32_t floatBits(float value) noexcept
    {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    struct CompactUniforms24Host
    {
        std::uint32_t w0 = 0;
        std::uint32_t w1 = 0;
        std::uint32_t w2 = 0;
        std::uint32_t w3 = 0;
        std::uint32_t t0 = 0;
        std::uint32_t t1 = 0;
    };
    static_assert(sizeof(CompactUniforms24Host) == 24);

    struct CompactUniforms16Host
    {
        std::array<std::uint32_t, 4> words{};
    };
    static_assert(sizeof(CompactUniforms16Host) == 16);

    struct RemapVertexHost
    {
        std::array<float, 2> position{};
        std::uint32_t vertexTag = 0;
    };
    static_assert(sizeof(RemapVertexHost) == 12);

    struct UploadedQuadTexture final
    {
        rachis::TextureHandle texture{};
        rachis::TextureViewHandle view{};
        rachis::BufferHandle staging{};
    };

    [[nodiscard]] UploadedQuadTexture create2x2RgbaTexture(rachis::Device& device,
                                                           const std::array<std::array<std::uint8_t, 4>, 4>& texels,
                                                           rachis::PipelineStage consumerStages,
                                                           std::string_view label)
    {
        auto texture = device.createTexture({
            .width = 2,
            .height = 2,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::copyDestination,
            .label = label,
        });
        REQUIRE(texture);
        auto view = device.createTextureView({
            .texture = *texture,
            .usage = rachis::TextureViewUsage::sampled,
            .dimension = rachis::TextureViewDimension::texture2D,
        });
        REQUIRE(view);

        const rachis::BufferTextureCopyRegion queryRegion{
            .extent = {.width = 2, .height = 2, .depth = 1},
        };
        const auto layout = device.queryTextureTransferLayout(*texture, queryRegion);
        REQUIRE(layout);

        auto staging = device.createBuffer({
            .size = layout->requiredBufferSize,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
            .label = label,
        });
        REQUIRE(staging);

        std::vector<std::byte> uploadBytes(static_cast<std::size_t>(layout->requiredBufferSize), std::byte{0});
        std::memcpy(uploadBytes.data(), texels[0].data(), 4);
        std::memcpy(uploadBytes.data() + 4, texels[1].data(), 4);
        std::memcpy(uploadBytes.data() + layout->bytesPerRow, texels[2].data(), 4);
        std::memcpy(uploadBytes.data() + layout->bytesPerRow + 4, texels[3].data(), 4);
        REQUIRE(device.writeBuffer(*staging, 0, uploadBytes).has_value());

        const auto queue = consumerStages == rachis::PipelineStage::computeShader ? rachis::QueueType::compute
                                                                                  : rachis::QueueType::graphics;
        auto command = rachis::test::acquireCommandBuffer(device, queue);
        REQUIRE(command);
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *texture,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessAfter = rachis::Access::copyDestination,
                                        .layoutAfter = rachis::TextureLayout::copyDestination,
                                    })
                    .has_value());
        const rachis::BufferTextureCopyRegion copyRegion{
            .bytesPerRow = layout->bytesPerRow,
            .rowsPerImage = layout->rowsPerImage,
            .extent = {.width = 2, .height = 2, .depth = 1},
        };
        REQUIRE(device.copyBufferToTexture(*command, *staging, *texture, std::array{copyRegion}).has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *texture,
                                        .stagesBefore = rachis::PipelineStage::copy,
                                        .stagesAfter = consumerStages,
                                        .accessBefore = rachis::Access::copyDestination,
                                        .accessAfter = rachis::Access::shaderResource,
                                        .layoutBefore = rachis::TextureLayout::copyDestination,
                                        .layoutAfter = rachis::TextureLayout::shaderResource,
                                    })
                    .has_value());
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted);
        REQUIRE(device.wait(*submitted).has_value());

        return UploadedQuadTexture{
            .texture = *texture,
            .view = *view,
            .staging = *staging,
        };
    }

    [[nodiscard]] std::array<std::byte, rachis::metalConstantBufferSize> makeLayoutFixturePayload(std::uint32_t marker)
    {
        std::array<std::byte, rachis::metalConstantBufferSize> bytes{};
        auto writeFloat = [&](std::size_t offset, float value) {
            std::memcpy(bytes.data() + offset, &value, sizeof(value));
        };
        writeFloat(rachis::metalNestedScalarOffset, 1.5F);
        writeFloat(rachis::metalNestedFloat3Offset + 0, 2.5F);
        writeFloat(rachis::metalNestedFloat3Offset + 4, 3.5F);
        writeFloat(rachis::metalNestedFloat3Offset + 8, 4.5F);
        writeFloat(rachis::metalNonSquareMatrixOffset + 0, 6.25F);
        writeFloat(rachis::metalArrayOffset + 4, 14.0F);
        std::memcpy(bytes.data() + rachis::metalMarkerOffset, &marker, sizeof(marker));
        return bytes;
    }

} // namespace

TEST_CASE("Metal binding layouts, pools, and immutable sets reject malformed state atomically",
          "[size:large][rhi][metal][binding]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto checkedCapabilities = device.capabilities();
    REQUIRE(checkedCapabilities.has_value());
    REQUIRE(checkedCapabilities->partiallyBoundBindingArrays);
    REQUIRE(checkedCapabilities->variableDescriptorCountBindingArrays);

    auto constants = device.createBuffer(rachis::BufferDesc{
        .size = 512,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal binding constants",
    });
    auto storage = device.createBuffer(rachis::BufferDesc{
        .size = 1024,
        .usage = rachis::BufferUsage::storage,
        .label = "Metal binding storage",
    });
    auto texture = device.createTexture(rachis::TextureDesc{
        .width = 16,
        .height = 16,
        .mipLevels = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
        .label = "Metal binding texture",
    });
    REQUIRE(constants);
    REQUIRE(storage);
    REQUIRE(texture);
    auto sampledView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *texture,
        .usage = rachis::TextureViewUsage::sampled,
        .dimension = rachis::TextureViewDimension::texture2D,
    });
    auto storageView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *texture,
        .usage = rachis::TextureViewUsage::storage,
        .dimension = rachis::TextureViewDimension::texture2D,
        .firstMipLevel = 1,
    });
    auto sampler = device.createSampler(rachis::SamplerDesc{.label = "Metal binding sampler"});
    REQUIRE(sampledView);
    REQUIRE(storageView);
    REQUIRE(sampler);

    const std::array layoutEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::vertex, 1},
        rachis::BindingLayoutEntry{2, rachis::BindingType::readOnlyStorageBuffer, rachis::ShaderStage::all, 1},
        rachis::BindingLayoutEntry{4, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{6, rachis::BindingType::sampledTexture, rachis::ShaderStage::pixel, 2},
        rachis::BindingLayoutEntry{8, rachis::BindingType::storageTexture, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{10, rachis::BindingType::sampler, rachis::ShaderStage::pixel, 2},
    };
    auto layout = device.createBindingLayout({.entries = layoutEntries, .label = "Metal complete layout"});
    REQUIRE(layout);

    const std::array writes{
        rachis::BindingWrite{0, 0, rachis::BufferBinding{*constants, 0, 512}},
        rachis::BindingWrite{2, 0, rachis::BufferBinding{*storage, 0, 256}},
        rachis::BindingWrite{4, 0, rachis::BufferBinding{*storage, 256, 256, 16}},
        rachis::BindingWrite{6, 0, *sampledView},
        rachis::BindingWrite{6, 1, *sampledView},
        rachis::BindingWrite{8, 0, *storageView},
        rachis::BindingWrite{10, 0, *sampler},
        rachis::BindingWrite{10, 1, *sampler},
    };
    auto set = rachis::test::createBindingSet(device,
                                              rachis::BindingSetDesc{
                                                  .layout = *layout,
                                                  .writes = writes,
                                                  .label = "Metal complete set",
                                              });
    REQUIRE(set);

    const std::array duplicateBindings{
        rachis::BindingLayoutEntry{1, rachis::BindingType::uniformBuffer},
        rachis::BindingLayoutEntry{1, rachis::BindingType::sampler},
    };
    CHECK_FALSE(device.createBindingLayout({.entries = duplicateBindings}).has_value());
    const std::array overlappingRegisters{
        rachis::BindingLayoutEntry{1, rachis::BindingType::sampledTexture, rachis::ShaderStage::pixel, 4},
        rachis::BindingLayoutEntry{3, rachis::BindingType::storageTexture},
    };
    CHECK_FALSE(device.createBindingLayout({.entries = overlappingRegisters}).has_value());
    const std::array overlappingBufferRegisters{
        rachis::BindingLayoutEntry{0, rachis::BindingType::dynamicUniformBuffer, rachis::ShaderStage::compute, 2},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    auto overlappingBuffers = device.createBindingLayout({.entries = overlappingBufferRegisters});
    REQUIRE_FALSE(overlappingBuffers.has_value());
    CHECK(overlappingBuffers.error().code == rachis::ErrorCode::invalidArgument);
    const std::array separateNamespaces{
        overlappingRegisters[0],
        rachis::BindingLayoutEntry{3, rachis::BindingType::readOnlyStorageBuffer},
    };
    auto separateLayout = device.createBindingLayout({.entries = separateNamespaces});
    REQUIRE(separateLayout);
    REQUIRE(device.destroyBindingLayout(*separateLayout));
    const std::array zeroCount{
        rachis::BindingLayoutEntry{0, rachis::BindingType::sampler, rachis::ShaderStage::pixel, 0},
    };
    CHECK_FALSE(device.createBindingLayout({.entries = zeroCount}).has_value());
    const std::array acceleration{
        rachis::BindingLayoutEntry{0, rachis::BindingType::accelerationStructure, rachis::ShaderStage::compute, 1},
    };
    auto accelerationLayout = device.createBindingLayout({.entries = acceleration});
    REQUIRE_FALSE(accelerationLayout);
    CHECK(accelerationLayout.error().code == rachis::ErrorCode::unsupported);
    const std::array feedback{
        rachis::BindingLayoutEntry{0, rachis::BindingType::minMipFeedback, rachis::ShaderStage::pixel, 1},
    };
    auto feedbackLayout = device.createBindingLayout({.entries = feedback});
    REQUIRE_FALSE(feedbackLayout);
    CHECK(feedbackLayout.error().code == rachis::ErrorCode::unsupported);

    const std::array variable{
        rachis::BindingLayoutEntry{
            .binding = 0,
            .type = rachis::BindingType::sampledTexture,
            .count = rachis::unboundedBindingCount,
            .flags = rachis::BindingFlag::variableDescriptorCount,
        },
    };
    auto variableLayout = device.createBindingLayout({.entries = variable});
    REQUIRE(variableLayout);
    REQUIRE(device.destroyBindingLayout(*variableLayout));
    const std::array variableWithoutUnbounded{
        rachis::BindingLayoutEntry{
            .binding = 0,
            .type = rachis::BindingType::sampledTexture,
            .count = 4,
            .flags = rachis::BindingFlag::variableDescriptorCount,
        },
    };
    CHECK_FALSE(device.createBindingLayout({.entries = variableWithoutUnbounded}).has_value());

    CHECK_FALSE(rachis::test::createBindingSet(
                    device,
                    rachis::BindingSetDesc{
                        .layout = *layout,
                        .writes = std::span<const rachis::BindingWrite>{writes}.first(writes.size() - 1),
                    })
                    .has_value());
    auto wrongTypeWrites = writes;
    wrongTypeWrites[0].resource = *sampler;
    CHECK_FALSE(rachis::test::createBindingSet(device,
                                               rachis::BindingSetDesc{
                                                   .layout = *layout,
                                                   .writes = wrongTypeWrites,
                                               })
                    .has_value());
    auto wrongViewWrites = writes;
    wrongViewWrites[3].resource = *storageView;
    CHECK_FALSE(rachis::test::createBindingSet(device,
                                               rachis::BindingSetDesc{
                                                   .layout = *layout,
                                                   .writes = wrongViewWrites,
                                               })
                    .has_value());
    auto misalignedWrites = writes;
    misalignedWrites[0].resource = rachis::BufferBinding{*constants, 1, 256};
    CHECK_FALSE(rachis::test::createBindingSet(device,
                                               rachis::BindingSetDesc{
                                                   .layout = *layout,
                                                   .writes = misalignedWrites,
                                               })
                    .has_value());

    auto emptyLayout = device.createBindingLayout({});
    REQUIRE(emptyLayout);
    auto missingPool = device.createBindingSet({.layout = *emptyLayout});
    REQUIRE_FALSE(missingPool);
    CHECK(missingPool.error().code == rachis::ErrorCode::invalidArgument);
    auto emptySet = rachis::test::createBindingSet(device, {.layout = *emptyLayout});
    REQUIRE(emptySet);

    const std::array tinySizes{
        rachis::BindingPoolSize{rachis::BindingType::uniformBuffer, 1},
    };
    auto tinyPool = device.createBindingPool({
        .maxBindingSets = 1,
        .sizes = tinySizes,
        .label = "Metal tiny pool",
    });
    REQUIRE(tinyPool);
    const std::array oneWrite{rachis::BindingWrite{0, 0, rachis::BufferBinding{*constants, 0, 256}}};
    const std::array oneEntry{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute, 1},
    };
    auto oneLayout = device.createBindingLayout({.entries = oneEntry});
    REQUIRE(oneLayout);
    auto first = device.createBindingSet({
        .layout = *oneLayout,
        .writes = oneWrite,
        .pool = *tinyPool,
    });
    REQUIRE(first);
    CHECK_FALSE(device
                    .createBindingSet({
                        .layout = *oneLayout,
                        .writes = oneWrite,
                        .pool = *tinyPool,
                    })
                    .has_value());
    REQUIRE(device.destroyBindingSet(*first));
    auto recycled = device.createBindingSet({
        .layout = *oneLayout,
        .writes = oneWrite,
        .pool = *tinyPool,
    });
    REQUIRE(recycled);

    const std::array versionSizes{
        rachis::BindingPoolSize{rachis::BindingType::uniformBuffer, 1},
    };
    auto versionPool = device.createBindingPool({
        .maxBindingSets = 2,
        .sizes = versionSizes,
        .label = "Metal one-descriptor pool",
    });
    REQUIRE(versionPool);
    auto versioned = device.createBindingSet({
        .layout = *oneLayout,
        .writes = oneWrite,
        .pool = *versionPool,
    });
    REQUIRE(versioned);
    REQUIRE(device.updateBindingSet(*versioned, oneWrite));
    REQUIRE(device.updateBindingSet(*versioned, oneWrite));
    const auto exhausted = device.createBindingSet({
        .layout = *oneLayout,
        .writes = oneWrite,
        .pool = *versionPool,
    });
    REQUIRE_FALSE(exhausted);
    CHECK(exhausted.error().code == rachis::ErrorCode::resourceExhausted);

    REQUIRE(device.destroyBindingLayout(*emptyLayout));
    REQUIRE(device.destroyBindingSet(*emptySet));
    REQUIRE(device.destroyBindingSet(*set));
    CHECK_FALSE(device.destroyBindingSet(*set).has_value());
    REQUIRE(device.destroyBindingSet(*recycled));
    REQUIRE(device.destroyBindingSet(*versioned));
}

TEST_CASE("Metal binding-set updates are copy-on-write and pipeline layouts reserve push constants",
          "[size:large][rhi][metal][binding][update]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    auto bufferA = device.createBuffer({
        .size = 256,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal update A",
    });
    auto bufferB = device.createBuffer({
        .size = 256,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal update B",
    });
    auto samplerA = device.createSampler({.label = "Metal update sampler A"});
    auto samplerB = device.createSampler({
        .minFilter = rachis::FilterMode::linear,
        .magFilter = rachis::FilterMode::linear,
        .label = "Metal update sampler B",
    });
    REQUIRE(bufferA);
    REQUIRE(bufferB);
    REQUIRE(samplerA);
    REQUIRE(samplerB);

    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::sampler, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{2, rachis::BindingType::dynamicUniformBuffer, rachis::ShaderStage::compute, 1},
    };
    auto layout = device.createBindingLayout({.entries = entries, .label = "Metal update layout"});
    REQUIRE(layout);
    const std::array initial{
        rachis::BindingWrite{0, 0, rachis::BufferBinding{*bufferA, 0, 256}},
        rachis::BindingWrite{1, 0, *samplerA},
        rachis::BindingWrite{2, 0, rachis::BufferBinding{*bufferA, 0, 256}},
    };
    const auto before = device.bindingUpdateStatistics();
    auto set = rachis::test::createBindingSet(device,
                                              rachis::BindingSetDesc{
                                                  .layout = *layout,
                                                  .writes = initial,
                                              });
    REQUIRE(set);
    const auto createdStats = device.bindingUpdateStatistics();
    CHECK(createdStats.resourceDescriptorsPrepared > before.resourceDescriptorsPrepared);
    CHECK(createdStats.samplerDescriptorsPrepared > before.samplerDescriptorsPrepared);

    CHECK_FALSE(device.updateBindingSet(*set, {}).has_value());
    const std::array update{
        rachis::BindingWrite{0, 0, rachis::BufferBinding{*bufferB, 0, 256}},
        rachis::BindingWrite{1, 0, *samplerB},
    };
    REQUIRE(device.updateBindingSet(*set, update));
    const auto updatedStats = device.bindingUpdateStatistics();
    CHECK(updatedStats.resourceTablePublishCount > createdStats.resourceTablePublishCount);

    const auto storageShaderBytes = loadPnmsBytes("MetalPushConstant.cs.metallib");
    const std::array storageRequirements{
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
    };
    const std::array storagePush{rachis::ShaderPushConstantRequirement{0, 16}};
    auto storageShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = storageShaderBytes,
        .entryPoint = "computeMain",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, storageRequirements, storagePush),
    });
    REQUIRE(storageShader);
    auto storageBindings = device.createBindingLayout({
        .entries =
            std::array{rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute}},
    });
    REQUIRE(storageBindings);
    auto updatePipelineLayout = device.createPipelineLayout({
        .bindingLayouts = std::array{*storageBindings, *layout},
        .pushConstantRanges = std::array{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16}},
    });
    REQUIRE(updatePipelineLayout);
    auto updatePipeline =
        device.createComputePipeline({.layout = *updatePipelineLayout, .computeShader = *storageShader});
    REQUIRE(updatePipeline);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command);
    REQUIRE(device.bindComputePipeline(*command, *updatePipeline).has_value());
    CHECK_FALSE(device.bindComputeBindingSet(*command, 1, *set).has_value());
    const std::array offsets{0U};
    REQUIRE(device.bindComputeBindingSet(*command, 1, *set, offsets).has_value());
    CHECK_FALSE(device.bindComputeBindingSet(*command, 1, rachis::BindingSetHandle{}).has_value());

    const std::array push{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16},
    };
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = std::array{*layout},
        .pushConstantRanges = push,
        .label = "Metal pipeline layout with binding 0 and push constants",
    });
    REQUIRE(pipelineLayout);
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout));

    std::vector<rachis::BindingLayoutEntry> fullBufferEntries;
    fullBufferEntries.reserve(31);
    for (std::uint32_t binding = 0; binding < 31; ++binding)
    {
        fullBufferEntries.push_back(
            rachis::BindingLayoutEntry{binding, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute, 1});
    }
    auto fullBuffersLayout = device.createBindingLayout({.entries = fullBufferEntries});
    REQUIRE(fullBuffersLayout);
    auto maxNoPushLayout = device.createPipelineLayout({.bindingLayouts = std::array{*fullBuffersLayout}});
    REQUIRE(maxNoPushLayout);
    REQUIRE(device.destroyPipelineLayout(*maxNoPushLayout));
    auto overflowWithPush =
        device.createPipelineLayout({.bindingLayouts = std::array{*fullBuffersLayout}, .pushConstantRanges = push});
    REQUIRE_FALSE(overflowWithPush);
    CHECK(overflowWithPush.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(device.destroyBindingLayout(*fullBuffersLayout));

    auto pushBindings = device.createBindingLayout({
        .entries =
            std::array{rachis::BindingLayoutEntry{1, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute}},
    });
    REQUIRE(pushBindings);
    pipelineLayout =
        device.createPipelineLayout({.bindingLayouts = std::array{*pushBindings}, .pushConstantRanges = push});
    REQUIRE(pipelineLayout);
    const std::array overlappingPush{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16},
        rachis::PushConstantRange{rachis::ShaderStage::compute, 8, 16},
    };
    CHECK_FALSE(device.createPipelineLayout({.pushConstantRanges = overlappingPush}).has_value());
    CHECK_FALSE(device
                    .createPipelineLayout({
                        .bindingLayouts = std::array{rachis::BindingLayoutHandle{}},
                    })
                    .has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout));
    CHECK_FALSE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingSet(*set));
}

TEST_CASE("Metal captured binding versions stay resident after public destroy",
          "[size:large][rhi][metal][binding][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    auto source = device.createBuffer(rachis::BufferDesc{
        .size = capturedBytes,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal captured source",
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = capturedBytes,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal captured readback",
    });
    REQUIRE(source);
    REQUIRE(readback);

    std::array<std::byte, capturedBytes> expected{};
    for (std::uint64_t index = 0; index < capturedBytes; ++index)
    {
        expected[index] = capturedPattern(index);
    }
    REQUIRE(device.writeBuffer(*source, 0, expected).has_value());

    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    auto layout = device.createBindingLayout({.entries = entries, .label = "Metal captured layout"});
    REQUIRE(layout);
    const std::array writes{
        rachis::BindingWrite{0, 0, rachis::BufferBinding{*source, 0, capturedBytes}},
    };
    auto set = rachis::test::createBindingSet(device,
                                              rachis::BindingSetDesc{
                                                  .layout = *layout,
                                                  .writes = writes,
                                                  .label = "Metal captured set",
                                              });
    REQUIRE(set);
    const auto storageShaderBytes = loadPnmsBytes("MetalPushConstant.cs.metallib");
    const std::array storageRequirements{
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
    };
    const std::array storagePush{rachis::ShaderPushConstantRequirement{0, 16}};
    auto storageShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = storageShaderBytes,
        .entryPoint = "computeMain",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, storageRequirements, storagePush),
    });
    REQUIRE(storageShader);
    auto storageBindings = device.createBindingLayout({
        .entries =
            std::array{rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute}},
    });
    REQUIRE(storageBindings);
    auto capturedPipelineLayout = device.createPipelineLayout({
        .bindingLayouts = std::array{*storageBindings, *layout},
        .pushConstantRanges = std::array{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16}},
    });
    REQUIRE(capturedPipelineLayout);
    auto capturedPipeline =
        device.createComputePipeline({.layout = *capturedPipelineLayout, .computeShader = *storageShader});
    REQUIRE(capturedPipeline);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command);
    REQUIRE(device.bindComputePipeline(*command, *capturedPipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 1, *set).has_value());
    REQUIRE(rachis::metal::copyCapturedBindingBuffer(*command, 1, *readback, capturedBytes).has_value());

    const auto staleSet = *set;
    const auto staleSource = *source;
    REQUIRE(device.destroyBindingSet(*set).has_value());
    REQUIRE(device.destroyBuffer(*source).has_value());
    CHECK_FALSE(device.destroyBindingSet(staleSet).has_value());
    CHECK_FALSE(device.destroyBuffer(staleSource).has_value());
    const auto staleUpdate = device.updateBindingSet(staleSet, writes);
    REQUIRE_FALSE(staleUpdate);
    CHECK(staleUpdate.error().code == rachis::ErrorCode::invalidArgument);

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    std::array<std::byte, capturedBytes> actual{};
    REQUIRE(device.readBuffer(*readback, 0, actual).has_value());
    CHECK(actual == expected);
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal uniform buffers accept 16-byte offset alignment and non-256-multiple sizes on GPU",
          "[size:large][rhi][metal][binding][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto caps = device.capabilities();
    REQUIRE(caps.has_value());
    CHECK(caps->limits.minUniformBufferOffsetAlignment == 16U);

    auto largeUbo = device.createBuffer({
        .size = 70000,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(largeUbo);
    auto staticUboLayout = device.createBindingLayout({
        .entries = std::array{rachis::BindingLayoutEntry{
            0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute, 1}},
    });
    auto dynamicUboLayout = device.createBindingLayout({
        .entries = std::array{rachis::BindingLayoutEntry{
            0, rachis::BindingType::dynamicUniformBuffer, rachis::ShaderStage::compute, 1}},
    });
    REQUIRE(staticUboLayout);
    REQUIRE(dynamicUboLayout);

    for (const std::uint64_t badOffset : {1ULL, 8ULL})
    {
        const auto rejected = rachis::test::createBindingSet(
            device,
            {.layout = *staticUboLayout,
             .writes = std::array{rachis::BindingWrite{0, 0, rachis::BufferBinding{*largeUbo, badOffset, 24}}}});
        REQUIRE_FALSE(rejected);
        CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
    }
    for (const std::uint64_t badSize : {0ULL, 65537ULL})
    {
        const auto rejected = rachis::test::createBindingSet(
            device,
            {.layout = *staticUboLayout,
             .writes = std::array{rachis::BindingWrite{0, 0, rachis::BufferBinding{*largeUbo, 0, badSize}}}});
        REQUIRE_FALSE(rejected);
        CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
    }

    auto validDynSet = rachis::test::createBindingSet(
        device,
        {.layout = *dynamicUboLayout,
         .writes = std::array{rachis::BindingWrite{0, 0, rachis::BufferBinding{*largeUbo, 0, 24}}}});
    REQUIRE(validDynSet);
    {
        auto cmd = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(cmd);
        for (const std::uint32_t badDyn : {1U, 8U})
        {
            const std::array dyn{badDyn};
            const auto rejected = device.bindComputeBindingSet(*cmd, 0, *validDynSet, dyn);
            REQUIRE_FALSE(rejected);
            CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
        }
    }

    {
        const auto fixturePnms = loadPnmsBytes("LayoutFixture.cs.metallib");
        const std::array fixtureReq{
            rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1},
            rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
        };
        auto fixtureShader = device.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = fixturePnms,
            .entryPoint = "computeMain",
            .shaderInterface = {.bindings = fixtureReq, .workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1}},
        });
        REQUIRE(fixtureShader);
        auto fixtureBindingLayout = device.createBindingLayout({
            .entries =
                std::array{
                    rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute, 1},
                    rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
                },
        });
        REQUIRE(fixtureBindingLayout);
        auto fixturePipeLayout = device.createPipelineLayout({.bindingLayouts = std::array{*fixtureBindingLayout}});
        REQUIRE(fixturePipeLayout);
        auto fixturePipeline =
            device.createComputePipeline({.layout = *fixturePipeLayout, .computeShader = *fixtureShader});
        REQUIRE(fixturePipeline);

        auto exact80Buffer = device.createBuffer({
            .size = rachis::metalConstantBufferSize,
            .usage = rachis::BufferUsage::constant,
            .memoryType = rachis::MemoryType::upload,
        });
        auto outBuffer = device.createBuffer({
            .size = 32,
            .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        });
        auto readback = device.createBuffer({
            .size = 32,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
        });
        REQUIRE(exact80Buffer);
        REQUIRE(outBuffer);
        REQUIRE(readback);

        const auto payload80 = makeLayoutFixturePayload(0xCAFE1234U);
        REQUIRE(device.writeBuffer(*exact80Buffer, 0, payload80).has_value());
        const auto minAlign = caps->limits.minUniformBufferOffsetAlignment;
        const auto payloadAt16 = makeLayoutFixturePayload(0xBEEF5678U);
        REQUIRE(device.writeBuffer(*largeUbo, minAlign, payloadAt16).has_value());

        for (const bool useOffset16 : {false, true})
        {
            const rachis::BufferBinding uboBinding =
                useOffset16 ? rachis::BufferBinding{*largeUbo, minAlign, rachis::metalConstantBufferSize}
                            : rachis::BufferBinding{*exact80Buffer, 0, rachis::metalConstantBufferSize};
            auto set = rachis::test::createBindingSet(
                device,
                {
                    .layout = *fixtureBindingLayout,
                    .writes =
                        std::array{
                            rachis::BindingWrite{0, 0, uboBinding},
                            rachis::BindingWrite{1, 0, rachis::BufferBinding{*outBuffer, 0, 32}},
                        },
                });
            REQUIRE(set);
            auto cmd = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
            REQUIRE(cmd);
            REQUIRE(device.bindComputePipeline(*cmd, *fixturePipeline).has_value());
            REQUIRE(device.bindComputeBindingSet(*cmd, 0, *set).has_value());
            REQUIRE(device.dispatch(*cmd, 1, 1, 1).has_value());
            REQUIRE(device
                        .bufferBarrier(*cmd,
                                       {
                                           .buffer = *outBuffer,
                                           .stagesBefore = rachis::PipelineStage::computeShader,
                                           .stagesAfter = rachis::PipelineStage::copy,
                                           .accessBefore = rachis::Access::unorderedAccess,
                                           .accessAfter = rachis::Access::copySource,
                                       })
                        .has_value());
            REQUIRE(device.copyBuffer(*cmd, {.source = *outBuffer, .destination = *readback, .size = 32}).has_value());
            auto pt = device.submit(std::move(*cmd));
            REQUIRE(pt);
            REQUIRE(device.wait(*pt).has_value());

            std::array<std::uint32_t, 8> words{};
            REQUIRE(device
                        .readBuffer(*readback,
                                    0,
                                    std::span<std::byte>{reinterpret_cast<std::byte*>(words.data()), sizeof(words)})
                        .has_value());
            CHECK(words[0] == floatBits(1.5F));
            CHECK(words[1] == floatBits(2.5F));
            CHECK(words[2] == floatBits(3.5F));
            CHECK(words[3] == floatBits(4.5F));
            CHECK(words[4] == floatBits(6.25F));
            CHECK(words[5] == floatBits(14.0F));
            CHECK(words[6] == (useOffset16 ? 0xBEEF5678U : 0xCAFE1234U));
            CHECK(words[7] == 0xA5A5A5A5U);
            REQUIRE(device.destroyBindingSet(*set).has_value());
        }
    }

    {
        const auto singleCsPnms = loadPnmsBytes("MetalRemapContract.single.cs.metallib");
        const std::array shaderReq{
            rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1},
            rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::sampledTexture, 1},
            rachis::ShaderBindingRequirement{0, 2, rachis::ShaderResourceType::sampler, 1},
            rachis::ShaderBindingRequirement{0, 3, rachis::ShaderResourceType::storageBuffer, 1},
        };
        const std::array pushReq{rachis::ShaderPushConstantRequirement{0, 16}};
        auto shader = device.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = singleCsPnms,
            .entryPoint = "pushAndBindingZeroCompute",
            .shaderInterface = {.bindings = shaderReq,
                                .pushConstants = pushReq,
                                .workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1}},
        });
        REQUIRE(shader);

        const auto tex =
            create2x2RgbaTexture(device,
                                 {{{10, 20, 30, 40}, {50, 60, 70, 80}, {90, 100, 110, 120}, {130, 140, 150, 160}}},
                                 rachis::PipelineStage::computeShader,
                                 "UBO alignment quad");
        auto sampler = device.createSampler({
            .minFilter = rachis::FilterMode::nearest,
            .magFilter = rachis::FilterMode::nearest,
        });
        auto outBuf = device.createBuffer({
            .size = 32,
            .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        });
        auto readback = device.createBuffer({
            .size = 32,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
        });
        REQUIRE(sampler);
        REQUIRE(outBuf);
        REQUIRE(readback);

        for (const std::uint64_t exactSize : {24ULL, 48ULL})
        {
            auto exactBuf = device.createBuffer({
                .size = exactSize,
                .usage = rachis::BufferUsage::constant,
                .memoryType = rachis::MemoryType::upload,
            });
            REQUIRE(exactBuf);
            const CompactUniforms24Host payload{
                .w0 = 0x11110000U + static_cast<std::uint32_t>(exactSize),
                .w1 = 100U + static_cast<std::uint32_t>(exactSize),
                .w2 = 7U,
                .w3 = 9U,
                .t0 = 300U + static_cast<std::uint32_t>(exactSize),
                .t1 = 400U + static_cast<std::uint32_t>(exactSize),
            };
            REQUIRE(device.writeBuffer(*exactBuf, 0, asByteSpan(payload)).has_value());

            auto bLayout = device.createBindingLayout({
                .entries =
                    std::array{
                        rachis::BindingLayoutEntry{
                            0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute, 1},
                        rachis::BindingLayoutEntry{
                            1, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute, 1},
                        rachis::BindingLayoutEntry{2, rachis::BindingType::sampler, rachis::ShaderStage::compute, 1},
                        rachis::BindingLayoutEntry{
                            3, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
                    },
            });
            REQUIRE(bLayout);
            auto pLayout = device.createPipelineLayout({
                .bindingLayouts = std::array{*bLayout},
                .pushConstantRanges = std::array{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16}},
            });
            REQUIRE(pLayout);
            auto pipe = device.createComputePipeline({.layout = *pLayout, .computeShader = *shader});
            REQUIRE(pipe);

            auto set = rachis::test::createBindingSet(
                device,
                {
                    .layout = *bLayout,
                    .writes =
                        std::array{
                            rachis::BindingWrite{0, 0, rachis::BufferBinding{*exactBuf, 0, exactSize}},
                            rachis::BindingWrite{1, 0, tex.view},
                            rachis::BindingWrite{2, 0, *sampler},
                            rachis::BindingWrite{3, 0, rachis::BufferBinding{*outBuf, 0, 32}},
                        },
                });
            REQUIRE(set);
            const std::array<std::uint32_t, 4> push{0x0000AAAAU, 10U, 20U, 30U};
            auto cmd = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
            REQUIRE(cmd);
            REQUIRE(device.bindComputePipeline(*cmd, *pipe).has_value());
            REQUIRE(device.bindComputeBindingSet(*cmd, 0, *set).has_value());
            REQUIRE(device.setComputePushConstants(*cmd, 0, asByteSpan(push)).has_value());
            REQUIRE(device.dispatch(*cmd, 1, 1, 1).has_value());
            REQUIRE(device
                        .bufferBarrier(*cmd,
                                       {
                                           .buffer = *outBuf,
                                           .stagesBefore = rachis::PipelineStage::computeShader,
                                           .stagesAfter = rachis::PipelineStage::copy,
                                           .accessBefore = rachis::Access::unorderedAccess,
                                           .accessAfter = rachis::Access::copySource,
                                       })
                        .has_value());
            REQUIRE(device.copyBuffer(*cmd, {.source = *outBuf, .destination = *readback, .size = 32}).has_value());
            auto pt = device.submit(std::move(*cmd));
            REQUIRE(pt);
            REQUIRE(device.wait(*pt).has_value());

            std::array<std::uint32_t, 8> actual{};
            REQUIRE(device
                        .readBuffer(*readback,
                                    0,
                                    std::span<std::byte>{reinterpret_cast<std::byte*>(actual.data()), sizeof(actual)})
                        .has_value());
            CHECK(actual[0] == (push[0] ^ payload.w0));
            CHECK(actual[1] == (push[1] + payload.w1));
            CHECK(actual[2] == (push[2] + payload.t0));
            CHECK(actual[3] == (push[3] + payload.t1));
            CHECK(actual[4] == 50U);
            CHECK(actual[5] == 60U);
            CHECK(actual[6] == 70U);
            CHECK(actual[7] == (80U + payload.w2 + payload.w3));
        }

        auto dynBLayout = device.createBindingLayout({
            .entries =
                std::array{
                    rachis::BindingLayoutEntry{
                        0, rachis::BindingType::dynamicUniformBuffer, rachis::ShaderStage::compute, 1},
                    rachis::BindingLayoutEntry{1, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute, 1},
                    rachis::BindingLayoutEntry{2, rachis::BindingType::sampler, rachis::ShaderStage::compute, 1},
                    rachis::BindingLayoutEntry{3, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
                },
        });
        REQUIRE(dynBLayout);
        auto dynPLayout = device.createPipelineLayout({
            .bindingLayouts = std::array{*dynBLayout},
            .pushConstantRanges = std::array{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16}},
        });
        REQUIRE(dynPLayout);
        auto dynPipe = device.createComputePipeline({.layout = *dynPLayout, .computeShader = *shader});
        REQUIRE(dynPipe);

        auto dynSet = rachis::test::createBindingSet(
            device,
            {
                .layout = *dynBLayout,
                .writes =
                    std::array{
                        rachis::BindingWrite{0, 0, rachis::BufferBinding{*largeUbo, 0, 24}},
                        rachis::BindingWrite{1, 0, tex.view},
                        rachis::BindingWrite{2, 0, *sampler},
                        rachis::BindingWrite{3, 0, rachis::BufferBinding{*outBuf, 0, 32}},
                    },
            });
        REQUIRE(dynSet);

        for (const std::uint32_t dynOffset : {0U, 16U, 32U, 48U, 256U})
        {
            const CompactUniforms24Host payload{
                .w0 = 0x22000000U | dynOffset,
                .w1 = 500U + dynOffset,
                .w2 = 3U,
                .w3 = 5U,
                .t0 = 700U + dynOffset,
                .t1 = 900U + dynOffset,
            };
            REQUIRE(device.writeBuffer(*largeUbo, dynOffset, asByteSpan(payload)).has_value());

            const std::array<std::uint32_t, 4> push{0x00FF00FFU, 1U, 2U, 3U};
            const std::array dynOffsets{dynOffset};
            auto cmd = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
            REQUIRE(cmd);
            REQUIRE(device.bindComputePipeline(*cmd, *dynPipe).has_value());
            REQUIRE(device.bindComputeBindingSet(*cmd, 0, *dynSet, dynOffsets).has_value());
            REQUIRE(device.setComputePushConstants(*cmd, 0, asByteSpan(push)).has_value());
            REQUIRE(device.dispatch(*cmd, 1, 1, 1).has_value());
            REQUIRE(device
                        .bufferBarrier(*cmd,
                                       {
                                           .buffer = *outBuf,
                                           .stagesBefore = rachis::PipelineStage::computeShader,
                                           .stagesAfter = rachis::PipelineStage::copy,
                                           .accessBefore = rachis::Access::unorderedAccess,
                                           .accessAfter = rachis::Access::copySource,
                                       })
                        .has_value());
            REQUIRE(device.copyBuffer(*cmd, {.source = *outBuf, .destination = *readback, .size = 32}).has_value());
            auto pt = device.submit(std::move(*cmd));
            REQUIRE(pt);
            REQUIRE(device.wait(*pt).has_value());

            std::array<std::uint32_t, 8> actual{};
            REQUIRE(device
                        .readBuffer(*readback,
                                    0,
                                    std::span<std::byte>{reinterpret_cast<std::byte*>(actual.data()), sizeof(actual)})
                        .has_value());
            CAPTURE(dynOffset);
            CHECK(actual[0] == (push[0] ^ payload.w0));
            CHECK(actual[1] == (push[1] + payload.w1));
            CHECK(actual[2] == (push[2] + payload.t0));
            CHECK(actual[3] == (push[3] + payload.t1));
            CHECK(actual[7] == (80U + payload.w2 + payload.w3));
        }
    }

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal compute and graphics pipelines execute with push constants and set 0 binding 0",
          "[size:large][rhi][metal][binding][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    const auto singleVsPnms = loadPnmsBytes("MetalRemapContract.single.vs.metallib");
    const auto singlePsPnms = loadPnmsBytes("MetalRemapContract.single.ps.metallib");
    const std::array vsReq{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1},
    };
    const std::array psReq{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1},
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::sampledTexture, 1},
        rachis::ShaderBindingRequirement{0, 2, rachis::ShaderResourceType::sampler, 1},
    };
    const std::array pushReq{rachis::ShaderPushConstantRequirement{0, 16}};
    const std::array vsLocations{0U, 1U};
    auto vs = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = singleVsPnms,
        .entryPoint = "pushAndBindingZeroVertex",
        .shaderInterface = {.bindings = vsReq, .pushConstants = pushReq, .vertexInputLocations = vsLocations},
    });
    auto ps = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = singlePsPnms,
        .entryPoint = "pushAndBindingZeroPixel",
        .shaderInterface = {.bindings = psReq, .pushConstants = pushReq},
    });
    REQUIRE(vs);
    REQUIRE(ps);

    const auto allGraphics = rachis::ShaderStage::vertex | rachis::ShaderStage::pixel;
    auto gfxBindingLayout = device.createBindingLayout({
        .entries =
            std::array{
                rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, allGraphics, 1},
                rachis::BindingLayoutEntry{1, rachis::BindingType::sampledTexture, rachis::ShaderStage::pixel, 1},
                rachis::BindingLayoutEntry{2, rachis::BindingType::sampler, rachis::ShaderStage::pixel, 1},
            },
    });
    REQUIRE(gfxBindingLayout);
    auto gfxPipeLayout = device.createPipelineLayout({
        .bindingLayouts = std::array{*gfxBindingLayout},
        .pushConstantRanges = std::array{rachis::PushConstantRange{allGraphics, 0, 16}},
    });
    REQUIRE(gfxPipeLayout);

    const std::array vertexAttrs{
        rachis::VertexAttribute{.format = rachis::VertexFormat::float32x2, .offset = 0, .shaderLocation = 0},
        rachis::VertexAttribute{.format = rachis::VertexFormat::uint32, .offset = 8, .shaderLocation = 1},
    };
    const std::array vertexBuffers{
        rachis::VertexBufferLayout{
            .stride = sizeof(RemapVertexHost),
            .stepMode = rachis::VertexStepMode::vertex,
            .attributes = vertexAttrs,
        },
    };
    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba32Uint},
    };
    auto gfxPipeline = device.createGraphicsPipeline({
        .layout = *gfxPipeLayout,
        .vertexShader = *vs,
        .pixelShader = *ps,
        .vertexBuffers = vertexBuffers,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
    });
    REQUIRE(gfxPipeline);

    auto exact16Ubo = device.createBuffer({
        .size = sizeof(CompactUniforms16Host),
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(exact16Ubo);
    const CompactUniforms16Host ubo16{.words = {100U, 200U, 300U, 400U}};
    REQUIRE(device.writeBuffer(*exact16Ubo, 0, asByteSpan(ubo16)).has_value());

    const std::array<RemapVertexHost, 3> fullscreenTri{{
        {.position = {-1.0F, -1.0F}, .vertexTag = 700U},
        {.position = {3.0F, -1.0F}, .vertexTag = 700U},
        {.position = {-1.0F, 3.0F}, .vertexTag = 700U},
    }};
    auto vbo = device.createBuffer({
        .size = sizeof(fullscreenTri),
        .usage = rachis::BufferUsage::vertex,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(vbo);
    REQUIRE(device.writeBuffer(*vbo, 0, asByteSpan(fullscreenTri)).has_value());

    const auto tex =
        create2x2RgbaTexture(device,
                             {{{11, 22, 33, 44}, {55, 66, 77, 88}, {99, 111, 122, 133}, {144, 155, 166, 177}}},
                             rachis::PipelineStage::pixelShader,
                             "Single graphics set0 tex");
    auto sampler = device.createSampler({
        .minFilter = rachis::FilterMode::nearest,
        .magFilter = rachis::FilterMode::nearest,
    });
    REQUIRE(sampler);

    auto set0 = rachis::test::createBindingSet(
        device,
        {
            .layout = *gfxBindingLayout,
            .writes =
                std::array{
                    rachis::BindingWrite{0, 0, rachis::BufferBinding{*exact16Ubo, 0, sizeof(CompactUniforms16Host)}},
                    rachis::BindingWrite{1, 0, tex.view},
                    rachis::BindingWrite{2, 0, *sampler},
                },
        });
    REQUIRE(set0);

    auto target = device.createTexture({
        .width = 4,
        .height = 4,
        .format = rachis::Format::rgba32Uint,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource,
    });
    REQUIRE(target);
    auto targetView = device.createTextureView({
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
        .dimension = rachis::TextureViewDimension::texture2D,
    });
    REQUIRE(targetView);
    const rachis::BufferTextureCopyRegion queryRegion{.extent = {.width = 4, .height = 4, .depth = 1}};
    const auto transfer = device.queryTextureTransferLayout(*target, queryRegion);
    REQUIRE(transfer);
    auto readback = device.createBuffer({
        .size = transfer->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(readback);

    const std::array<std::uint32_t, 4> push{10U, 20U, 30U, 40U};
    auto cmd = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(cmd);
    REQUIRE(device
                .textureBarrier(*cmd,
                                {
                                    .texture = *target,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array attachments{
        rachis::ColorAttachment{.view = *targetView, .loadOperation = rachis::LoadOperation::clear},
    };
    REQUIRE(device.beginRendering(*cmd, {.colorAttachments = attachments}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*cmd, *gfxPipeline).has_value());
    REQUIRE(device.setViewport(*cmd, {.width = 4.0F, .height = 4.0F}).has_value());
    REQUIRE(device.setScissor(*cmd, {.width = 4, .height = 4}).has_value());
    REQUIRE(device.bindVertexBuffer(*cmd, 0, *vbo, 0).has_value());
    REQUIRE(device.bindGraphicsBindingSet(*cmd, 0, *set0).has_value());
    REQUIRE(device.setGraphicsPushConstants(*cmd, allGraphics, 0, asByteSpan(push)).has_value());
    REQUIRE(device.draw(*cmd, 3, 1, 0, 0).has_value());
    REQUIRE(device.endRendering(*cmd).has_value());
    REQUIRE(device
                .textureBarrier(*cmd,
                                {
                                    .texture = *target,
                                    .stagesBefore = rachis::PipelineStage::renderTarget,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::renderTarget,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    const rachis::BufferTextureCopyRegion copyRegion{
        .bytesPerRow = transfer->bytesPerRow,
        .rowsPerImage = transfer->rowsPerImage,
        .extent = {.width = 4, .height = 4, .depth = 1},
    };
    REQUIRE(device.copyTextureToBuffer(*cmd, *target, *readback, std::array{copyRegion}).has_value());
    auto pt = device.submit(std::move(*cmd));
    REQUIRE(pt);
    REQUIRE(device.wait(*pt).has_value());

    std::array<std::uint32_t, 4> pixel{};
    REQUIRE(
        device.readBuffer(*readback, 0, std::span<std::byte>{reinterpret_cast<std::byte*>(pixel.data()), sizeof(pixel)})
            .has_value());
    const std::uint32_t expectedPackedTex = 99U | (111U << 8U);
    CHECK(pixel[0] == (700U + push[0] + ubo16.words[0]));
    CHECK(pixel[1] == (ubo16.words[0] + ubo16.words[3] + ubo16.words[2]));
    CHECK(pixel[2] == (ubo16.words[1] + push[2]));
    CHECK(pixel[3] == (expectedPackedTex + ubo16.words[3] + push[3]));

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal multi-set compute and graphics pipelines remap colliding bindings and preserve partial rebinds",
          "[size:large][rhi][metal][binding][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    {
        const auto multiCsPnms = loadPnmsBytes("MetalRemapContract.multi.cs.metallib");
        const std::array shaderReq{
            rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1},
            rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::sampledTexture, 1},
            rachis::ShaderBindingRequirement{0, 2, rachis::ShaderResourceType::sampler, 1},
            rachis::ShaderBindingRequirement{0, 3, rachis::ShaderResourceType::readOnlyStorageBuffer, 1},
            rachis::ShaderBindingRequirement{1, 0, rachis::ShaderResourceType::uniformBuffer, 1},
            rachis::ShaderBindingRequirement{1, 1, rachis::ShaderResourceType::sampledTexture, 1},
            rachis::ShaderBindingRequirement{1, 2, rachis::ShaderResourceType::sampler, 1},
            rachis::ShaderBindingRequirement{1, 3, rachis::ShaderResourceType::storageBuffer, 1},
        };
        const std::array pushReq{rachis::ShaderPushConstantRequirement{0, 16}};
        auto cs = device.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = multiCsPnms,
            .entryPoint = "multiSetCompute",
            .shaderInterface = {.bindings = shaderReq,
                                .pushConstants = pushReq,
                                .workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1}},
        });
        REQUIRE(cs);

        auto set0Layout = device.createBindingLayout({
            .entries =
                std::array{
                    rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute, 1},
                    rachis::BindingLayoutEntry{1, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute, 1},
                    rachis::BindingLayoutEntry{2, rachis::BindingType::sampler, rachis::ShaderStage::compute, 1},
                    rachis::BindingLayoutEntry{
                        3, rachis::BindingType::readOnlyStorageBuffer, rachis::ShaderStage::compute, 1},
                },
        });
        auto set1Layout = device.createBindingLayout({
            .entries =
                std::array{
                    rachis::BindingLayoutEntry{
                        0, rachis::BindingType::dynamicUniformBuffer, rachis::ShaderStage::compute, 1},
                    rachis::BindingLayoutEntry{1, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute, 1},
                    rachis::BindingLayoutEntry{2, rachis::BindingType::sampler, rachis::ShaderStage::compute, 1},
                    rachis::BindingLayoutEntry{
                        3, rachis::BindingType::dynamicStorageBuffer, rachis::ShaderStage::compute, 1},
                },
        });
        REQUIRE(set0Layout);
        REQUIRE(set1Layout);

        auto pipeLayout = device.createPipelineLayout({
            .bindingLayouts = std::array{*set0Layout, *set1Layout},
            .pushConstantRanges = std::array{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16}},
        });
        REQUIRE(pipeLayout);
        auto pipeline = device.createComputePipeline({.layout = *pipeLayout, .computeShader = *cs});
        REQUIRE(pipeline);

        auto set0UboBuf = device.createBuffer({
            .size = 64,
            .usage = rachis::BufferUsage::constant,
            .memoryType = rachis::MemoryType::upload,
        });
        auto set1UboBuf = device.createBuffer({
            .size = 128,
            .usage = rachis::BufferUsage::constant,
            .memoryType = rachis::MemoryType::upload,
        });
        auto set0InBuf = device.createBuffer({
            .size = 16,
            .usage = rachis::BufferUsage::storage,
            .memoryType = rachis::MemoryType::upload,
        });
        auto set1OutBuf = device.createBuffer({
            .size = 512,
            .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        });
        auto readback = device.createBuffer({
            .size = 512,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
        });
        REQUIRE(set0UboBuf);
        REQUIRE(set1UboBuf);
        REQUIRE(set0InBuf);
        REQUIRE(set1OutBuf);
        REQUIRE(readback);

        const CompactUniforms24Host set0Ubo{
            .w0 = 1000U, .w1 = 1100U, .w2 = 1200U, .w3 = 13U, .t0 = 1400U, .t1 = 0x55AAU};
        const CompactUniforms24Host set1UboA{
            .w0 = 2000U, .w1 = 2100U, .w2 = 2200U, .w3 = 23U, .t0 = 0x0F0FU, .t1 = 2500U};
        const CompactUniforms24Host set1UboB{
            .w0 = 3000U, .w1 = 3100U, .w2 = 3200U, .w3 = 33U, .t0 = 0x33CCU, .t1 = 3500U};
        REQUIRE(device.writeBuffer(*set0UboBuf, 16, asByteSpan(set0Ubo)).has_value());
        REQUIRE(device.writeBuffer(*set1UboBuf, 16, asByteSpan(set1UboA)).has_value());
        REQUIRE(device.writeBuffer(*set1UboBuf, 64, asByteSpan(set1UboB)).has_value());
        const std::uint32_t set0InWord = 777U;
        REQUIRE(device.writeBuffer(*set0InBuf, 0, asByteSpan(set0InWord)).has_value());

        const auto tex0 =
            create2x2RgbaTexture(device,
                                 {{{12, 34, 56, 78}, {90, 91, 92, 93}, {94, 95, 96, 97}, {98, 99, 100, 101}}},
                                 rachis::PipelineStage::computeShader,
                                 "Multi compute set0 tex");
        const auto tex1A = create2x2RgbaTexture(device,
                                                {{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 150, 210}}},
                                                rachis::PipelineStage::computeShader,
                                                "Multi compute set1A tex");
        const auto tex1B = create2x2RgbaTexture(device,
                                                {{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 175, 225}}},
                                                rachis::PipelineStage::computeShader,
                                                "Multi compute set1B tex");

        auto repeatSampler = device.createSampler({
            .minFilter = rachis::FilterMode::nearest,
            .magFilter = rachis::FilterMode::nearest,
            .addressU = rachis::AddressMode::repeat,
            .addressV = rachis::AddressMode::repeat,
        });
        auto clampSampler = device.createSampler({
            .minFilter = rachis::FilterMode::nearest,
            .magFilter = rachis::FilterMode::nearest,
            .addressU = rachis::AddressMode::clampToEdge,
            .addressV = rachis::AddressMode::clampToEdge,
        });
        REQUIRE(repeatSampler);
        REQUIRE(clampSampler);

        auto set0 = rachis::test::createBindingSet(
            device,
            {
                .layout = *set0Layout,
                .writes =
                    std::array{
                        rachis::BindingWrite{
                            0, 0, rachis::BufferBinding{*set0UboBuf, 16, sizeof(CompactUniforms24Host)}},
                        rachis::BindingWrite{1, 0, tex0.view},
                        rachis::BindingWrite{2, 0, *repeatSampler},
                        rachis::BindingWrite{3, 0, rachis::BufferBinding{*set0InBuf, 0, 16}},
                    },
            });
        auto set1A = rachis::test::createBindingSet(
            device,
            {
                .layout = *set1Layout,
                .writes =
                    std::array{
                        rachis::BindingWrite{
                            0, 0, rachis::BufferBinding{*set1UboBuf, 0, sizeof(CompactUniforms24Host)}},
                        rachis::BindingWrite{1, 0, tex1A.view},
                        rachis::BindingWrite{2, 0, *clampSampler},
                        rachis::BindingWrite{3, 0, rachis::BufferBinding{*set1OutBuf, 0, 32}},
                    },
            });
        auto set1B = rachis::test::createBindingSet(
            device,
            {
                .layout = *set1Layout,
                .writes =
                    std::array{
                        rachis::BindingWrite{
                            0, 0, rachis::BufferBinding{*set1UboBuf, 0, sizeof(CompactUniforms24Host)}},
                        rachis::BindingWrite{1, 0, tex1B.view},
                        rachis::BindingWrite{2, 0, *clampSampler},
                        rachis::BindingWrite{3, 0, rachis::BufferBinding{*set1OutBuf, 0, 32}},
                    },
            });
        REQUIRE(set0);
        REQUIRE(set1A);
        REQUIRE(set1B);

        const std::array<std::uint32_t, 4> pushA{11U, 22U, 33U, 44U};
        const std::array<std::uint32_t, 4> pushB{55U, 66U, 77U, 88U};

        auto cmd = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(cmd);
        REQUIRE(device.bindComputePipeline(*cmd, *pipeline).has_value());
        REQUIRE(device.bindComputeBindingSet(*cmd, 0, *set0).has_value());
        REQUIRE(device.bindComputeBindingSet(*cmd, 1, *set1A, std::array{16U, 0U}).has_value());
        REQUIRE(device.setComputePushConstants(*cmd, 0, asByteSpan(pushA)).has_value());
        REQUIRE(device.dispatch(*cmd, 1, 1, 1).has_value());

        REQUIRE(device.bindComputeBindingSet(*cmd, 1, *set1B, std::array{64U, 256U}).has_value());
        REQUIRE(device.setComputePushConstants(*cmd, 0, asByteSpan(pushB)).has_value());
        REQUIRE(device.dispatch(*cmd, 1, 1, 1).has_value());

        REQUIRE(device
                    .bufferBarrier(*cmd,
                                   {
                                       .buffer = *set1OutBuf,
                                       .stagesBefore = rachis::PipelineStage::computeShader,
                                       .stagesAfter = rachis::PipelineStage::copy,
                                       .accessBefore = rachis::Access::unorderedAccess,
                                       .accessAfter = rachis::Access::copySource,
                                   })
                    .has_value());
        REQUIRE(device.copyBuffer(*cmd, {.source = *set1OutBuf, .destination = *readback, .size = 512}).has_value());
        auto pt = device.submit(std::move(*cmd));
        REQUIRE(pt);
        REQUIRE(device.wait(*pt).has_value());

        std::array<std::uint32_t, 8> actualA{};
        std::array<std::uint32_t, 8> actualB{};
        REQUIRE(device
                    .readBuffer(*readback,
                                0,
                                std::span<std::byte>{reinterpret_cast<std::byte*>(actualA.data()), sizeof(actualA)})
                    .has_value());
        REQUIRE(device
                    .readBuffer(*readback,
                                256,
                                std::span<std::byte>{reinterpret_cast<std::byte*>(actualB.data()), sizeof(actualB)})
                    .has_value());

        CHECK(actualA[0] == (pushA[0] + set0Ubo.w0));
        CHECK(actualA[1] == (pushA[1] + set1UboA.w0));
        CHECK(actualA[2] == (set0Ubo.t0 + set0InWord));
        CHECK(actualA[3] == (set1UboA.t1 + pushA[2]));
        CHECK(actualA[4] == (12U | (34U << 8U)));
        CHECK(actualA[5] == (150U | (210U << 8U)));
        CHECK(actualA[6] == (set0Ubo.t1 ^ set1UboA.t0));
        CHECK(actualA[7] == (pushA[3] + set0Ubo.w3 + set1UboA.w3));

        CHECK(actualB[0] == (pushB[0] + set0Ubo.w0));
        CHECK(actualB[1] == (pushB[1] + set1UboB.w0));
        CHECK(actualB[2] == (set0Ubo.t0 + set0InWord));
        CHECK(actualB[3] == (set1UboB.t1 + pushB[2]));
        CHECK(actualB[4] == (12U | (34U << 8U)));
        CHECK(actualB[5] == (175U | (225U << 8U)));
        CHECK(actualB[6] == (set0Ubo.t1 ^ set1UboB.t0));
        CHECK(actualB[7] == (pushB[3] + set0Ubo.w3 + set1UboB.w3));
    }

    {
        const auto multiVsPnms = loadPnmsBytes("MetalRemapContract.multi.vs.metallib");
        const auto multiPsPnms = loadPnmsBytes("MetalRemapContract.multi.ps.metallib");
        const std::array vsReq{
            rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1},
            rachis::ShaderBindingRequirement{1, 0, rachis::ShaderResourceType::uniformBuffer, 1},
        };
        const std::array psReq{
            rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1},
            rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::sampledTexture, 1},
            rachis::ShaderBindingRequirement{0, 2, rachis::ShaderResourceType::sampler, 1},
            rachis::ShaderBindingRequirement{1, 0, rachis::ShaderResourceType::uniformBuffer, 1},
            rachis::ShaderBindingRequirement{1, 1, rachis::ShaderResourceType::sampledTexture, 1},
            rachis::ShaderBindingRequirement{1, 2, rachis::ShaderResourceType::sampler, 1},
        };
        const std::array pushReq{rachis::ShaderPushConstantRequirement{0, 16}};
        const std::array vsLocations{0U, 1U};
        auto vs = device.createShaderModule({
            .stage = rachis::ShaderStage::vertex,
            .bytecode = multiVsPnms,
            .entryPoint = "multiSetVertex",
            .shaderInterface = {.bindings = vsReq, .pushConstants = pushReq, .vertexInputLocations = vsLocations},
        });
        auto ps = device.createShaderModule({
            .stage = rachis::ShaderStage::pixel,
            .bytecode = multiPsPnms,
            .entryPoint = "multiSetPixel",
            .shaderInterface = {.bindings = psReq, .pushConstants = pushReq},
        });
        REQUIRE(vs);
        REQUIRE(ps);

        const auto allGraphics = rachis::ShaderStage::vertex | rachis::ShaderStage::pixel;
        auto gfxSetLayout = device.createBindingLayout({
            .entries =
                std::array{
                    rachis::BindingLayoutEntry{0, rachis::BindingType::dynamicUniformBuffer, allGraphics, 1},
                    rachis::BindingLayoutEntry{1, rachis::BindingType::sampledTexture, rachis::ShaderStage::pixel, 1},
                    rachis::BindingLayoutEntry{2, rachis::BindingType::sampler, rachis::ShaderStage::pixel, 1},
                },
        });
        REQUIRE(gfxSetLayout);
        auto gfxPipeLayout = device.createPipelineLayout({
            .bindingLayouts = std::array{*gfxSetLayout, *gfxSetLayout},
            .pushConstantRanges = std::array{rachis::PushConstantRange{allGraphics, 0, 16}},
        });
        REQUIRE(gfxPipeLayout);

        const std::array vertexAttrs{
            rachis::VertexAttribute{.format = rachis::VertexFormat::float32x2, .offset = 0, .shaderLocation = 0},
            rachis::VertexAttribute{.format = rachis::VertexFormat::uint32, .offset = 8, .shaderLocation = 1},
        };
        const std::array vertexBuffers{
            rachis::VertexBufferLayout{
                .stride = sizeof(RemapVertexHost),
                .stepMode = rachis::VertexStepMode::vertex,
                .attributes = vertexAttrs,
            },
        };
        const std::array colorTargets{
            rachis::ColorTargetState{.format = rachis::Format::rgba32Uint},
        };
        auto gfxPipeline = device.createGraphicsPipeline({
            .layout = *gfxPipeLayout,
            .vertexShader = *vs,
            .pixelShader = *ps,
            .vertexBuffers = vertexBuffers,
            .cullMode = rachis::CullMode::none,
            .colorTargets = colorTargets,
        });
        REQUIRE(gfxPipeline);

        auto uboArena = device.createBuffer({
            .size = 256,
            .usage = rachis::BufferUsage::constant,
            .memoryType = rachis::MemoryType::upload,
        });
        REQUIRE(uboArena);
        const CompactUniforms24Host set0Ubo{.w0 = 4000U, .w1 = 41U, .w2 = 42U, .w3 = 43U, .t0 = 440U, .t1 = 450U};
        const CompactUniforms24Host set1UboA{.w0 = 5000U, .w1 = 51U, .w2 = 52U, .w3 = 53U, .t0 = 540U, .t1 = 550U};
        const CompactUniforms24Host set1UboB{.w0 = 6000U, .w1 = 61U, .w2 = 62U, .w3 = 63U, .t0 = 640U, .t1 = 650U};
        REQUIRE(device.writeBuffer(*uboArena, 16, asByteSpan(set0Ubo)).has_value());
        REQUIRE(device.writeBuffer(*uboArena, 64, asByteSpan(set1UboA)).has_value());
        REQUIRE(device.writeBuffer(*uboArena, 112, asByteSpan(set1UboB)).has_value());

        const std::array<RemapVertexHost, 3> fullscreenTri{{
            {.position = {-1.0F, -1.0F}, .vertexTag = 900U},
            {.position = {3.0F, -1.0F}, .vertexTag = 900U},
            {.position = {-1.0F, 3.0F}, .vertexTag = 900U},
        }};
        auto vbo = device.createBuffer({
            .size = sizeof(fullscreenTri),
            .usage = rachis::BufferUsage::vertex,
            .memoryType = rachis::MemoryType::upload,
        });
        REQUIRE(vbo);
        REQUIRE(device.writeBuffer(*vbo, 0, asByteSpan(fullscreenTri)).has_value());

        const auto tex0 = create2x2RgbaTexture(device,
                                               {{{21, 43, 65, 87}, {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}},
                                               rachis::PipelineStage::pixelShader,
                                               "Multi graphics set0 tex");
        const auto tex1A = create2x2RgbaTexture(device,
                                                {{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 180, 240}}},
                                                rachis::PipelineStage::pixelShader,
                                                "Multi graphics set1A tex");
        const auto tex1B = create2x2RgbaTexture(device,
                                                {{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 195, 250}}},
                                                rachis::PipelineStage::pixelShader,
                                                "Multi graphics set1B tex");

        auto repeatSampler = device.createSampler({
            .minFilter = rachis::FilterMode::nearest,
            .magFilter = rachis::FilterMode::nearest,
            .addressU = rachis::AddressMode::repeat,
            .addressV = rachis::AddressMode::repeat,
        });
        auto clampSampler = device.createSampler({
            .minFilter = rachis::FilterMode::nearest,
            .magFilter = rachis::FilterMode::nearest,
            .addressU = rachis::AddressMode::clampToEdge,
            .addressV = rachis::AddressMode::clampToEdge,
        });
        REQUIRE(repeatSampler);
        REQUIRE(clampSampler);

        auto set0 = rachis::test::createBindingSet(
            device,
            {
                .layout = *gfxSetLayout,
                .writes =
                    std::array{
                        rachis::BindingWrite{0, 0, rachis::BufferBinding{*uboArena, 0, sizeof(CompactUniforms24Host)}},
                        rachis::BindingWrite{1, 0, tex0.view},
                        rachis::BindingWrite{2, 0, *repeatSampler},
                    },
            });
        auto set1A = rachis::test::createBindingSet(
            device,
            {
                .layout = *gfxSetLayout,
                .writes =
                    std::array{
                        rachis::BindingWrite{0, 0, rachis::BufferBinding{*uboArena, 0, sizeof(CompactUniforms24Host)}},
                        rachis::BindingWrite{1, 0, tex1A.view},
                        rachis::BindingWrite{2, 0, *clampSampler},
                    },
            });
        auto set1B = rachis::test::createBindingSet(
            device,
            {
                .layout = *gfxSetLayout,
                .writes =
                    std::array{
                        rachis::BindingWrite{0, 0, rachis::BufferBinding{*uboArena, 0, sizeof(CompactUniforms24Host)}},
                        rachis::BindingWrite{1, 0, tex1B.view},
                        rachis::BindingWrite{2, 0, *clampSampler},
                    },
            });
        REQUIRE(set0);
        REQUIRE(set1A);
        REQUIRE(set1B);

        auto target = device.createTexture({
            .width = 8,
            .height = 4,
            .format = rachis::Format::rgba32Uint,
            .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource,
        });
        REQUIRE(target);
        auto targetView = device.createTextureView({
            .texture = *target,
            .usage = rachis::TextureViewUsage::renderTarget,
            .dimension = rachis::TextureViewDimension::texture2D,
        });
        REQUIRE(targetView);
        const rachis::BufferTextureCopyRegion queryRegion{.extent = {.width = 8, .height = 4, .depth = 1}};
        const auto transfer = device.queryTextureTransferLayout(*target, queryRegion);
        REQUIRE(transfer);
        auto readback = device.createBuffer({
            .size = transfer->requiredBufferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
        });
        REQUIRE(readback);

        const std::array<std::uint32_t, 4> pushA{1U, 2U, 3U, 0x00001111U};
        const std::array<std::uint32_t, 4> pushB{4U, 5U, 6U, 0x00002222U};

        auto cmd = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(cmd);
        REQUIRE(device
                    .textureBarrier(*cmd,
                                    {
                                        .texture = *target,
                                        .stagesAfter = rachis::PipelineStage::renderTarget,
                                        .accessAfter = rachis::Access::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::renderTarget,
                                    })
                    .has_value());
        const std::array attachments{
            rachis::ColorAttachment{.view = *targetView, .loadOperation = rachis::LoadOperation::clear},
        };
        REQUIRE(device.beginRendering(*cmd, {.colorAttachments = attachments}).has_value());
        REQUIRE(device.bindGraphicsPipeline(*cmd, *gfxPipeline).has_value());
        REQUIRE(device.setViewport(*cmd, {.width = 8.0F, .height = 4.0F}).has_value());
        REQUIRE(device.bindVertexBuffer(*cmd, 0, *vbo, 0).has_value());

        REQUIRE(device.setScissor(*cmd, {.x = 0, .y = 0, .width = 4, .height = 4}).has_value());
        REQUIRE(device.bindGraphicsBindingSets(*cmd, 0, std::array{*set0, *set1A}, std::array{16U, 64U}).has_value());
        REQUIRE(device.setGraphicsPushConstants(*cmd, allGraphics, 0, asByteSpan(pushA)).has_value());
        REQUIRE(device.draw(*cmd, 3, 1, 0, 0).has_value());

        REQUIRE(device.setScissor(*cmd, {.x = 4, .y = 0, .width = 4, .height = 4}).has_value());
        REQUIRE(device.bindGraphicsBindingSet(*cmd, 1, *set1B, std::array{112U}).has_value());
        REQUIRE(device.setGraphicsPushConstants(*cmd, allGraphics, 0, asByteSpan(pushB)).has_value());
        REQUIRE(device.draw(*cmd, 3, 1, 0, 0).has_value());

        REQUIRE(device.endRendering(*cmd).has_value());
        REQUIRE(device
                    .textureBarrier(*cmd,
                                    {
                                        .texture = *target,
                                        .stagesBefore = rachis::PipelineStage::renderTarget,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::renderTarget,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                    })
                    .has_value());
        const rachis::BufferTextureCopyRegion copyRegion{
            .bytesPerRow = transfer->bytesPerRow,
            .rowsPerImage = transfer->rowsPerImage,
            .extent = {.width = 8, .height = 4, .depth = 1},
        };
        REQUIRE(device.copyTextureToBuffer(*cmd, *target, *readback, std::array{copyRegion}).has_value());
        auto pt = device.submit(std::move(*cmd));
        REQUIRE(pt);
        REQUIRE(device.wait(*pt).has_value());

        std::array<std::uint32_t, 4> leftPixel{};
        std::array<std::uint32_t, 4> rightPixel{};
        REQUIRE(device
                    .readBuffer(*readback,
                                0,
                                std::span<std::byte>{reinterpret_cast<std::byte*>(leftPixel.data()), sizeof(leftPixel)})
                    .has_value());
        REQUIRE(
            device
                .readBuffer(*readback,
                            4 * sizeof(std::uint32_t) * 4,
                            std::span<std::byte>{reinterpret_cast<std::byte*>(rightPixel.data()), sizeof(rightPixel)})
                .has_value());

        const std::uint32_t packed0 = 21U | (43U << 8U);
        const std::uint32_t packed1A = 180U | (240U << 8U);
        const std::uint32_t packed1B = 195U | (250U << 8U);

        CHECK(leftPixel[0] == (900U + pushA[0] + set0Ubo.w0 + set1UboA.w0));
        CHECK(leftPixel[1] == (set0Ubo.w0 + set0Ubo.t0 + set0Ubo.t1));
        CHECK(leftPixel[2] == (set1UboA.w0 + set1UboA.t1 + set1UboA.t0 + pushA[2]));
        CHECK(leftPixel[3] == (((packed0 << 16U) | packed1A) ^ pushA[3]));

        CHECK(rightPixel[0] == (900U + pushB[0] + set0Ubo.w0 + set1UboB.w0));
        CHECK(rightPixel[1] == (set0Ubo.w0 + set0Ubo.t0 + set0Ubo.t1));
        CHECK(rightPixel[2] == (set1UboB.w0 + set1UboB.t1 + set1UboB.t0 + pushB[2]));
        CHECK(rightPixel[3] == (((packed0 << 16U) | packed1B) ^ pushB[3]));
    }

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal variable-count binding set updates report accurate published descriptor statistics",
          "[size:large][rhi][metal][binding]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    auto texA = create2x2RgbaTexture(device,
                                     {{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}}},
                                     rachis::PipelineStage::computeShader,
                                     "VarStats Tex A");
    auto texB = create2x2RgbaTexture(device,
                                     {{{21, 22, 23, 24}, {25, 26, 27, 28}, {29, 30, 31, 32}, {33, 34, 35, 36}}},
                                     rachis::PipelineStage::computeShader,
                                     "VarStats Tex B");
    auto samplerA = device.createSampler({.label = "VarStats Sampler A"});
    auto samplerB = device.createSampler({
        .minFilter = rachis::FilterMode::linear,
        .magFilter = rachis::FilterMode::linear,
        .label = "VarStats Sampler B",
    });
    REQUIRE(samplerA);
    REQUIRE(samplerB);

    const std::array entries{
        rachis::BindingLayoutEntry{
            .binding = 0,
            .type = rachis::BindingType::sampler,
            .visibility = rachis::ShaderStage::compute,
            .count = 1,
        },
        rachis::BindingLayoutEntry{
            .binding = 1,
            .type = rachis::BindingType::sampledTexture,
            .visibility = rachis::ShaderStage::compute,
            .count = rachis::unboundedBindingCount,
            .flags = rachis::BindingFlag::partiallyBound | rachis::BindingFlag::variableDescriptorCount,
        },
    };
    auto layout = device.createBindingLayout({.entries = entries, .label = "VarStats layout"});
    REQUIRE(layout);

    const std::array initialWrites{
        rachis::BindingWrite{0, 0, *samplerA},
        rachis::BindingWrite{1, 0, texA.view},
    };
    auto set = rachis::test::createBindingSet(device,
                                              {
                                                  .layout = *layout,
                                                  .writes = initialWrites,
                                                  .variableDescriptorCount = 3,
                                                  .label = "VarStats set",
                                              });
    REQUIRE(set);

    const auto before = device.bindingUpdateStatistics();

    const std::array outOfBoundsWrite{
        rachis::BindingWrite{1, 3, texB.view},
    };
    auto rejected = device.updateBindingSet(*set, outOfBoundsWrite);
    REQUIRE_FALSE(rejected);
    CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
    const auto afterRejected = device.bindingUpdateStatistics();
    CHECK(afterRejected.resourceDescriptorsPublished == before.resourceDescriptorsPublished);
    CHECK(afterRejected.samplerDescriptorsPublished == before.samplerDescriptorsPublished);

    const std::array validResourceUpdate{
        rachis::BindingWrite{1, 2, texB.view},
    };
    REQUIRE(device.updateBindingSet(*set, validResourceUpdate).has_value());
    const auto afterResource = device.bindingUpdateStatistics();
    CHECK(afterResource.resourceDescriptorsPrepared - before.resourceDescriptorsPrepared == 1);
    CHECK(afterResource.resourceDescriptorsPublished - before.resourceDescriptorsPublished == 3);
    CHECK(afterResource.resourceTablePublishCount - before.resourceTablePublishCount == 1);
    CHECK(afterResource.samplerDescriptorsPrepared == before.samplerDescriptorsPrepared);
    CHECK(afterResource.samplerDescriptorsPublished - before.samplerDescriptorsPublished == 1);

    const std::array validSamplerUpdate{
        rachis::BindingWrite{0, 0, *samplerB},
    };
    REQUIRE(device.updateBindingSet(*set, validSamplerUpdate).has_value());
    const auto afterSampler = device.bindingUpdateStatistics();
    CHECK(afterSampler.samplerDescriptorsPrepared - afterResource.samplerDescriptorsPrepared == 1);
    CHECK(afterSampler.samplerDescriptorsPublished - afterResource.samplerDescriptorsPublished == 1);
    CHECK(afterSampler.samplerTablePublishCount - afterResource.samplerTablePublishCount == 1);
    CHECK(afterSampler.resourceDescriptorsPrepared == afterResource.resourceDescriptorsPrepared);
    CHECK(afterSampler.resourceDescriptorsPublished - afterResource.resourceDescriptorsPublished == 3);

    REQUIRE(device.destroyBindingSet(*set).has_value());
    REQUIRE(device.destroyBindingLayout(*layout).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal variable descriptor arrays reject over-limit counts and keep placeholders sparse",
          "[size:large][rhi][metal][binding]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto checkedCapabilities = device.capabilities();
    REQUIRE(checkedCapabilities.has_value());
    REQUIRE(checkedCapabilities->partiallyBoundBindingArrays);
    REQUIRE(checkedCapabilities->variableDescriptorCountBindingArrays);

    auto storage = device.createBuffer(rachis::BufferDesc{
        .size = 64,
        .usage = rachis::BufferUsage::storage,
        .label = "Metal variable storage",
    });
    REQUIRE(storage);
    const std::array variable{
        rachis::BindingLayoutEntry{
            .binding = 1,
            .type = rachis::BindingType::storageBuffer,
            .visibility = rachis::ShaderStage::compute,
            .count = rachis::unboundedBindingCount,
            .flags = rachis::BindingFlag::variableDescriptorCount | rachis::BindingFlag::partiallyBound,
        },
    };
    auto layout = device.createBindingLayout({.entries = variable});
    REQUIRE(layout);
    const std::array writes{rachis::BindingWrite{
        .binding = 1,
        .resource = rachis::BufferBinding{.buffer = *storage, .structureByteStride = sizeof(std::uint32_t)},
    }};
    auto set =
        rachis::test::createBindingSet(device, {.layout = *layout, .writes = writes, .variableDescriptorCount = 2});
    REQUIRE(set);

    auto overLimit = rachis::test::createBindingSet(
        device,
        {.layout = *layout,
         .writes = writes,
         .variableDescriptorCount = checkedCapabilities->limits.maxVariableBufferDescriptors + 1});
    REQUIRE_FALSE(overLimit);
    CHECK(overLimit.error().code == rachis::ErrorCode::invalidArgument);

    REQUIRE(device.destroyBindingSet(*set).has_value());
    REQUIRE(device.destroyBindingLayout(*layout).has_value());
    REQUIRE(device.destroyBuffer(*storage).has_value());
}

TEST_CASE("Metal variable descriptor arrays reach each published class limit and reject one more",
          "[size:large][rhi][metal][binding]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto capabilities = device.capabilities();
    REQUIRE(capabilities.has_value());
    REQUIRE(capabilities->partiallyBoundBindingArrays);
    REQUIRE(capabilities->variableDescriptorCountBindingArrays);
    const auto& limits = capabilities->limits;
    CHECK(limits.maxVariableBufferDescriptors == 31);
    CHECK(limits.maxVariableResourceDescriptors == 128);
    CHECK(limits.maxVariableSamplerDescriptors == 16);

    struct VariableArrayCase final
    {
        rachis::BindingType type = rachis::BindingType::uniformBuffer;
        std::uint32_t limit = 0;
    };
    const std::array cases{
        VariableArrayCase{rachis::BindingType::uniformBuffer, limits.maxVariableBufferDescriptors},
        VariableArrayCase{rachis::BindingType::readOnlyStorageBuffer, limits.maxVariableBufferDescriptors},
        VariableArrayCase{rachis::BindingType::storageBuffer, limits.maxVariableBufferDescriptors},
        VariableArrayCase{rachis::BindingType::sampledTexture, limits.maxVariableResourceDescriptors},
        VariableArrayCase{rachis::BindingType::storageTexture, limits.maxVariableResourceDescriptors},
        VariableArrayCase{rachis::BindingType::sampler, limits.maxVariableSamplerDescriptors},
    };
    for (const auto& variableCase : cases)
    {
        for (const std::uint32_t binding : {0U, 1U})
        {
            INFO("type=" << static_cast<int>(variableCase.type) << " limit=" << variableCase.limit
                         << " binding=" << binding);
            REQUIRE(variableCase.limit > binding);
            const std::array entries{
                rachis::BindingLayoutEntry{
                    .binding = binding,
                    .type = variableCase.type,
                    .visibility = rachis::ShaderStage::compute,
                    .count = rachis::unboundedBindingCount,
                    .flags = rachis::BindingFlag::variableDescriptorCount | rachis::BindingFlag::partiallyBound,
                },
            };
            auto layout = device.createBindingLayout({.entries = entries});
            REQUIRE(layout);
            const auto reachable = variableCase.limit - binding;
            auto atLimit =
                rachis::test::createBindingSet(device, {.layout = *layout, .variableDescriptorCount = reachable});
            CHECK(atLimit.has_value());
            if (atLimit)
            {
                CHECK(device.destroyBindingSet(*atLimit).has_value());
            }
            auto overLimit =
                rachis::test::createBindingSet(device, {.layout = *layout, .variableDescriptorCount = reachable + 1});
            REQUIRE_FALSE(overLimit);
            CHECK(overLimit.error().code == rachis::ErrorCode::invalidArgument);
            REQUIRE(device.destroyBindingLayout(*layout).has_value());
        }
    }
    CHECK(device.diagnosticCounts().errors == 0);
}
