#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <thread>
#include <utility>
#include <vector>

namespace
{

    [[nodiscard]] rachis::DeviceDesc vulkanDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::vulkan,
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

#ifdef RACHIS_SPIRV_DIR
    [[nodiscard]] std::vector<std::byte> loadSpirv(const char* name)
    {
        return loadBytes(std::filesystem::path(RACHIS_SPIRV_DIR) / name);
    }
#endif

} // namespace

TEST_CASE("Vulkan binding layouts, pools, and immutable sets reject malformed state atomically",
          "[size:medium][rhi][vulkan][binding]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    auto constants = device.createBuffer(rachis::BufferDesc{
        .size = 512,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan binding constants",
    });
    auto storage = device.createBuffer(rachis::BufferDesc{
        .size = 1024,
        .usage = rachis::BufferUsage::storage,
        .label = "Vulkan binding storage",
    });
    auto texture = device.createTexture(rachis::TextureDesc{
        .width = 16,
        .height = 16,
        .mipLevels = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
        .label = "Vulkan binding texture",
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
    auto sampler = device.createSampler(rachis::SamplerDesc{.label = "Vulkan binding sampler"});
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
    auto layout = device.createBindingLayout({.entries = layoutEntries, .label = "Vulkan complete layout"});
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
                                                  .label = "Vulkan complete set",
                                              });
    REQUIRE(set);

    const std::array duplicateBindings{
        rachis::BindingLayoutEntry{1, rachis::BindingType::uniformBuffer},
        rachis::BindingLayoutEntry{1, rachis::BindingType::sampler},
    };
    CHECK_FALSE(device.createBindingLayout({.entries = duplicateBindings}).has_value());
    const std::array overlappingRegisters{
        rachis::BindingLayoutEntry{1, rachis::BindingType::sampledTexture, rachis::ShaderStage::pixel, 4},
        rachis::BindingLayoutEntry{3, rachis::BindingType::readOnlyStorageBuffer},
    };
    CHECK_FALSE(device.createBindingLayout({.entries = overlappingRegisters}).has_value());
    const std::array zeroCount{
        rachis::BindingLayoutEntry{0, rachis::BindingType::sampler, rachis::ShaderStage::pixel, 0},
    };
    CHECK_FALSE(device.createBindingLayout({.entries = zeroCount}).has_value());
    const std::array variable{
        rachis::BindingLayoutEntry{
            .binding = 0,
            .type = rachis::BindingType::sampledTexture,
            .count = rachis::unboundedBindingCount,
            .flags = rachis::BindingFlag::variableDescriptorCount,
        },
    };
    auto variableLayout = device.createBindingLayout({.entries = variable});
    if (rachis::test::requireCapabilities(device).variableDescriptorCountBindingArrays)
    {
        REQUIRE(variableLayout);
        REQUIRE(device.destroyBindingLayout(*variableLayout));
    }
    else
    {
        REQUIRE_FALSE(variableLayout);
        CHECK(variableLayout.error().code == rachis::ErrorCode::unsupported);
    }
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
        .label = "Vulkan tiny pool",
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

    REQUIRE(device.destroyBindingLayout(*emptyLayout));
    REQUIRE(device.destroyBindingSet(*emptySet));
    REQUIRE(device.destroyBindingSet(*set));
    CHECK_FALSE(device.destroyBindingSet(*set).has_value());
}

TEST_CASE("Vulkan binding-set updates are copy-on-write and pipeline layouts accept push constants",
          "[size:medium][rhi][vulkan][binding][update]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    auto bufferA = device.createBuffer({
        .size = 256,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan update A",
    });
    auto bufferB = device.createBuffer({
        .size = 256,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan update B",
    });
    auto samplerA = device.createSampler({.label = "Vulkan update sampler A"});
    auto samplerB = device.createSampler({
        .minFilter = rachis::FilterMode::linear,
        .magFilter = rachis::FilterMode::linear,
        .label = "Vulkan update sampler B",
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
    auto layout = device.createBindingLayout({.entries = entries, .label = "Vulkan update layout"});
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

    const std::array push{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16},
    };
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = std::array{*layout},
        .pushConstantRanges = push,
        .label = "Vulkan pipeline layout",
    });
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

#ifdef RACHIS_SPIRV_DIR
TEST_CASE("Vulkan binding pools hold every copy-on-write version that their descriptor capacity admits",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][binding][update]")
{
    constexpr std::uint32_t versionCapacity = 8;

    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    const auto bytecode = loadBytes(std::filesystem::path(RACHIS_SPIRV_DIR) / "VulkanComputeWrite.spv");
    REQUIRE_FALSE(bytecode.empty());
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Vulkan version-capacity shader",
        .shaderInterface = rachis::test::computeWriteShaderInterface(),
    });
    REQUIRE(shader.has_value());

    auto storage = device.createBuffer({
        .size = 256,
        .usage = rachis::BufferUsage::storage,
        .label = "Vulkan version-capacity storage",
    });
    auto constants = device.createBuffer({
        .size = 256,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan version-capacity constants",
    });
    REQUIRE(storage.has_value());
    REQUIRE(constants.has_value());

    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute, 1},
    };
    auto layout = device.createBindingLayout({.entries = entries, .label = "Vulkan version-capacity layout"});
    REQUIRE(layout.has_value());
    const std::array poolSizes{
        rachis::BindingPoolSize{rachis::BindingType::storageBuffer, versionCapacity},
        rachis::BindingPoolSize{rachis::BindingType::uniformBuffer, versionCapacity},
    };
    auto pool = device.createBindingPool({
        .maxBindingSets = 1,
        .sizes = poolSizes,
        .label = "Vulkan version-capacity pool",
    });
    REQUIRE(pool.has_value());
    const std::array writes{
        rachis::BindingWrite{0, 0, rachis::BufferBinding{*storage, 0, 256, sizeof(std::uint32_t)}},
        rachis::BindingWrite{1, 0, rachis::BufferBinding{*constants, 0, 256}},
    };
    auto set = device.createBindingSet({
        .layout = *layout,
        .writes = writes,
        .label = "Vulkan version-capacity set",
        .pool = *pool,
    });
    REQUIRE(set.has_value());

    const std::array pushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)},
    };
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = std::array{*layout},
        .pushConstantRanges = pushConstantRanges,
        .label = "Vulkan version-capacity pipeline layout",
    });
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Vulkan version-capacity pipeline",
    });
    REQUIRE(pipeline.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    for (std::uint32_t round = 1; round < versionCapacity; ++round)
    {
        CAPTURE(round);
        REQUIRE(device.bindComputeBindingSet(*command, 0, *set).has_value());
        auto updated = device.updateBindingSet(*set, writes);
        if (!updated)
        {
            FAIL(updated.error().message);
        }
    }
    REQUIRE(device.bindComputeBindingSet(*command, 0, *set).has_value());
    auto exhausted = device.updateBindingSet(*set, writes);
    REQUIRE_FALSE(exhausted.has_value());
    CHECK(exhausted.error().code == rachis::ErrorCode::resourceExhausted);

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    REQUIRE(device.updateBindingSet(*set, writes).has_value());

    REQUIRE(device.destroyBindingSet(*set).has_value());
    REQUIRE(device.destroyBindingPool(*pool).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*layout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBuffer(*storage).has_value());
    REQUIRE(device.destroyBuffer(*constants).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan sampled-texture bindings read textures that barriers leave in the common layout",
          "[size:medium][rhi][vulkan][binding][layout]")
{
    constexpr std::uint32_t rowPitch = 256;
    constexpr std::uint64_t uploadSize = static_cast<std::uint64_t>(rowPitch) * 2;
    constexpr std::uint64_t outputSize = 4 * sizeof(float);
    constexpr std::array uploadedValues{0.125F, 0.375F, 0.625F, 0.875F};

    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    const auto bytecode = loadBytes(std::filesystem::path(RACHIS_SPIRV_DIR) / "VulkanSampledTexture.spv");
    REQUIRE_FALSE(bytecode.empty());
    const std::array requirements{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::sampledTexture, 1},
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::sampler, 1},
        rachis::ShaderBindingRequirement{0, 2, rachis::ShaderResourceType::storageBuffer, 1},
    };
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Vulkan common-layout sampling shader",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, requirements),
    });
    REQUIRE(shader.has_value());
    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::sampler, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{2, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    auto layout = device.createBindingLayout({.entries = entries, .label = "Vulkan common-layout sampling layout"});
    REQUIRE(layout.has_value());
    const std::array poolSizes{
        rachis::BindingPoolSize{rachis::BindingType::sampledTexture, 1},
        rachis::BindingPoolSize{rachis::BindingType::sampler, 1},
        rachis::BindingPoolSize{rachis::BindingType::storageBuffer, 1},
    };
    auto pool = device.createBindingPool({
        .maxBindingSets = 1,
        .sizes = poolSizes,
        .label = "Vulkan common-layout sampling pool",
    });
    REQUIRE(pool.has_value());
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = std::array{*layout},
        .label = "Vulkan common-layout sampling pipeline layout",
    });
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Vulkan common-layout sampling pipeline",
    });
    REQUIRE(pipeline.has_value());
    auto sampler = device.createSampler({.label = "Vulkan common-layout point sampler"});
    REQUIRE(sampler.has_value());

    auto upload = device.createBuffer({
        .size = uploadSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan common-layout upload",
    });
    auto output = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .label = "Vulkan common-layout samples",
    });
    auto readback = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan common-layout readback",
    });
    REQUIRE(upload.has_value());
    REQUIRE(output.has_value());
    REQUIRE(readback.has_value());

    const auto createFloatTexture = [&] {
        std::array<std::byte, uploadSize> bytes{};
        std::memcpy(bytes.data(), uploadedValues.data(), 2 * sizeof(float));
        std::memcpy(bytes.data() + rowPitch, uploadedValues.data() + 2, 2 * sizeof(float));
        REQUIRE(device.writeBuffer(*upload, 0, bytes).has_value());
        auto texture = device.createTexture({
            .width = 2,
            .height = 2,
            .format = rachis::Format::r32Float,
            .usage = rachis::TextureUsage::copyDestination | rachis::TextureUsage::sampled,
            .label = "Vulkan common-layout r32Float texture",
        });
        REQUIRE(texture.has_value());
        return *texture;
    };
    const auto copyUpload = [&](rachis::CommandBuffer& command, rachis::TextureHandle texture) {
        REQUIRE(device
                    .copyBufferToTexture(command,
                                         {
                                             .buffer = *upload,
                                             .bytesPerRow = rowPitch,
                                             .rowsPerImage = 2,
                                             .texture = texture,
                                             .extent = {2, 2, 1},
                                         })
                    .has_value());
    };
    const auto sampleAndRead = [&](rachis::CommandBuffer command, rachis::TextureViewHandle view) {
        const std::array writes{
            rachis::BindingWrite{0, 0, view},
            rachis::BindingWrite{1, 0, *sampler},
            rachis::BindingWrite{2, 0, rachis::BufferBinding{*output, 0, outputSize, sizeof(float)}},
        };
        auto set = device.createBindingSet({
            .layout = *layout,
            .writes = writes,
            .label = "Vulkan common-layout sampling set",
            .pool = *pool,
        });
        REQUIRE(set.has_value());
        REQUIRE(device
                    .bufferBarrier(command,
                                   {
                                       .buffer = *output,
                                       .stagesAfter = rachis::PipelineStage::computeShader,
                                       .accessAfter = rachis::Access::unorderedAccess,
                                   })
                    .has_value());
        REQUIRE(device.bindComputePipeline(command, *pipeline).has_value());
        REQUIRE(device.bindComputeBindingSet(command, 0, *set).has_value());
        REQUIRE(device.dispatch(command, 1).has_value());
        REQUIRE(device
                    .bufferBarrier(command,
                                   {
                                       .buffer = *output,
                                       .stagesBefore = rachis::PipelineStage::computeShader,
                                       .stagesAfter = rachis::PipelineStage::copy,
                                       .accessBefore = rachis::Access::unorderedAccess,
                                       .accessAfter = rachis::Access::copySource,
                                   })
                    .has_value());
        REQUIRE(
            device.copyBuffer(command, {.source = *output, .destination = *readback, .size = outputSize}).has_value());
        auto submitted = device.submit(std::move(command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());
        std::array<float, 4> sampled{};
        REQUIRE(device.readBuffer(*readback, 0, std::as_writable_bytes(std::span(sampled))).has_value());
        REQUIRE(device.destroyBindingSet(*set).has_value());
        return sampled;
    };

    std::vector<rachis::TextureHandle> textures;
    std::vector<rachis::TextureViewHandle> views;
    SECTION("a copy into a texture kept in common, then a common-to-common barrier to shader reads")
    {
        const auto texture = createFloatTexture();
        textures.push_back(texture);
        auto view = device.createTextureView({.texture = texture, .usage = rachis::TextureViewUsage::sampled});
        REQUIRE(view.has_value());
        views.push_back(*view);
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = texture,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessAfter = rachis::Access::copyDestination,
                                        .layoutAfter = rachis::TextureLayout::common,
                                    })
                    .has_value());
        copyUpload(*command, texture);
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = texture,
                                        .stagesBefore = rachis::PipelineStage::copy,
                                        .stagesAfter = rachis::PipelineStage::computeShader,
                                        .accessBefore = rachis::Access::copyDestination,
                                        .accessAfter = rachis::Access::shaderResource,
                                        .layoutBefore = rachis::TextureLayout::common,
                                        .layoutAfter = rachis::TextureLayout::common,
                                    })
                    .has_value());
        const auto sampled = sampleAndRead(std::move(*command), *view);
        CHECK(sampled == uploadedValues);
    }
    SECTION("a barrier from undefined straight to shader reads in common")
    {
        const auto texture = createFloatTexture();
        textures.push_back(texture);
        auto view = device.createTextureView({.texture = texture, .usage = rachis::TextureViewUsage::sampled});
        REQUIRE(view.has_value());
        views.push_back(*view);
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = texture,
                                        .stagesAfter = rachis::PipelineStage::computeShader,
                                        .accessAfter = rachis::Access::shaderResource,
                                        .layoutAfter = rachis::TextureLayout::common,
                                    })
                    .has_value());
        static_cast<void>(sampleAndRead(std::move(*command), *view));
    }
    SECTION("mip 1 after generateMipmaps leaves the texture in common")
    {
        REQUIRE(rachis::test::requireCapabilities(device).mipmapGeneration);
        constexpr std::array<std::uint8_t, 4> baseReds{0, 64, 128, 192};
        std::array<std::byte, uploadSize> bytes{};
        for (std::size_t texel = 0; texel < baseReds.size(); ++texel)
        {
            const auto offset = (texel / 2) * rowPitch + (texel % 2) * 4;
            bytes[offset] = static_cast<std::byte>(baseReds[texel]);
            bytes[offset + 3] = std::byte{255};
        }
        REQUIRE(device.writeBuffer(*upload, 0, bytes).has_value());
        auto texture = device.createTexture({
            .width = 2,
            .height = 2,
            .mipLevels = 2,
            .format = rachis::Format::rgba8Unorm,
            .usage =
                rachis::TextureUsage::copyDestination | rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
            .label = "Vulkan common-layout generated mips",
        });
        REQUIRE(texture.has_value());
        textures.push_back(*texture);
        auto view = device.createTextureView({
            .texture = *texture,
            .usage = rachis::TextureViewUsage::sampled,
            .firstMipLevel = 1,
            .mipLevelCount = 1,
        });
        REQUIRE(view.has_value());
        views.push_back(*view);
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *texture,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessAfter = rachis::Access::copyDestination,
                                        .layoutAfter = rachis::TextureLayout::copyDestination,
                                    })
                    .has_value());
        copyUpload(*command, *texture);
        REQUIRE(device
                    .generateMipmaps(*command,
                                     {
                                         .texture = *texture,
                                         .baseLayout = rachis::TextureLayout::copyDestination,
                                         .finalLayout = rachis::TextureLayout::common,
                                     })
                    .has_value());
        const auto sampled = sampleAndRead(std::move(*command), *view);
        constexpr float expected = 96.0F / 255.0F;
        for (const auto value : sampled)
        {
            CHECK(value >= expected - 1.0F / 255.0F);
            CHECK(value <= expected + 1.0F / 255.0F);
        }
    }
    SECTION("a texture in the shaderResource layout")
    {
        const auto texture = createFloatTexture();
        textures.push_back(texture);
        auto view = device.createTextureView({.texture = texture, .usage = rachis::TextureViewUsage::sampled});
        REQUIRE(view.has_value());
        views.push_back(*view);
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = texture,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessAfter = rachis::Access::copyDestination,
                                        .layoutAfter = rachis::TextureLayout::copyDestination,
                                    })
                    .has_value());
        copyUpload(*command, texture);
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = texture,
                                        .stagesBefore = rachis::PipelineStage::copy,
                                        .stagesAfter = rachis::PipelineStage::computeShader,
                                        .accessBefore = rachis::Access::copyDestination,
                                        .accessAfter = rachis::Access::shaderResource,
                                        .layoutBefore = rachis::TextureLayout::copyDestination,
                                        .layoutAfter = rachis::TextureLayout::shaderResource,
                                    })
                    .has_value());
        const auto sampled = sampleAndRead(std::move(*command), *view);
        CHECK(sampled == uploadedValues);
    }

    for (const auto view : views)
    {
        REQUIRE(device.destroyTextureView(view).has_value());
    }
    for (const auto texture : textures)
    {
        REQUIRE(device.destroyTexture(texture).has_value());
    }
    REQUIRE(device.destroySampler(*sampler).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingPool(*pool).has_value());
    REQUIRE(device.destroyBindingLayout(*layout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*output).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
#endif

TEST_CASE("Vulkan binding pools without native descriptor sizes serve descriptor-free binding sets",
          "[size:medium][rhi][vulkan][binding]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    auto layout = device.createBindingLayout({.label = "Vulkan descriptor-free layout"});
    REQUIRE(layout.has_value());
    const std::array feedbackSizes{rachis::BindingPoolSize{rachis::BindingType::minMipFeedback, 4}};
    const std::array<std::span<const rachis::BindingPoolSize>, 2> poolSizes{
        std::span<const rachis::BindingPoolSize>{},
        std::span<const rachis::BindingPoolSize>{feedbackSizes},
    };
    for (const auto sizes : poolSizes)
    {
        CAPTURE(sizes.size());
        auto pool = device.createBindingPool({
            .maxBindingSets = 2,
            .sizes = sizes,
            .label = "Vulkan descriptor-free pool",
        });
        REQUIRE(pool.has_value());
        auto set = device.createBindingSet({
            .layout = *layout,
            .label = "Vulkan descriptor-free set",
            .pool = *pool,
        });
        REQUIRE(set.has_value());
        REQUIRE(device.destroyBindingSet(*set).has_value());
        REQUIRE(device.destroyBindingPool(*pool).has_value());
    }
    REQUIRE(device.destroyBindingLayout(*layout).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan binding pools synchronize concurrent descriptor allocation, copy-on-write updates, and retirement",
          "[size:medium][rhi][vulkan][binding][concurrency]")
{
    constexpr std::size_t workerCount = 4;
    constexpr std::size_t setsPerWorker = 4;
    constexpr std::size_t iterationCount = 12;

    constexpr std::uint32_t poolCapacity =
        static_cast<std::uint32_t>(workerCount * (setsPerWorker + 1) + (1 + workerCount));

    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    auto bufferA = device.createBuffer({
        .size = 256,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan concurrent binding buffer A",
    });
    auto bufferB = device.createBuffer({
        .size = 256,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan concurrent binding buffer B",
    });
    auto samplerA = device.createSampler({.label = "Vulkan concurrent sampler A"});
    auto samplerB = device.createSampler({
        .minFilter = rachis::FilterMode::linear,
        .magFilter = rachis::FilterMode::linear,
        .label = "Vulkan concurrent sampler B",
    });
    REQUIRE(bufferA);
    REQUIRE(bufferB);
    REQUIRE(samplerA);
    REQUIRE(samplerB);

    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::sampler, rachis::ShaderStage::compute, 1},
    };
    auto layout = device.createBindingLayout({
        .entries = entries,
        .label = "Vulkan concurrent binding layout",
    });
    REQUIRE(layout);

    const std::array poolSizes{
        rachis::BindingPoolSize{rachis::BindingType::uniformBuffer, poolCapacity},
        rachis::BindingPoolSize{rachis::BindingType::sampler, poolCapacity},
    };
    auto sharedPool = device.createBindingPool({
        .maxBindingSets = poolCapacity,
        .sizes = poolSizes,
        .label = "Vulkan shared concurrent binding pool",
    });
    REQUIRE(sharedPool);

    const std::array writesA{
        rachis::BindingWrite{0, 0, rachis::BufferBinding{*bufferA, 0, 256}},
        rachis::BindingWrite{1, 0, *samplerA},
    };
    const std::array writesB{
        rachis::BindingWrite{0, 0, rachis::BufferBinding{*bufferB, 0, 256}},
        rachis::BindingWrite{1, 0, *samplerB},
    };

    auto contendedSet = device.createBindingSet({
        .layout = *layout,
        .writes = writesA,
        .label = "Vulkan contended shared set",
        .pool = *sharedPool,
    });
    REQUIRE(contendedSet);

    struct WorkerOutcome final
    {
        bool allPerThreadSucceeded = true;
        bool unexpectedContendedError = false;
    };
    std::array<WorkerOutcome, workerCount> outcomes{};
    std::array<std::atomic<std::uint32_t>, iterationCount> contendedSuccesses{};
    std::atomic<bool> stopCollector{false};
    std::barrier roundBarrier(static_cast<std::ptrdiff_t>(workerCount));

    {
        std::jthread collector([&device, &stopCollector] {
            while (!stopCollector.load(std::memory_order_acquire))
            {
                device.collectGarbage();
                static_cast<void>(device.bindingUpdateStatistics());
                std::this_thread::yield();
            }
        });

        std::vector<std::jthread> workers;
        workers.reserve(workerCount);
        for (std::size_t workerIndex = 0; workerIndex < workerCount; ++workerIndex)
        {
            workers.emplace_back([&, workerIndex] {
                auto& outcome = outcomes[workerIndex];
                for (std::size_t iteration = 0; iteration < iterationCount; ++iteration)
                {
                    roundBarrier.arrive_and_wait();

                    std::array<rachis::BindingSetHandle, setsPerWorker> localSets{};
                    std::size_t createdCount = 0;
                    for (std::size_t slot = 0; slot < setsPerWorker; ++slot)
                    {
                        auto allocated = device.createBindingSet({
                            .layout = *layout,
                            .writes = writesA,
                            .pool = *sharedPool,
                        });
                        if (!allocated)
                        {
                            outcome.allPerThreadSucceeded = false;
                            break;
                        }
                        localSets[createdCount++] = *allocated;
                    }

                    for (std::size_t slot = 0; slot < createdCount; ++slot)
                    {
                        if (!device.updateBindingSet(localSets[slot], writesB) ||
                            !device.updateBindingSet(localSets[slot], writesA))
                        {
                            outcome.allPerThreadSucceeded = false;
                        }
                    }

                    roundBarrier.arrive_and_wait();
                    const auto& contendedWrites = ((iteration + workerIndex) % 2U == 0U) ? writesB : writesA;
                    auto contendedResult = device.updateBindingSet(*contendedSet, contendedWrites);
                    if (contendedResult)
                    {
                        contendedSuccesses[iteration].fetch_add(1U, std::memory_order_relaxed);
                    }
                    else if (contendedResult.error().code != rachis::ErrorCode::invalidState)
                    {
                        outcome.unexpectedContendedError = true;
                    }

                    for (std::size_t slot = 0; slot < createdCount; ++slot)
                    {
                        if (!device.destroyBindingSet(localSets[slot]))
                        {
                            outcome.allPerThreadSucceeded = false;
                        }
                    }
                }
            });
        }
        workers.clear();
        stopCollector.store(true, std::memory_order_release);
    }

    for (std::size_t workerIndex = 0; workerIndex < workerCount; ++workerIndex)
    {
        CHECK(outcomes[workerIndex].allPerThreadSucceeded);
        CHECK_FALSE(outcomes[workerIndex].unexpectedContendedError);
    }
    for (std::size_t iteration = 0; iteration < iterationCount; ++iteration)
    {
        CHECK(contendedSuccesses[iteration].load(std::memory_order_relaxed) >= 1U);
    }

    REQUIRE(device.destroyBindingSet(*contendedSet));
    device.collectGarbage();

    std::vector<rachis::BindingSetHandle> refill;
    refill.reserve(poolCapacity);
    for (std::uint32_t index = 0; index < poolCapacity; ++index)
    {
        auto set = device.createBindingSet({
            .layout = *layout,
            .writes = writesA,
            .pool = *sharedPool,
        });
        REQUIRE(set);
        refill.push_back(*set);
    }
    auto overflow = device.createBindingSet({
        .layout = *layout,
        .writes = writesA,
        .pool = *sharedPool,
    });
    REQUIRE_FALSE(overflow);
    CHECK(overflow.error().code == rachis::ErrorCode::resourceExhausted);

    for (const auto handle : refill)
    {
        REQUIRE(device.destroyBindingSet(handle));
    }
    REQUIRE(device.destroyBindingPool(*sharedPool));
    REQUIRE(device.destroyBindingLayout(*layout));

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

#ifdef RACHIS_SPIRV_DIR
TEST_CASE("Vulkan default-visibility layouts and mixed-stage barriers stay validation-clean on any feature set",
          "[size:medium][rhi][vulkan][binding][validation]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);

    const auto bytecode = loadSpirv("VulkanComputeWrite.spv");
    REQUIRE_FALSE(bytecode.empty());
    auto shader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Vulkan default-visibility compute",
        .shaderInterface = rachis::test::computeWriteShaderInterface(),
    });
    REQUIRE(shader.has_value());
    constexpr std::uint64_t storageSize = 64 * sizeof(std::uint32_t);
    auto storage = device.createBuffer(rachis::BufferDesc{
        .size = storageSize,
        .usage = rachis::BufferUsage::storage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan default-visibility storage",
    });
    REQUIRE(storage.has_value());

    const std::array entries{
        rachis::BindingLayoutEntry{.binding = 0, .type = rachis::BindingType::storageBuffer},
    };
    auto bindingLayout = device.createBindingLayout({.entries = entries, .label = "Vulkan default-visibility layout"});
    REQUIRE(bindingLayout.has_value());
    const std::array graphicsEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::allGraphics},
    };
    auto graphicsLayout = device.createBindingLayout({.entries = graphicsEntries});
    REQUIRE(graphicsLayout.has_value());
    const std::array writes{
        rachis::BindingWrite{
            .binding = 0,
            .resource =
                rachis::BufferBinding{
                    .buffer = *storage,
                    .size = storageSize,
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device, {.layout = *bindingLayout, .writes = writes});
    REQUIRE(bindingSet.has_value());
    const std::array bindingLayouts{*bindingLayout};
    const std::array pushConstantRanges{
        rachis::PushConstantRange{.visibility = rachis::ShaderStage::all, .size = 2 * sizeof(std::uint32_t)},
    };
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = pushConstantRanges,
        .label = "Vulkan default-visibility pipeline layout",
    });
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *shader});
    REQUIRE(pipeline.has_value());

    constexpr auto everyShaderStage = rachis::PipelineStage::vertexShader | rachis::PipelineStage::pixelShader |
                                      rachis::PipelineStage::computeShader | rachis::PipelineStage::geometryShader |
                                      rachis::PipelineStage::hullShader | rachis::PipelineStage::domainShader |
                                      rachis::PipelineStage::taskShader | rachis::PipelineStage::meshShader |
                                      rachis::PipelineStage::rayTracingShader;
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(
        device
            .bufferBarrier(*command,
                           rachis::BufferBarrier{
                               .buffer = *storage,
                               .stagesBefore = rachis::PipelineStage::all,
                               .stagesAfter = rachis::PipelineStage::computeShader | rachis::PipelineStage::meshShader |
                                              rachis::PipelineStage::rayTracingShader,
                               .accessBefore = rachis::Access::common,
                               .accessAfter = rachis::Access::unorderedAccess,
                           })
            .has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
    const std::array constants{3U, 7U};
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(constants))).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
    REQUIRE(device
                .resourceBarriers(*command,
                                  {.globalBarriers = std::array{rachis::GlobalBarrier{
                                       .stagesBefore = everyShaderStage,
                                       .stagesAfter = everyShaderStage,
                                       .accessBefore = rachis::Access::unorderedAccess,
                                       .accessAfter = rachis::Access::shaderResource,
                                   }}})
                .has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);

    const std::array optionalStages{
        std::pair{rachis::ShaderStage::geometry, capabilities.geometryShaders},
        std::pair{rachis::ShaderStage::hull, capabilities.tessellationShaders},
        std::pair{rachis::ShaderStage::domain, capabilities.tessellationShaders},
        std::pair{rachis::ShaderStage::task, capabilities.taskShaders},
        std::pair{rachis::ShaderStage::mesh, capabilities.meshShaders},
        std::pair{rachis::ShaderStage::rayGeneration, capabilities.rayTracing},
    };
    for (const auto& [stage, enabled] : optionalStages)
    {
        const std::array entry{rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, stage}};
        auto explicitLayout = device.createBindingLayout({.entries = entry});
        CHECK(explicitLayout.has_value() == enabled);
        if (explicitLayout.has_value())
        {
            REQUIRE(device.destroyBindingLayout(*explicitLayout).has_value());
        }
        else
        {
            CHECK(explicitLayout.error().code == rachis::ErrorCode::unsupported);
        }
    }

    auto rejectedCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(rejectedCommand.has_value());
    const auto meshOnly = device.resourceBarriers(*rejectedCommand,
                                                  {.globalBarriers = std::array{rachis::GlobalBarrier{
                                                       .stagesBefore = rachis::PipelineStage::computeShader,
                                                       .stagesAfter = rachis::PipelineStage::meshShader,
                                                       .accessBefore = rachis::Access::unorderedAccess,
                                                       .accessAfter = rachis::Access::shaderResource,
                                                   }}});
    CHECK(meshOnly.has_value() == capabilities.meshShaders);
    if (!meshOnly.has_value())
    {
        CHECK(meshOnly.error().code == rachis::ErrorCode::unsupported);
    }

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*graphicsLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
}
#endif
