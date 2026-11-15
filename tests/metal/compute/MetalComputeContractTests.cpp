#include "Internal/MetalShaderManifest.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace
{

    constexpr std::uint32_t imageSize = 8;
    constexpr std::uint32_t valueCount = 65;
    constexpr std::uint32_t expectedSum = 6112;

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

    [[nodiscard]] std::uint32_t readU32(rachis::Device& device, rachis::BufferHandle buffer, std::uint64_t offset = 0)
    {
        std::uint32_t value = 0;
        REQUIRE(device.readBuffer(buffer, offset, std::as_writable_bytes(std::span{&value, 1})).has_value());
        return value;
    }

} // namespace

TEST_CASE("Metal dispatch rejects zero work, over-limit groups, and indirect arguments",
          "[size:large][rhi][metal][compute][validation]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto checkedCapabilities = device.capabilities();
    REQUIRE(checkedCapabilities.has_value());
    auto layout = device.createPipelineLayout({});
    REQUIRE(layout);
    const auto bytecode = loadBytes(pnmsPath("MetalPushConstant.cs.metallib"));
    REQUIRE_FALSE(bytecode.empty());
    const std::array shaderBindings{
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
    };
    const std::array shaderPush{rachis::ShaderPushConstantRequirement{0, 16}};
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "computeMain",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, shaderBindings, shaderPush),
    });
    REQUIRE(shader);
    const std::array layoutEntries{
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = layoutEntries});
    REQUIRE(bindingLayout);
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = std::array{*bindingLayout},
        .pushConstantRanges = std::array{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16}},
    });
    REQUIRE(pipelineLayout);
    auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *shader});
    REQUIRE(pipeline);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command);
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    auto zero = device.dispatch(*command, 0);
    REQUIRE_FALSE(zero);
    CHECK(zero.error().code == rachis::ErrorCode::invalidArgument);
    const auto limit = checkedCapabilities->limits.maxComputeWorkgroupCount[0];
    auto over = device.dispatch(*command, limit + 1);
    REQUIRE_FALSE(over);
    CHECK(over.error().code == rachis::ErrorCode::invalidArgument);
    auto indirect = device.dispatchIndirect(*command, rachis::BufferHandle{});
    REQUIRE_FALSE(indirect);
    CHECK(indirect.error().code == rachis::ErrorCode::invalidArgument);
}

TEST_CASE("Metal scatter, reduce, and storage-image dispatch match GPU readback and survive copy/graphics splits",
          "[size:large][rhi][metal][compute][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    const auto scatterBytes = loadBytes(pnmsPath("MetalComputeStorage.scatterMain.metallib"));
    const auto reduceBytes = loadBytes(pnmsPath("MetalComputeStorage.reduceMain.metallib"));
    REQUIRE_FALSE(scatterBytes.empty());
    REQUIRE_FALSE(reduceBytes.empty());
    auto scatterPayload = rachis::metal::unpackShaderManifest(scatterBytes);
    REQUIRE(scatterPayload);
    const auto scatterJson =
        std::string_view(reinterpret_cast<const char*>(scatterPayload->json.data()), scatterPayload->json.size());
    auto reflected = rachis::metal::parseSlangReflectionJson(scatterJson, "scatterMain", rachis::ShaderStage::compute);
    REQUIRE(reflected);
    REQUIRE(reflected->resources.size() == 2);
    CHECK(reflected->resources[0].type == rachis::ShaderResourceType::storageBuffer);
    CHECK(reflected->resources[0].metalIndex == 1);
    CHECK(reflected->resources[1].type == rachis::ShaderResourceType::storageTexture);
    CHECK(reflected->resources[1].metalIndex == 0);
    const std::array shaderBindings{
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageTexture, 1},
    };
    auto scatterShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = scatterBytes,
        .entryPoint = "scatterMain",
        .shaderInterface = rachis::test::computeShaderInterface({8, 8, 1}, shaderBindings),
    });
    auto reduceShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = reduceBytes,
        .entryPoint = "reduceMain",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, shaderBindings),
    });
    if (!scatterShader)
    {
        UNSCOPED_INFO(scatterShader.error().message);
    }
    if (!reduceShader)
    {
        UNSCOPED_INFO(reduceShader.error().message);
    }
    REQUIRE(scatterShader);
    REQUIRE(reduceShader);

    constexpr std::uint64_t storageBytes = valueCount * sizeof(std::uint32_t);
    auto storage = device.createBuffer({
        .size = storageBytes,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Metal compute storage",
    });
    auto readback = device.createBuffer({
        .size = storageBytes,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal compute readback",
    });
    auto image = device.createTexture({
        .width = imageSize,
        .height = imageSize,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::storage | rachis::TextureUsage::copySource,
        .label = "Metal compute image",
    });
    REQUIRE(storage);
    REQUIRE(readback);
    REQUIRE(image);
    auto imageView = device.createTextureView({
        .texture = *image,
        .usage = rachis::TextureViewUsage::storage,
        .dimension = rachis::TextureViewDimension::texture2D,
    });
    REQUIRE(imageView);
    const rachis::BufferTextureCopyRegion imageCopy{
        .extent = {.width = imageSize, .height = imageSize, .depth = 1},
    };
    auto imageLayout = device.queryTextureTransferLayout(*image, imageCopy);
    REQUIRE(imageLayout);
    auto imageReadback = device.createBuffer({
        .size = imageLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal compute image readback",
    });
    REQUIRE(imageReadback);

    const std::array layoutEntries{
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageTexture, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = layoutEntries});
    REQUIRE(bindingLayout);
    auto set = rachis::test::createBindingSet(
        device,
        {.layout = *bindingLayout,
         .writes = std::array{
             rachis::BindingWrite{1, 0, rachis::BufferBinding{*storage, 0, storageBytes, sizeof(std::uint32_t)}},
             rachis::BindingWrite{0, 0, *imageView},
         }});
    REQUIRE(set);
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = std::array{*bindingLayout}});
    REQUIRE(pipelineLayout);
    auto scatter = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *scatterShader});
    auto reduce = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *reduceShader});
    REQUIRE(scatter);
    REQUIRE(reduce);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *image,
                                    .stagesAfter = rachis::PipelineStage::computeShader,
                                    .accessAfter = rachis::Access::unorderedAccess,
                                    .layoutAfter = rachis::TextureLayout::unorderedAccess,
                                })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *scatter).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *set).has_value());
    REQUIRE(device.dispatch(*command, 1, 1, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *storage,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *reduce).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *set).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *storage,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *image,
                                    .stagesBefore = rachis::PipelineStage::computeShader,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::unorderedAccess,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::unorderedAccess,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(
        device.copyBuffer(*command, {.source = *storage, .destination = *readback, .size = storageBytes}).has_value());
    const rachis::BufferTextureCopyRegion copyRegion{
        .bytesPerRow = imageLayout->bytesPerRow,
        .rowsPerImage = imageLayout->rowsPerImage,
        .extent = {.width = imageSize, .height = imageSize, .depth = 1},
    };
    REQUIRE(device.copyTextureToBuffer(*command, *image, *imageReadback, std::array{copyRegion}).has_value());

    const auto triangleVs = loadBytes(pnmsPath("BootstrapTriangle.vs.metallib"));
    const auto trianglePs = loadBytes(pnmsPath("BootstrapTriangle.ps.metallib"));
    auto vertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = triangleVs,
        .entryPoint = "vertexMain",
        .label = "Metal compute-to-graphics vertex",
    });
    auto pixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = trianglePs,
        .entryPoint = "pixelMain",
        .label = "Metal compute-to-graphics pixel",
    });
    REQUIRE(vertex);
    REQUIRE(pixel);
    auto graphicsLayout = device.createPipelineLayout({});
    REQUIRE(graphicsLayout);
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    auto graphics = device.createGraphicsPipeline({
        .layout = *graphicsLayout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .colorTargets = colorTargets,
        .label = "Metal compute-to-graphics pipeline",
    });
    REQUIRE(graphics);
    auto color = device.createTexture({
        .width = 16,
        .height = 16,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource,
        .label = "Metal compute-to-graphics color",
    });
    REQUIRE(color);
    auto colorView = device.createTextureView({
        .texture = *color,
        .usage = rachis::TextureViewUsage::renderTarget,
        .dimension = rachis::TextureViewDimension::texture2D,
    });
    REQUIRE(colorView);
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *color,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array attachment{rachis::ColorAttachment{
        .view = *colorView,
        .loadOperation = rachis::LoadOperation::clear,
        .clearValue = {.alpha = 1.0F},
    }};
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachment}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *graphics).has_value());
    REQUIRE(device.setViewports(*command, std::array{rachis::Viewport{.width = 16.0F, .height = 16.0F}}).has_value());
    REQUIRE(device.setScissors(*command, std::array{rachis::ScissorRect{.width = 16, .height = 16}}).has_value());
    REQUIRE(device.draw(*command, 3, 1, 0, 0).has_value());
    REQUIRE(device.endRendering(*command).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    CHECK(readU32(device, *readback, 0) == 1);
    CHECK(readU32(device, *readback, 63 * sizeof(std::uint32_t)) == 63 * 3 + 1);
    CHECK(readU32(device, *readback, 64 * sizeof(std::uint32_t)) == expectedSum);
    auto mapped = device.mapBuffer(*imageReadback);
    REQUIRE(mapped);
    const auto* origin = mapped->data;
    CHECK(std::to_integer<std::uint8_t>(origin[0]) <= 2);
    CHECK(std::to_integer<std::uint8_t>(origin[2]) >= 60);
    const auto* corner = mapped->data + static_cast<std::size_t>(7) * imageLayout->bytesPerRow + std::size_t{7} * 4;
    CHECK(std::to_integer<std::uint8_t>(corner[0]) >= 250);
    CHECK(std::to_integer<std::uint8_t>(corner[1]) >= 250);
    REQUIRE(device.unmapBuffer(*imageReadback).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal compute-to-copy producer-consumer ordering holds across queue timelines",
          "[size:large][rhi][metal][compute][gpu][queue]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto bytecode = loadBytes(pnmsPath("MetalPushConstant.cs.metallib"));
    REQUIRE_FALSE(bytecode.empty());
    const std::array shaderBindings{
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
    };
    const std::array shaderPush{rachis::ShaderPushConstantRequirement{0, 16}};
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "computeMain",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, shaderBindings, shaderPush),
    });
    REQUIRE(shader);
    constexpr std::uint64_t outputSize = sizeof(std::uint32_t);
    auto storage = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
    });
    auto readback = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(storage);
    REQUIRE(readback);
    const std::array layoutEntries{
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = layoutEntries});
    REQUIRE(bindingLayout);
    auto set = rachis::test::createBindingSet(
        device,
        {.layout = *bindingLayout,
         .writes = std::array{
             rachis::BindingWrite{1, 0, rachis::BufferBinding{*storage, 0, outputSize, sizeof(std::uint32_t)}}}});
    REQUIRE(set);
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = std::array{*bindingLayout},
        .pushConstantRanges = std::array{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16}},
    });
    REQUIRE(pipelineLayout);
    auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *shader});
    REQUIRE(pipeline);

    auto producer = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(producer);
    REQUIRE(device.bindComputePipeline(*producer, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*producer, 0, *set).has_value());
    const std::array<std::uint32_t, 4> constants{0x5150U, 0, 0, 0};
    REQUIRE(device.setComputePushConstants(*producer, 0, std::as_bytes(std::span{constants})).has_value());
    REQUIRE(device.dispatch(*producer, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*producer,
                               rachis::BufferBarrier{
                                   .buffer = *storage,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    auto computePoint = device.submit(std::move(*producer));
    REQUIRE(computePoint);

    auto consumer = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(consumer);
    REQUIRE(device
                .bufferBarrier(*consumer,
                               rachis::BufferBarrier{
                                   .buffer = *storage,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::copySource,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(
        device.copyBuffer(*consumer, {.source = *storage, .destination = *readback, .size = outputSize}).has_value());
    auto copyPoint = device.submit(std::move(*consumer), std::array{*computePoint});
    REQUIRE(copyPoint);
    REQUIRE(device.wait(*copyPoint).has_value());
    CHECK(readU32(device, *readback) == 0x5150U);
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal dynamic storage offsets and layout compatibility survive dispatch and rejected rebinding",
          "[size:large][rhi][metal][compute][gpu][regression]")
{
    const auto setIndex = GENERATE(0U, 1U, 3U);
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto shaderName = setIndex == 0U ? std::string{"MetalPushConstant.cs.metallib"}
                                           : "MetalPushConstant.set" + std::to_string(setIndex) + ".cs.metallib";
    const auto bytecode = loadBytes(pnmsPath(shaderName));
    REQUIRE_FALSE(bytecode.empty());
    const std::array requirements{
        rachis::ShaderBindingRequirement{setIndex, 1, rachis::ShaderResourceType::storageBuffer, 1},
    };
    const std::array push{rachis::ShaderPushConstantRequirement{0, 16}};
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "computeMain",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, requirements, push),
    });
    REQUIRE(shader);
    const std::array entries{
        rachis::BindingLayoutEntry{1, rachis::BindingType::dynamicStorageBuffer, rachis::ShaderStage::compute},
    };
    auto bindings = device.createBindingLayout({.entries = entries});
    REQUIRE(bindings);
    auto emptyBindings = device.createBindingLayout({});
    REQUIRE(emptyBindings);
    auto extraBindings = device.createBindingLayout({
        .entries =
            std::array{rachis::BindingLayoutEntry{2, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute}},
    });
    REQUIRE(extraBindings);
    std::vector<rachis::BindingLayoutHandle> layouts(setIndex + 1, *emptyBindings);
    layouts[0] = *extraBindings;
    layouts[setIndex] = *bindings;
    auto multiSetLayout = device.createPipelineLayout({.bindingLayouts = std::array{*bindings, *bindings}});
    REQUIRE(multiSetLayout);
    REQUIRE(device.destroyPipelineLayout(*multiSetLayout));
    std::vector<rachis::BindingLayoutEntry> sixteenBuffers;
    sixteenBuffers.reserve(16);
    for (std::uint32_t index = 0; index < 16; ++index)
    {
        sixteenBuffers.push_back(
            rachis::BindingLayoutEntry{index, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute});
    }
    auto sixteenLayout = device.createBindingLayout({.entries = sixteenBuffers});
    REQUIRE(sixteenLayout);
    CHECK_FALSE(device.createPipelineLayout({.bindingLayouts = std::array{*sixteenLayout, *sixteenLayout}}));
    REQUIRE(device.destroyBindingLayout(*sixteenLayout));
    auto layout = device.createPipelineLayout({
        .bindingLayouts = layouts,
        .pushConstantRanges = std::array{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16}},
    });
    REQUIRE(layout);
    auto pipeline = device.createComputePipeline({.layout = *layout, .computeShader = *shader});
    REQUIRE(pipeline);
    auto output = device.createBuffer({
        .size = 1024,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto readback = device.createBuffer({
        .size = 1024,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(output);
    REQUIRE(readback);
    const std::array<std::byte, 1024> zero{};
    REQUIRE(device.writeBuffer(*output, 0, zero));
    auto equivalentBindings = device.createBindingLayout({.entries = entries});
    REQUIRE(equivalentBindings);
    auto incompatibleBindings = device.createBindingLayout({
        .entries = std::array{rachis::BindingLayoutEntry{
            2, rachis::BindingType::dynamicStorageBuffer, rachis::ShaderStage::compute}},
    });
    REQUIRE(incompatibleBindings);
    auto incompatibleSet = rachis::test::createBindingSet(
        device,
        {
            .layout = *incompatibleBindings,
            .writes = std::array{rachis::BindingWrite{2, 0, rachis::BufferBinding{*output, 0, 4, 4}}},
        });
    REQUIRE(incompatibleSet);
    auto set = rachis::test::createBindingSet(
        device,
        {
            .layout = *equivalentBindings,
            .writes = std::array{rachis::BindingWrite{1, 0, rachis::BufferBinding{*output, 0, 4, 4}}},
        });
    REQUIRE(set);
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command);
    REQUIRE(device.bindComputePipeline(*command, *pipeline));
    if (setIndex != 0)
    {
        auto extraSet = rachis::test::createBindingSet(
            device,
            {
                .layout = *extraBindings,
                .writes = std::array{rachis::BindingWrite{2, 0, rachis::BufferBinding{*output, 0, 4, 4}}},
            });
        REQUIRE(extraSet);
        REQUIRE(device.bindComputeBindingSet(*command, 0, *extraSet));
        REQUIRE(device.destroyBindingSet(*extraSet));
    }
    auto incompatibleBind = device.bindComputeBindingSet(*command, setIndex, *incompatibleSet, std::array{0U});
    REQUIRE_FALSE(incompatibleBind);
    CHECK(incompatibleBind.error().code == rachis::ErrorCode::invalidArgument);
    auto rejectedLayout = device.dispatch(*command, 1);
    REQUIRE_FALSE(rejectedLayout);
    CHECK(rejectedLayout.error().code == rachis::ErrorCode::invalidState);
    for (std::uint32_t index = 1; index <= 3; ++index)
    {
        const std::array offsets{index * 256U};
        REQUIRE(device.bindComputeBindingSet(*command, setIndex, *set, offsets));
        CHECK_FALSE(device.bindComputeBindingSet(*command, setIndex, *set, std::array{1024U}));
        const std::array<std::uint32_t, 4> values{0x12340000U + index, 0, 0, 0};
        REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span{values})));
        REQUIRE(device.dispatch(*command, 1));
    }
    REQUIRE(device.bufferBarrier(*command,
                                 {
                                     .buffer = *output,
                                     .stagesBefore = rachis::PipelineStage::computeShader,
                                     .stagesAfter = rachis::PipelineStage::copy,
                                     .accessBefore = rachis::Access::unorderedAccess,
                                     .accessAfter = rachis::Access::copySource,
                                 }));
    REQUIRE(device.copyBuffer(*command, {.source = *output, .destination = *readback, .size = 1024}));
    REQUIRE(device.destroyBindingSet(*set));
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted));
    CHECK(readU32(device, *readback, 0) == 0);
    for (std::uint32_t index = 1; index <= 3; ++index)
    {
        CHECK(readU32(device, *readback, std::uint64_t{index} * 256U) == 0x12340000U + index);
    }
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal compute binding sets require a bound pipeline, match its layout, and reset on pipeline switches",
          "[size:large][rhi][metal][compute][validation][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto bytecode = loadBytes(pnmsPath("MetalPushConstant.cs.metallib"));
    REQUIRE_FALSE(bytecode.empty());
    const std::array requirements{
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
    };
    const std::array push{rachis::ShaderPushConstantRequirement{0, 16}};
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "computeMain",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, requirements, push),
    });
    REQUIRE(shader);
    const std::array entries{
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindings = device.createBindingLayout({.entries = entries});
    REQUIRE(bindings);
    auto otherBindings = device.createBindingLayout({
        .entries =
            std::array{rachis::BindingLayoutEntry{2, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute}},
    });
    REQUIRE(otherBindings);
    auto layout = device.createPipelineLayout({
        .bindingLayouts = std::array{*bindings},
        .pushConstantRanges = std::array{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16}},
    });
    REQUIRE(layout);
    auto first = device.createComputePipeline({.layout = *layout, .computeShader = *shader});
    auto second = device.createComputePipeline({.layout = *layout, .computeShader = *shader});
    REQUIRE(first);
    REQUIRE(second);
    auto output = device.createBuffer({
        .size = 512,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto readback = device.createBuffer({
        .size = 512,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(output);
    REQUIRE(readback);
    const std::array<std::byte, 512> zero{};
    REQUIRE(device.writeBuffer(*output, 0, zero));
    auto firstSet = rachis::test::createBindingSet(
        device,
        {
            .layout = *bindings,
            .writes = std::array{rachis::BindingWrite{1, 0, rachis::BufferBinding{*output, 0, 4, 4}}},
        });
    auto secondSet = rachis::test::createBindingSet(
        device,
        {
            .layout = *bindings,
            .writes = std::array{rachis::BindingWrite{1, 0, rachis::BufferBinding{*output, 256, 4, 4}}},
        });
    auto otherSet = rachis::test::createBindingSet(
        device,
        {
            .layout = *otherBindings,
            .writes = std::array{rachis::BindingWrite{2, 0, rachis::BufferBinding{*output, 0, 4, 4}}},
        });
    REQUIRE(firstSet);
    REQUIRE(secondSet);
    REQUIRE(otherSet);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command);
    auto unbound = device.bindComputeBindingSet(*command, 0, *firstSet);
    REQUIRE_FALSE(unbound);
    CHECK(unbound.error().code == rachis::ErrorCode::invalidState);

    REQUIRE(device.bindComputePipeline(*command, *first));
    auto outside = device.bindComputeBindingSet(*command, 1, *firstSet);
    REQUIRE_FALSE(outside);
    CHECK(outside.error().code == rachis::ErrorCode::invalidArgument);
    auto overflow = device.bindComputeBindingSets(*command, 0, std::array{*firstSet, *firstSet});
    REQUIRE_FALSE(overflow);
    CHECK(overflow.error().code == rachis::ErrorCode::invalidArgument);
    auto mismatched = device.bindComputeBindingSet(*command, 0, *otherSet);
    REQUIRE_FALSE(mismatched);
    CHECK(mismatched.error().code == rachis::ErrorCode::invalidArgument);
    auto missing = device.dispatch(*command, 1);
    REQUIRE_FALSE(missing);
    CHECK(missing.error().code == rachis::ErrorCode::invalidState);

    REQUIRE(device.bindComputeBindingSet(*command, 0, *firstSet));
    auto replaced = device.bindComputeBindingSet(*command, 0, *otherSet);
    REQUIRE_FALSE(replaced);
    CHECK(replaced.error().code == rachis::ErrorCode::invalidArgument);
    const std::array<std::uint32_t, 4> firstValues{0x51A70001U, 0, 0, 0};
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span{firstValues})));
    REQUIRE(device.dispatch(*command, 1));

    REQUIRE(device.bindComputePipeline(*command, *second));
    auto stale = device.dispatch(*command, 1);
    REQUIRE_FALSE(stale);
    CHECK(stale.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.bindComputeBindingSet(*command, 0, *secondSet));
    REQUIRE(device.bindComputePipeline(*command, *second));
    auto rebound = device.dispatch(*command, 1);
    REQUIRE_FALSE(rebound);
    CHECK(rebound.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.bindComputeBindingSet(*command, 0, *secondSet));
    const std::array<std::uint32_t, 4> secondValues{0x51A70002U, 0, 0, 0};
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span{secondValues})));
    REQUIRE(device.dispatch(*command, 1));
    REQUIRE(device.bufferBarrier(*command,
                                 {
                                     .buffer = *output,
                                     .stagesBefore = rachis::PipelineStage::computeShader,
                                     .stagesAfter = rachis::PipelineStage::copy,
                                     .accessBefore = rachis::Access::unorderedAccess,
                                     .accessAfter = rachis::Access::copySource,
                                 }));
    REQUIRE(device.copyBuffer(*command, {.source = *output, .destination = *readback, .size = 512}));
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted));
    CHECK(readU32(device, *readback, 0) == 0x51A70001U);
    CHECK(readU32(device, *readback, 256) == 0x51A70002U);
    CHECK(device.diagnosticCounts().errors == 0);
}
