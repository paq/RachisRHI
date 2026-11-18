#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string_view>
#include <vector>

namespace
{

    constexpr std::uint32_t targetWidth = 64;
    constexpr std::uint32_t targetHeight = 64;

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

    [[nodiscard]] rachis::ShaderInterfaceDesc cubeComputeInterface(
        std::array<rachis::ShaderBindingRequirement, 2>& bindings,
        std::array<rachis::ShaderPushConstantRequirement, 1>& push) noexcept
    {
        bindings = {
            rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::readOnlyStorageBuffer, 1},
            rachis::ShaderBindingRequirement{0, 2, rachis::ShaderResourceType::storageBuffer, 1},
        };
        push = {rachis::ShaderPushConstantRequirement{0, 16}};
        return {.bindings = bindings, .pushConstants = push, .workgroupSize = rachis::ShaderWorkgroupSize{64, 1, 1}};
    }

    [[nodiscard]] bool nearByte(std::byte actual, std::uint8_t expected, std::uint8_t tolerance = 2) noexcept
    {
        const auto value = std::to_integer<int>(actual);
        const auto target = static_cast<int>(expected);
        return std::abs(value - target) <= static_cast<int>(tolerance);
    }

} // namespace

TEST_CASE("Metal vertex-stage layered rendering writes independent array layers on Apple GPU",
          "[size:large][rhi][metal][layered]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto capabilities = device.capabilities();
    REQUIRE(capabilities);
    REQUIRE(rachis::hasAny(capabilities->layeredRenderingModes, rachis::LayeredRenderingMode::vertexStage));
    REQUIRE(capabilities->limits.maxFramebufferLayers >= 2);

    const auto vs = loadBytes(pnmsPath("MetalLayeredRendering.vs.metallib"));
    const auto ps = loadBytes(pnmsPath("MetalLayeredRendering.ps.metallib"));
    REQUIRE_FALSE(vs.empty());
    REQUIRE_FALSE(ps.empty());
    auto vertex =
        device.createShaderModule({.stage = rachis::ShaderStage::vertex, .bytecode = vs, .entryPoint = "vertexMain"});
    auto pixel =
        device.createShaderModule({.stage = rachis::ShaderStage::pixel, .bytecode = ps, .entryPoint = "pixelMain"});
    INFO((vertex ? "" : vertex.error().message));
    REQUIRE(vertex);
    REQUIRE(pixel);
    auto layout = device.createPipelineLayout({});
    REQUIRE(layout);
    const std::array targets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    auto pipeline = device.createGraphicsPipeline({
        .layout = *layout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .cullMode = rachis::CullMode::none,
        .colorTargets = targets,
    });
    REQUIRE(pipeline);

    auto layered = device.createTexture(rachis::TextureDesc{
        .width = targetWidth,
        .height = targetHeight,
        .depthOrArrayLayers = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
        .label = "Metal layered color",
    });
    REQUIRE(layered);
    auto view = device.createTextureView(rachis::TextureViewDesc{
        .texture = *layered,
        .usage = rachis::TextureViewUsage::renderTarget,
        .dimension = rachis::TextureViewDimension::texture2DArray,
        .arrayLayerCount = 2,
    });
    REQUIRE(view);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                {.texture = *layered,
                                 .stagesAfter = rachis::PipelineStage::renderTarget,
                                 .accessAfter = rachis::Access::renderTarget,
                                 .layoutAfter = rachis::TextureLayout::renderTarget})
                .has_value());
    const std::array attachments{rachis::ColorAttachment{
        .view = *view,
        .loadOperation = rachis::LoadOperation::clear,
        .clearValue = {0, 0, 0, 0},
    }};
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *pipeline).has_value());
    REQUIRE(device
                .setViewport(*command,
                             {.width = static_cast<float>(targetWidth), .height = static_cast<float>(targetHeight)})
                .has_value());
    REQUIRE(device.setScissor(*command, {.width = targetWidth, .height = targetHeight}).has_value());
    REQUIRE(device.draw(*command, 3, 2).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {.texture = *layered,
                                 .stagesBefore = rachis::PipelineStage::renderTarget,
                                 .stagesAfter = rachis::PipelineStage::copy,
                                 .accessBefore = rachis::Access::renderTarget,
                                 .accessAfter = rachis::Access::copySource,
                                 .layoutBefore = rachis::TextureLayout::renderTarget,
                                 .layoutAfter = rachis::TextureLayout::copySource})
                .has_value());

    std::array<std::vector<std::byte>, 2> layers{};
    for (std::uint32_t layer = 0; layer < 2; ++layer)
    {
        const rachis::BufferTextureCopyRegion region{
            .textureSubresource = {.arrayLayer = layer},
            .extent = {targetWidth, targetHeight, 1},
        };
        auto transfer = device.queryTextureTransferLayout(*layered, region);
        REQUIRE(transfer);
        auto readback = device.createBuffer(rachis::BufferDesc{
            .size = transfer->requiredBufferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
        });
        REQUIRE(readback);
        REQUIRE(device
                    .copyTextureToBuffer(*command,
                                         {.buffer = *readback,
                                          .bytesPerRow = transfer->bytesPerRow,
                                          .rowsPerImage = transfer->rowsPerImage,
                                          .texture = *layered,
                                          .textureSubresource = {.arrayLayer = layer},
                                          .extent = {targetWidth, targetHeight, 1}})
                    .has_value());
        layers[layer].resize(static_cast<std::size_t>(transfer->requiredBufferSize));
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted);
        REQUIRE(device.wait(*submitted).has_value());
        auto mapped = device.mapBuffer(*readback);
        REQUIRE(mapped);
        std::memcpy(layers[layer].data(), mapped->data, layers[layer].size());
        REQUIRE(device.unmapBuffer(*readback).has_value());
        REQUIRE(device.destroyBuffer(*readback).has_value());
        command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command);
    }

    const auto sample = [](const std::vector<std::byte>& pixels, std::uint32_t pitch) {
        const auto offset =
            static_cast<std::size_t>(targetHeight / 2) * pitch + static_cast<std::size_t>(targetWidth / 2) * 4;
        return pixels.data() + offset;
    };
    const rachis::BufferTextureCopyRegion layoutRegion{.extent = {targetWidth, targetHeight, 1}};
    auto layout0 = device.queryTextureTransferLayout(*layered, layoutRegion);
    REQUIRE(layout0);
    CHECK(nearByte(sample(layers[0], layout0->bytesPerRow)[0], 255));
    CHECK(nearByte(sample(layers[0], layout0->bytesPerRow)[1], 0));
    CHECK(nearByte(sample(layers[1], layout0->bytesPerRow)[0], 0));
    CHECK(nearByte(sample(layers[1], layout0->bytesPerRow)[1], 255));
}

TEST_CASE("Metal reports exact view-index and layer limits and rejects unmapped raster families",
          "[size:large][rhi][metal][raster]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    const auto triangleVs = loadBytes(pnmsPath("BootstrapTriangle.vs.metallib"));
    const auto trianglePs = loadBytes(pnmsPath("BootstrapTriangle.ps.metallib"));
    const auto cube = loadBytes(pnmsPath("BootstrapCube.cs.metallib"));
    REQUIRE_FALSE(triangleVs.empty());
    REQUIRE_FALSE(trianglePs.empty());
    REQUIRE_FALSE(cube.empty());
    auto vertex = device.createShaderModule(
        {.stage = rachis::ShaderStage::vertex, .bytecode = triangleVs, .entryPoint = "vertexMain"});
    auto pixel = device.createShaderModule(
        {.stage = rachis::ShaderStage::pixel, .bytecode = trianglePs, .entryPoint = "pixelMain"});
    std::array<rachis::ShaderBindingRequirement, 2> bindings{};
    std::array<rachis::ShaderPushConstantRequirement, 1> push{};
    auto compute = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = cube,
        .entryPoint = "computeMain",
        .shaderInterface = cubeComputeInterface(bindings, push),
    });
    REQUIRE(vertex);
    REQUIRE(pixel);
    REQUIRE(compute);
    auto layout = device.createPipelineLayout({});
    REQUIRE(layout);
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};

    auto geometry = device.createGraphicsPipeline({
        .layout = *layout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .geometryShader = *vertex,
        .colorTargets = colorTargets,
    });
    REQUIRE_FALSE(geometry);
    CHECK(geometry.error().code == rachis::ErrorCode::unsupported);

    auto tessellation = device.createGraphicsPipeline({
        .layout = *layout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .hullShader = *vertex,
        .domainShader = *pixel,
        .topology = rachis::PrimitiveTopology::patchList,
        .patchControlPoints = 3,
        .colorTargets = colorTargets,
    });
    REQUIRE_FALSE(tessellation);
    CHECK(tessellation.error().code == rachis::ErrorCode::unsupported);

    auto mesh = device.createGraphicsPipeline({
        .layout = *layout,
        .meshShader = *compute,
        .colorTargets = colorTargets,
    });
    REQUIRE_FALSE(mesh);
    CHECK(mesh.error().code == (rachis::test::requireCapabilities(device).meshShaders
                                    ? rachis::ErrorCode::invalidArgument
                                    : rachis::ErrorCode::unsupported));

    auto fans = device.createGraphicsPipeline({
        .layout = *layout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .topology = rachis::PrimitiveTopology::triangleFan,
        .colorTargets = colorTargets,
    });
    REQUIRE_FALSE(fans);
    CHECK(fans.error().code == rachis::ErrorCode::unsupported);

    auto wireframe = device.createGraphicsPipeline({
        .layout = *layout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .fillMode = rachis::FillMode::wireframe,
        .colorTargets = colorTargets,
    });
    REQUIRE_FALSE(wireframe);
    CHECK(wireframe.error().code == rachis::ErrorCode::unsupported);

    auto unclipped = device.createGraphicsPipeline({
        .layout = *layout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .depthClipEnabled = false,
        .colorTargets = colorTargets,
    });
    REQUIRE_FALSE(unclipped);
    CHECK(unclipped.error().code == rachis::ErrorCode::unsupported);

    auto conservative = device.createGraphicsPipeline({
        .layout = *layout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .colorTargets = colorTargets,
        .conservativeRasterization = rachis::ConservativeRasterizationMode::overestimate,
    });
    REQUIRE_FALSE(conservative);
    CHECK(conservative.error().code == rachis::ErrorCode::unsupported);

    auto viewMask = device.createGraphicsPipeline({
        .layout = *layout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .colorTargets = colorTargets,
        .viewMask = 0b11,
    });
    REQUIRE_FALSE(viewMask);
    CHECK(viewMask.error().code == rachis::ErrorCode::unsupported);

    auto target = device.createTexture(rachis::TextureDesc{
        .width = 8,
        .height = 8,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
    });
    REQUIRE(target);
    auto targetView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    REQUIRE(targetView);
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    const std::array attachments{rachis::ColorAttachment{.view = *targetView}};
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
    const std::array twoViewports{
        rachis::Viewport{.width = 4, .height = 4},
        rachis::Viewport{.x = 4, .width = 4, .height = 4},
    };
    auto extraViewport = device.setViewports(*command, twoViewports);
    REQUIRE_FALSE(extraViewport);
    CHECK(extraViewport.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(device.endRendering(*command).has_value());
}
