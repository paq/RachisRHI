#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <span>
#include <vector>

namespace
{

    constexpr std::uint32_t targetWidth = 64;
    constexpr std::uint32_t targetHeight = 64;
    constexpr std::uint32_t rowPitch = 256;
    constexpr std::uint64_t readbackSize = static_cast<std::uint64_t>(rowPitch) * targetHeight;

    void printDiagnostic(const rachis::DiagnosticMessage& message, void*) noexcept
    {
        static_cast<void>(std::fprintf(stderr, "D3D12 [%u]: ", static_cast<unsigned int>(message.nativeId)));
        if (!message.description.empty())
        {
            static_cast<void>(std::fwrite(message.description.data(), 1, message.description.size(), stderr));
        }
        static_cast<void>(std::fputc('\n', stderr));
    }

    [[nodiscard]] std::vector<std::byte> loadShader(const char* path)
    {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        const auto end = stream.tellg();
        if (end <= 0)
        {
            return {};
        }
        std::vector<std::byte> result(static_cast<std::size_t>(end));
        stream.seekg(0, std::ios::beg);
        stream.read(reinterpret_cast<char*>(result.data()), end);
        return stream ? result : std::vector<std::byte>{};
    }

    [[nodiscard]] const std::byte* pixel(const rachis::MappedMemory& mapped, std::uint32_t x, std::uint32_t y)
    {
        return mapped.data + static_cast<std::uint64_t>(y) * rowPitch + static_cast<std::uint64_t>(x) * 4U;
    }

    void checkColor(const std::byte* value, std::array<std::byte, 4> expected)
    {
        CHECK(value[0] == expected[0]);
        CHECK(value[1] == expected[1]);
        CHECK(value[2] == expected[2]);
        CHECK(value[3] == expected[3]);
    }

} // namespace

TEST_CASE("viewport and scissor arrays route geometry atomically", "[size:medium][rhi][graphics][multi-viewport][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).multiViewport);

    const auto vertexBytecode = loadShader(RACHIS_TEST_GEOMETRY_VERTEX_SHADER_PATH);
    const auto geometryBytecode = loadShader(RACHIS_TEST_MULTI_VIEWPORT_GEOMETRY_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_GEOMETRY_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(geometryBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());

    auto vertexShader = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .entryPoint = "pointVertex",
        .label = "Multi-viewport vertex shader",
        .shaderInterface = {},
    });
    auto geometryShader = device.createShaderModule({
        .stage = rachis::ShaderStage::geometry,
        .bytecode = geometryBytecode,
        .entryPoint = "viewportArrayMain",
        .label = "Multi-viewport geometry shader",
        .shaderInterface = {},
    });
    auto pixelShader = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "geometryPixelMain",
        .label = "Multi-viewport pixel shader",
        .shaderInterface = {},
    });
    REQUIRE(vertexShader.has_value());
    REQUIRE(geometryShader.has_value());
    REQUIRE(pixelShader.has_value());

    auto layout = device.createPipelineLayout({.label = "Multi-viewport pipeline layout"});
    REQUIRE(layout.has_value());
    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    auto pipeline = device.createGraphicsPipeline({
        .layout = *layout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .geometryShader = *geometryShader,
        .topology = rachis::PrimitiveTopology::pointList,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .label = "Multi-viewport routing pipeline",
    });
    REQUIRE(pipeline.has_value());

    auto target = device.createTexture({
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
        .label = "Multi-viewport target",
    });
    auto readback = device.createBuffer({
        .size = readbackSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Multi-viewport readback",
    });
    REQUIRE(target.has_value());
    REQUIRE(readback.has_value());
    auto targetView = device.createTextureView({
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
        .label = "Multi-viewport target view",
    });
    REQUIRE(targetView.has_value());

    const std::array viewports{
        rachis::Viewport{.width = 32.0F, .height = 64.0F, .maxDepth = 1.0F},
        rachis::Viewport{.x = 32.0F, .width = 32.0F, .height = 64.0F, .maxDepth = 1.0F},
    };
    const std::array scissors{
        rachis::ScissorRect{.width = 32, .height = 64},
        rachis::ScissorRect{.x = 32, .width = 32, .height = 64},
    };

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    auto outsideRendering = device.setViewports(*command, viewports);
    REQUIRE_FALSE(outsideRendering.has_value());
    CHECK(outsideRendering.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *target,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array attachments{
        rachis::ColorAttachment{
            .view = *targetView,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.0F, 0.0F, 0.0F, 1.0F},
        },
    };
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *pipeline).has_value());

    const auto emptyViewports = device.setViewports(*command, {});
    REQUIRE_FALSE(emptyViewports.has_value());
    CHECK(emptyViewports.error().code == rachis::ErrorCode::invalidArgument);
    const auto emptyScissors = device.setScissors(*command, {});
    REQUIRE_FALSE(emptyScissors.has_value());
    CHECK(emptyScissors.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(device.setViewports(*command, viewports).has_value());
    REQUIRE(device.setScissors(*command, scissors).has_value());

    auto invalidViewports = viewports;
    invalidViewports[1].width = 0.0F;
    auto invalidViewportBatch = device.setViewports(*command, invalidViewports);
    REQUIRE_FALSE(invalidViewportBatch.has_value());
    CHECK(invalidViewportBatch.error().code == rachis::ErrorCode::invalidArgument);

    auto invalidScissors = scissors;
    invalidScissors[1].height = 0;
    auto invalidScissorBatch = device.setScissors(*command, invalidScissors);
    REQUIRE_FALSE(invalidScissorBatch.has_value());
    CHECK(invalidScissorBatch.error().code == rachis::ErrorCode::invalidArgument);

    const std::array<rachis::Viewport, rachis::maxViewportCount + 1> excessiveViewports{};
    auto excessive = device.setViewports(*command, excessiveViewports);
    REQUIRE_FALSE(excessive.has_value());
    CHECK(excessive.error().code == rachis::ErrorCode::unsupported);
    REQUIRE(device.draw(*command, 1).has_value());

    REQUIRE(device.setViewport(*command, viewports[0]).has_value());
    auto mismatchedViewportCount = device.draw(*command, 1);
    REQUIRE_FALSE(mismatchedViewportCount.has_value());
    CHECK(mismatchedViewportCount.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.setViewports(*command, viewports).has_value());
    REQUIRE(device.setScissor(*command, scissors[0]).has_value());
    auto mismatchedScissorCount = device.draw(*command, 1);
    REQUIRE_FALSE(mismatchedScissorCount.has_value());
    CHECK(mismatchedScissorCount.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.setScissors(*command, scissors).has_value());
    REQUIRE(device.draw(*command, 1).has_value());

    REQUIRE(device.endRendering(*command).has_value());
    REQUIRE(device
                .textureBarrier(*command,
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
    REQUIRE(device
                .copyTextureToBuffer(*command,
                                     {
                                         .buffer = *readback,
                                         .bytesPerRow = rowPitch,
                                         .rowsPerImage = targetHeight,
                                         .texture = *target,
                                         .extent = {targetWidth, targetHeight, 1},
                                     })
                .has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    checkColor(pixel(*mapped, 16, 32), {std::byte{255}, std::byte{0}, std::byte{0}, std::byte{255}});
    checkColor(pixel(*mapped, 48, 32), {std::byte{0}, std::byte{255}, std::byte{0}, std::byte{255}});
    checkColor(pixel(*mapped, 0, 0), {std::byte{0}, std::byte{0}, std::byte{0}, std::byte{255}});
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*layout).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    REQUIRE(device.destroyShaderModule(*geometryShader).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
