#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
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

    [[nodiscard]] bool nearByte(std::byte actual, std::uint8_t expected) noexcept
    {
        const auto value = std::to_integer<std::uint8_t>(actual);
        const auto lower = expected == 0 ? 0U : static_cast<std::uint32_t>(expected) - 1U;
        const auto upper = (std::min)(static_cast<std::uint32_t>(expected) + 1U, 255U);
        return value >= lower && value <= upper;
    }

} // namespace

TEST_CASE("dual-source blending validates one target and executes the second pixel output",
          "[size:medium][rhi][graphics][blend][dual-source][warp]")
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
    REQUIRE(rachis::test::requireCapabilities(device).dualSourceBlending);

    const auto vertexBytecode = loadShader(RACHIS_TEST_GRAPHICS_DIRECT_VERTEX_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_DUAL_SOURCE_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());

    auto vertexShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .entryPoint = "directVertex",
        .shaderInterface = {},
    });
    auto pixelShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "main",
        .shaderInterface = {},
    });
    REQUIRE(vertexShader.has_value());
    REQUIRE(pixelShader.has_value());

    auto layout = device.createPipelineLayout({});
    REQUIRE(layout.has_value());

    const rachis::BlendState dualSourceBlend{
        .enabled = true,
        .color =
            {
                .sourceFactor = rachis::BlendFactor::source1,
                .destinationFactor = rachis::BlendFactor::oneMinusSource1,
            },
        .alpha =
            {
                .sourceFactor = rachis::BlendFactor::source1Alpha,
                .destinationFactor = rachis::BlendFactor::oneMinusSource1Alpha,
            },
    };
    const std::array dualSourceTargets{
        rachis::ColorTargetState{
            .format = rachis::Format::rgba8Unorm,
            .blend = dualSourceBlend,
        },
    };
    const std::array tooManyTargets{
        dualSourceTargets[0],
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    auto tooMany = device.createGraphicsPipeline(rachis::GraphicsPipelineDesc{
        .layout = *layout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = tooManyTargets,
    });
    REQUIRE_FALSE(tooMany.has_value());
    CHECK(tooMany.error().code == rachis::ErrorCode::unsupported);

    auto unsupportedOperationTargets = dualSourceTargets;
    unsupportedOperationTargets[0].blend.color.operation = rachis::BlendOperation::minimum;
    auto unsupportedOperation = device.createGraphicsPipeline(rachis::GraphicsPipelineDesc{
        .layout = *layout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = unsupportedOperationTargets,
    });
    REQUIRE_FALSE(unsupportedOperation.has_value());
    CHECK(unsupportedOperation.error().code == rachis::ErrorCode::unsupported);

    auto pipeline = device.createGraphicsPipeline(rachis::GraphicsPipelineDesc{
        .layout = *layout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = dualSourceTargets,
    });
    if (!pipeline)
    {
        INFO(pipeline.error().message);
    }
    REQUIRE(pipeline.has_value());

    auto target = device.createTexture(rachis::TextureDesc{
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
    });
    REQUIRE(target.has_value());
    auto targetView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = readbackSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(targetView.has_value());
    REQUIRE(readback.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
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
            .clearValue = {0.0F, 1.0F, 0.25F, 1.0F},
        },
    };
    REQUIRE(device.beginRendering(*command, rachis::RenderingDesc{.colorAttachments = attachments}).has_value());
    REQUIRE(device
                .setViewport(*command,
                             rachis::Viewport{
                                 .width = static_cast<float>(targetWidth),
                                 .height = static_cast<float>(targetHeight),
                             })
                .has_value());
    REQUIRE(device.setScissor(*command, rachis::ScissorRect{.width = targetWidth, .height = targetHeight}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *pipeline).has_value());
    REQUIRE(device.draw(*command, 3).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
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
                                     rachis::BufferTextureCopy{
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
    const auto* corner = mapped->data;
    const auto* center = mapped->data + static_cast<std::uint64_t>(targetHeight / 2) * rowPitch +
                         static_cast<std::uint64_t>(targetWidth / 2) * 4;
    CHECK(nearByte(corner[0], 0));
    CHECK(nearByte(corner[1], 255));
    CHECK(nearByte(corner[2], 64));
    CHECK(nearByte(corner[3], 255));
    CAPTURE(std::to_integer<std::uint8_t>(center[0]),
            std::to_integer<std::uint8_t>(center[1]),
            std::to_integer<std::uint8_t>(center[2]),
            std::to_integer<std::uint8_t>(center[3]));
    CHECK(nearByte(center[0], 64));
    CHECK(nearByte(center[1], 128));
    CHECK(nearByte(center[2], 112));
    CHECK(nearByte(center[3], 255));
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*layout).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
