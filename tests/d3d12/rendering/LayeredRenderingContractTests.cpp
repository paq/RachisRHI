#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
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
    constexpr std::uint64_t layerBytes = static_cast<std::uint64_t>(rowPitch) * targetHeight;
    constexpr std::uint64_t readbackBytes = 2 * layerBytes;

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

    [[nodiscard]] bool layerHasColor(const rachis::MappedMemory& mapped,
                                     std::uint64_t layerOffset,
                                     std::array<std::byte, 4> expected) noexcept
    {
        for (std::uint32_t y = 0; y < targetHeight; ++y)
        {
            for (std::uint32_t x = 0; x < targetWidth; ++x)
            {
                const auto offset =
                    layerOffset + static_cast<std::uint64_t>(y) * rowPitch + static_cast<std::uint64_t>(x) * 4;
                const std::span actual{mapped.data + static_cast<std::size_t>(offset), expected.size()};
                if (!std::ranges::equal(actual, expected))
                {
                    return false;
                }
            }
        }
        return true;
    }

} // namespace

TEST_CASE("vertex-stage layered rendering selects array attachments on WARP",
          "[size:medium][rhi][graphics][layered][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .diagnosticCallback = &printDiagnostic,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    const auto capabilities = *device.capabilities();
    REQUIRE(rachis::hasAny(capabilities.layeredRenderingModes, rachis::LayeredRenderingMode::vertexStage));

    const auto vertexBytecode = loadShader(RACHIS_TEST_LAYERED_RENDERING_VERTEX_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_LAYERED_RENDERING_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    auto vertexShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .label = "Layered-rendering vertex shader",
        .shaderInterface = {},
    });
    auto pixelShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .label = "Layered-rendering pixel shader",
        .shaderInterface = {},
    });
    REQUIRE(vertexShader.has_value());
    REQUIRE(pixelShader.has_value());
    auto layout = device.createPipelineLayout(rachis::PipelineLayoutDesc{.label = "Layered-rendering layout"});
    REQUIRE(layout.has_value());
    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    auto pipeline = device.createGraphicsPipeline(rachis::GraphicsPipelineDesc{
        .layout = *layout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .label = "Layered-rendering pipeline",
    });
    REQUIRE(pipeline.has_value());

    auto layeredTarget = device.createTexture(rachis::TextureDesc{
        .width = targetWidth,
        .height = targetHeight,
        .depthOrArrayLayers = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
        .label = "Layered-rendering target",
    });
    auto singleTarget = device.createTexture(rachis::TextureDesc{
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .label = "Single-layer mismatch target",
    });
    auto singleDepth = device.createTexture(rachis::TextureDesc{
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::d32Float,
        .usage = rachis::TextureUsage::depthStencil,
        .label = "Single-layer mismatch depth",
    });
    REQUIRE(layeredTarget.has_value());
    REQUIRE(singleTarget.has_value());
    REQUIRE(singleDepth.has_value());
    auto layeredView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *layeredTarget,
        .usage = rachis::TextureViewUsage::renderTarget,
        .dimension = rachis::TextureViewDimension::texture2DArray,
        .arrayLayerCount = 2,
        .label = "Layered-rendering target view",
    });
    auto singleView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *singleTarget,
        .usage = rachis::TextureViewUsage::renderTarget,
        .label = "Single-layer mismatch view",
    });
    auto singleDepthView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *singleDepth,
        .usage = rachis::TextureViewUsage::depthStencil,
        .aspect = rachis::TextureAspect::depth,
        .label = "Single-layer mismatch depth view",
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = readbackBytes,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Layered-rendering readback",
    });
    REQUIRE(layeredView.has_value());
    REQUIRE(singleView.has_value());
    REQUIRE(singleDepthView.has_value());
    REQUIRE(readback.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    const std::array transitions{
        rachis::TextureBarrier{
            .texture = *layeredTarget,
            .stagesAfter = rachis::PipelineStage::renderTarget,
            .accessAfter = rachis::Access::renderTarget,
            .layoutAfter = rachis::TextureLayout::renderTarget,
        },
        rachis::TextureBarrier{
            .texture = *singleTarget,
            .stagesAfter = rachis::PipelineStage::renderTarget,
            .accessAfter = rachis::Access::renderTarget,
            .layoutAfter = rachis::TextureLayout::renderTarget,
        },
        rachis::TextureBarrier{
            .texture = *singleDepth,
            .stagesAfter = rachis::PipelineStage::depthStencil,
            .accessAfter = rachis::Access::depthStencilWrite,
            .layoutAfter = rachis::TextureLayout::depthStencilWrite,
            .subresources = {.aspect = rachis::TextureAspect::depth},
        },
    };
    REQUIRE(
        device.resourceBarriers(*command, rachis::ResourceBarrierBatch{.textureBarriers = transitions}).has_value());

    const std::array mismatchedAttachments{
        rachis::ColorAttachment{.view = *layeredView},
        rachis::ColorAttachment{.view = *singleView},
    };
    auto mismatched = device.beginRendering(*command, rachis::RenderingDesc{.colorAttachments = mismatchedAttachments});
    REQUIRE_FALSE(mismatched.has_value());
    CHECK(mismatched.error().code == rachis::ErrorCode::invalidArgument);

    const std::array layeredColorAttachment{
        rachis::ColorAttachment{.view = *layeredView},
    };
    auto mismatchedDepth =
        device.beginRendering(*command,
                              rachis::RenderingDesc{
                                  .colorAttachments = layeredColorAttachment,
                                  .depthStencilAttachment = rachis::DepthStencilAttachment{.view = *singleDepthView},
                              });
    REQUIRE_FALSE(mismatchedDepth.has_value());
    CHECK(mismatchedDepth.error().code == rachis::ErrorCode::invalidArgument);

    const std::array attachments{
        rachis::ColorAttachment{
            .view = *layeredView,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.0F, 0.0F, 0.0F, 0.0F},
        },
    };
    REQUIRE(device.beginRendering(*command, rachis::RenderingDesc{.colorAttachments = attachments}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *pipeline).has_value());
    REQUIRE(device
                .setViewport(*command,
                             rachis::Viewport{
                                 .width = static_cast<float>(targetWidth),
                                 .height = static_cast<float>(targetHeight),
                             })
                .has_value());
    REQUIRE(device.setScissor(*command, rachis::ScissorRect{.width = targetWidth, .height = targetHeight}).has_value());
    REQUIRE(device.draw(*command, 3, 2).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *layeredTarget,
                                    .stagesBefore = rachis::PipelineStage::renderTarget,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::renderTarget,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    for (std::uint32_t layer = 0; layer < 2; ++layer)
    {
        REQUIRE(device
                    .copyTextureToBuffer(*command,
                                         rachis::BufferTextureCopy{
                                             .buffer = *readback,
                                             .bufferOffset = layer * layerBytes,
                                             .bytesPerRow = rowPitch,
                                             .rowsPerImage = targetHeight,
                                             .texture = *layeredTarget,
                                             .textureSubresource = {.arrayLayer = layer},
                                             .extent = {targetWidth, targetHeight, 1},
                                         })
                    .has_value());
    }

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    REQUIRE(mapped->size == readbackBytes);
    CHECK(layerHasColor(*mapped, 0, {std::byte{0xFF}, std::byte{0}, std::byte{0}, std::byte{0xFF}}));
    CHECK(layerHasColor(*mapped, layerBytes, {std::byte{0}, std::byte{0xFF}, std::byte{0}, std::byte{0xFF}}));
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyTextureView(*singleDepthView).has_value());
    REQUIRE(device.destroyTextureView(*singleView).has_value());
    REQUIRE(device.destroyTextureView(*layeredView).has_value());
    REQUIRE(device.destroyTexture(*singleTarget).has_value());
    REQUIRE(device.destroyTexture(*singleDepth).has_value());
    REQUIRE(device.destroyTexture(*layeredTarget).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*layout).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
