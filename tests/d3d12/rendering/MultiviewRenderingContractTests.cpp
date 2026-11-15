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
    constexpr std::uint32_t targetLayers = 3;
    constexpr std::uint32_t rowPitch = 256;
    constexpr std::uint64_t layerBytes = static_cast<std::uint64_t>(rowPitch) * targetHeight;
    constexpr std::uint64_t readbackBytes = targetLayers * layerBytes;
    constexpr std::uint32_t twoViewMask = 0b110;

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
                                     std::uint32_t layer,
                                     std::array<std::byte, 4> expected) noexcept
    {
        const auto layerOffset = static_cast<std::uint64_t>(layer) * layerBytes;
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

TEST_CASE("D3D12 view instancing renders two masked views on WARP", "[size:medium][rhi][graphics][multiview][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .diagnosticCallback = &printDiagnostic,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    const auto capabilities = *device.capabilities();
    REQUIRE(capabilities.multiviewRendering);
    REQUIRE(capabilities.limits.maxFramebufferLayers >= targetLayers);

    const auto vertexBytecode = loadShader(RACHIS_TEST_MULTIVIEW_RENDERING_VERTEX_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_MULTIVIEW_RENDERING_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    auto vertexShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .label = "Multiview vertex shader",
        .shaderInterface = {},
    });
    auto pixelShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .label = "Multiview pixel shader",
        .shaderInterface = {},
    });
    REQUIRE(vertexShader.has_value());
    REQUIRE(pixelShader.has_value());
    auto layout = device.createPipelineLayout(rachis::PipelineLayoutDesc{.label = "Multiview layout"});
    auto cache = device.createPipelineCache(rachis::PipelineCacheDesc{.label = "Multiview pipeline cache"});
    REQUIRE(layout.has_value());
    REQUIRE(cache.has_value());

    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    const rachis::GraphicsPipelineDesc pipelineDesc{
        .layout = *layout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .viewMask = twoViewMask,
        .label = "Multiview pipeline",
        .cache = *cache,
        .cacheKey = {.high = 0x4D554C5449564945ULL, .low = 0x5752454E44455231ULL},
    };
    auto firstPipeline = device.createGraphicsPipeline(pipelineDesc);
    REQUIRE(firstPipeline.has_value());
    REQUIRE(device.destroyPipeline(*firstPipeline).has_value());
    auto pipeline = device.createGraphicsPipeline(pipelineDesc);
    REQUIRE(pipeline.has_value());

    auto nonMultiviewPipeline = device.createGraphicsPipeline(rachis::GraphicsPipelineDesc{
        .layout = *layout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .label = "Non-multiview pipeline",
    });
    REQUIRE(nonMultiviewPipeline.has_value());

    auto invalidMaskPipeline = device.createGraphicsPipeline(rachis::GraphicsPipelineDesc{
        .layout = *layout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .viewMask = 0b101,
    });
    REQUIRE_FALSE(invalidMaskPipeline.has_value());
    CHECK(invalidMaskPipeline.error().code == rachis::ErrorCode::invalidArgument);

    auto excessiveMaskPipeline = device.createGraphicsPipeline(rachis::GraphicsPipelineDesc{
        .layout = *layout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .viewMask = 0b1'1111,
    });
    REQUIRE_FALSE(excessiveMaskPipeline.has_value());
    CHECK(excessiveMaskPipeline.error().code == rachis::ErrorCode::unsupported);

    auto target = device.createTexture(rachis::TextureDesc{
        .width = targetWidth,
        .height = targetHeight,
        .depthOrArrayLayers = targetLayers,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
        .label = "Multiview target",
    });
    REQUIRE(target.has_value());
    auto targetView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
        .dimension = rachis::TextureViewDimension::texture2DArray,
        .arrayLayerCount = targetLayers,
        .label = "Multiview target view",
    });
    auto insufficientView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
        .dimension = rachis::TextureViewDimension::texture2DArray,
        .arrayLayerCount = 2,
        .label = "Insufficient multiview target view",
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = readbackBytes,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Multiview readback",
    });
    REQUIRE(targetView.has_value());
    REQUIRE(insufficientView.has_value());
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
            .clearValue = {0.0F, 0.0F, 0.0F, 1.0F},
        },
    };
    const std::array insufficientAttachments{
        rachis::ColorAttachment{.view = *insufficientView},
    };

    auto noncontiguousScope =
        device.beginRendering(*command, rachis::RenderingDesc{.colorAttachments = attachments, .viewMask = 0b101});
    REQUIRE_FALSE(noncontiguousScope.has_value());
    CHECK(noncontiguousScope.error().code == rachis::ErrorCode::invalidArgument);

    auto insufficientScope = device.beginRendering(
        *command, rachis::RenderingDesc{.colorAttachments = insufficientAttachments, .viewMask = twoViewMask});
    REQUIRE_FALSE(insufficientScope.has_value());
    CHECK(insufficientScope.error().code == rachis::ErrorCode::invalidArgument);

    REQUIRE(
        device.beginRendering(*command, rachis::RenderingDesc{.colorAttachments = attachments, .viewMask = twoViewMask})
            .has_value());
    auto mismatchedPipeline = device.bindGraphicsPipeline(*command, *nonMultiviewPipeline);
    REQUIRE_FALSE(mismatchedPipeline.has_value());
    CHECK(mismatchedPipeline.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(device.endRendering(*command).has_value());

    REQUIRE(device.beginRendering(*command, rachis::RenderingDesc{.colorAttachments = attachments}).has_value());
    mismatchedPipeline = device.bindGraphicsPipeline(*command, *pipeline);
    REQUIRE_FALSE(mismatchedPipeline.has_value());
    CHECK(mismatchedPipeline.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(device.endRendering(*command).has_value());

    REQUIRE(
        device.beginRendering(*command, rachis::RenderingDesc{.colorAttachments = attachments, .viewMask = twoViewMask})
            .has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *pipeline).has_value());
    REQUIRE(device
                .setViewport(*command,
                             rachis::Viewport{
                                 .width = static_cast<float>(targetWidth),
                                 .height = static_cast<float>(targetHeight),
                             })
                .has_value());
    REQUIRE(device.setScissor(*command, rachis::ScissorRect{.width = targetWidth, .height = targetHeight}).has_value());
    REQUIRE(device.draw(*command, 3, 1).has_value());
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
    for (std::uint32_t layer = 0; layer < targetLayers; ++layer)
    {
        REQUIRE(device
                    .copyTextureToBuffer(*command,
                                         rachis::BufferTextureCopy{
                                             .buffer = *readback,
                                             .bufferOffset = layer * layerBytes,
                                             .bytesPerRow = rowPitch,
                                             .rowsPerImage = targetHeight,
                                             .texture = *target,
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
    CHECK(layerHasColor(*mapped, 0, {std::byte{0}, std::byte{0}, std::byte{0}, std::byte{0xFF}}));
    CHECK(layerHasColor(*mapped, 1, {std::byte{0xFF}, std::byte{0}, std::byte{0}, std::byte{0xFF}}));
    CHECK(layerHasColor(*mapped, 2, {std::byte{0}, std::byte{0xFF}, std::byte{0}, std::byte{0xFF}}));
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyTextureView(*insufficientView).has_value());
    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    REQUIRE(device.destroyPipeline(*nonMultiviewPipeline).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineCache(*cache).has_value());
    REQUIRE(device.destroyPipelineLayout(*layout).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
