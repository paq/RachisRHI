#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <span>
#include <utility>
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

    [[nodiscard]] bool pixelIsBlack(const std::byte* pixel) noexcept
    {
        return pixel[0] == std::byte{0} && pixel[1] == std::byte{0} && pixel[2] == std::byte{0};
    }

} // namespace

TEST_CASE("dynamic depth bounds are capability gated, bind invalidated, and affect WARP rasterization",
          "[size:medium][rhi][graphics][depth-stencil][depth-bounds][warp]")
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
    REQUIRE(rachis::test::requireCapabilities(device).depthBoundsTest);

    const auto vertexBytecode = loadShader(RACHIS_TEST_GRAPHICS_DIRECT_VERTEX_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_GRAPHICS_PIXEL_SHADER_PATH);
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
        .entryPoint = "pixelMain",
        .shaderInterface = rachis::test::graphicsPixelShaderInterface(),
    });
    REQUIRE(vertexShader.has_value());
    REQUIRE(pixelShader.has_value());

    const std::array pushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::pixel, 0, 4 * sizeof(float)},
    };
    auto layout = device.createPipelineLayout(rachis::PipelineLayoutDesc{
        .pushConstantRanges = pushConstantRanges,
    });
    REQUIRE(layout.has_value());
    REQUIRE(rachis::test::requireCapabilities(device).pipelineCaches);
    auto pipelineCache = device.createPipelineCache({.label = "Depth-bounds pipeline cache"});
    REQUIRE(pipelineCache.has_value());
    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    const auto staticDesc = rachis::GraphicsPipelineDesc{
        .layout = *layout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .depthStencil =
            {
                .format = rachis::Format::d32Float,
                .depthTestEnabled = true,
                .depthCompare = rachis::CompareOperation::always,
            },
    };
    auto staticPipeline = device.createGraphicsPipeline(staticDesc);
    auto depthBoundsDesc = staticDesc;
    depthBoundsDesc.depthStencil.depthBoundsTestEnabled = true;
    depthBoundsDesc.cache = *pipelineCache;
    depthBoundsDesc.cacheKey = {.high = 0x4445505448424F55ULL, .low = 0x4E44535445535431ULL};
    auto depthBoundsPipeline = device.createGraphicsPipeline(depthBoundsDesc);
    auto missingDepthDesc = depthBoundsDesc;
    missingDepthDesc.depthStencil.format = rachis::Format::unknown;
    missingDepthDesc.depthStencil.depthTestEnabled = false;
    REQUIRE(staticPipeline.has_value());
    REQUIRE(depthBoundsPipeline.has_value());
    auto cachedDepthBoundsPipeline = device.createGraphicsPipeline(depthBoundsDesc);
    REQUIRE(cachedDepthBoundsPipeline.has_value());
    REQUIRE(device.destroyPipeline(*cachedDepthBoundsPipeline).has_value());
    auto missingDepth = device.createGraphicsPipeline(missingDepthDesc);
    REQUIRE_FALSE(missingDepth.has_value());
    CHECK(missingDepth.error().code == rachis::ErrorCode::invalidArgument);

    auto target = device.createTexture(rachis::TextureDesc{
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
    });
    auto depth = device.createTexture(rachis::TextureDesc{
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::d32Float,
        .usage = rachis::TextureUsage::depthStencil,
    });
    REQUIRE(target.has_value());
    REQUIRE(depth.has_value());
    auto targetView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    auto depthView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *depth,
        .usage = rachis::TextureViewUsage::depthStencil,
        .aspect = rachis::TextureAspect::depth,
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = readbackSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(targetView.has_value());
    REQUIRE(depthView.has_value());
    REQUIRE(readback.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    CHECK_FALSE(device.setDepthBounds(*command, {}).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *target,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *depth,
                                    .stagesAfter = rachis::PipelineStage::depthStencil,
                                    .accessAfter = rachis::Access::depthStencilWrite,
                                    .layoutAfter = rachis::TextureLayout::depthStencilWrite,
                                    .subresources = {.aspect = rachis::TextureAspect::depth},
                                })
                .has_value());
    const std::array attachments{
        rachis::ColorAttachment{
            .view = *targetView,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {},
        },
    };
    const rachis::DepthStencilAttachment depthAttachment{
        .view = *depthView,
        .depthLoadOperation = rachis::LoadOperation::clear,
        .depthClearValue = 0.5F,
    };
    REQUIRE(device
                .beginRendering(*command,
                                rachis::RenderingDesc{
                                    .colorAttachments = attachments,
                                    .depthStencilAttachment = depthAttachment,
                                })
                .has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *staticPipeline).has_value());
    CHECK_FALSE(device.setDepthBounds(*command, {}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *depthBoundsPipeline).has_value());
    REQUIRE(device
                .setViewport(*command,
                             rachis::Viewport{
                                 .width = static_cast<float>(targetWidth),
                                 .height = static_cast<float>(targetHeight),
                             })
                .has_value());
    const std::array tint{0.125F, 0.5F, 0.75F, 1.0F};
    REQUIRE(device.setGraphicsPushConstants(*command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(tint)))
                .has_value());
    REQUIRE(device
                .setScissor(*command,
                            rachis::ScissorRect{
                                .width = targetWidth / 2,
                                .height = targetHeight,
                            })
                .has_value());
    CHECK_FALSE(device.draw(*command, 3).has_value());
    CHECK_FALSE(device
                    .setDepthBounds(*command,
                                    rachis::DepthBounds{
                                        .minimum = std::numeric_limits<float>::quiet_NaN(),
                                    })
                    .has_value());
    CHECK_FALSE(device
                    .setDepthBounds(*command,
                                    rachis::DepthBounds{
                                        .maximum = std::numeric_limits<float>::infinity(),
                                    })
                    .has_value());
    CHECK_FALSE(device.setDepthBounds(*command, {.minimum = -0.25F, .maximum = 0.75F}).has_value());
    CHECK_FALSE(device.setDepthBounds(*command, {.minimum = 0.25F, .maximum = 1.25F}).has_value());
    CHECK_FALSE(device.setDepthBounds(*command, {.minimum = 0.75F, .maximum = 0.25F}).has_value());
    CHECK_FALSE(device.draw(*command, 3).has_value());
    REQUIRE(device.setDepthBounds(*command, {.minimum = 0.25F, .maximum = 0.75F}).has_value());
    REQUIRE(device.draw(*command, 3).has_value());

    REQUIRE(device.bindGraphicsPipeline(*command, *depthBoundsPipeline).has_value());
    CHECK_FALSE(device.draw(*command, 3).has_value());
    REQUIRE(device.setDepthBounds(*command, {.minimum = 0.75F, .maximum = 1.0F}).has_value());
    REQUIRE(device
                .setScissor(*command,
                            rachis::ScissorRect{
                                .x = targetWidth / 2,
                                .width = targetWidth / 2,
                                .height = targetHeight,
                            })
                .has_value());
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
    const auto leftOffset =
        static_cast<std::uint64_t>(targetHeight / 2) * rowPitch + static_cast<std::uint64_t>(targetWidth * 3 / 8) * 4;
    const auto rightOffset =
        static_cast<std::uint64_t>(targetHeight / 2) * rowPitch + static_cast<std::uint64_t>(targetWidth * 5 / 8) * 4;
    CHECK_FALSE(pixelIsBlack(mapped->data + leftOffset));
    CHECK(pixelIsBlack(mapped->data + rightOffset));
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyTextureView(*depthView).has_value());
    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*depth).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    REQUIRE(device.destroyPipeline(*depthBoundsPipeline).has_value());
    REQUIRE(device.destroyPipeline(*staticPipeline).has_value());
    REQUIRE(device.destroyPipelineCache(*pipelineCache).has_value());
    REQUIRE(device.destroyPipelineLayout(*layout).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
