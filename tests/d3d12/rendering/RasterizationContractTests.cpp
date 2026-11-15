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
    constexpr std::uint64_t rasterizationReadbackSize = 3 * readbackSize;

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
        return value + 1U >= expected && value <= static_cast<std::uint32_t>(expected) + 1U;
    }

} // namespace

TEST_CASE("rasterization state validates and renders on WARP", "[size:medium][rhi][graphics][rasterization][warp]")
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
    REQUIRE(rachis::test::requireCapabilities(device).wireframeRasterization);
    REQUIRE(rachis::test::requireCapabilities(device).conservativeRasterization);
    REQUIRE(rachis::test::requireCapabilities(device).unclippedDepth);

    const auto directBytecode = loadShader(RACHIS_TEST_GRAPHICS_DIRECT_VERTEX_SHADER_PATH);
    const auto outsideDepthBytecode = loadShader(RACHIS_TEST_GRAPHICS_UNCLIPPED_VERTEX_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_GRAPHICS_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(directBytecode.empty());
    REQUIRE_FALSE(outsideDepthBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    auto directShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = directBytecode,
        .entryPoint = "directVertex",
        .shaderInterface = {},
    });
    auto outsideDepthShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = outsideDepthBytecode,
        .entryPoint = "unclippedVertex",
        .shaderInterface = {},
    });
    auto pixelShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "pixelMain",
        .shaderInterface = rachis::test::graphicsPixelShaderInterface(),
    });
    REQUIRE(directShader.has_value());
    REQUIRE(outsideDepthShader.has_value());
    REQUIRE(pixelShader.has_value());

    const std::array pushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::pixel, 0, 4 * sizeof(float)},
    };
    auto layout = device.createPipelineLayout(rachis::PipelineLayoutDesc{
        .pushConstantRanges = pushConstantRanges,
    });
    REQUIRE(layout.has_value());
    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };

    const auto baseDesc = rachis::GraphicsPipelineDesc{
        .layout = *layout,
        .vertexShader = *directShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
    };
    auto invalidDesc = baseDesc;
    invalidDesc.fillMode = static_cast<rachis::FillMode>(255);
    CHECK_FALSE(device.createGraphicsPipeline(invalidDesc).has_value());
    invalidDesc = baseDesc;
    invalidDesc.depthBiasClamp = std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(device.createGraphicsPipeline(invalidDesc).has_value());
    invalidDesc = baseDesc;
    invalidDesc.depthBiasSlopeScale = std::numeric_limits<float>::infinity();
    CHECK_FALSE(device.createGraphicsPipeline(invalidDesc).has_value());
    invalidDesc = baseDesc;
    invalidDesc.alphaToCoverageEnabled = true;
    CHECK_FALSE(device.createGraphicsPipeline(invalidDesc).has_value());
    invalidDesc = baseDesc;
    invalidDesc.sampleCount = 4;
    invalidDesc.sampleMask = 1U << 4U;
    CHECK_FALSE(device.createGraphicsPipeline(invalidDesc).has_value());
    invalidDesc = baseDesc;
    invalidDesc.conservativeRasterization = static_cast<rachis::ConservativeRasterizationMode>(255);
    CHECK_FALSE(device.createGraphicsPipeline(invalidDesc).has_value());
    invalidDesc = baseDesc;
    invalidDesc.conservativeRasterization = rachis::ConservativeRasterizationMode::overestimate;
    invalidDesc.fillMode = rachis::FillMode::wireframe;
    CHECK_FALSE(device.createGraphicsPipeline(invalidDesc).has_value());
    invalidDesc = baseDesc;
    invalidDesc.conservativeRasterization = rachis::ConservativeRasterizationMode::overestimate;
    invalidDesc.topology = rachis::PrimitiveTopology::lineList;
    CHECK_FALSE(device.createGraphicsPipeline(invalidDesc).has_value());

    auto wireframeDesc = baseDesc;
    wireframeDesc.fillMode = rachis::FillMode::wireframe;
    wireframeDesc.depthBias = -1;
    wireframeDesc.depthBiasClamp = 1.0F;
    wireframeDesc.depthBiasSlopeScale = 1.0F;
    auto wireframePipeline = device.createGraphicsPipeline(wireframeDesc);
    auto clippedDesc = baseDesc;
    clippedDesc.vertexShader = *outsideDepthShader;
    auto clippedPipeline = device.createGraphicsPipeline(clippedDesc);
    auto unclippedDesc = clippedDesc;
    unclippedDesc.depthClipEnabled = false;
    auto unclippedPipeline = device.createGraphicsPipeline(unclippedDesc);
    auto ordinaryPipeline = device.createGraphicsPipeline(baseDesc);
    auto conservativeDesc = baseDesc;
    conservativeDesc.conservativeRasterization = rachis::ConservativeRasterizationMode::overestimate;
    auto conservativePipeline = device.createGraphicsPipeline(conservativeDesc);
    REQUIRE(wireframePipeline.has_value());
    REQUIRE(clippedPipeline.has_value());
    REQUIRE(unclippedPipeline.has_value());
    REQUIRE(ordinaryPipeline.has_value());
    REQUIRE(conservativePipeline.has_value());

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
    REQUIRE(targetView.has_value());
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = rasterizationReadbackSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(readback.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *target,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array attachments{
        rachis::ColorAttachment{
            .view = *targetView,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.0F, 0.0F, 0.0F, 0.0F},
        },
    };
    REQUIRE(device.beginRendering(*command, rachis::RenderingDesc{.colorAttachments = attachments}).has_value());
    const std::array tint{0.125F, 0.5F, 0.75F, 1.0F};
    for (const auto [pipeline, scissor] : std::array{
             std::pair{*wireframePipeline, rachis::ScissorRect{.width = targetWidth, .height = targetHeight}},
             std::pair{*clippedPipeline, rachis::ScissorRect{.width = targetWidth / 2, .height = targetHeight}},
             std::pair{*unclippedPipeline,
                       rachis::ScissorRect{
                           .x = targetWidth / 2,
                           .width = targetWidth / 2,
                           .height = targetHeight,
                       }},
         })
    {
        REQUIRE(device.bindGraphicsPipeline(*command, pipeline).has_value());
        REQUIRE(device
                    .setViewport(*command,
                                 rachis::Viewport{
                                     .width = static_cast<float>(targetWidth),
                                     .height = static_cast<float>(targetHeight),
                                 })
                    .has_value());
        REQUIRE(device.setScissor(*command, scissor).has_value());
        REQUIRE(device.setGraphicsPushConstants(*command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(tint)))
                    .has_value());
        REQUIRE(device.draw(*command, 3).has_value());
    }
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

    const auto recordSubpixelTriangle = [&](rachis::PipelineHandle pipeline, std::uint64_t bufferOffset) {
        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = *target,
                                        .stagesBefore = rachis::PipelineStage::copy,
                                        .stagesAfter = rachis::PipelineStage::renderTarget,
                                        .accessBefore = rachis::Access::copySource,
                                        .accessAfter = rachis::Access::renderTarget,
                                        .layoutBefore = rachis::TextureLayout::copySource,
                                        .layoutAfter = rachis::TextureLayout::renderTarget,
                                    })
                    .has_value());
        REQUIRE(device.beginRendering(*command, rachis::RenderingDesc{.colorAttachments = attachments}).has_value());
        REQUIRE(device.bindGraphicsPipeline(*command, pipeline).has_value());

        REQUIRE(device
                    .setViewport(*command,
                                 rachis::Viewport{
                                     .x = 32.6F,
                                     .y = 32.1F,
                                     .width = 1.0F,
                                     .height = 1.0F,
                                 })
                    .has_value());
        REQUIRE(
            device.setScissor(*command, rachis::ScissorRect{.width = targetWidth, .height = targetHeight}).has_value());
        REQUIRE(device.setGraphicsPushConstants(*command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(tint)))
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
                                             .bufferOffset = bufferOffset,
                                             .bytesPerRow = rowPitch,
                                             .rowsPerImage = targetHeight,
                                             .texture = *target,
                                             .extent = {targetWidth, targetHeight, 1},
                                         })
                    .has_value());
    };
    recordSubpixelTriangle(*ordinaryPipeline, readbackSize);
    recordSubpixelTriangle(*conservativePipeline, 2 * readbackSize);

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    const auto* clippedInterior = mapped->data + static_cast<std::uint64_t>(targetHeight / 2) * rowPitch +
                                  static_cast<std::uint64_t>(targetWidth * 3 / 8) * 4;
    const auto* unclippedInterior = mapped->data + static_cast<std::uint64_t>(targetHeight / 2) * rowPitch +
                                    static_cast<std::uint64_t>(targetWidth * 5 / 8) * 4;
    CHECK(clippedInterior[0] == std::byte{0});
    CHECK(clippedInterior[1] == std::byte{0});
    CHECK(clippedInterior[2] == std::byte{0});
    CHECK(nearByte(unclippedInterior[0], 32));
    CHECK(nearByte(unclippedInterior[1], 128));
    CHECK(nearByte(unclippedInterior[2], 191));
    std::size_t wireframePixelCount = 0;
    for (std::uint32_t y = 0; y < targetHeight; ++y)
    {
        for (std::uint32_t x = 0; x < targetWidth / 2; ++x)
        {
            const auto* pixel =
                mapped->data + static_cast<std::uint64_t>(y) * rowPitch + static_cast<std::uint64_t>(x) * 4;
            wireframePixelCount +=
                pixel[0] != std::byte{0} || pixel[1] != std::byte{0} || pixel[2] != std::byte{0} ? 1U : 0U;
        }
    }
    CHECK(wireframePixelCount > 0);
    std::size_t ordinaryPixelCount = 0;
    std::size_t conservativePixelCount = 0;
    for (std::uint32_t y = 0; y < targetHeight; ++y)
    {
        for (std::uint32_t x = 0; x < targetWidth; ++x)
        {
            const auto pixelOffset = static_cast<std::uint64_t>(y) * rowPitch + static_cast<std::uint64_t>(x) * 4;
            const auto* ordinaryPixel = mapped->data + readbackSize + pixelOffset;
            const auto* conservativePixel = mapped->data + 2 * readbackSize + pixelOffset;
            ordinaryPixelCount += ordinaryPixel[0] != std::byte{0} ? 1U : 0U;
            conservativePixelCount += conservativePixel[0] != std::byte{0} ? 1U : 0U;
        }
    }
    CHECK(ordinaryPixelCount == 0);
    CHECK(conservativePixelCount > ordinaryPixelCount);
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    REQUIRE(device.destroyPipeline(*wireframePipeline).has_value());
    REQUIRE(device.destroyPipeline(*clippedPipeline).has_value());
    REQUIRE(device.destroyPipeline(*unclippedPipeline).has_value());
    REQUIRE(device.destroyPipeline(*ordinaryPipeline).has_value());
    REQUIRE(device.destroyPipeline(*conservativePipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*layout).has_value());
    REQUIRE(device.destroyShaderModule(*directShader).has_value());
    REQUIRE(device.destroyShaderModule(*outsideDepthShader).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("point lists without a geometry stage draw one-pixel points on WARP",
          "[size:medium][rhi][graphics][rasterization][points][warp]")
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

    constexpr std::uint32_t pointTargetSize = 8;
    const auto vertexBytecode = loadShader(RACHIS_TEST_POINT_LIST_VERTEX_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_POINT_LIST_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    auto vertexShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .entryPoint = "vertexMain",
        .shaderInterface = {},
    });
    auto pixelShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "pixelMain",
        .shaderInterface = {},
    });
    auto layout = device.createPipelineLayout({});
    REQUIRE(vertexShader.has_value());
    REQUIRE(pixelShader.has_value());
    REQUIRE(layout.has_value());
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    auto pipeline = device.createGraphicsPipeline(rachis::GraphicsPipelineDesc{
        .layout = *layout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .topology = rachis::PrimitiveTopology::pointList,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
    });
    REQUIRE(pipeline.has_value());

    auto target = device.createTexture(rachis::TextureDesc{
        .width = pointTargetSize,
        .height = pointTargetSize,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
    });
    REQUIRE(target.has_value());
    auto targetView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = static_cast<std::uint64_t>(rowPitch) * pointTargetSize,
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
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array attachments{
        rachis::ColorAttachment{
            .view = *targetView,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {},
        },
    };
    REQUIRE(device.beginRendering(*command, rachis::RenderingDesc{.colorAttachments = attachments}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *pipeline).has_value());
    REQUIRE(device
                .setViewport(*command,
                             rachis::Viewport{
                                 .width = static_cast<float>(pointTargetSize),
                                 .height = static_cast<float>(pointTargetSize),
                             })
                .has_value());
    REQUIRE(device.setScissor(*command, rachis::ScissorRect{.width = pointTargetSize, .height = pointTargetSize})
                .has_value());
    REQUIRE(device.draw(*command, 4).has_value());
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
                                         .rowsPerImage = pointTargetSize,
                                         .texture = *target,
                                         .extent = {pointTargetSize, pointTargetSize, 1},
                                     })
                .has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    std::vector<std::pair<std::uint32_t, std::uint32_t>> lit;
    for (std::uint32_t y = 0; y < pointTargetSize; ++y)
    {
        for (std::uint32_t x = 0; x < pointTargetSize; ++x)
        {
            const auto* pixel =
                mapped->data + static_cast<std::uint64_t>(y) * rowPitch + static_cast<std::uint64_t>(x) * 4;
            if (pixel[0] != std::byte{0})
            {
                lit.emplace_back(x, y);
            }
        }
    }
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> expected{{1, 3}, {3, 3}, {5, 3}, {7, 3}};
    CHECK(lit == expected);
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*layout).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
