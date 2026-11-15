#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <vector>

namespace
{

    constexpr std::uint32_t targetSize = 64;

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

    struct OffscreenTarget final
    {
        rachis::TextureHandle texture{};
        rachis::TextureViewHandle view{};
        rachis::BufferHandle readback{};
        rachis::TextureTransferLayout layout{};
    };

    [[nodiscard]] OffscreenTarget makeColorTarget(rachis::Device& device, std::string_view label)
    {
        OffscreenTarget target{};
        auto texture = device.createTexture(rachis::TextureDesc{
            .width = targetSize,
            .height = targetSize,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource,
            .label = label,
        });
        REQUIRE(texture);
        target.texture = *texture;
        auto view = device.createTextureView(rachis::TextureViewDesc{
            .texture = target.texture,
            .usage = rachis::TextureViewUsage::renderTarget,
            .dimension = rachis::TextureViewDimension::texture2D,
        });
        REQUIRE(view);
        target.view = *view;
        const rachis::BufferTextureCopyRegion region{
            .extent = {.width = targetSize, .height = targetSize, .depth = 1},
        };
        auto layout = device.queryTextureTransferLayout(target.texture, region);
        REQUIRE(layout);
        target.layout = *layout;
        auto readback = device.createBuffer(rachis::BufferDesc{
            .size = layout->requiredBufferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Metal raster readback",
        });
        REQUIRE(readback);
        target.readback = *readback;
        return target;
    }

    void copyColorToReadback(rachis::Device& device, rachis::CommandBuffer& command, const OffscreenTarget& target)
    {
        REQUIRE(device
                    .textureBarrier(command,
                                    rachis::TextureBarrier{
                                        .texture = target.texture,
                                        .stagesBefore = rachis::PipelineStage::renderTarget,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::renderTarget,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                    })
                    .has_value());
        const rachis::BufferTextureCopyRegion region{
            .bytesPerRow = target.layout.bytesPerRow,
            .rowsPerImage = target.layout.rowsPerImage,
            .extent = {.width = targetSize, .height = targetSize, .depth = 1},
        };
        REQUIRE(device.copyTextureToBuffer(command, target.texture, target.readback, std::array{region}).has_value());
    }

    [[nodiscard]] std::vector<std::byte> readTarget(rachis::Device& device, const OffscreenTarget& target)
    {
        auto mapped = device.mapBuffer(target.readback);
        REQUIRE(mapped);
        std::vector<std::byte> bytes(mapped->data,
                                     mapped->data + static_cast<std::size_t>(target.layout.requiredBufferSize));
        REQUIRE(device.unmapBuffer(target.readback).has_value());
        return bytes;
    }

    [[nodiscard]] const std::byte* pixelAt(const std::vector<std::byte>& bytes,
                                           const rachis::TextureTransferLayout& layout,
                                           std::uint32_t x,
                                           std::uint32_t y) noexcept
    {
        return bytes.data() + static_cast<std::size_t>(y) * layout.bytesPerRow + static_cast<std::size_t>(x) * 4;
    }

    [[nodiscard]] bool pixelNear(const std::byte* pixel,
                                 std::array<std::uint8_t, 4> expected,
                                 std::uint8_t slop = 2) noexcept
    {
        for (std::uint32_t channel = 0; channel < 4; ++channel)
        {
            const auto value = std::to_integer<std::uint8_t>(pixel[channel]);
            const auto lower = expected[channel] > slop ? static_cast<std::uint32_t>(expected[channel] - slop) : 0U;
            const auto upper = (std::min)(static_cast<std::uint32_t>(expected[channel]) + slop, 255U);
            if (value < lower || value > upper)
            {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] std::size_t coloredPixelCount(const std::vector<std::byte>& bytes,
                                                const rachis::TextureTransferLayout& layout) noexcept
    {
        std::size_t count = 0;
        for (std::uint32_t y = 0; y < targetSize; ++y)
        {
            for (std::uint32_t x = 0; x < targetSize; ++x)
            {
                const auto* pixel = pixelAt(bytes, layout, x, y);
                if (std::to_integer<std::uint8_t>(pixel[0]) != 0 || std::to_integer<std::uint8_t>(pixel[1]) != 0 ||
                    std::to_integer<std::uint8_t>(pixel[2]) != 0)
                {
                    ++count;
                }
            }
        }
        return count;
    }

    [[nodiscard]] rachis::PipelineHandle makeTrianglePipeline(rachis::Device& device,
                                                              rachis::PipelineLayoutHandle layout,
                                                              rachis::CullMode cullMode = rachis::CullMode::back,
                                                              rachis::BlendState blend = {},
                                                              rachis::DepthStencilState depth = {})
    {
        const auto triangleVs = loadBytes(pnmsPath("BootstrapTriangle.vs.metallib"));
        const auto trianglePs = loadBytes(pnmsPath("BootstrapTriangle.ps.metallib"));
        REQUIRE_FALSE(triangleVs.empty());
        REQUIRE_FALSE(trianglePs.empty());
        auto vertex = device.createShaderModule({
            .stage = rachis::ShaderStage::vertex,
            .bytecode = triangleVs,
            .entryPoint = "vertexMain",
            .label = "Metal raster vertex",
        });
        auto pixel = device.createShaderModule({
            .stage = rachis::ShaderStage::pixel,
            .bytecode = trianglePs,
            .entryPoint = "pixelMain",
            .label = "Metal raster pixel",
        });
        REQUIRE(vertex);
        REQUIRE(pixel);
        const std::array colorTargets{rachis::ColorTargetState{
            .format = rachis::Format::rgba8Unorm,
            .blend = blend,
        }};
        auto pipeline = device.createGraphicsPipeline({
            .layout = layout,
            .vertexShader = *vertex,
            .pixelShader = *pixel,
            .cullMode = cullMode,
            .colorTargets = colorTargets,
            .depthStencil = depth,
            .label = "Metal raster pipeline",
        });
        if (!pipeline)
        {
            UNSCOPED_INFO(pipeline.error().message);
        }
        REQUIRE(pipeline);
        return *pipeline;
    }

    void bindDefaultDrawState(rachis::Device& device,
                              rachis::CommandBuffer& command,
                              std::uint32_t width = targetSize,
                              std::uint32_t height = targetSize)
    {
        REQUIRE(device
                    .setViewports(command,
                                  std::array{rachis::Viewport{
                                      .width = static_cast<float>(width),
                                      .height = static_cast<float>(height),
                                  }})
                    .has_value());
        REQUIRE(device
                    .setScissors(command,
                                 std::array{rachis::ScissorRect{
                                     .width = width,
                                     .height = height,
                                 }})
                    .has_value());
    }

} // namespace

TEST_CASE("Metal rendering rejects nested scopes, compute queues, and unsupported view masks",
          "[size:large][rhi][metal][graphics][validation]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    auto target = makeColorTarget(device, "Metal raster validation color");
    auto layout = device.createPipelineLayout({});
    REQUIRE(layout);

    auto compute = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(compute);
    const std::array color{rachis::ColorAttachment{.view = target.view}};
    auto computeBegin = device.beginRendering(*compute, {.colorAttachments = color});
    REQUIRE_FALSE(computeBegin);
    CHECK(computeBegin.error().code == rachis::ErrorCode::invalidState);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device.beginRendering(*command, {.colorAttachments = color}).has_value());
    auto nested = device.beginRendering(*command, {.colorAttachments = color});
    REQUIRE_FALSE(nested);
    CHECK(nested.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.endRendering(*command).has_value());

    auto viewMask = device.beginRendering(*command, {.colorAttachments = color, .viewMask = 1});
    REQUIRE_FALSE(viewMask);
    CHECK(viewMask.error().code == rachis::ErrorCode::unsupported);

    auto empty = device.beginRendering(*command, {});
    REQUIRE_FALSE(empty);
    CHECK(empty.error().code == rachis::ErrorCode::invalidState);
}

TEST_CASE("Metal offscreen triangle, scissor, front-face cull, load/store, blend, and clears match GPU readback",
          "[size:large][rhi][metal][graphics][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    auto layout = device.createPipelineLayout({});
    REQUIRE(layout);
    auto pipeline = makeTrianglePipeline(device, *layout);
    auto culled = makeTrianglePipeline(device, *layout, rachis::CullMode::front);
    const rachis::BlendState additive{
        .enabled = true,
        .color = {.sourceFactor = rachis::BlendFactor::one, .destinationFactor = rachis::BlendFactor::one},
        .alpha = {.sourceFactor = rachis::BlendFactor::one, .destinationFactor = rachis::BlendFactor::one},
    };
    auto blended = makeTrianglePipeline(device, *layout, rachis::CullMode::back, additive);

    auto color = makeColorTarget(device, "Metal raster color");
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = color.texture,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array attachment{rachis::ColorAttachment{
        .view = color.view,
        .loadOperation = rachis::LoadOperation::clear,
        .clearValue = {.red = 0.0F, .green = 0.0F, .blue = 0.0F, .alpha = 1.0F},
    }};
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachment}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, pipeline).has_value());
    bindDefaultDrawState(device, *command);
    REQUIRE(device.draw(*command, 3, 1, 0, 0).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    copyColorToReadback(device, *command, color);
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    auto triangleBytes = readTarget(device, color);
    CHECK(coloredPixelCount(triangleBytes, color.layout) > 200);
    CHECK(pixelNear(pixelAt(triangleBytes, color.layout, 0, 0), {0, 0, 0, 255}));
    const auto* center = pixelAt(triangleBytes, color.layout, targetSize / 2, targetSize / 2 + 8);
    CHECK((std::to_integer<std::uint8_t>(center[0]) > 8 || std::to_integer<std::uint8_t>(center[1]) > 8 ||
           std::to_integer<std::uint8_t>(center[2]) > 8));

    command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = color.texture,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachment}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, culled).has_value());
    bindDefaultDrawState(device, *command);
    REQUIRE(device.draw(*command, 3, 1, 0, 0).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    copyColorToReadback(device, *command, color);
    submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    auto culledBytes = readTarget(device, color);
    CHECK(coloredPixelCount(culledBytes, color.layout) == 0);

    command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = color.texture,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachment}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, pipeline).has_value());
    bindDefaultDrawState(device, *command);
    REQUIRE(device
                .setScissors(*command,
                             std::array{rachis::ScissorRect{
                                 .x = 0,
                                 .y = 0,
                                 .width = 12,
                                 .height = 12,
                             }})
                .has_value());
    REQUIRE(device.draw(*command, 3, 1, 0, 0).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    copyColorToReadback(device, *command, color);
    submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    auto scissorBytes = readTarget(device, color);
    CHECK(pixelNear(pixelAt(scissorBytes, color.layout, targetSize - 1, targetSize - 1), {0, 0, 0, 255}));
    CHECK(coloredPixelCount(scissorBytes, color.layout) < coloredPixelCount(triangleBytes, color.layout));

    command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = color.texture,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array magenta{rachis::ColorAttachment{
        .view = color.view,
        .loadOperation = rachis::LoadOperation::clear,
        .clearValue = {.red = 1.0F, .green = 0.0F, .blue = 1.0F, .alpha = 1.0F},
    }};
    REQUIRE(device.beginRendering(*command, {.colorAttachments = magenta}).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    const std::array loaded{rachis::ColorAttachment{
        .view = color.view,
        .loadOperation = rachis::LoadOperation::load,
    }};
    REQUIRE(device.beginRendering(*command, {.colorAttachments = loaded}).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    copyColorToReadback(device, *command, color);
    submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    auto loadedBytes = readTarget(device, color);
    CHECK(pixelNear(pixelAt(loadedBytes, color.layout, 4, 4), {255, 0, 255, 255}));

    command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = color.texture,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array halfRed{rachis::ColorAttachment{
        .view = color.view,
        .loadOperation = rachis::LoadOperation::clear,
        .clearValue = {.red = 0.5F, .green = 0.0F, .blue = 0.0F, .alpha = 1.0F},
    }};
    REQUIRE(device.beginRendering(*command, {.colorAttachments = halfRed}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, blended).has_value());
    bindDefaultDrawState(device, *command);
    REQUIRE(device.draw(*command, 3, 1, 0, 0).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    copyColorToReadback(device, *command, color);
    submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    auto blendBytes = readTarget(device, color);
    CHECK(pixelNear(pixelAt(blendBytes, color.layout, 0, 0), {128, 0, 0, 255}, 4));
    const auto* blendCenter = pixelAt(blendBytes, color.layout, targetSize / 2, targetSize / 2 + 8);
    CHECK(std::to_integer<std::uint8_t>(blendCenter[0]) > 128);

    command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = color.texture,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachment}).has_value());
    REQUIRE(device.clearColorAttachment(*command, 0, {.red = 0.0F, .green = 1.0F, .blue = 0.0F, .alpha = 1.0F}, {})
                .has_value());
    REQUIRE(device.endRendering(*command).has_value());
    copyColorToReadback(device, *command, color);
    submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    auto cleared = readTarget(device, color);
    CHECK(pixelNear(pixelAt(cleared, color.layout, 8, 8), {0, 255, 0, 255}));
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal depth test, indexed vertex buffers, and instance identity produce offscreen GPU results",
          "[size:large][rhi][metal][graphics][gpu][depth][index][instance]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    auto layout = device.createPipelineLayout({});
    REQUIRE(layout);

    const rachis::DepthStencilState depthState{
        .format = rachis::Format::d32Float,
        .depthTestEnabled = true,
        .depthWriteEnabled = true,
        .depthCompare = rachis::CompareOperation::less,
    };
    auto pipeline = makeTrianglePipeline(device, *layout, rachis::CullMode::back, {}, depthState);
    auto color = makeColorTarget(device, "Metal depth color");
    auto depthTexture = device.createTexture(rachis::TextureDesc{
        .width = targetSize,
        .height = targetSize,
        .format = rachis::Format::d32Float,
        .usage = rachis::TextureUsage::depthStencil,
        .label = "Metal raster depth",
    });
    REQUIRE(depthTexture);
    auto depthView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *depthTexture,
        .usage = rachis::TextureViewUsage::depthStencil,
        .dimension = rachis::TextureViewDimension::texture2D,
        .aspect = rachis::TextureAspect::depth,
    });
    REQUIRE(depthView);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = color.texture,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *depthTexture,
                                    .stagesAfter = rachis::PipelineStage::depthStencil,
                                    .accessAfter = rachis::Access::depthStencilWrite,
                                    .layoutAfter = rachis::TextureLayout::depthStencilWrite,
                                    .subresources = {.aspect = rachis::TextureAspect::depth},
                                })
                .has_value());
    const std::array colorAttachment{rachis::ColorAttachment{
        .view = color.view,
        .loadOperation = rachis::LoadOperation::clear,
        .clearValue = {.alpha = 1.0F},
    }};
    const rachis::DepthStencilAttachment depthAttachment{
        .view = *depthView,
        .depthLoadOperation = rachis::LoadOperation::clear,
        .depthClearValue = 1.0F,
    };
    REQUIRE(device
                .beginRendering(*command,
                                {
                                    .colorAttachments = colorAttachment,
                                    .depthStencilAttachment = depthAttachment,
                                })
                .has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, pipeline).has_value());
    REQUIRE(device
                .setViewports(*command,
                              std::array{rachis::Viewport{
                                  .width = static_cast<float>(targetSize),
                                  .height = static_cast<float>(targetSize),
                                  .minDepth = 0.8F,
                                  .maxDepth = 0.9F,
                              }})
                .has_value());
    REQUIRE(device
                .setScissors(*command,
                             std::array{rachis::ScissorRect{
                                 .width = targetSize,
                                 .height = targetSize,
                             }})
                .has_value());
    REQUIRE(device.draw(*command, 3, 1, 0, 0).has_value());
    REQUIRE(device
                .setViewports(*command,
                              std::array{rachis::Viewport{
                                  .width = static_cast<float>(targetSize),
                                  .height = static_cast<float>(targetSize),
                                  .minDepth = 0.1F,
                                  .maxDepth = 0.2F,
                              }})
                .has_value());
    REQUIRE(device.draw(*command, 3, 1, 0, 0).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    copyColorToReadback(device, *command, color);
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    auto depthBytes = readTarget(device, color);
    CHECK(coloredPixelCount(depthBytes, color.layout) > 200);

    const auto vertexVs = loadBytes(pnmsPath("MetalRasterVertex.vs.metallib"));
    const auto vertexPs = loadBytes(pnmsPath("MetalRasterVertex.ps.metallib"));
    REQUIRE_FALSE(vertexVs.empty());
    REQUIRE_FALSE(vertexPs.empty());
    const std::array vertexLocations{0U, 1U};
    auto meshVertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexVs,
        .entryPoint = "vertexMain",
        .label = "Metal vertex-input vertex",
        .shaderInterface = rachis::test::shaderInterface({}, {}, vertexLocations),
    });
    auto meshPixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = vertexPs,
        .entryPoint = "pixelMain",
        .label = "Metal vertex-input pixel",
    });
    if (!meshVertex)
    {
        UNSCOPED_INFO(meshVertex.error().message);
    }
    REQUIRE(meshVertex);
    REQUIRE(meshPixel);
    const std::array meshAttributes{
        rachis::VertexAttribute{rachis::VertexFormat::float32x2, 0, 0},
        rachis::VertexAttribute{rachis::VertexFormat::float32x3, 8, 1},
    };
    const std::array meshLayouts{rachis::VertexBufferLayout{
        .stride = 20,
        .attributes = meshAttributes,
    }};
    const std::array meshTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    auto meshPipeline = device.createGraphicsPipeline({
        .layout = *layout,
        .vertexShader = *meshVertex,
        .pixelShader = *meshPixel,
        .vertexBuffers = meshLayouts,
        .cullMode = rachis::CullMode::none,
        .colorTargets = meshTargets,
        .label = "Metal indexed vertex pipeline",
    });
    if (!meshPipeline)
    {
        UNSCOPED_INFO(meshPipeline.error().message);
    }
    REQUIRE(meshPipeline);
    const std::array misalignedLayouts{rachis::VertexBufferLayout{
        .stride = 22,
        .attributes = meshAttributes,
    }};
    const auto misalignedPipeline = device.createGraphicsPipeline({
        .layout = *layout,
        .vertexShader = *meshVertex,
        .pixelShader = *meshPixel,
        .vertexBuffers = misalignedLayouts,
        .cullMode = rachis::CullMode::none,
        .colorTargets = meshTargets,
    });
    REQUIRE_FALSE(misalignedPipeline.has_value());
    CHECK(misalignedPipeline.error().code == rachis::ErrorCode::invalidArgument);

    struct MeshVertex final
    {
        float position[2];
        float color[3];
    };
    const std::array vertices{
        MeshVertex{{0.0F, 0.72F}, {1.0F, 0.0F, 0.0F}},
        MeshVertex{{-0.72F, -0.72F}, {0.0F, 1.0F, 0.0F}},
        MeshVertex{{0.72F, -0.72F}, {0.0F, 0.0F, 1.0F}},
    };
    const std::array<std::uint16_t, 3> indices{0, 1, 2};
    auto vertexBuffer = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(vertices),
        .usage = rachis::BufferUsage::vertex | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal raster vertices",
    });
    auto indexBuffer = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(indices),
        .usage = rachis::BufferUsage::index | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal raster indices",
    });
    REQUIRE(vertexBuffer);
    REQUIRE(indexBuffer);
    REQUIRE(device.writeBuffer(*vertexBuffer, 0, std::as_bytes(std::span{vertices})).has_value());
    REQUIRE(device.writeBuffer(*indexBuffer, 0, std::as_bytes(std::span{indices})).has_value());

    command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = color.texture,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    REQUIRE(device.beginRendering(*command, {.colorAttachments = colorAttachment}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *meshPipeline).has_value());
    bindDefaultDrawState(device, *command);
    const auto misalignedVertices = device.bindVertexBuffer(*command, 0, *vertexBuffer, 2);
    REQUIRE_FALSE(misalignedVertices.has_value());
    CHECK(misalignedVertices.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(device.bindVertexBuffers(*command, 0, std::array{rachis::VertexBufferBinding{*vertexBuffer}}).has_value());
    REQUIRE(device.bindIndexBuffer(*command, *indexBuffer, rachis::IndexFormat::uint16, 0).has_value());
    REQUIRE(device.drawIndexed(*command, 3, 1, 0, 0, 0).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    copyColorToReadback(device, *command, color);
    submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    auto indexedBytes = readTarget(device, color);
    CHECK(coloredPixelCount(indexedBytes, color.layout) > 200);

    const auto instanceVs = loadBytes(pnmsPath("MetalRasterInstance.vs.metallib"));
    const auto instancePs = loadBytes(pnmsPath("MetalRasterInstance.ps.metallib"));
    REQUIRE_FALSE(instanceVs.empty());
    REQUIRE_FALSE(instancePs.empty());
    auto instanceVertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = instanceVs,
        .entryPoint = "vertexMain",
        .label = "Metal instance vertex",
    });
    auto instancePixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = instancePs,
        .entryPoint = "pixelMain",
        .label = "Metal instance pixel",
    });
    REQUIRE(instanceVertex);
    REQUIRE(instancePixel);
    auto instancePipeline = device.createGraphicsPipeline({
        .layout = *layout,
        .vertexShader = *instanceVertex,
        .pixelShader = *instancePixel,
        .cullMode = rachis::CullMode::none,
        .colorTargets = meshTargets,
        .label = "Metal instance pipeline",
    });
    REQUIRE(instancePipeline);
    command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = color.texture,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    REQUIRE(device.beginRendering(*command, {.colorAttachments = colorAttachment}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *instancePipeline).has_value());
    bindDefaultDrawState(device, *command);
    REQUIRE(device.draw(*command, 3, 2, 0, 0).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    copyColorToReadback(device, *command, color);
    submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    auto instanceBytes = readTarget(device, color);
    bool sawRed = false;
    bool sawBlue = false;
    for (std::uint32_t y = 0; y < targetSize; ++y)
    {
        for (std::uint32_t x = 0; x < targetSize; ++x)
        {
            const auto* pixel = pixelAt(instanceBytes, color.layout, x, y);
            const auto red = std::to_integer<std::uint8_t>(pixel[0]);
            const auto blue = std::to_integer<std::uint8_t>(pixel[2]);
            sawRed = sawRed || (red > 200 && blue < 40);
            sawBlue = sawBlue || (blue > 200 && red < 40);
        }
    }
    CHECK(sawRed);
    CHECK(sawBlue);
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal indexed draws reject index ranges outside the bound index buffer",
          "[size:large][rhi][metal][graphics][validation][index][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    auto layout = device.createPipelineLayout({});
    REQUIRE(layout);
    auto pipeline = makeTrianglePipeline(device, *layout);
    auto color = makeColorTarget(device, "Metal index range color");
    const std::array<std::uint16_t, 3> indices{0, 1, 2};
    auto indexBuffer = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(indices) + 1,
        .usage = rachis::BufferUsage::index | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal index range indices",
    });
    REQUIRE(indexBuffer);
    REQUIRE(device.writeBuffer(*indexBuffer, 0, std::as_bytes(std::span{indices})).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = color.texture,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array attachment{rachis::ColorAttachment{
        .view = color.view,
        .loadOperation = rachis::LoadOperation::clear,
        .clearValue = {.alpha = 1.0F},
    }};
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachment}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, pipeline).has_value());
    bindDefaultDrawState(device, *command);

    auto partialUint32 = device.bindIndexBuffer(*command, *indexBuffer, rachis::IndexFormat::uint32, 4);
    REQUIRE_FALSE(partialUint32);
    CHECK(partialUint32.error().code == rachis::ErrorCode::invalidArgument);
    auto partialUint16 = device.bindIndexBuffer(*command, *indexBuffer, rachis::IndexFormat::uint16, 6);
    REQUIRE_FALSE(partialUint16);
    CHECK(partialUint16.error().code == rachis::ErrorCode::invalidArgument);
    auto unbound = device.drawIndexed(*command, 3, 1, 0, 0, 0);
    REQUIRE_FALSE(unbound);
    CHECK(unbound.error().code == rachis::ErrorCode::invalidState);

    REQUIRE(device.bindIndexBuffer(*command, *indexBuffer, rachis::IndexFormat::uint16, 0).has_value());
    auto tooMany = device.drawIndexed(*command, 4, 1, 0, 0, 0);
    REQUIRE_FALSE(tooMany);
    CHECK(tooMany.error().code == rachis::ErrorCode::invalidArgument);
    auto pastEnd = device.drawIndexed(*command, 2, 1, 2, 0, 0);
    REQUIRE_FALSE(pastEnd);
    CHECK(pastEnd.error().code == rachis::ErrorCode::invalidArgument);
    auto wrapped = device.drawIndexed(*command, std::numeric_limits<std::uint32_t>::max(), 1, 1, 0, 0);
    REQUIRE_FALSE(wrapped);
    CHECK(wrapped.error().code == rachis::ErrorCode::invalidArgument);
    auto startPastEnd = device.drawIndexed(*command, 1, 1, 3, 0, 0);
    REQUIRE_FALSE(startPastEnd);
    CHECK(startPastEnd.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(device.drawIndexed(*command, 1, 1, 2, 0, 0).has_value());
    REQUIRE(device.drawIndexed(*command, 3, 1, 0, 0, 0).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    copyColorToReadback(device, *command, color);
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    auto bytes = readTarget(device, color);
    CHECK(coloredPixelCount(bytes, color.layout) > 200);
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal indexed indirect draws require an index buffer bound after the current pipeline",
          "[size:large][rhi][metal][graphics][validation][index][indirect][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    auto layout = device.createPipelineLayout({});
    REQUIRE(layout);
    auto firstPipeline = makeTrianglePipeline(device, *layout);
    auto secondPipeline = makeTrianglePipeline(device, *layout);
    auto color = makeColorTarget(device, "Metal indexed indirect rebind color");
    const std::array<std::uint16_t, 3> staleIndices{0, 0, 0};
    const std::array<std::uint32_t, 4> freshIndices{0, 0, 1, 2};
    auto staleBuffer = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(staleIndices),
        .usage = rachis::BufferUsage::index | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal indexed indirect stale indices",
    });
    auto freshBuffer = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(freshIndices),
        .usage = rachis::BufferUsage::index | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal indexed indirect fresh indices",
    });
    const rachis::DrawIndexedIndirectArguments indexed{3, 1, 0, 0, 0};
    auto arguments = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(indexed),
        .usage = rachis::BufferUsage::indirect | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal indexed indirect rebind arguments",
    });
    REQUIRE(staleBuffer);
    REQUIRE(freshBuffer);
    REQUIRE(arguments);
    REQUIRE(device.writeBuffer(*staleBuffer, 0, std::as_bytes(std::span{staleIndices})).has_value());
    REQUIRE(device.writeBuffer(*freshBuffer, 0, std::as_bytes(std::span{freshIndices})).has_value());
    REQUIRE(device.writeBuffer(*arguments, 0, std::as_bytes(std::span{&indexed, 1})).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = color.texture,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array attachment{rachis::ColorAttachment{
        .view = color.view,
        .loadOperation = rachis::LoadOperation::clear,
        .clearValue = {.alpha = 1.0F},
    }};
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachment}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, firstPipeline).has_value());
    bindDefaultDrawState(device, *command);
    REQUIRE(device.bindIndexBuffer(*command, *staleBuffer, rachis::IndexFormat::uint16, 0).has_value());
    REQUIRE(device.drawIndexed(*command, 3, 1, 0, 0, 0).has_value());
    REQUIRE(device.drawIndexedIndirect(*command, {.argumentBuffer = *arguments}).has_value());

    REQUIRE(device.bindGraphicsPipeline(*command, secondPipeline).has_value());
    bindDefaultDrawState(device, *command);
    auto indirect = device.drawIndexedIndirect(*command, {.argumentBuffer = *arguments});
    REQUIRE_FALSE(indirect);
    CHECK(indirect.error().code == rachis::ErrorCode::invalidState);
    auto empty = device.drawIndexedIndirect(*command, {.argumentBuffer = *arguments, .maxDrawCount = 0});
    REQUIRE_FALSE(empty);
    CHECK(empty.error().code == rachis::ErrorCode::invalidState);
    auto direct = device.drawIndexed(*command, 3, 1, 0, 0, 0);
    REQUIRE_FALSE(direct);
    CHECK(direct.error().code == rachis::ErrorCode::invalidState);

    REQUIRE(device.bindIndexBuffer(*command, *freshBuffer, rachis::IndexFormat::uint32, 4).has_value());
    REQUIRE(device.drawIndexedIndirect(*command, {.argumentBuffer = *arguments, .maxDrawCount = 0}).has_value());
    REQUIRE(device.drawIndexedIndirect(*command, {.argumentBuffer = *arguments}).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    copyColorToReadback(device, *command, color);
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    auto bytes = readTarget(device, color);
    CHECK(coloredPixelCount(bytes, color.layout) > 200);
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal graphics binding sets must match the bound pipeline layout when they are bound",
          "[size:large][rhi][metal][graphics][binding][validation][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    auto bindings = device.createBindingLayout({
        .entries =
            std::array{rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::vertex}},
    });
    auto otherBindings = device.createBindingLayout({
        .entries =
            std::array{rachis::BindingLayoutEntry{1, rachis::BindingType::uniformBuffer, rachis::ShaderStage::vertex}},
    });
    REQUIRE(bindings);
    REQUIRE(otherBindings);
    auto layout = device.createPipelineLayout({.bindingLayouts = std::array{*bindings}});
    REQUIRE(layout);
    auto pipeline = makeTrianglePipeline(device, *layout);
    auto uniforms = device.createBuffer(rachis::BufferDesc{
        .size = 256,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal graphics binding uniforms",
    });
    REQUIRE(uniforms);
    auto set = rachis::test::createBindingSet(
        device,
        {
            .layout = *bindings,
            .writes = std::array{rachis::BindingWrite{0, 0, rachis::BufferBinding{*uniforms, 0, 256}}},
        });
    auto otherSet = rachis::test::createBindingSet(
        device,
        {
            .layout = *otherBindings,
            .writes = std::array{rachis::BindingWrite{1, 0, rachis::BufferBinding{*uniforms, 0, 256}}},
        });
    REQUIRE(set);
    REQUIRE(otherSet);
    auto color = makeColorTarget(device, "Metal graphics binding color");

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = color.texture,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array attachment{rachis::ColorAttachment{
        .view = color.view,
        .loadOperation = rachis::LoadOperation::clear,
        .clearValue = {.alpha = 1.0F},
    }};
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachment}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, pipeline).has_value());
    bindDefaultDrawState(device, *command);
    auto mismatched = device.bindGraphicsBindingSet(*command, 0, *otherSet);
    REQUIRE_FALSE(mismatched);
    CHECK(mismatched.error().code == rachis::ErrorCode::invalidArgument);
    auto outside = device.bindGraphicsBindingSet(*command, 1, *set);
    REQUIRE_FALSE(outside);
    CHECK(outside.error().code == rachis::ErrorCode::invalidArgument);
    auto missing = device.draw(*command, 3, 1, 0, 0);
    REQUIRE_FALSE(missing);
    CHECK(missing.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.bindGraphicsBindingSet(*command, 0, *set).has_value());
    auto replaced = device.bindGraphicsBindingSet(*command, 0, *otherSet);
    REQUIRE_FALSE(replaced);
    CHECK(replaced.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(device.draw(*command, 3, 1, 0, 0).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    copyColorToReadback(device, *command, color);
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    auto bytes = readTarget(device, color);
    CHECK(coloredPixelCount(bytes, color.layout) > 200);
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal indexed strips restart at the strip index format and reject strips without one",
          "[size:large][rhi][metal][graphics][gpu][index][primitive-restart][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto capabilities = device.capabilities();
    REQUIRE(capabilities);
    CHECK(capabilities->primitiveRestart);
    auto layout = device.createPipelineLayout({});
    REQUIRE(layout);

    const auto vertexVs = loadBytes(pnmsPath("MetalRasterVertex.vs.metallib"));
    const auto vertexPs = loadBytes(pnmsPath("MetalRasterVertex.ps.metallib"));
    REQUIRE_FALSE(vertexVs.empty());
    REQUIRE_FALSE(vertexPs.empty());
    const std::array vertexLocations{0U, 1U};
    auto vertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexVs,
        .entryPoint = "vertexMain",
        .label = "Metal restart vertex",
        .shaderInterface = rachis::test::shaderInterface({}, {}, vertexLocations),
    });
    auto pixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = vertexPs,
        .entryPoint = "pixelMain",
        .label = "Metal restart pixel",
    });
    REQUIRE(vertex);
    REQUIRE(pixel);
    const std::array attributes{
        rachis::VertexAttribute{rachis::VertexFormat::float32x2, 0, 0},
        rachis::VertexAttribute{rachis::VertexFormat::float32x3, 8, 1},
    };
    const std::array vertexLayouts{rachis::VertexBufferLayout{
        .stride = 20,
        .attributes = attributes,
    }};
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    const rachis::GraphicsPipelineDesc stripDesc{
        .layout = *layout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .vertexBuffers = vertexLayouts,
        .topology = rachis::PrimitiveTopology::triangleStrip,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .label = "Metal restart strip pipeline",
    };
    auto unrestartedStrip = device.createGraphicsPipeline(stripDesc);
    auto uint16Desc = stripDesc;
    uint16Desc.stripIndexFormat = rachis::IndexFormat::uint16;
    auto uint16Strip = device.createGraphicsPipeline(uint16Desc);
    auto uint32Desc = stripDesc;
    uint32Desc.stripIndexFormat = rachis::IndexFormat::uint32;
    auto uint32Strip = device.createGraphicsPipeline(uint32Desc);
    auto lineDesc = uint16Desc;
    lineDesc.topology = rachis::PrimitiveTopology::lineStrip;
    auto lineStrip = device.createGraphicsPipeline(lineDesc);
    REQUIRE(unrestartedStrip);
    REQUIRE(uint16Strip);
    REQUIRE(uint32Strip);
    REQUIRE(lineStrip);
    auto listDesc = uint16Desc;
    listDesc.topology = rachis::PrimitiveTopology::triangleList;
    auto list = device.createGraphicsPipeline(listDesc);
    REQUIRE_FALSE(list);
    CHECK(list.error().code == rachis::ErrorCode::invalidArgument);
    auto invalidFormatDesc = uint16Desc;
    invalidFormatDesc.stripIndexFormat = static_cast<rachis::IndexFormat>(255);
    auto invalidFormat = device.createGraphicsPipeline(invalidFormatDesc);
    REQUIRE_FALSE(invalidFormat);
    CHECK(invalidFormat.error().code == rachis::ErrorCode::invalidArgument);

    struct StripVertex final
    {
        float position[2];
        float color[3];
    };
    const std::array vertices{
        StripVertex{{-0.9F, -0.6F}, {1.0F, 0.0F, 0.0F}},
        StripVertex{{-0.5F, 0.6F}, {1.0F, 0.0F, 0.0F}},
        StripVertex{{-0.1F, -0.6F}, {1.0F, 0.0F, 0.0F}},
        StripVertex{{0.1F, -0.6F}, {0.0F, 1.0F, 0.0F}},
        StripVertex{{0.5F, 0.6F}, {0.0F, 1.0F, 0.0F}},
        StripVertex{{0.9F, -0.6F}, {0.0F, 1.0F, 0.0F}},
    };
    const std::array<std::uint16_t, 7> uint16Indices{0, 1, 2, 0xFFFFU, 3, 4, 5};
    const std::array<std::uint32_t, 7> uint32Indices{0, 1, 2, 0xFFFFFFFFU, 3, 4, 5};
    const rachis::DrawIndexedIndirectArguments indirectArguments{
        .indexCount = static_cast<std::uint32_t>(uint16Indices.size()),
        .instanceCount = 1,
    };
    auto vertexBuffer = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(vertices),
        .usage = rachis::BufferUsage::vertex,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal restart vertices",
    });
    auto uint16Buffer = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(uint16Indices),
        .usage = rachis::BufferUsage::index,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal restart uint16 indices",
    });
    auto uint32Buffer = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(uint32Indices),
        .usage = rachis::BufferUsage::index,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal restart uint32 indices",
    });
    auto indirectBuffer = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(indirectArguments),
        .usage = rachis::BufferUsage::indirect,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal restart indirect arguments",
    });
    REQUIRE(vertexBuffer);
    REQUIRE(uint16Buffer);
    REQUIRE(uint32Buffer);
    REQUIRE(indirectBuffer);
    REQUIRE(device.writeBuffer(*vertexBuffer, 0, std::as_bytes(std::span{vertices})).has_value());
    REQUIRE(device.writeBuffer(*uint16Buffer, 0, std::as_bytes(std::span{uint16Indices})).has_value());
    REQUIRE(device.writeBuffer(*uint32Buffer, 0, std::as_bytes(std::span{uint32Indices})).has_value());
    REQUIRE(device.writeBuffer(*indirectBuffer, 0, std::as_bytes(std::span{&indirectArguments, 1})).has_value());

    auto color = makeColorTarget(device, "Metal restart color");
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = color.texture,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array colorAttachment{rachis::ColorAttachment{
        .view = color.view,
        .loadOperation = rachis::LoadOperation::clear,
        .clearValue = {.alpha = 1.0F},
    }};
    const std::array vertexBindings{rachis::VertexBufferBinding{*vertexBuffer}};
    REQUIRE(device.beginRendering(*command, {.colorAttachments = colorAttachment}).has_value());

    REQUIRE(device.bindGraphicsPipeline(*command, *unrestartedStrip).has_value());
    bindDefaultDrawState(device, *command);
    REQUIRE(device.bindVertexBuffers(*command, 0, vertexBindings).has_value());
    REQUIRE(device.bindIndexBuffer(*command, *uint16Buffer, rachis::IndexFormat::uint16).has_value());
    auto unrestartedIndexed = device.drawIndexed(*command, static_cast<std::uint32_t>(uint16Indices.size()));
    REQUIRE_FALSE(unrestartedIndexed);
    CHECK(unrestartedIndexed.error().code == rachis::ErrorCode::unsupported);
    auto unrestartedIndirect = device.drawIndexedIndirect(*command, {.argumentBuffer = *indirectBuffer});
    REQUIRE_FALSE(unrestartedIndirect);
    CHECK(unrestartedIndirect.error().code == rachis::ErrorCode::unsupported);
    REQUIRE(device.draw(*command, 3).has_value());

    REQUIRE(device.bindGraphicsPipeline(*command, *uint16Strip).has_value());
    bindDefaultDrawState(device, *command);
    REQUIRE(device.bindVertexBuffers(*command, 0, vertexBindings).has_value());
    auto mismatched16 = device.bindIndexBuffer(*command, *uint32Buffer, rachis::IndexFormat::uint32);
    REQUIRE_FALSE(mismatched16);
    CHECK(mismatched16.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(device.bindIndexBuffer(*command, *uint16Buffer, rachis::IndexFormat::uint16).has_value());
    REQUIRE(device.drawIndexed(*command, static_cast<std::uint32_t>(uint16Indices.size())).has_value());
    REQUIRE(device.drawIndexedIndirect(*command, {.argumentBuffer = *indirectBuffer}).has_value());

    REQUIRE(device.bindGraphicsPipeline(*command, *uint32Strip).has_value());
    bindDefaultDrawState(device, *command);
    REQUIRE(device.bindVertexBuffers(*command, 0, vertexBindings).has_value());
    auto staleIndirect = device.drawIndexedIndirect(*command, {.argumentBuffer = *indirectBuffer});
    REQUIRE_FALSE(staleIndirect);
    CHECK(staleIndirect.error().code == rachis::ErrorCode::invalidState);
    auto mismatched32 = device.bindIndexBuffer(*command, *uint16Buffer, rachis::IndexFormat::uint16);
    REQUIRE_FALSE(mismatched32);
    CHECK(mismatched32.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(device.bindIndexBuffer(*command, *uint32Buffer, rachis::IndexFormat::uint32).has_value());
    REQUIRE(device.drawIndexed(*command, static_cast<std::uint32_t>(uint32Indices.size())).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    copyColorToReadback(device, *command, color);
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());

    const auto bytes = readTarget(device, color);
    CHECK(pixelNear(pixelAt(bytes, color.layout, targetSize / 4, targetSize / 2), {255, 0, 0, 255}));
    CHECK(pixelNear(pixelAt(bytes, color.layout, 3 * targetSize / 4, targetSize / 2), {0, 255, 0, 255}));
    CHECK(pixelNear(pixelAt(bytes, color.layout, targetSize / 2, targetSize / 2), {0, 0, 0, 255}));
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal ignores depth state on pipelines without a depth-stencil format",
          "[size:large][rhi][metal][graphics][gpu][depth][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    auto layout = device.createPipelineLayout({});
    REQUIRE(layout);
    const rachis::DepthStencilState depthWithoutTarget{
        .depthTestEnabled = true,
        .depthWriteEnabled = true,
        .depthCompare = rachis::CompareOperation::never,
    };
    auto pipeline = makeTrianglePipeline(device, *layout, rachis::CullMode::none, {}, depthWithoutTarget);
    auto color = makeColorTarget(device, "Metal depthless color");

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = color.texture,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array colorAttachment{rachis::ColorAttachment{
        .view = color.view,
        .loadOperation = rachis::LoadOperation::clear,
        .clearValue = {.alpha = 1.0F},
    }};
    REQUIRE(device.beginRendering(*command, {.colorAttachments = colorAttachment}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, pipeline).has_value());
    bindDefaultDrawState(device, *command);
    REQUIRE(device.draw(*command, 3).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    copyColorToReadback(device, *command, color);
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    CHECK(coloredPixelCount(readTarget(device, color), color.layout) > 200);
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal clearing one attachment preserves earlier clears of other attachments",
          "[size:large][rhi][metal][raster][gpu][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const std::array targets{makeColorTarget(device, "clear A"), makeColorTarget(device, "clear B")};
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    for (const auto& target : targets)
    {
        REQUIRE(device.textureBarrier(*command,
                                      {
                                          .texture = target.texture,
                                          .stagesAfter = rachis::PipelineStage::renderTarget,
                                          .accessAfter = rachis::Access::renderTarget,
                                          .layoutAfter = rachis::TextureLayout::renderTarget,
                                      }));
    }
    const std::array attachments{
        rachis::ColorAttachment{
            .view = targets[0].view, .loadOperation = rachis::LoadOperation::clear, .clearValue = {.alpha = 1.0F}},
        rachis::ColorAttachment{
            .view = targets[1].view, .loadOperation = rachis::LoadOperation::clear, .clearValue = {.alpha = 1.0F}},
    };
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}));
    REQUIRE(device.clearColorAttachment(*command, 0, {.red = 1.0F, .alpha = 1.0F}, {}));
    REQUIRE(device.clearColorAttachment(*command, 1, {.green = 1.0F, .alpha = 1.0F}, {}));
    REQUIRE(device.endRendering(*command));
    for (const auto& target : targets)
    {
        copyColorToReadback(device, *command, target);
    }
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted));
    const auto a = readTarget(device, targets[0]);
    const auto b = readTarget(device, targets[1]);
    CHECK(pixelNear(pixelAt(a, targets[0].layout, 32, 32), {255, 0, 0, 255}));
    CHECK(pixelNear(pixelAt(b, targets[1].layout, 32, 32), {0, 255, 0, 255}));
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal mid-pass clears keep the depth of an attachment that discards at the end of the pass",
          "[size:large][rhi][metal][raster][gpu][depth][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    auto layout = device.createPipelineLayout({});
    REQUIRE(layout);
    const rachis::DepthStencilState depthState{
        .format = rachis::Format::d32Float,
        .depthTestEnabled = true,
        .depthWriteEnabled = true,
        .depthCompare = rachis::CompareOperation::less,
    };
    auto pipeline = makeTrianglePipeline(device, *layout, rachis::CullMode::back, {}, depthState);
    auto color = makeColorTarget(device, "Metal discarded depth color");
    auto depthTexture = device.createTexture(rachis::TextureDesc{
        .width = targetSize,
        .height = targetSize,
        .format = rachis::Format::d32Float,
        .usage = rachis::TextureUsage::depthStencil,
        .label = "Metal discarded depth",
    });
    REQUIRE(depthTexture);
    auto depthView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *depthTexture,
        .usage = rachis::TextureViewUsage::depthStencil,
        .dimension = rachis::TextureViewDimension::texture2D,
        .aspect = rachis::TextureAspect::depth,
    });
    REQUIRE(depthView);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device.textureBarrier(*command,
                                  rachis::TextureBarrier{
                                      .texture = color.texture,
                                      .stagesAfter = rachis::PipelineStage::renderTarget,
                                      .accessAfter = rachis::Access::renderTarget,
                                      .layoutAfter = rachis::TextureLayout::renderTarget,
                                  }));
    REQUIRE(device.textureBarrier(*command,
                                  rachis::TextureBarrier{
                                      .texture = *depthTexture,
                                      .stagesAfter = rachis::PipelineStage::depthStencil,
                                      .accessAfter = rachis::Access::depthStencilWrite,
                                      .layoutAfter = rachis::TextureLayout::depthStencilWrite,
                                      .subresources = {.aspect = rachis::TextureAspect::depth},
                                  }));
    const std::array colorAttachment{rachis::ColorAttachment{
        .view = color.view,
        .loadOperation = rachis::LoadOperation::clear,
        .clearValue = {.alpha = 1.0F},
    }};
    const rachis::DepthStencilAttachment primedDepth{
        .view = *depthView,
        .depthLoadOperation = rachis::LoadOperation::clear,
        .depthClearValue = 1.0F,
    };
    REQUIRE(
        device.beginRendering(*command, {.colorAttachments = colorAttachment, .depthStencilAttachment = primedDepth}));
    REQUIRE(device.endRendering(*command));
    REQUIRE(device.textureBarrier(*command,
                                  rachis::TextureBarrier{
                                      .texture = *depthTexture,
                                      .stagesBefore = rachis::PipelineStage::depthStencil,
                                      .stagesAfter = rachis::PipelineStage::depthStencil,
                                      .accessBefore = rachis::Access::depthStencilWrite,
                                      .accessAfter = rachis::Access::depthStencilWrite,
                                      .layoutBefore = rachis::TextureLayout::depthStencilWrite,
                                      .layoutAfter = rachis::TextureLayout::depthStencilWrite,
                                      .subresources = {.aspect = rachis::TextureAspect::depth},
                                  }));
    const rachis::DepthStencilAttachment discardedDepth{
        .view = *depthView,
        .depthLoadOperation = rachis::LoadOperation::clear,
        .depthStoreOperation = rachis::StoreOperation::discard,
        .depthClearValue = 1.0F,
    };
    REQUIRE(device.beginRendering(*command,
                                  {.colorAttachments = colorAttachment, .depthStencilAttachment = discardedDepth}));
    REQUIRE(device.bindGraphicsPipeline(*command, pipeline));
    REQUIRE(device.setViewports(*command,
                                std::array{rachis::Viewport{
                                    .width = static_cast<float>(targetSize),
                                    .height = static_cast<float>(targetSize),
                                    .minDepth = 0.2F,
                                    .maxDepth = 0.2F,
                                }}));
    REQUIRE(device.setScissors(*command, std::array{rachis::ScissorRect{.width = targetSize, .height = targetSize}}));
    REQUIRE(device.draw(*command, 3, 1, 0, 0));
    REQUIRE(device.clearColorAttachment(*command, 0, {.green = 1.0F, .alpha = 1.0F}));
    REQUIRE(device.setViewports(*command,
                                std::array{rachis::Viewport{
                                    .width = static_cast<float>(targetSize),
                                    .height = static_cast<float>(targetSize),
                                    .minDepth = 0.8F,
                                    .maxDepth = 0.8F,
                                }}));
    REQUIRE(device.draw(*command, 3, 1, 0, 0));
    REQUIRE(device.endRendering(*command));
    copyColorToReadback(device, *command, color);
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted));
    const auto bytes = readTarget(device, color);
    CHECK(pixelNear(pixelAt(bytes, color.layout, targetSize / 2, targetSize / 2 + 8), {0, 255, 0, 255}));
    CHECK(pixelNear(pixelAt(bytes, color.layout, 0, 0), {0, 255, 0, 255}));
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal attachment clears validate rectangles and clear a target that several rectangles cover",
          "[size:large][rhi][metal][raster][gpu][validation]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    auto color = makeColorTarget(device, "Metal clear rectangle color");
    auto depthTexture = device.createTexture(rachis::TextureDesc{
        .width = targetSize,
        .height = targetSize,
        .format = rachis::Format::d32Float,
        .usage = rachis::TextureUsage::depthStencil,
        .label = "Metal clear rectangle depth",
    });
    REQUIRE(depthTexture);
    auto depthView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *depthTexture,
        .usage = rachis::TextureViewUsage::depthStencil,
        .dimension = rachis::TextureViewDimension::texture2D,
        .aspect = rachis::TextureAspect::depth,
    });
    REQUIRE(depthView);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device.textureBarrier(*command,
                                  rachis::TextureBarrier{
                                      .texture = color.texture,
                                      .stagesAfter = rachis::PipelineStage::renderTarget,
                                      .accessAfter = rachis::Access::renderTarget,
                                      .layoutAfter = rachis::TextureLayout::renderTarget,
                                  }));
    REQUIRE(device.textureBarrier(*command,
                                  rachis::TextureBarrier{
                                      .texture = *depthTexture,
                                      .stagesAfter = rachis::PipelineStage::depthStencil,
                                      .accessAfter = rachis::Access::depthStencilWrite,
                                      .layoutAfter = rachis::TextureLayout::depthStencilWrite,
                                      .subresources = {.aspect = rachis::TextureAspect::depth},
                                  }));
    const std::array colorAttachment{rachis::ColorAttachment{
        .view = color.view,
        .loadOperation = rachis::LoadOperation::clear,
        .clearValue = {.red = 1.0F, .alpha = 1.0F},
    }};
    const rachis::DepthStencilAttachment depthAttachment{
        .view = *depthView,
        .depthLoadOperation = rachis::LoadOperation::clear,
        .depthClearValue = 1.0F,
    };
    REQUIRE(device.beginRendering(*command,
                                  {.colorAttachments = colorAttachment, .depthStencilAttachment = depthAttachment}));

    const std::array negative{rachis::ScissorRect{.x = -1, .width = 1, .height = 1}};
    const std::array empty{rachis::ScissorRect{.x = 1, .y = 1, .height = 1}};
    const std::array outside{rachis::ScissorRect{.x = targetSize - 1, .width = 2, .height = 1}};
    const std::array partial{rachis::ScissorRect{.x = 1, .y = 1, .width = 2, .height = 2}};
    const std::array gap{
        rachis::ScissorRect{.width = targetSize / 2, .height = targetSize},
        rachis::ScissorRect{.x = targetSize / 2 + 1, .width = targetSize / 2 - 1, .height = targetSize},
    };
    const std::array halves{
        rachis::ScissorRect{.width = targetSize / 2 + 4, .height = targetSize},
        rachis::ScissorRect{
            .x = targetSize / 2, .y = targetSize / 2, .width = targetSize / 2, .height = targetSize / 2},
        rachis::ScissorRect{.x = targetSize / 2, .width = targetSize / 2, .height = targetSize / 2},
    };
    for (const auto rectangles : {std::span<const rachis::ScissorRect>{negative},
                                  std::span<const rachis::ScissorRect>{empty},
                                  std::span<const rachis::ScissorRect>{outside}})
    {
        auto colorClear = device.clearColorAttachment(*command, 0, {.green = 1.0F, .alpha = 1.0F}, rectangles);
        REQUIRE_FALSE(colorClear);
        CHECK(colorClear.error().code == rachis::ErrorCode::invalidArgument);
        auto depthClear =
            device.clearDepthStencilAttachment(*command, rachis::DepthStencilClearAspect::depth, 0.5F, 0, rectangles);
        REQUIRE_FALSE(depthClear);
        CHECK(depthClear.error().code == rachis::ErrorCode::invalidArgument);
    }
    for (const auto rectangles :
         {std::span<const rachis::ScissorRect>{partial}, std::span<const rachis::ScissorRect>{gap}})
    {
        auto colorClear = device.clearColorAttachment(*command, 0, {.green = 1.0F, .alpha = 1.0F}, rectangles);
        REQUIRE_FALSE(colorClear);
        CHECK(colorClear.error().code == rachis::ErrorCode::unsupported);
        auto depthClear =
            device.clearDepthStencilAttachment(*command, rachis::DepthStencilClearAspect::depth, 0.5F, 0, rectangles);
        REQUIRE_FALSE(depthClear);
        CHECK(depthClear.error().code == rachis::ErrorCode::unsupported);
    }
    REQUIRE(device.clearColorAttachment(*command, 0, {.green = 1.0F, .alpha = 1.0F}, halves));
    REQUIRE(device.clearDepthStencilAttachment(*command, rachis::DepthStencilClearAspect::depth, 0.5F, 0, halves));
    REQUIRE(device.endRendering(*command));
    copyColorToReadback(device, *command, color);
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted));
    const auto bytes = readTarget(device, color);
    for (const auto& [x, y] : std::array{std::array{0U, 0U},
                                         std::array{targetSize - 1, 0U},
                                         std::array{0U, targetSize - 1},
                                         std::array{targetSize - 1, targetSize - 1}})
    {
        CHECK(pixelNear(pixelAt(bytes, color.layout, x, y), {0, 255, 0, 255}));
    }
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal mid-pass clears reapply scissors clipped to the render target",
          "[size:large][rhi][metal][raster][gpu][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    auto layout = device.createPipelineLayout({});
    REQUIRE(layout);
    auto pipeline = makeTrianglePipeline(device, *layout);
    auto color = makeColorTarget(device, "Metal clipped scissor color");
    const std::array attachment{rachis::ColorAttachment{
        .view = color.view,
        .loadOperation = rachis::LoadOperation::clear,
        .clearValue = {.alpha = 1.0F},
    }};
    const auto render = [&](std::span<const rachis::ScissorRect> scissors, bool clearFirst) {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command);
        REQUIRE(device.textureBarrier(*command,
                                      rachis::TextureBarrier{
                                          .texture = color.texture,
                                          .stagesAfter = rachis::PipelineStage::renderTarget,
                                          .accessAfter = rachis::Access::renderTarget,
                                          .layoutAfter = rachis::TextureLayout::renderTarget,
                                      }));
        REQUIRE(device.beginRendering(*command, {.colorAttachments = attachment}));
        REQUIRE(device.bindGraphicsPipeline(*command, pipeline));
        bindDefaultDrawState(device, *command);
        for (const auto& scissor : scissors)
        {
            REQUIRE(device.setScissor(*command, scissor));
        }
        if (clearFirst)
        {
            REQUIRE(device.clearColorAttachment(*command, 0, {.alpha = 1.0F}));
        }
        REQUIRE(device.draw(*command, 3, 1, 0, 0));
        REQUIRE(device.endRendering(*command));
        copyColorToReadback(device, *command, color);
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted);
        REQUIRE(device.wait(*submitted));
        return readTarget(device, color);
    };
    const auto coloredIn = [&](const std::vector<std::byte>& bytes, std::uint32_t left, std::uint32_t top) {
        std::size_t count = 0;
        for (std::uint32_t y = top; y < targetSize; ++y)
        {
            for (std::uint32_t x = left; x < targetSize; ++x)
            {
                const auto* pixel = pixelAt(bytes, color.layout, x, y);
                if (std::to_integer<std::uint8_t>(pixel[0]) != 0 || std::to_integer<std::uint8_t>(pixel[1]) != 0 ||
                    std::to_integer<std::uint8_t>(pixel[2]) != 0)
                {
                    ++count;
                }
            }
        }
        return count;
    };

    const auto reference = render({}, false);
    const auto quadrant = coloredIn(reference, targetSize / 2, targetSize / 2);
    REQUIRE(quadrant > 0);

    const std::array partlyOutside{
        rachis::ScissorRect{.x = targetSize / 2, .y = targetSize / 2, .width = targetSize, .height = targetSize}};
    const auto clipped = render(partlyOutside, true);
    CHECK(coloredPixelCount(clipped, color.layout) == quadrant);
    CHECK(coloredIn(clipped, targetSize / 2, targetSize / 2) == quadrant);

    const std::array oversized{rachis::ScissorRect{.width = 4096, .height = 4096}};
    const auto whole = render(oversized, true);
    CHECK(coloredPixelCount(whole, color.layout) == coloredPixelCount(reference, color.layout));

    const std::array outside{
        rachis::ScissorRect{.x = targetSize, .y = targetSize, .width = targetSize / 4, .height = targetSize / 4}};
    const auto culled = render(outside, true);
    CHECK(coloredPixelCount(culled, color.layout) == 0);

    const std::array outsideThenOversized{outside[0], oversized[0]};
    const auto restored = render(outsideThenOversized, true);
    CHECK(coloredPixelCount(restored, color.layout) == coloredPixelCount(reference, color.layout));
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal depth and stencil clear values and stencil references follow the portable ranges",
          "[size:large][rhi][metal][graphics][validation]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    auto color = makeColorTarget(device, "Metal value range color");
    auto depthTexture = device.createTexture(rachis::TextureDesc{
        .width = targetSize,
        .height = targetSize,
        .format = rachis::Format::d32Float,
        .usage = rachis::TextureUsage::depthStencil,
        .label = "Metal value range depth",
    });
    REQUIRE(depthTexture);
    auto depthView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *depthTexture,
        .usage = rachis::TextureViewUsage::depthStencil,
        .dimension = rachis::TextureViewDimension::texture2D,
        .aspect = rachis::TextureAspect::depth,
    });
    REQUIRE(depthView);
    const std::array colorAttachment{rachis::ColorAttachment{.view = color.view}};

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    auto outside = device.setStencilReference(*command, rachis::StencilFace::frontAndBack, 7);
    REQUIRE_FALSE(outside);
    CHECK(outside.error().code == rachis::ErrorCode::invalidState);
    auto outsideFace = device.setStencilReference(*command, static_cast<rachis::StencilFace>(255), 7);
    REQUIRE_FALSE(outsideFace);
    CHECK(outsideFace.error().code == rachis::ErrorCode::invalidArgument);

    const auto rejectedBegin = [&](const rachis::DepthStencilAttachment& attachment) {
        auto begun = device.beginRendering(*command,
                                           {.colorAttachments = colorAttachment, .depthStencilAttachment = attachment});
        REQUIRE_FALSE(begun);
        CHECK(begun.error().code == rachis::ErrorCode::invalidArgument);
    };
    rejectedBegin({
        .view = *depthView,
        .depthLoadOperation = rachis::LoadOperation::clear,
        .depthClearValue = 1.5F,
    });
    rejectedBegin({
        .view = *depthView,
        .depthLoadOperation = rachis::LoadOperation::clear,
        .depthClearValue = -0.25F,
    });
    rejectedBegin({
        .view = *depthView,
        .depthLoadOperation = rachis::LoadOperation::clear,
        .depthClearValue = std::numeric_limits<float>::quiet_NaN(),
    });
    rejectedBegin({
        .view = *depthView,
        .stencilClearValue = 256,
    });

    REQUIRE(device.beginRendering(*command,
                                  {
                                      .colorAttachments = colorAttachment,
                                      .depthStencilAttachment =
                                          rachis::DepthStencilAttachment{
                                              .view = *depthView,
                                              .depthClearValue = std::numeric_limits<float>::quiet_NaN(),
                                          },
                                  }));

    auto badFace = device.setStencilReference(*command, static_cast<rachis::StencilFace>(255), 7);
    REQUIRE_FALSE(badFace);
    CHECK(badFace.error().code == rachis::ErrorCode::invalidArgument);
    auto wideReference = device.setStencilReference(*command, rachis::StencilFace::frontAndBack, 256);
    REQUIRE_FALSE(wideReference);
    CHECK(wideReference.error().code == rachis::ErrorCode::invalidArgument);
    auto frontOnly = device.setStencilReference(*command, rachis::StencilFace::front, 7);
    REQUIRE_FALSE(frontOnly);
    CHECK(frontOnly.error().code == rachis::ErrorCode::unsupported);
    REQUIRE(device.setStencilReference(*command, rachis::StencilFace::frontAndBack, 255));

    const auto rejectedClear =
        [&](rachis::DepthStencilClearAspect aspects, float depth, std::uint32_t stencil, rachis::ErrorCode expected) {
            auto cleared = device.clearDepthStencilAttachment(*command, aspects, depth, stencil);
            REQUIRE_FALSE(cleared);
            CHECK(cleared.error().code == expected);
        };
    rejectedClear(rachis::DepthStencilClearAspect::none, 0.5F, 0, rachis::ErrorCode::invalidArgument);
    rejectedClear(static_cast<rachis::DepthStencilClearAspect>(1U << 7U), 0.5F, 0, rachis::ErrorCode::invalidArgument);
    rejectedClear(rachis::DepthStencilClearAspect::depth | static_cast<rachis::DepthStencilClearAspect>(1U << 7U),
                  0.5F,
                  0,
                  rachis::ErrorCode::invalidArgument);
    rejectedClear(rachis::DepthStencilClearAspect::depth, 1.5F, 0, rachis::ErrorCode::invalidArgument);
    rejectedClear(rachis::DepthStencilClearAspect::depth, -0.5F, 0, rachis::ErrorCode::invalidArgument);
    rejectedClear(rachis::DepthStencilClearAspect::depth,
                  std::numeric_limits<float>::infinity(),
                  0,
                  rachis::ErrorCode::invalidArgument);
    rejectedClear(rachis::DepthStencilClearAspect::stencil, 0.0F, 256, rachis::ErrorCode::invalidArgument);
    rejectedClear(rachis::DepthStencilClearAspect::stencil,
                  std::numeric_limits<float>::quiet_NaN(),
                  1,
                  rachis::ErrorCode::unsupported);
    REQUIRE(device.clearDepthStencilAttachment(
        *command, rachis::DepthStencilClearAspect::depth, 0.5F, std::numeric_limits<std::uint32_t>::max()));
    REQUIRE(device.clearDepthStencilAttachment(*command, rachis::DepthStencilClearAspect::depth, 1.0F, 0));
    REQUIRE(device.endRendering(*command));
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted));
    CHECK(device.diagnosticCounts().errors == 0);
}
