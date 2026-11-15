#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <utility>
#include <vector>

namespace
{

    constexpr std::uint32_t targetWidth = 64;
    constexpr std::uint32_t targetHeight = 64;
    constexpr std::uint32_t rowPitch = 256;
    constexpr std::uint64_t readbackSize = static_cast<std::uint64_t>(rowPitch) * targetHeight;

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

    [[nodiscard]] bool nearByte(std::byte actual, std::uint8_t expected) noexcept
    {
        const auto value = std::to_integer<std::uint8_t>(actual);
        const auto lower = expected == 0 ? 0U : static_cast<std::uint32_t>(expected) - 1U;
        const auto upper = (std::min)(static_cast<std::uint32_t>(expected) + 1U, 255U);
        return value >= lower && value <= upper;
    }

    [[nodiscard]] const std::byte* pixelAt(const std::byte* data, std::uint32_t x, std::uint32_t y) noexcept
    {
        return data + static_cast<std::uint64_t>(y) * rowPitch + static_cast<std::uint64_t>(x) * 4;
    }

    [[nodiscard]] bool insideRect(const rachis::ScissorRect& rect, std::uint32_t x, std::uint32_t y) noexcept
    {
        const auto left = static_cast<std::uint32_t>(rect.x);
        const auto top = static_cast<std::uint32_t>(rect.y);
        return x >= left && x - left < rect.width && y >= top && y - top < rect.height;
    }

    [[nodiscard]] bool colorLayerMatches(const std::byte* layer,
                                         const rachis::ScissorRect& rect,
                                         std::array<std::byte, 4> inside,
                                         std::array<std::byte, 4> outside) noexcept
    {
        for (std::uint32_t y = 0; y < targetHeight; ++y)
        {
            for (std::uint32_t x = 0; x < targetWidth; ++x)
            {
                const auto& expected = insideRect(rect, x, y) ? inside : outside;
                if (!std::ranges::equal(std::span{pixelAt(layer, x, y), expected.size()}, expected))
                {
                    return false;
                }
            }
        }
        return true;
    }

    [[nodiscard]] bool depthLayerMatches(const std::byte* layer,
                                         const rachis::ScissorRect& rect,
                                         float inside,
                                         float outside) noexcept
    {
        for (std::uint32_t y = 0; y < targetHeight; ++y)
        {
            for (std::uint32_t x = 0; x < targetWidth; ++x)
            {
                float value = 0.0F;
                std::memcpy(&value, pixelAt(layer, x, y), sizeof(value));
                if (value != (insideRect(rect, x, y) ? inside : outside))
                {
                    return false;
                }
            }
        }
        return true;
    }

    [[nodiscard]] rachis::Result<rachis::PipelineHandle> createInstancedInputPipeline(
        rachis::Device& device,
        rachis::PipelineLayoutHandle layout,
        rachis::ShaderModuleHandle vertexShader,
        rachis::ShaderModuleHandle pixelShader,
        std::uint32_t instanceStride,
        std::span<const rachis::VertexAttribute> instanceAttributes)
    {
        const std::array positionAttributes{
            rachis::VertexAttribute{.format = rachis::VertexFormat::float32x2, .shaderLocation = 0},
        };
        const std::array vertexLayouts{
            rachis::VertexBufferLayout{.stride = 2 * sizeof(float), .attributes = positionAttributes},
            rachis::VertexBufferLayout{
                .stride = instanceStride,
                .stepMode = rachis::VertexStepMode::instance,
                .attributes = instanceAttributes,
            },
        };
        const std::array colorTargets{
            rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
        };
        return device.createGraphicsPipeline(rachis::GraphicsPipelineDesc{
            .layout = layout,
            .vertexShader = vertexShader,
            .pixelShader = pixelShader,
            .vertexBuffers = vertexLayouts,
            .cullMode = rachis::CullMode::none,
            .colorTargets = colorTargets,
        });
    }

    [[nodiscard]] rachis::Result<rachis::BufferHandle> createFilledVertexBuffer(rachis::Device& device,
                                                                                std::span<const std::byte> bytes)
    {
        auto buffer = device.createBuffer(rachis::BufferDesc{
            .size = bytes.size(),
            .usage = rachis::BufferUsage::vertex,
            .memoryType = rachis::MemoryType::upload,
        });
        if (!buffer)
        {
            return buffer;
        }
        auto mapped = device.mapBuffer(*buffer);
        if (!mapped)
        {
            static_cast<void>(device.destroyBuffer(*buffer));
            return std::unexpected(std::move(mapped.error()));
        }
        std::memcpy(mapped->data, bytes.data(), bytes.size());
        if (auto unmapped = device.unmapBuffer(*buffer); !unmapped)
        {
            static_cast<void>(device.destroyBuffer(*buffer));
            return std::unexpected(std::move(unmapped.error()));
        }
        return buffer;
    }

} // namespace

TEST_CASE("Vulkan offscreen rendering executes direct and indexed draws with Y-up viewports",
          "[size:medium][rhi][vulkan][graphics]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    const auto directVertexBytecode =
        loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanGraphicsTriangle.direct.vs.spv");
    const auto inputVertexBytecode =
        loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanGraphicsTriangle.input.vs.spv");
    const auto pixelBytecode =
        loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanGraphicsTriangle.ps.spv");
    REQUIRE_FALSE(directVertexBytecode.empty());
    REQUIRE_FALSE(inputVertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());

    auto directVertexShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = directVertexBytecode,
        .entryPoint = "directVertex",
        .label = "Vulkan direct triangle vertex",
        .shaderInterface = {},
    });
    auto inputVertexShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = inputVertexBytecode,
        .entryPoint = "inputVertex",
        .label = "Vulkan input triangle vertex",
        .shaderInterface = rachis::test::graphicsInputVertexShaderInterface(),
    });
    auto pixelShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "pixelMain",
        .label = "Vulkan triangle pixel",
        .shaderInterface = rachis::test::graphicsPixelShaderInterface(),
    });
    REQUIRE(directVertexShader.has_value());
    REQUIRE(inputVertexShader.has_value());
    REQUIRE(pixelShader.has_value());

    const std::array pushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::pixel, 0, 4 * sizeof(float)},
    };
    auto pipelineLayout = device.createPipelineLayout(rachis::PipelineLayoutDesc{
        .pushConstantRanges = pushConstantRanges,
        .label = "Vulkan graphics triangle pipeline layout",
    });
    REQUIRE(pipelineLayout.has_value());

    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    auto directPipeline = device.createGraphicsPipeline(rachis::GraphicsPipelineDesc{
        .layout = *pipelineLayout,
        .vertexShader = *directVertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .depthStencil =
            {
                .format = rachis::Format::d32Float,
                .depthTestEnabled = true,
                .depthWriteEnabled = true,
                .depthCompare = rachis::CompareOperation::lessEqual,
            },
        .label = "Vulkan direct triangle pipeline",
    });
    REQUIRE(directPipeline.has_value());

    const std::array vertexAttributes{
        rachis::VertexAttribute{
            .format = rachis::VertexFormat::float32x2,
            .offset = 0,
            .shaderLocation = 0,
        },
    };
    const std::array vertexLayouts{
        rachis::VertexBufferLayout{
            .stride = 2 * sizeof(float),
            .attributes = vertexAttributes,
        },
    };
    auto indexedPipeline = device.createGraphicsPipeline(rachis::GraphicsPipelineDesc{
        .layout = *pipelineLayout,
        .vertexShader = *inputVertexShader,
        .pixelShader = *pixelShader,
        .vertexBuffers = vertexLayouts,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .depthStencil =
            {
                .format = rachis::Format::d32Float,
                .depthTestEnabled = true,
                .depthWriteEnabled = true,
                .depthCompare = rachis::CompareOperation::lessEqual,
            },
        .label = "Vulkan indexed triangle pipeline",
    });
    REQUIRE(indexedPipeline.has_value());

    const std::array incompatibleTargets{
        rachis::ColorTargetState{.format = rachis::Format::bgra8Unorm},
    };
    auto incompatiblePipeline = device.createGraphicsPipeline(rachis::GraphicsPipelineDesc{
        .layout = *pipelineLayout,
        .vertexShader = *directVertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = incompatibleTargets,
        .depthStencil = {.format = rachis::Format::d32Float},
    });
    REQUIRE(incompatiblePipeline.has_value());

    auto target = device.createTexture(rachis::TextureDesc{
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
        .label = "Vulkan graphics contract target",
    });
    REQUIRE(target.has_value());
    auto targetView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
        .label = "Vulkan graphics contract target view",
    });
    REQUIRE(targetView.has_value());
    auto depth = device.createTexture(rachis::TextureDesc{
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::d32Float,
        .usage = rachis::TextureUsage::depthStencil,
        .label = "Vulkan graphics contract depth",
    });
    REQUIRE(depth.has_value());
    auto depthView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *depth,
        .usage = rachis::TextureViewUsage::depthStencil,
        .aspect = rachis::TextureAspect::depth,
        .label = "Vulkan graphics contract depth view",
    });
    REQUIRE(depthView.has_value());
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = readbackSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan graphics contract readback",
    });
    REQUIRE(readback.has_value());

    const std::array vertices{
        std::array{0.0F, 0.75F},
        std::array{0.75F, -0.75F},
        std::array{-0.75F, -0.75F},
    };
    const std::array<std::uint16_t, 3> indices{0, 1, 2};
    auto vertexBuffer = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(vertices),
        .usage = rachis::BufferUsage::vertex,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan graphics contract vertices",
    });
    auto indexBuffer = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(indices),
        .usage = rachis::BufferUsage::index,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan graphics contract indices",
    });
    REQUIRE(vertexBuffer.has_value());
    REQUIRE(indexBuffer.has_value());
    auto mappedVertices = device.mapBuffer(*vertexBuffer);
    auto mappedIndices = device.mapBuffer(*indexBuffer);
    REQUIRE(mappedVertices.has_value());
    REQUIRE(mappedIndices.has_value());
    std::memcpy(mappedVertices->data, vertices.data(), sizeof(vertices));
    std::memcpy(mappedIndices->data, indices.data(), sizeof(indices));
    REQUIRE(device.unmapBuffer(*vertexBuffer).has_value());
    REQUIRE(device.unmapBuffer(*indexBuffer).has_value());

    auto copyCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(copyCommand.has_value());
    const std::array colorAttachments{
        rachis::ColorAttachment{
            .view = *targetView,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.0F, 0.0F, 0.0F, 0.0F},
        },
    };
    const rachis::DepthStencilAttachment depthAttachment{
        .view = *depthView,
        .depthLoadOperation = rachis::LoadOperation::clear,
        .depthClearValue = 1.0F,
    };
    CHECK_FALSE(device
                    .beginRendering(*copyCommand,
                                    rachis::RenderingDesc{
                                        .colorAttachments = colorAttachments,
                                        .depthStencilAttachment = depthAttachment,
                                    })
                    .has_value());

    auto unfinishedCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(unfinishedCommand.has_value());
    REQUIRE(device
                .beginRendering(*unfinishedCommand,
                                rachis::RenderingDesc{
                                    .colorAttachments = colorAttachments,
                                    .depthStencilAttachment = depthAttachment,
                                })
                .has_value());
    CHECK_FALSE(device.submit(std::move(*unfinishedCommand)).has_value());
    CHECK(unfinishedCommand->isValid());
    REQUIRE(device.endRendering(*unfinishedCommand).has_value());

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
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *depth,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::depthStencil,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::depthStencilWrite,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::depthStencilWrite,
                                    .subresources = {.aspect = rachis::TextureAspect::depth},
                                })
                .has_value());
    auto invalidColorAttachments = colorAttachments;
    invalidColorAttachments[0].storeOperation = static_cast<rachis::StoreOperation>(255);
    CHECK_FALSE(device
                    .beginRendering(*command,
                                    rachis::RenderingDesc{
                                        .colorAttachments = invalidColorAttachments,
                                    })
                    .has_value());
    REQUIRE(device
                .beginRendering(*command,
                                rachis::RenderingDesc{
                                    .colorAttachments = colorAttachments,
                                    .depthStencilAttachment = depthAttachment,
                                })
                .has_value());
    CHECK_FALSE(device
                    .beginRendering(*command,
                                    rachis::RenderingDesc{
                                        .colorAttachments = colorAttachments,
                                        .depthStencilAttachment = depthAttachment,
                                    })
                    .has_value());
    CHECK_FALSE(device.bindGraphicsPipeline(*command, *incompatiblePipeline).has_value());
    CHECK_FALSE(device.draw(*command, 3).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *directPipeline).has_value());
    CHECK_FALSE(device.draw(*command, 3).has_value());
    REQUIRE(device
                .setViewport(*command,
                             rachis::Viewport{
                                 .width = static_cast<float>(targetWidth),
                                 .height = static_cast<float>(targetHeight),
                             })
                .has_value());
    REQUIRE(device
                .setScissor(*command,
                            rachis::ScissorRect{
                                .width = targetWidth,
                                .height = targetHeight,
                            })
                .has_value());
    const std::array tint{0.125F, 0.5F, 0.75F, 1.0F};
    REQUIRE(device.setGraphicsPushConstants(*command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(tint)))
                .has_value());
    REQUIRE(device.draw(*command, 3).has_value());

    REQUIRE(device.bindGraphicsPipeline(*command, *indexedPipeline).has_value());
    CHECK_FALSE(device.drawIndexed(*command, 3).has_value());
    REQUIRE(device.bindVertexBuffer(*command, 0, *vertexBuffer).has_value());
    REQUIRE(device.bindIndexBuffer(*command, *indexBuffer, rachis::IndexFormat::uint16).has_value());
    REQUIRE(device.setGraphicsPushConstants(*command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(tint)))
                .has_value());
    REQUIRE(device
                .setViewport(*command,
                             rachis::Viewport{
                                 .width = static_cast<float>(targetWidth),
                                 .height = static_cast<float>(targetHeight),
                             })
                .has_value());
    REQUIRE(device
                .setScissor(*command,
                            rachis::ScissorRect{
                                .width = targetWidth,
                                .height = targetHeight,
                            })
                .has_value());
    REQUIRE(device.drawIndexed(*command, 3).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    CHECK_FALSE(device.endRendering(*command).has_value());
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

    const auto stalePipeline = *directPipeline;
    REQUIRE(device.destroyPipeline(*directPipeline).has_value());
    REQUIRE(device.destroyPipeline(*indexedPipeline).has_value());
    REQUIRE(device.destroyPipeline(*incompatiblePipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*directVertexShader).has_value());
    REQUIRE(device.destroyShaderModule(*inputVertexShader).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    REQUIRE(device.destroyTextureView(*depthView).has_value());
    REQUIRE(device.destroyTexture(*depth).has_value());
    REQUIRE(device.destroyBuffer(*vertexBuffer).has_value());
    REQUIRE(device.destroyBuffer(*indexBuffer).has_value());
    CHECK_FALSE(device.destroyPipeline(stalePipeline).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    const auto* corner = mapped->data;
    const auto* center = pixelAt(mapped->data, targetWidth / 2, targetHeight / 2);
    const auto* uprightCorner = pixelAt(mapped->data, 14, 52);
    const auto* flippedCorner = pixelAt(mapped->data, 14, 12);
    CHECK(corner[0] == std::byte{0});
    CHECK(corner[1] == std::byte{0});
    CHECK(corner[2] == std::byte{0});
    CHECK(corner[3] == std::byte{0});
    CHECK(nearByte(center[0], 32));
    CHECK(nearByte(center[1], 128));
    CHECK(nearByte(center[2], 191));
    CHECK(nearByte(center[3], 255));
    CHECK(nearByte(uprightCorner[0], 32));
    CHECK(nearByte(uprightCorner[1], 128));
    CHECK(nearByte(uprightCorner[2], 191));
    CHECK(nearByte(uprightCorner[3], 255));
    CHECK(flippedCorner[0] == std::byte{0});
    CHECK(flippedCorner[1] == std::byte{0});
    CHECK(flippedCorner[2] == std::byte{0});
    CHECK(flippedCorner[3] == std::byte{0});
    REQUIRE(device.unmapBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan color attachment clears replace the active render target",
          "[size:medium][rhi][vulkan][graphics][clear]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    auto target = device.createTexture(rachis::TextureDesc{
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
        .label = "Vulkan clear target",
    });
    REQUIRE(target.has_value());
    auto targetView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    REQUIRE(targetView.has_value());
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = readbackSize,
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
    const std::array colorAttachments{
        rachis::ColorAttachment{
            .view = *targetView,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.0F, 0.0F, 0.0F, 1.0F},
        },
    };
    REQUIRE(device.beginRendering(*command, rachis::RenderingDesc{.colorAttachments = colorAttachments}).has_value());
    REQUIRE(device.clearColorAttachment(*command, 0, {1.0F, 0.0F, 0.25F, 1.0F}).has_value());
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

    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    CHECK(nearByte(mapped->data[0], 255));
    CHECK(nearByte(mapped->data[1], 0));
    CHECK(nearByte(mapped->data[2], 64));
    CHECK(nearByte(mapped->data[3], 255));
    REQUIRE(device.unmapBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
}

TEST_CASE("Vulkan attachment clears inside multiview scopes clear only the views in the mask",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][graphics][clear][multiview]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (rachis::test::skipWithoutCapability(capabilities.multiviewRendering, "multiviewRendering"))
    {
        return;
    }
    REQUIRE(capabilities.maxMultiviewViewCount >= 3);

    struct Scenario final
    {
        std::uint32_t layerCount = 0;
        std::uint32_t viewMask = 0;
    };
    constexpr std::array scenarios{
        Scenario{.layerCount = 3, .viewMask = 0b110},
        Scenario{.layerCount = 2, .viewMask = 0b11},
    };
    constexpr std::array blue{std::byte{0}, std::byte{0}, std::byte{0xFF}, std::byte{0xFF}};
    constexpr std::array red{std::byte{0xFF}, std::byte{0}, std::byte{0}, std::byte{0xFF}};
    constexpr std::array green{std::byte{0}, std::byte{0xFF}, std::byte{0}, std::byte{0xFF}};
    const std::array subRects{rachis::ScissorRect{.x = 8, .y = 16, .width = 24, .height = 32}};

    for (const auto& scenario : scenarios)
    {
        INFO("viewMask " << scenario.viewMask << " on " << scenario.layerCount << " layers");
        auto color = device.createTexture(rachis::TextureDesc{
            .width = targetWidth,
            .height = targetHeight,
            .depthOrArrayLayers = scenario.layerCount,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
            .label = "Vulkan multiview clear color",
        });
        auto depth = device.createTexture(rachis::TextureDesc{
            .width = targetWidth,
            .height = targetHeight,
            .depthOrArrayLayers = scenario.layerCount,
            .format = rachis::Format::d32Float,
            .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::depthStencil,
            .label = "Vulkan multiview clear depth",
        });
        REQUIRE(color.has_value());
        REQUIRE(depth.has_value());
        auto colorView = device.createTextureView(rachis::TextureViewDesc{
            .texture = *color,
            .usage = rachis::TextureViewUsage::renderTarget,
            .dimension = rachis::TextureViewDimension::texture2DArray,
            .arrayLayerCount = scenario.layerCount,
        });
        auto depthView = device.createTextureView(rachis::TextureViewDesc{
            .texture = *depth,
            .usage = rachis::TextureViewUsage::depthStencil,
            .dimension = rachis::TextureViewDimension::texture2DArray,
            .aspect = rachis::TextureAspect::depth,
            .arrayLayerCount = scenario.layerCount,
        });
        REQUIRE(colorView.has_value());
        REQUIRE(depthView.has_value());
        auto readback = device.createBuffer(rachis::BufferDesc{
            .size = readbackSize * scenario.layerCount * 2,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
        });
        REQUIRE(readback.has_value());

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = *color,
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
        const std::array prefillAttachments{
            rachis::ColorAttachment{
                .view = *colorView,
                .loadOperation = rachis::LoadOperation::clear,
                .clearValue = {0.0F, 0.0F, 1.0F, 1.0F},
            },
        };
        REQUIRE(device
                    .beginRendering(*command,
                                    rachis::RenderingDesc{
                                        .colorAttachments = prefillAttachments,
                                        .depthStencilAttachment =
                                            rachis::DepthStencilAttachment{
                                                .view = *depthView,
                                                .depthLoadOperation = rachis::LoadOperation::clear,
                                                .depthClearValue = 1.0F,
                                            },
                                    })
                    .has_value());
        REQUIRE(device.endRendering(*command).has_value());
        const std::array loadAttachments{rachis::ColorAttachment{.view = *colorView}};
        REQUIRE(device
                    .beginRendering(*command,
                                    rachis::RenderingDesc{
                                        .colorAttachments = loadAttachments,
                                        .depthStencilAttachment = rachis::DepthStencilAttachment{.view = *depthView},
                                        .viewMask = scenario.viewMask,
                                    })
                    .has_value());
        REQUIRE(device.clearColorAttachment(*command, 0, {1.0F, 0.0F, 0.0F, 1.0F}).has_value());
        REQUIRE(device.clearColorAttachment(*command, 0, {0.0F, 1.0F, 0.0F, 1.0F}, subRects).has_value());
        REQUIRE(
            device.clearDepthStencilAttachment(*command, rachis::DepthStencilClearAspect::depth, 0.5F, 0).has_value());
        REQUIRE(device.clearDepthStencilAttachment(*command, rachis::DepthStencilClearAspect::depth, 0.25F, 0, subRects)
                    .has_value());
        REQUIRE(device.endRendering(*command).has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = *color,
                                        .stagesBefore = rachis::PipelineStage::renderTarget,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::renderTarget,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                    })
                    .has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = *depth,
                                        .stagesBefore = rachis::PipelineStage::depthStencil,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::depthStencilWrite,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::depthStencilWrite,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                        .subresources = {.aspect = rachis::TextureAspect::depth},
                                    })
                    .has_value());
        for (std::uint32_t layer = 0; layer < scenario.layerCount; ++layer)
        {
            REQUIRE(device
                        .copyTextureToBuffer(*command,
                                             rachis::BufferTextureCopy{
                                                 .buffer = *readback,
                                                 .bufferOffset = readbackSize * layer,
                                                 .bytesPerRow = rowPitch,
                                                 .rowsPerImage = targetHeight,
                                                 .texture = *color,
                                                 .textureSubresource = {.arrayLayer = layer},
                                                 .extent = {targetWidth, targetHeight, 1},
                                             })
                        .has_value());
            REQUIRE(device
                        .copyTextureToBuffer(*command,
                                             rachis::BufferTextureCopy{
                                                 .buffer = *readback,
                                                 .bufferOffset = readbackSize * (scenario.layerCount + layer),
                                                 .bytesPerRow = rowPitch,
                                                 .rowsPerImage = targetHeight,
                                                 .texture = *depth,
                                                 .textureSubresource =
                                                     {
                                                         .arrayLayer = layer,
                                                         .aspect = rachis::TextureAspect::depth,
                                                     },
                                                 .extent = {targetWidth, targetHeight, 1},
                                             })
                        .has_value());
        }
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());

        auto mapped = device.mapBuffer(*readback);
        REQUIRE(mapped.has_value());
        for (std::uint32_t layer = 0; layer < scenario.layerCount; ++layer)
        {
            INFO("layer " << layer);
            const bool inMask = ((scenario.viewMask >> layer) & 1U) != 0;
            const auto* const colorLayer = mapped->data + readbackSize * layer;
            const auto* const depthLayer = mapped->data + readbackSize * (scenario.layerCount + layer);
            CHECK(colorLayerMatches(colorLayer, subRects[0], inMask ? green : blue, inMask ? red : blue));
            CHECK(depthLayerMatches(depthLayer, subRects[0], inMask ? 0.25F : 1.0F, inMask ? 0.5F : 1.0F));
        }
        REQUIRE(device.unmapBuffer(*readback).has_value());
        REQUIRE(device.destroyTextureView(*colorView).has_value());
        REQUIRE(device.destroyTextureView(*depthView).has_value());
        REQUIRE(device.destroyTexture(*color).has_value());
        REQUIRE(device.destroyTexture(*depth).has_value());
        REQUIRE(device.destroyBuffer(*readback).has_value());
    }

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan vertex inputs take one attribute per location that matrix and array inputs consume",
          "[size:medium][rhi][vulkan][graphics][vertex-input]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    const auto matrixVertexBytecode =
        loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VertexInput.matrix.vs.spv");
    const auto arrayVertexBytecode =
        loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VertexInput.array.vs.spv");
    const auto pixelBytecode = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VertexInput.ps.spv");
    REQUIRE_FALSE(matrixVertexBytecode.empty());
    REQUIRE_FALSE(arrayVertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());

    constexpr std::array baseLocations{0U, 1U};
    constexpr std::array matrixLocations{0U, 1U, 2U, 3U, 4U};
    constexpr std::array arrayLocations{0U, 1U, 2U};
    auto matrixBaseOnly = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = matrixVertexBytecode,
        .entryPoint = "matrixVertexMain",
        .shaderInterface = rachis::test::shaderInterface({}, {}, baseLocations),
    });
    REQUIRE_FALSE(matrixBaseOnly.has_value());
    CHECK(matrixBaseOnly.error().code == rachis::ErrorCode::invalidArgument);
    auto arrayBaseOnly = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = arrayVertexBytecode,
        .entryPoint = "arrayVertexMain",
        .shaderInterface = rachis::test::shaderInterface({}, {}, baseLocations),
    });
    REQUIRE_FALSE(arrayBaseOnly.has_value());
    CHECK(arrayBaseOnly.error().code == rachis::ErrorCode::invalidArgument);

    auto matrixShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = matrixVertexBytecode,
        .entryPoint = "matrixVertexMain",
        .shaderInterface = rachis::test::shaderInterface({}, {}, matrixLocations),
    });
    if (!matrixShader)
    {
        UNSCOPED_INFO(matrixShader.error().message);
    }
    auto arrayShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = arrayVertexBytecode,
        .entryPoint = "arrayVertexMain",
        .shaderInterface = rachis::test::shaderInterface({}, {}, arrayLocations),
    });
    if (!arrayShader)
    {
        UNSCOPED_INFO(arrayShader.error().message);
    }
    auto pixelShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "pixelMain",
        .shaderInterface = {},
    });
    auto pipelineLayout = device.createPipelineLayout({});
    REQUIRE(matrixShader.has_value());
    REQUIRE(arrayShader.has_value());
    REQUIRE(pixelShader.has_value());
    REQUIRE(pipelineLayout.has_value());

    constexpr std::uint32_t matrixStride = 16 * sizeof(float);
    constexpr std::uint32_t arrayStride = 8 * sizeof(float);
    const std::array matrixAttributes{
        rachis::VertexAttribute{.format = rachis::VertexFormat::float32x4, .offset = 0, .shaderLocation = 1},
        rachis::VertexAttribute{.format = rachis::VertexFormat::float32x4, .offset = 16, .shaderLocation = 2},
        rachis::VertexAttribute{.format = rachis::VertexFormat::float32x4, .offset = 32, .shaderLocation = 3},
        rachis::VertexAttribute{.format = rachis::VertexFormat::float32x4, .offset = 48, .shaderLocation = 4},
    };
    const std::array arrayAttributes{
        rachis::VertexAttribute{.format = rachis::VertexFormat::float32x4, .offset = 0, .shaderLocation = 1},
        rachis::VertexAttribute{.format = rachis::VertexFormat::float32x4, .offset = 16, .shaderLocation = 2},
    };
    auto matrixBasePipeline = createInstancedInputPipeline(
        device, *pipelineLayout, *matrixShader, *pixelShader, matrixStride, std::span(matrixAttributes).first(1));
    REQUIRE_FALSE(matrixBasePipeline.has_value());
    CHECK(matrixBasePipeline.error().code == rachis::ErrorCode::validationError);
    auto arrayBasePipeline = createInstancedInputPipeline(
        device, *pipelineLayout, *arrayShader, *pixelShader, arrayStride, std::span(arrayAttributes).first(1));
    REQUIRE_FALSE(arrayBasePipeline.has_value());
    CHECK(arrayBasePipeline.error().code == rachis::ErrorCode::validationError);
    auto matrixPipeline = createInstancedInputPipeline(
        device, *pipelineLayout, *matrixShader, *pixelShader, matrixStride, matrixAttributes);
    if (!matrixPipeline)
    {
        UNSCOPED_INFO(matrixPipeline.error().message);
    }
    auto arrayPipeline =
        createInstancedInputPipeline(device, *pipelineLayout, *arrayShader, *pixelShader, arrayStride, arrayAttributes);
    if (!arrayPipeline)
    {
        UNSCOPED_INFO(arrayPipeline.error().message);
    }
    REQUIRE(matrixPipeline.has_value());
    REQUIRE(arrayPipeline.has_value());

    const std::array positions{
        std::array{-0.5F, 0.5F},
        std::array{-0.1F, -0.5F},
        std::array{-0.9F, -0.5F},
        std::array{0.5F, 0.5F},
        std::array{0.9F, -0.5F},
        std::array{0.1F, -0.5F},
    };
    const std::array halfIdentity{
        std::array{0.5F, 0.0F, 0.0F, 0.0F},
        std::array{0.0F, 0.5F, 0.0F, 0.0F},
        std::array{0.0F, 0.0F, 0.5F, 0.0F},
        std::array{0.0F, 0.0F, 0.0F, 0.5F},
    };
    const std::array weights{
        std::array{0.5F, 0.0F, 0.5F, 0.5F},
        std::array{0.0F, 0.25F, 0.25F, 0.5F},
    };
    auto positionBuffer = createFilledVertexBuffer(device, std::as_bytes(std::span(positions)));
    auto matrixBuffer = createFilledVertexBuffer(device, std::as_bytes(std::span(halfIdentity)));
    auto weightBuffer = createFilledVertexBuffer(device, std::as_bytes(std::span(weights)));
    REQUIRE(positionBuffer.has_value());
    REQUIRE(matrixBuffer.has_value());
    REQUIRE(weightBuffer.has_value());

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
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array colorAttachments{
        rachis::ColorAttachment{
            .view = *targetView,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.0F, 0.0F, 0.0F, 0.0F},
        },
    };
    REQUIRE(device.beginRendering(*command, rachis::RenderingDesc{.colorAttachments = colorAttachments}).has_value());
    REQUIRE(device
                .setViewport(*command,
                             rachis::Viewport{
                                 .width = static_cast<float>(targetWidth),
                                 .height = static_cast<float>(targetHeight),
                             })
                .has_value());
    REQUIRE(device.setScissor(*command, rachis::ScissorRect{.width = targetWidth, .height = targetHeight}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *matrixPipeline).has_value());
    const std::array matrixBindings{
        rachis::VertexBufferBinding{.buffer = *positionBuffer},
        rachis::VertexBufferBinding{.buffer = *matrixBuffer},
    };
    REQUIRE(device.bindVertexBuffers(*command, 0, matrixBindings).has_value());
    REQUIRE(device.draw(*command, 3).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *arrayPipeline).has_value());
    const std::array arrayBindings{
        rachis::VertexBufferBinding{.buffer = *positionBuffer},
        rachis::VertexBufferBinding{.buffer = *weightBuffer},
    };
    REQUIRE(device.bindVertexBuffers(*command, 0, arrayBindings).has_value());
    REQUIRE(device.draw(*command, 3, 1, 3).has_value());
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

    REQUIRE(device.destroyPipeline(*matrixPipeline).has_value());
    REQUIRE(device.destroyPipeline(*arrayPipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*matrixShader).has_value());
    REQUIRE(device.destroyShaderModule(*arrayShader).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    REQUIRE(device.destroyBuffer(*positionBuffer).has_value());
    REQUIRE(device.destroyBuffer(*matrixBuffer).has_value());
    REQUIRE(device.destroyBuffer(*weightBuffer).has_value());
    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    const auto* matrixPixel = pixelAt(mapped->data, targetWidth / 4, targetHeight / 2);
    const auto* arrayPixel = pixelAt(mapped->data, 3 * targetWidth / 4, targetHeight / 2);
    CHECK(nearByte(matrixPixel[0], 128));
    CHECK(nearByte(matrixPixel[1], 128));
    CHECK(nearByte(matrixPixel[2], 128));
    CHECK(nearByte(matrixPixel[3], 128));
    CHECK(nearByte(arrayPixel[0], 128));
    CHECK(nearByte(arrayPixel[1], 64));
    CHECK(nearByte(arrayPixel[2], 191));
    CHECK(nearByte(arrayPixel[3], 255));
    REQUIRE(device.unmapBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan vertex input rejects strides and offsets that break format alignment",
          "[size:medium][rhi][vulkan][graphics]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    const auto inputVertexBytecode =
        loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanGraphicsTriangle.input.vs.spv");
    const auto pixelBytecode =
        loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanGraphicsTriangle.ps.spv");
    REQUIRE_FALSE(inputVertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    auto inputVertexShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = inputVertexBytecode,
        .entryPoint = "inputVertex",
        .shaderInterface = rachis::test::graphicsInputVertexShaderInterface(),
    });
    auto pixelShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "pixelMain",
        .shaderInterface = rachis::test::graphicsPixelShaderInterface(),
    });
    REQUIRE(inputVertexShader.has_value());
    REQUIRE(pixelShader.has_value());
    const std::array pushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::pixel, 0, 4 * sizeof(float)},
    };
    auto pipelineLayout = device.createPipelineLayout(rachis::PipelineLayoutDesc{
        .pushConstantRanges = pushConstantRanges,
    });
    REQUIRE(pipelineLayout.has_value());

    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    const auto makeDesc = [&](std::span<const rachis::VertexBufferLayout> vertexBuffers) {
        return rachis::GraphicsPipelineDesc{
            .layout = *pipelineLayout,
            .vertexShader = *inputVertexShader,
            .pixelShader = *pixelShader,
            .vertexBuffers = vertexBuffers,
            .cullMode = rachis::CullMode::none,
            .colorTargets = colorTargets,
        };
    };
    const auto checkRejected = [&](rachis::VertexFormat format, std::uint32_t offset, std::uint32_t stride) {
        const std::array attributes{rachis::VertexAttribute{.format = format, .offset = offset}};
        const std::array layouts{rachis::VertexBufferLayout{.stride = stride, .attributes = attributes}};
        const auto rejected = device.createGraphicsPipeline(makeDesc(layouts));
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
    };
    checkRejected(rachis::VertexFormat::float32x2, 2, 12);
    checkRejected(rachis::VertexFormat::float32x2, 0, 10);
    checkRejected(rachis::VertexFormat::unorm8x2, 1, 4);
    checkRejected(rachis::VertexFormat::uint16x2, 2, 8);
    checkRejected(rachis::VertexFormat::unorm8x4, 2, 8);

    const std::array floatAttributes{rachis::VertexAttribute{.format = rachis::VertexFormat::float32x2, .offset = 4}};
    const std::array floatLayouts{rachis::VertexBufferLayout{.stride = 12, .attributes = floatAttributes}};
    const std::array packedAttributes{rachis::VertexAttribute{.format = rachis::VertexFormat::unorm8x2, .offset = 2}};
    const std::array packedLayouts{rachis::VertexBufferLayout{.stride = 4, .attributes = packedAttributes}};
    auto floatPipeline = device.createGraphicsPipeline(makeDesc(floatLayouts));
    auto packedPipeline = device.createGraphicsPipeline(makeDesc(packedLayouts));
    REQUIRE(floatPipeline.has_value());
    REQUIRE(packedPipeline.has_value());

    const std::array floatVertexData{
        0.0F,
        0.0F,
        0.0F,
        0.75F,
        0.0F,
        0.75F,
        -0.75F,
        0.0F,
        -0.75F,
        -0.75F,
    };
    const std::array<std::uint8_t, 16> packedVertexData{
        0,
        0,
        0,
        0,
        128,
        255,
        0,
        0,
        255,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    };
    auto floatVertices = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(floatVertexData),
        .usage = rachis::BufferUsage::vertex,
        .memoryType = rachis::MemoryType::upload,
    });
    auto packedVertices = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(packedVertexData),
        .usage = rachis::BufferUsage::vertex,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(floatVertices.has_value());
    REQUIRE(packedVertices.has_value());
    REQUIRE(device.writeBuffer(*floatVertices, 0, std::as_bytes(std::span{floatVertexData})).has_value());
    REQUIRE(device.writeBuffer(*packedVertices, 0, std::as_bytes(std::span{packedVertexData})).has_value());

    auto target = device.createTexture(rachis::TextureDesc{
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
    });
    REQUIRE(target.has_value());
    auto targetView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    REQUIRE(targetView.has_value());

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
    const std::array colorAttachments{
        rachis::ColorAttachment{
            .view = *targetView,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.0F, 0.0F, 0.0F, 0.0F},
        },
    };
    REQUIRE(device.beginRendering(*command, rachis::RenderingDesc{.colorAttachments = colorAttachments}).has_value());
    const std::array tint{0.125F, 0.5F, 0.75F, 1.0F};
    const auto bindDrawState = [&](rachis::PipelineHandle pipeline) {
        REQUIRE(device.bindGraphicsPipeline(*command, pipeline).has_value());
        REQUIRE(device
                    .setViewport(*command,
                                 rachis::Viewport{
                                     .width = static_cast<float>(targetWidth),
                                     .height = static_cast<float>(targetHeight),
                                 })
                    .has_value());
        REQUIRE(
            device.setScissor(*command, rachis::ScissorRect{.width = targetWidth, .height = targetHeight}).has_value());
        REQUIRE(device.setGraphicsPushConstants(*command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(tint)))
                    .has_value());
    };

    bindDrawState(*floatPipeline);
    const auto misalignedFloat = device.bindVertexBuffer(*command, 0, *floatVertices, 2);
    REQUIRE_FALSE(misalignedFloat.has_value());
    CHECK(misalignedFloat.error().code == rachis::ErrorCode::invalidArgument);
    CHECK_FALSE(device.draw(*command, 3).has_value());
    REQUIRE(device.bindVertexBuffer(*command, 0, *floatVertices, 4).has_value());
    REQUIRE(device.draw(*command, 3).has_value());

    bindDrawState(*packedPipeline);
    const auto oddPacked = device.bindVertexBuffer(*command, 0, *packedVertices, 1);
    REQUIRE_FALSE(oddPacked.has_value());
    CHECK(oddPacked.error().code == rachis::ErrorCode::invalidArgument);
    CHECK_FALSE(device.draw(*command, 3).has_value());
    REQUIRE(device.bindVertexBuffer(*command, 0, *packedVertices, 2).has_value());
    REQUIRE(device.draw(*command, 3).has_value());
    REQUIRE(device.endRendering(*command).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    REQUIRE(device.destroyBuffer(*floatVertices).has_value());
    REQUIRE(device.destroyBuffer(*packedVertices).has_value());
    REQUIRE(device.destroyPipeline(*floatPipeline).has_value());
    REQUIRE(device.destroyPipeline(*packedPipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*inputVertexShader).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    device.collectGarbage();

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan graphics pipelines push and draw through layouts that repeat a stage across push-constant ranges",
          "[size:medium][rhi][vulkan][graphics][push-constants]")
{
    constexpr std::uint32_t cellSize = 32;
    constexpr std::uint32_t cellCount = 3;
    constexpr std::uint32_t cellRowPitch = 512;

    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    const auto spirvDirectory = std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR);
    const auto directVertexBytecode = loadBytes(spirvDirectory / "VulkanGraphicsTriangle.direct.vs.spv");
    const auto lowVertexBytecode = loadBytes(spirvDirectory / "VulkanPushConstantRanges.low.vs.spv");
    const auto highVertexBytecode = loadBytes(spirvDirectory / "VulkanPushConstantRanges.high.vs.spv");
    const auto contiguousPixelBytecode = loadBytes(spirvDirectory / "VulkanPushConstantRanges.contiguous.ps.spv");
    const auto splitPixelBytecode = loadBytes(spirvDirectory / "VulkanPushConstantRanges.split.ps.spv");
    REQUIRE_FALSE(directVertexBytecode.empty());
    REQUIRE_FALSE(lowVertexBytecode.empty());
    REQUIRE_FALSE(highVertexBytecode.empty());
    REQUIRE_FALSE(contiguousPixelBytecode.empty());
    REQUIRE_FALSE(splitPixelBytecode.empty());

    static constexpr std::array lowVertexPush{rachis::ShaderPushConstantRequirement{0, 16}};
    static constexpr std::array highVertexPush{rachis::ShaderPushConstantRequirement{16, 16}};
    static constexpr std::array contiguousPixelPush{rachis::ShaderPushConstantRequirement{0, 32}};
    static constexpr std::array splitPixelPush{
        rachis::ShaderPushConstantRequirement{0, 16},
        rachis::ShaderPushConstantRequirement{32, 16},
    };
    auto directVertex = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = directVertexBytecode,
        .entryPoint = "directVertex",
        .label = "Vulkan push-range direct vertex",
        .shaderInterface = {},
    });
    auto lowVertex = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = lowVertexBytecode,
        .entryPoint = "vertexMain",
        .label = "Vulkan push-range low vertex",
        .shaderInterface = rachis::test::shaderInterface({}, lowVertexPush),
    });
    auto highVertex = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = highVertexBytecode,
        .entryPoint = "vertexMain",
        .label = "Vulkan push-range high vertex",
        .shaderInterface = rachis::test::shaderInterface({}, highVertexPush),
    });
    auto contiguousPixel = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::pixel,
        .bytecode = contiguousPixelBytecode,
        .entryPoint = "pixelMain",
        .label = "Vulkan push-range contiguous pixel",
        .shaderInterface = rachis::test::shaderInterface({}, contiguousPixelPush),
    });
    auto splitPixel = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::pixel,
        .bytecode = splitPixelBytecode,
        .entryPoint = "pixelMain",
        .label = "Vulkan push-range split pixel",
        .shaderInterface = rachis::test::shaderInterface({}, splitPixelPush),
    });
    REQUIRE(directVertex.has_value());
    REQUIRE(lowVertex.has_value());
    REQUIRE(highVertex.has_value());
    REQUIRE(contiguousPixel.has_value());
    REQUIRE(splitPixel.has_value());

    struct PushWrite final
    {
        rachis::ShaderStage visibility = rachis::ShaderStage::none;
        std::uint32_t offset = 0;
        std::array<float, 4> values = {};
    };
    struct PushCase final
    {
        std::vector<rachis::PushConstantRange> ranges;
        rachis::ShaderModuleHandle vertexShader;
        rachis::ShaderModuleHandle pixelShader;
        std::vector<PushWrite> writes;
        std::array<std::uint8_t, 4> expected = {};
    };
    const std::array cases{
        PushCase{
            .ranges =
                {
                    {.visibility = rachis::ShaderStage::vertex | rachis::ShaderStage::pixel, .offset = 0, .size = 16},
                    {.visibility = rachis::ShaderStage::pixel, .offset = 16, .size = 16},
                },
            .vertexShader = *lowVertex,
            .pixelShader = *contiguousPixel,
            .writes =
                {
                    {rachis::ShaderStage::vertex | rachis::ShaderStage::pixel, 0, {0.0F, 0.0F, 0.25F, 0.0F}},
                    {rachis::ShaderStage::pixel, 16, {0.5F, 0.25F, 0.25F, 1.0F}},
                },
            .expected = {128, 64, 128, 255},
        },
        PushCase{
            .ranges =
                {
                    {.visibility = rachis::ShaderStage::pixel, .offset = 0, .size = 16},
                    {.visibility = rachis::ShaderStage::pixel, .offset = 32, .size = 16},
                },
            .vertexShader = *directVertex,
            .pixelShader = *splitPixel,
            .writes =
                {
                    {rachis::ShaderStage::pixel, 0, {0.25F, 0.0F, 0.0F, 0.0F}},
                    {rachis::ShaderStage::pixel, 32, {0.0F, 0.5F, 0.75F, 1.0F}},
                },
            .expected = {64, 128, 191, 255},
        },
        PushCase{
            .ranges =
                {
                    {.visibility = rachis::ShaderStage::pixel, .offset = 0, .size = 16},
                    {.visibility = rachis::ShaderStage::vertex, .offset = 16, .size = 16},
                    {.visibility = rachis::ShaderStage::pixel, .offset = 32, .size = 16},
                },
            .vertexShader = *highVertex,
            .pixelShader = *splitPixel,
            .writes =
                {
                    {rachis::ShaderStage::pixel, 0, {0.75F, 0.0F, 0.0F, 0.0F}},
                    {rachis::ShaderStage::vertex, 16, {0.0F, 0.0F, 0.0F, 0.0F}},
                    {rachis::ShaderStage::pixel, 32, {0.0F, 0.25F, 0.0F, 1.0F}},
                },
            .expected = {191, 64, 0, 255},
        },
    };
    REQUIRE(cases.size() == cellCount);

    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    std::vector<rachis::PipelineLayoutHandle> layouts;
    std::vector<rachis::PipelineHandle> pipelines;
    for (const auto& pushCase : cases)
    {
        auto layout = device.createPipelineLayout(rachis::PipelineLayoutDesc{
            .pushConstantRanges = pushCase.ranges,
            .label = "Vulkan push-range pipeline layout",
        });
        REQUIRE(layout.has_value());
        layouts.push_back(*layout);
        auto pipeline = device.createGraphicsPipeline(rachis::GraphicsPipelineDesc{
            .layout = *layout,
            .vertexShader = pushCase.vertexShader,
            .pixelShader = pushCase.pixelShader,
            .cullMode = rachis::CullMode::none,
            .colorTargets = colorTargets,
            .label = "Vulkan push-range pipeline",
        });
        REQUIRE(pipeline.has_value());
        pipelines.push_back(*pipeline);
    }

    auto target = device.createTexture(rachis::TextureDesc{
        .width = cellSize * cellCount,
        .height = cellSize,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
        .label = "Vulkan push-range target",
    });
    REQUIRE(target.has_value());
    auto targetView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
        .label = "Vulkan push-range target view",
    });
    REQUIRE(targetView.has_value());
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = static_cast<std::uint64_t>(cellRowPitch) * cellSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan push-range readback",
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
    const std::array colorAttachments{
        rachis::ColorAttachment{
            .view = *targetView,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.0F, 0.0F, 0.0F, 0.0F},
        },
    };
    REQUIRE(device.beginRendering(*command, rachis::RenderingDesc{.colorAttachments = colorAttachments}).has_value());
    for (std::uint32_t cell = 0; cell < cellCount; ++cell)
    {
        CAPTURE(cell);
        REQUIRE(device.bindGraphicsPipeline(*command, pipelines[cell]).has_value());
        REQUIRE(device
                    .setViewport(*command,
                                 rachis::Viewport{
                                     .x = static_cast<float>(cell * cellSize),
                                     .width = static_cast<float>(cellSize),
                                     .height = static_cast<float>(cellSize),
                                 })
                    .has_value());
        REQUIRE(device
                    .setScissor(*command,
                                rachis::ScissorRect{
                                    .x = static_cast<std::int32_t>(cell * cellSize),
                                    .width = cellSize,
                                    .height = cellSize,
                                })
                    .has_value());
        for (const auto& write : cases[cell].writes)
        {
            REQUIRE(device
                        .setGraphicsPushConstants(
                            *command, write.visibility, write.offset, std::as_bytes(std::span(write.values)))
                        .has_value());
        }
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
                                         .bytesPerRow = cellRowPitch,
                                         .rowsPerImage = cellSize,
                                         .texture = *target,
                                         .extent = {cellSize * cellCount, cellSize, 1},
                                     })
                .has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    for (std::uint32_t cell = 0; cell < cellCount; ++cell)
    {
        CAPTURE(cell);
        const auto* center = mapped->data + static_cast<std::uint64_t>(cellSize / 2) * cellRowPitch +
                             static_cast<std::uint64_t>(cell * cellSize + cellSize / 2) * 4;
        CHECK(nearByte(center[0], cases[cell].expected[0]));
        CHECK(nearByte(center[1], cases[cell].expected[1]));
        CHECK(nearByte(center[2], cases[cell].expected[2]));
        CHECK(nearByte(center[3], cases[cell].expected[3]));
    }
    REQUIRE(device.unmapBuffer(*readback).has_value());

    for (const auto pipeline : pipelines)
    {
        REQUIRE(device.destroyPipeline(pipeline).has_value());
    }
    for (const auto layout : layouts)
    {
        REQUIRE(device.destroyPipelineLayout(layout).has_value());
    }
    REQUIRE(device.destroyShaderModule(*directVertex).has_value());
    REQUIRE(device.destroyShaderModule(*lowVertex).has_value());
    REQUIRE(device.destroyShaderModule(*highVertex).has_value());
    REQUIRE(device.destroyShaderModule(*contiguousPixel).has_value());
    REQUIRE(device.destroyShaderModule(*splitPixel).has_value());
    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    device.collectGarbage();

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
