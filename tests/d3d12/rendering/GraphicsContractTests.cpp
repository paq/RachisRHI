#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
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

    [[nodiscard]] bool nearByte(std::byte actual, std::uint8_t expected) noexcept
    {
        const auto value = std::to_integer<std::uint8_t>(actual);
        const auto lower = expected == 0 ? 0U : static_cast<std::uint32_t>(expected) - 1U;
        const auto upper = (std::min)(static_cast<std::uint32_t>(expected) + 1U, 255U);
        return value >= lower && value <= upper;
    }

} // namespace

TEST_CASE("graphics rendering executes direct and indexed draws and retains recorded objects",
          "[size:medium][rhi][graphics][pipeline][warp]")
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

    const auto directVertexBytecode = loadShader(RACHIS_TEST_GRAPHICS_DIRECT_VERTEX_SHADER_PATH);
    const auto inputVertexBytecode = loadShader(RACHIS_TEST_GRAPHICS_INPUT_VERTEX_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_GRAPHICS_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(directVertexBytecode.empty());
    REQUIRE_FALSE(inputVertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());

    auto directVertexShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = directVertexBytecode,
        .entryPoint = "directVertex",
        .label = "Direct triangle vertex shader",
        .shaderInterface = {},
    });
    auto inputVertexShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = inputVertexBytecode,
        .entryPoint = "inputVertex",
        .label = "Input triangle vertex shader",
        .shaderInterface = rachis::test::graphicsInputVertexShaderInterface(),
    });
    auto pixelShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "pixelMain",
        .label = "Triangle pixel shader",
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
        .label = "Graphics triangle pipeline layout",
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
        .label = "Direct triangle pipeline",
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
        .label = "Indexed triangle pipeline",
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
    CHECK_FALSE(device
                    .createGraphicsPipeline(rachis::GraphicsPipelineDesc{
                        .layout = *pipelineLayout,
                        .vertexShader = *pixelShader,
                        .pixelShader = *pixelShader,
                        .colorTargets = colorTargets,
                        .depthStencil = {.format = rachis::Format::d32Float},
                    })
                    .has_value());

    auto target = device.createTexture(rachis::TextureDesc{
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
        .label = "Graphics contract target",
    });
    REQUIRE(target.has_value());
    auto targetView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
        .label = "Graphics contract target view",
    });
    REQUIRE(targetView.has_value());
    auto depth = device.createTexture(rachis::TextureDesc{
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::d32Float,
        .usage = rachis::TextureUsage::depthStencil,
        .label = "Graphics contract depth",
    });
    REQUIRE(depth.has_value());
    auto depthView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *depth,
        .usage = rachis::TextureViewUsage::depthStencil,
        .aspect = rachis::TextureAspect::depth,
        .label = "Graphics contract depth view",
    });
    REQUIRE(depthView.has_value());
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = readbackSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Graphics contract readback",
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
        .label = "Graphics contract vertices",
    });
    auto indexBuffer = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(indices),
        .usage = rachis::BufferUsage::index,
        .memoryType = rachis::MemoryType::upload,
        .label = "Graphics contract indices",
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
    const std::array discardColorAttachments{
        rachis::ColorAttachment{
            .view = *targetView,
            .loadOperation = rachis::LoadOperation::discard,
            .storeOperation = rachis::StoreOperation::discard,
        },
    };
    const rachis::DepthStencilAttachment discardDepthAttachment{
        .view = *depthView,
        .depthLoadOperation = rachis::LoadOperation::discard,
        .depthStoreOperation = rachis::StoreOperation::discard,
    };
    REQUIRE(device
                .beginRendering(*command,
                                rachis::RenderingDesc{
                                    .colorAttachments = discardColorAttachments,
                                    .depthStencilAttachment = discardDepthAttachment,
                                })
                .has_value());
    REQUIRE(device.endRendering(*command).has_value());
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
    REQUIRE(device.destroyTexture(*target).has_value());
    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*depth).has_value());
    REQUIRE(device.destroyTextureView(*depthView).has_value());
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
    const auto* center = mapped->data + static_cast<std::uint64_t>(targetHeight / 2) * rowPitch +
                         static_cast<std::uint64_t>(targetWidth / 2) * 4;
    CHECK(corner[0] == std::byte{0});
    CHECK(corner[1] == std::byte{0});
    CHECK(corner[2] == std::byte{0});
    CHECK(corner[3] == std::byte{0});
    CHECK(nearByte(center[0], 32));
    CHECK(nearByte(center[1], 128));
    CHECK(nearByte(center[2], 191));
    CHECK(nearByte(center[3], 255));
    REQUIRE(device.unmapBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("graphics blending and stencil state produce validated pixels",
          "[size:medium][rhi][graphics][blend][stencil][warp]")
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

    const rachis::StencilFaceState stencilWriteFace{
        .compare = rachis::CompareOperation::always,
        .passOperation = rachis::StencilOperation::replace,
    };
    const std::array stencilWriteTargets{
        rachis::ColorTargetState{
            .format = rachis::Format::rgba8Unorm,
            .writeMask = rachis::ColorWriteMask::none,
        },
    };
    auto stencilWritePipeline = device.createGraphicsPipeline(rachis::GraphicsPipelineDesc{
        .layout = *layout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = stencilWriteTargets,
        .depthStencil =
            {
                .format = rachis::Format::d24UnormS8Uint,
                .stencilTestEnabled = true,
                .frontFace = stencilWriteFace,
                .backFace = stencilWriteFace,
            },
    });
    REQUIRE(stencilWritePipeline.has_value());

    const rachis::StencilFaceState stencilTestFace{
        .compare = rachis::CompareOperation::equal,
    };
    const std::array blendTargets{
        rachis::ColorTargetState{
            .format = rachis::Format::rgba8Unorm,
            .blend =
                {
                    .enabled = true,
                    .color =
                        {
                            .sourceFactor = rachis::BlendFactor::constant,
                            .destinationFactor = rachis::BlendFactor::oneMinusConstant,
                        },
                    .alpha =
                        {
                            .sourceFactor = rachis::BlendFactor::one,
                            .destinationFactor = rachis::BlendFactor::zero,
                        },
                },
        },
    };
    auto blendStencilPipeline = device.createGraphicsPipeline(rachis::GraphicsPipelineDesc{
        .layout = *layout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = blendTargets,
        .depthStencil =
            {
                .format = rachis::Format::d24UnormS8Uint,
                .stencilTestEnabled = true,
                .frontFace = stencilTestFace,
                .backFace = stencilTestFace,
                .stencilWriteMask = 0,
            },
    });
    REQUIRE(blendStencilPipeline.has_value());

    auto invalidBlendTargets = blendTargets;
    invalidBlendTargets[0].blend.color.operation = static_cast<rachis::BlendOperation>(255);
    CHECK_FALSE(device
                    .createGraphicsPipeline(rachis::GraphicsPipelineDesc{
                        .layout = *layout,
                        .vertexShader = *vertexShader,
                        .pixelShader = *pixelShader,
                        .colorTargets = invalidBlendTargets,
                    })
                    .has_value());
    CHECK_FALSE(device
                    .createGraphicsPipeline(rachis::GraphicsPipelineDesc{
                        .layout = *layout,
                        .vertexShader = *vertexShader,
                        .pixelShader = *pixelShader,
                        .colorTargets = blendTargets,
                        .depthStencil =
                            {
                                .format = rachis::Format::d32Float,
                                .stencilTestEnabled = true,
                            },
                    })
                    .has_value());

    auto target = device.createTexture(rachis::TextureDesc{
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
    });
    auto depthStencil = device.createTexture(rachis::TextureDesc{
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::d24UnormS8Uint,
        .usage = rachis::TextureUsage::depthStencil,
    });
    REQUIRE(target.has_value());
    REQUIRE(depthStencil.has_value());
    auto targetView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    auto depthStencilView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *depthStencil,
        .usage = rachis::TextureViewUsage::depthStencil,
        .aspect = rachis::TextureAspect::depth,
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = readbackSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(targetView.has_value());
    REQUIRE(depthStencilView.has_value());
    REQUIRE(readback.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    CHECK_FALSE(device.setBlendConstants(*command, {}).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *target,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    for (const auto aspect : {rachis::TextureAspect::depth, rachis::TextureAspect::stencil})
    {
        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = *depthStencil,
                                        .stagesAfter = rachis::PipelineStage::depthStencil,
                                        .accessAfter = rachis::Access::depthStencilWrite,
                                        .layoutAfter = rachis::TextureLayout::depthStencilWrite,
                                        .subresources = {.aspect = aspect},
                                    })
                    .has_value());
    }

    const std::array attachments{
        rachis::ColorAttachment{
            .view = *targetView,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.0F, 0.0F, 1.0F, 1.0F},
        },
    };
    const rachis::DepthStencilAttachment depthStencilAttachment{
        .view = *depthStencilView,
        .depthLoadOperation = rachis::LoadOperation::clear,
        .depthStoreOperation = rachis::StoreOperation::discard,
        .depthClearValue = 1.0F,
        .stencilLoadOperation = rachis::LoadOperation::clear,
        .stencilStoreOperation = rachis::StoreOperation::store,
        .stencilClearValue = 0,
    };
    REQUIRE(device
                .beginRendering(*command,
                                rachis::RenderingDesc{
                                    .colorAttachments = attachments,
                                    .depthStencilAttachment = depthStencilAttachment,
                                })
                .has_value());
    REQUIRE(device
                .setViewport(*command,
                             rachis::Viewport{
                                 .width = static_cast<float>(targetWidth),
                                 .height = static_cast<float>(targetHeight),
                             })
                .has_value());
    REQUIRE(device.setScissor(*command, rachis::ScissorRect{.width = targetWidth, .height = targetHeight}).has_value());
    const std::array red{1.0F, 0.0F, 0.0F, 1.0F};
    REQUIRE(device.bindGraphicsPipeline(*command, *stencilWritePipeline).has_value());
    REQUIRE(device.setGraphicsPushConstants(*command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(red)))
                .has_value());
    CHECK_FALSE(device.draw(*command, 3).has_value());
    REQUIRE(device.setStencilReference(*command, 7).has_value());
    REQUIRE(device.draw(*command, 3).has_value());

    REQUIRE(device.bindGraphicsPipeline(*command, *blendStencilPipeline).has_value());
    REQUIRE(device.setGraphicsPushConstants(*command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(red)))
                .has_value());
    REQUIRE(device.setStencilReference(*command, 8).has_value());
    CHECK_FALSE(device.draw(*command, 3).has_value());
    REQUIRE(device.setBlendConstants(*command, {0.25F, 0.25F, 0.25F, 0.25F}).has_value());
    REQUIRE(device.draw(*command, 3).has_value());
    CHECK_FALSE(device.setStencilReference(*command, 256).has_value());
    REQUIRE(device.setStencilReference(*command, 7).has_value());
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
    CHECK(corner[0] == std::byte{0});
    CHECK(corner[1] == std::byte{0});
    CHECK(corner[2] == std::byte{255});
    CHECK(corner[3] == std::byte{255});
    CAPTURE(std::to_integer<std::uint8_t>(center[0]),
            std::to_integer<std::uint8_t>(center[1]),
            std::to_integer<std::uint8_t>(center[2]),
            std::to_integer<std::uint8_t>(center[3]));
    CHECK(nearByte(center[0], 64));
    CHECK(nearByte(center[1], 0));
    CHECK(nearByte(center[2], 191));
    CHECK(nearByte(center[3], 255));
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyTextureView(*depthStencilView).has_value());
    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*depthStencil).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    REQUIRE(device.destroyPipeline(*blendStencilPipeline).has_value());
    REQUIRE(device.destroyPipeline(*stencilWritePipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*layout).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
