#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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

    [[nodiscard]] bool pixelMatches(const std::byte* pixel, const std::array<std::byte, 4>& expected) noexcept
    {
        return pixel[0] == expected[0] && pixel[1] == expected[1] && pixel[2] == expected[2] && pixel[3] == expected[3];
    }

} // namespace

TEST_CASE("independent stencil references are complete, bind invalidated, and affect WARP rasterization",
          "[size:medium][rhi][graphics][depth-stencil][independent-stencil-reference][warp]")
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
    REQUIRE(rachis::test::requireCapabilities(device).independentFrontAndBackStencilReference);

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
    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    const rachis::StencilFaceState writeStencil{
        .compare = rachis::CompareOperation::always,
        .passOperation = rachis::StencilOperation::replace,
    };
    const auto writeDesc = rachis::GraphicsPipelineDesc{
        .layout = *layout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .frontFace = rachis::FrontFace::counterClockwise,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .depthStencil =
            {
                .format = rachis::Format::d24UnormS8Uint,
                .stencilTestEnabled = true,
                .frontFace = writeStencil,
                .backFace = writeStencil,
            },
    };
    auto counterClockwisePipeline = device.createGraphicsPipeline(writeDesc);
    auto clockwiseDesc = writeDesc;
    clockwiseDesc.frontFace = rachis::FrontFace::clockwise;
    auto clockwisePipeline = device.createGraphicsPipeline(clockwiseDesc);

    auto compareDesc = writeDesc;
    compareDesc.depthStencil.frontFace = {.compare = rachis::CompareOperation::equal};
    compareDesc.depthStencil.backFace = {.compare = rachis::CompareOperation::equal};
    compareDesc.depthStencil.stencilWriteMask = 0;
    auto comparePipeline = device.createGraphicsPipeline(compareDesc);
    REQUIRE(counterClockwisePipeline.has_value());
    REQUIRE(clockwisePipeline.has_value());
    REQUIRE(comparePipeline.has_value());

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
    CHECK_FALSE(device.setStencilReference(*command, rachis::StencilFace::front, 7).has_value());
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
            .clearValue = {0.0F, 0.0F, 0.0F, 1.0F},
        },
    };
    const rachis::DepthStencilAttachment depthStencilAttachment{
        .view = *depthStencilView,
        .depthLoadOperation = rachis::LoadOperation::discard,
        .depthStoreOperation = rachis::StoreOperation::discard,
        .stencilLoadOperation = rachis::LoadOperation::clear,
        .stencilStoreOperation = rachis::StoreOperation::store,
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
    REQUIRE(device.bindGraphicsPipeline(*command, *counterClockwisePipeline).has_value());
    const std::array black{0.0F, 0.0F, 0.0F, 1.0F};
    REQUIRE(device.setGraphicsPushConstants(*command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(black)))
                .has_value());
    REQUIRE(device
                .setScissor(*command,
                            rachis::ScissorRect{
                                .width = targetWidth / 2,
                                .height = targetHeight,
                            })
                .has_value());
    CHECK_FALSE(device.setStencilReference(*command, static_cast<rachis::StencilFace>(255), 7).has_value());
    CHECK_FALSE(device.setStencilReference(*command, rachis::StencilFace::front, 256).has_value());
    REQUIRE(device.setStencilReference(*command, rachis::StencilFace::front, 7).has_value());
    CHECK_FALSE(device.draw(*command, 3).has_value());
    REQUIRE(device.setStencilReference(*command, rachis::StencilFace::back, 11).has_value());
    REQUIRE(device.draw(*command, 3).has_value());

    REQUIRE(device.bindGraphicsPipeline(*command, *clockwisePipeline).has_value());
    REQUIRE(device.setGraphicsPushConstants(*command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(black)))
                .has_value());
    REQUIRE(device
                .setScissor(*command,
                            rachis::ScissorRect{
                                .x = targetWidth / 2,
                                .width = targetWidth / 2,
                                .height = targetHeight,
                            })
                .has_value());
    CHECK_FALSE(device.draw(*command, 3).has_value());
    REQUIRE(device.setStencilReference(*command, rachis::StencilFace::front, 7).has_value());
    REQUIRE(device.setStencilReference(*command, rachis::StencilFace::back, 11).has_value());
    REQUIRE(device.draw(*command, 3).has_value());

    REQUIRE(device.bindGraphicsPipeline(*command, *comparePipeline).has_value());
    REQUIRE(device.setScissor(*command, rachis::ScissorRect{.width = targetWidth, .height = targetHeight}).has_value());
    const std::array green{0.0F, 1.0F, 0.0F, 1.0F};
    REQUIRE(device.setGraphicsPushConstants(*command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(green)))
                .has_value());
    REQUIRE(device.setStencilReference(*command, 7).has_value());
    REQUIRE(device.draw(*command, 3).has_value());
    const std::array red{1.0F, 0.0F, 0.0F, 1.0F};
    REQUIRE(device.setGraphicsPushConstants(*command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(red)))
                .has_value());
    REQUIRE(device.setStencilReference(*command, 11).has_value());
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
    constexpr std::array greenPixel{std::byte{0}, std::byte{255}, std::byte{0}, std::byte{255}};
    constexpr std::array redPixel{std::byte{255}, std::byte{0}, std::byte{0}, std::byte{255}};
    const bool leftGreen = pixelMatches(mapped->data + leftOffset, greenPixel);
    const bool leftRed = pixelMatches(mapped->data + leftOffset, redPixel);
    const bool rightGreen = pixelMatches(mapped->data + rightOffset, greenPixel);
    const bool rightRed = pixelMatches(mapped->data + rightOffset, redPixel);
    const bool distinctFaceColors = (leftGreen && rightRed) || (leftRed && rightGreen);
    CHECK(distinctFaceColors);
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyTextureView(*depthStencilView).has_value());
    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*depthStencil).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    REQUIRE(device.destroyPipeline(*comparePipeline).has_value());
    REQUIRE(device.destroyPipeline(*clockwisePipeline).has_value());
    REQUIRE(device.destroyPipeline(*counterClockwisePipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*layout).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
}
