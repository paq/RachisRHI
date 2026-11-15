#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

    struct InstanceData final
    {
        std::array<float, 2> offset;
        std::array<std::uint8_t, 4> color;
    };

    static_assert(sizeof(InstanceData) == 12);

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

    void checkPixel(const std::byte* pixel, std::array<std::uint8_t, 4> expected)
    {
        for (std::size_t channel = 0; channel < expected.size(); ++channel)
        {
            CHECK(std::to_integer<std::uint8_t>(pixel[channel]) == expected[channel]);
        }
    }

    void checkPixelNear(const std::byte* pixel, std::array<std::uint8_t, 4> expected)
    {
        for (std::size_t channel = 0; channel < expected.size(); ++channel)
        {
            const auto value = std::to_integer<int>(pixel[channel]);
            CHECK(value >= expected[channel] - 1);
            CHECK(value <= expected[channel] + 1);
        }
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

TEST_CASE("vertex input batches are atomic and support packed instanced streams",
          "[size:medium][rhi][graphics][vertex-input][warp]")
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

    const auto vertexInput = rachis::test::requireCapabilities(device).vertexInput;
    CHECK(vertexInput.maxBufferStride >= sizeof(InstanceData));
    CHECK(vertexInput.maxAttributeOffset >= 8);
    CHECK(vertexInput.maxInstanceStepRate >= 2);
    CHECK(vertexInput.supports(rachis::VertexFormat::float32x2));
    CHECK(vertexInput.supports(rachis::VertexFormat::unorm8x4));
    CHECK_FALSE(vertexInput.supports(static_cast<rachis::VertexFormat>(255)));

    const auto vertexBytecode = loadShader(RACHIS_TEST_VERTEX_INPUT_VERTEX_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_VERTEX_INPUT_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());

    auto vertexShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .entryPoint = "vertexMain",
        .shaderInterface = rachis::test::shaderInterface({}, {}, std::array{0U, 1U, 2U}),
    });
    auto pixelShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "pixelMain",
        .shaderInterface = {},
    });
    auto pipelineLayout = device.createPipelineLayout({});
    REQUIRE(vertexShader.has_value());
    REQUIRE(pixelShader.has_value());
    REQUIRE(pipelineLayout.has_value());

    const std::array vertexAttributes{
        rachis::VertexAttribute{.format = rachis::VertexFormat::float32x2, .shaderLocation = 0},
    };
    const std::array instanceAttributes{
        rachis::VertexAttribute{.format = rachis::VertexFormat::float32x2, .shaderLocation = 1},
        rachis::VertexAttribute{
            .format = rachis::VertexFormat::unorm8x4,
            .offset = 2 * sizeof(float),
            .shaderLocation = 2,
        },
    };
    const std::array vertexLayouts{
        rachis::VertexBufferLayout{
            .stride = 2 * sizeof(float),
            .attributes = vertexAttributes,
        },
        rachis::VertexBufferLayout{
            .stride = sizeof(InstanceData),
            .stepMode = rachis::VertexStepMode::instance,
            .instanceStepRate = 2,
            .attributes = instanceAttributes,
        },
    };
    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    const rachis::GraphicsPipelineDesc pipelineDesc{
        .layout = *pipelineLayout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .vertexBuffers = vertexLayouts,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
    };
    auto pipeline = device.createGraphicsPipeline(pipelineDesc);
    REQUIRE(pipeline.has_value());

    auto invalidStepLayouts = vertexLayouts;
    invalidStepLayouts[0].stepMode = static_cast<rachis::VertexStepMode>(255);
    auto invalidPipelineDesc = pipelineDesc;
    invalidPipelineDesc.vertexBuffers = invalidStepLayouts;
    CHECK_FALSE(device.createGraphicsPipeline(invalidPipelineDesc).has_value());
    auto invalidRateLayouts = vertexLayouts;
    invalidRateLayouts[1].instanceStepRate = 0;
    invalidPipelineDesc.vertexBuffers = invalidRateLayouts;
    CHECK_FALSE(device.createGraphicsPipeline(invalidPipelineDesc).has_value());
    invalidRateLayouts = vertexLayouts;
    invalidRateLayouts[0].instanceStepRate = 2;
    invalidPipelineDesc.vertexBuffers = invalidRateLayouts;
    CHECK_FALSE(device.createGraphicsPipeline(invalidPipelineDesc).has_value());

    const std::array vertices{
        std::array{0.0F, 0.3F},
        std::array{0.3F, -0.3F},
        std::array{-0.3F, -0.3F},
    };
    const std::array instances{
        InstanceData{.offset = {-0.5F, 0.0F}, .color = {255, 0, 0, 255}},
        InstanceData{.offset = {0.5F, 0.0F}, .color = {0, 255, 0, 255}},
    };
    auto vertexBuffer = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(vertices),
        .usage = rachis::BufferUsage::vertex,
        .memoryType = rachis::MemoryType::upload,
    });
    auto instanceBuffer = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(instances),
        .usage = rachis::BufferUsage::vertex,
        .memoryType = rachis::MemoryType::upload,
    });
    auto wrongUsageBuffer = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(instances),
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto staleBuffer = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(instances),
        .usage = rachis::BufferUsage::vertex,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(vertexBuffer.has_value());
    REQUIRE(instanceBuffer.has_value());
    REQUIRE(wrongUsageBuffer.has_value());
    REQUIRE(staleBuffer.has_value());
    auto mappedVertices = device.mapBuffer(*vertexBuffer);
    auto mappedInstances = device.mapBuffer(*instanceBuffer);
    REQUIRE(mappedVertices.has_value());
    REQUIRE(mappedInstances.has_value());
    std::memcpy(mappedVertices->data, vertices.data(), sizeof(vertices));
    std::memcpy(mappedInstances->data, instances.data(), sizeof(instances));
    REQUIRE(device.unmapBuffer(*vertexBuffer).has_value());
    REQUIRE(device.unmapBuffer(*instanceBuffer).has_value());
    const auto staleHandle = *staleBuffer;
    REQUIRE(device.destroyBuffer(*staleBuffer).has_value());

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
    REQUIRE(device.bindGraphicsPipeline(*command, *pipeline).has_value());
    REQUIRE(device
                .setViewport(*command,
                             rachis::Viewport{
                                 .width = static_cast<float>(targetWidth),
                                 .height = static_cast<float>(targetHeight),
                             })
                .has_value());
    REQUIRE(device.setScissor(*command, rachis::ScissorRect{.width = targetWidth, .height = targetHeight}).has_value());

    CHECK_FALSE(device.bindVertexBuffers(*command, 0, {}).has_value());
    const std::array overflowingBindings{
        rachis::VertexBufferBinding{.buffer = *vertexBuffer},
        rachis::VertexBufferBinding{.buffer = *instanceBuffer},
    };
    CHECK_FALSE(device.bindVertexBuffers(*command, 1, overflowingBindings).has_value());
    const std::array staleBindings{
        rachis::VertexBufferBinding{.buffer = *vertexBuffer},
        rachis::VertexBufferBinding{.buffer = staleHandle},
    };
    CHECK_FALSE(device.bindVertexBuffers(*command, 0, staleBindings).has_value());
    const std::array invalidBindings{
        rachis::VertexBufferBinding{.buffer = *vertexBuffer},
        rachis::VertexBufferBinding{.buffer = *wrongUsageBuffer},
    };
    CHECK_FALSE(device.bindVertexBuffers(*command, 0, invalidBindings).has_value());
    REQUIRE(device.bindVertexBuffer(*command, 1, *instanceBuffer).has_value());
    CHECK_FALSE(device.draw(*command, 3, 4).has_value());

    const std::array validBindings{
        rachis::VertexBufferBinding{.buffer = *vertexBuffer},
        rachis::VertexBufferBinding{.buffer = *instanceBuffer},
    };
    REQUIRE(device.bindVertexBuffers(*command, 0, validBindings).has_value());
    REQUIRE(device.draw(*command, 3, 4).has_value());
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

    REQUIRE(device.destroyBuffer(*vertexBuffer).has_value());
    REQUIRE(device.destroyBuffer(*instanceBuffer).has_value());
    REQUIRE(device.destroyBuffer(*wrongUsageBuffer).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    const auto* left = mapped->data + static_cast<std::uint64_t>(targetHeight / 2) * rowPitch +
                       static_cast<std::uint64_t>(targetWidth / 4) * 4;
    const auto* right = mapped->data + static_cast<std::uint64_t>(targetHeight / 2) * rowPitch +
                        static_cast<std::uint64_t>(3 * targetWidth / 4) * 4;
    checkPixel(left, {255, 0, 0, 255});
    checkPixel(right, {0, 255, 0, 255});
    REQUIRE(device.unmapBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("vertex inputs take one attribute per location that matrix and array inputs consume",
          "[size:medium][rhi][graphics][vertex-input][warp]")
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

    const auto matrixVertexBytecode = loadShader(RACHIS_TEST_VERTEX_INPUT_MATRIX_VERTEX_SHADER_PATH);
    const auto arrayVertexBytecode = loadShader(RACHIS_TEST_VERTEX_INPUT_ARRAY_VERTEX_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_VERTEX_INPUT_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(matrixVertexBytecode.empty());
    REQUIRE_FALSE(arrayVertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());

    auto matrixShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = matrixVertexBytecode,
        .entryPoint = "matrixVertexMain",
        .shaderInterface = rachis::test::shaderInterface({}, {}, std::array{0U, 1U, 2U, 3U, 4U}),
    });
    auto arrayShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = arrayVertexBytecode,
        .entryPoint = "arrayVertexMain",
        .shaderInterface = rachis::test::shaderInterface({}, {}, std::array{0U, 1U, 2U}),
    });
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
    auto arrayPipeline =
        createInstancedInputPipeline(device, *pipelineLayout, *arrayShader, *pixelShader, arrayStride, arrayAttributes);
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
    const auto* matrixPixel = mapped->data + static_cast<std::uint64_t>(targetHeight / 2) * rowPitch +
                              static_cast<std::uint64_t>(targetWidth / 4) * 4;
    const auto* arrayPixel = mapped->data + static_cast<std::uint64_t>(targetHeight / 2) * rowPitch +
                             static_cast<std::uint64_t>(3 * targetWidth / 4) * 4;
    checkPixelNear(matrixPixel, {128, 128, 128, 128});
    checkPixelNear(arrayPixel, {128, 64, 191, 255});
    REQUIRE(device.unmapBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("vertex input rejects strides and offsets that break format alignment",
          "[size:medium][rhi][graphics][vertex-input][warp]")
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

    const auto vertexBytecode = loadShader(RACHIS_TEST_VERTEX_INPUT_VERTEX_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_VERTEX_INPUT_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    auto vertexShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .entryPoint = "vertexMain",
        .shaderInterface = rachis::test::shaderInterface({}, {}, std::array{0U, 1U, 2U}),
    });
    auto pixelShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "pixelMain",
        .shaderInterface = {},
    });
    auto pipelineLayout = device.createPipelineLayout({});
    REQUIRE(vertexShader.has_value());
    REQUIRE(pixelShader.has_value());
    REQUIRE(pipelineLayout.has_value());

    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    const auto makeDesc = [&](std::span<const rachis::VertexBufferLayout> vertexBuffers) {
        return rachis::GraphicsPipelineDesc{
            .layout = *pipelineLayout,
            .vertexShader = *vertexShader,
            .pixelShader = *pixelShader,
            .vertexBuffers = vertexBuffers,
            .cullMode = rachis::CullMode::none,
            .colorTargets = colorTargets,
        };
    };
    const auto checkRejected = [&](std::span<const rachis::VertexBufferLayout> vertexBuffers) {
        const auto rejected = device.createGraphicsPipeline(makeDesc(vertexBuffers));
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
    };

    const std::array instanceAttributes{
        rachis::VertexAttribute{.format = rachis::VertexFormat::float32x2, .shaderLocation = 1},
        rachis::VertexAttribute{
            .format = rachis::VertexFormat::unorm8x4,
            .offset = 2 * sizeof(float),
            .shaderLocation = 2,
        },
    };
    const rachis::VertexBufferLayout instanceLayout{
        .stride = sizeof(InstanceData),
        .stepMode = rachis::VertexStepMode::instance,
        .attributes = instanceAttributes,
    };
    const std::array positionAtZero{
        rachis::VertexAttribute{.format = rachis::VertexFormat::float32x2, .shaderLocation = 0},
    };
    const std::array positionAtTwo{
        rachis::VertexAttribute{.format = rachis::VertexFormat::float32x2, .offset = 2, .shaderLocation = 0},
    };
    const std::array positionAtFour{
        rachis::VertexAttribute{.format = rachis::VertexFormat::float32x2, .offset = 4, .shaderLocation = 0},
    };
    const std::array packedPositionAtOne{
        rachis::VertexAttribute{.format = rachis::VertexFormat::unorm8x2, .offset = 1, .shaderLocation = 0},
    };
    const std::array mixedInstanceAttributes{
        rachis::VertexAttribute{.format = rachis::VertexFormat::unorm8x2, .offset = 0, .shaderLocation = 1},
        rachis::VertexAttribute{.format = rachis::VertexFormat::unorm8x4, .offset = 2, .shaderLocation = 2},
    };
    const std::array packedAttributes{
        rachis::VertexAttribute{.format = rachis::VertexFormat::unorm8x2, .offset = 0, .shaderLocation = 0},
        rachis::VertexAttribute{.format = rachis::VertexFormat::unorm8x2, .offset = 2, .shaderLocation = 1},
        rachis::VertexAttribute{.format = rachis::VertexFormat::unorm8x2, .offset = 4, .shaderLocation = 2},
    };

    checkRejected(std::array{rachis::VertexBufferLayout{.stride = 12, .attributes = positionAtTwo}, instanceLayout});
    checkRejected(std::array{rachis::VertexBufferLayout{.stride = 10, .attributes = positionAtZero}, instanceLayout});
    checkRejected(
        std::array{rachis::VertexBufferLayout{.stride = 4, .attributes = packedPositionAtOne}, instanceLayout});
    checkRejected(std::array{
        rachis::VertexBufferLayout{.stride = 2 * sizeof(float), .attributes = positionAtZero},
        rachis::VertexBufferLayout{
            .stride = 6,
            .stepMode = rachis::VertexStepMode::instance,
            .attributes = mixedInstanceAttributes,
        },
    });

    const std::array alignedLayouts{
        rachis::VertexBufferLayout{.stride = 12, .attributes = positionAtFour},
        instanceLayout,
    };
    const std::array packedLayouts{
        rachis::VertexBufferLayout{.stride = 6, .attributes = packedAttributes},
    };
    auto alignedPipeline = device.createGraphicsPipeline(makeDesc(alignedLayouts));
    auto packedPipeline = device.createGraphicsPipeline(makeDesc(packedLayouts));
    REQUIRE(alignedPipeline.has_value());
    REQUIRE(packedPipeline.has_value());

    const std::array alignedVertexData{
        0.0F,
        0.0F,
        0.0F,
        0.0F,
        0.5F,
        0.0F,
        0.0F,
        0.5F,
        -0.5F,
        0.0F,
        -0.5F,
        -0.5F,
    };
    const std::array instanceData{
        InstanceData{.offset = {0.0F, 0.0F}, .color = {255, 255, 255, 255}},
        InstanceData{.offset = {0.0F, 0.0F}, .color = {255, 255, 255, 255}},
    };
    const std::array<std::uint8_t, 20> packedVertexData{
        0, 0, 128, 255, 0, 0, 255, 255, 0, 0, 0, 0, 255, 0, 255, 255, 255, 255, 0, 0,
    };
    auto alignedVertices = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(alignedVertexData) + 4,
        .usage = rachis::BufferUsage::vertex,
        .memoryType = rachis::MemoryType::upload,
    });
    auto instances = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(instanceData),
        .usage = rachis::BufferUsage::vertex,
        .memoryType = rachis::MemoryType::upload,
    });
    auto packedVertices = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(packedVertexData),
        .usage = rachis::BufferUsage::vertex,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(alignedVertices.has_value());
    REQUIRE(instances.has_value());
    REQUIRE(packedVertices.has_value());
    REQUIRE(device.writeBuffer(*alignedVertices, 4, std::as_bytes(std::span{alignedVertexData})).has_value());
    REQUIRE(device.writeBuffer(*instances, 0, std::as_bytes(std::span{instanceData})).has_value());
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
    };

    bindDrawState(*alignedPipeline);
    REQUIRE(device.bindVertexBuffer(*command, 1, *instances).has_value());
    const auto misalignedVertex = device.bindVertexBuffer(*command, 0, *alignedVertices, 2);
    REQUIRE_FALSE(misalignedVertex.has_value());
    CHECK(misalignedVertex.error().code == rachis::ErrorCode::invalidArgument);
    const std::array misalignedBatch{
        rachis::VertexBufferBinding{.buffer = *alignedVertices, .offset = 4},
        rachis::VertexBufferBinding{.buffer = *instances, .offset = 2},
    };
    const auto misalignedInstance = device.bindVertexBuffers(*command, 0, misalignedBatch);
    REQUIRE_FALSE(misalignedInstance.has_value());
    CHECK(misalignedInstance.error().code == rachis::ErrorCode::invalidArgument);
    CHECK_FALSE(device.draw(*command, 3).has_value());
    REQUIRE(device.bindVertexBuffer(*command, 0, *alignedVertices, 4).has_value());
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

    REQUIRE(device.destroyBuffer(*alignedVertices).has_value());
    REQUIRE(device.destroyBuffer(*instances).has_value());
    REQUIRE(device.destroyBuffer(*packedVertices).has_value());
    REQUIRE(device.destroyPipeline(*alignedPipeline).has_value());
    REQUIRE(device.destroyPipeline(*packedPipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    device.collectGarbage();

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
