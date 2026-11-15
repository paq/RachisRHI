#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

namespace
{

    constexpr std::uint32_t targetWidth = 64;
    constexpr std::uint32_t targetHeight = 64;
    constexpr std::uint32_t rowPitch = 256;
    constexpr std::uint64_t readbackSize = static_cast<std::uint64_t>(rowPitch) * targetHeight;

    struct Vertex final
    {
        std::array<float, 2> position;
        std::array<std::uint8_t, 4> color;
    };

    static_assert(sizeof(Vertex) == 12);

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

} // namespace

TEST_CASE("primitive restart cuts indexed strips and rejects mismatched index formats",
          "[size:medium][rhi][graphics][primitive-restart][warp]")
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
    REQUIRE(rachis::test::requireCapabilities(device).primitiveRestart);

    const auto vertexBytecode = loadShader(RACHIS_TEST_PRIMITIVE_RESTART_VERTEX_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_PRIMITIVE_RESTART_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    auto vertexShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .entryPoint = "vertexMain",
        .shaderInterface = rachis::test::shaderInterface({}, {}, std::array{0U, 1U}),
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

    const std::array attributes{
        rachis::VertexAttribute{.format = rachis::VertexFormat::float32x2, .shaderLocation = 0},
        rachis::VertexAttribute{
            .format = rachis::VertexFormat::unorm8x4,
            .offset = 2 * sizeof(float),
            .shaderLocation = 1,
        },
    };
    const std::array vertexLayouts{
        rachis::VertexBufferLayout{.stride = sizeof(Vertex), .attributes = attributes},
    };
    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    const rachis::GraphicsPipelineDesc basePipelineDesc{
        .layout = *pipelineLayout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .vertexBuffers = vertexLayouts,
        .topology = rachis::PrimitiveTopology::triangleStrip,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
    };
    auto disabledPipeline = device.createGraphicsPipeline(basePipelineDesc);
    REQUIRE(disabledPipeline.has_value());

    auto uint16PipelineDesc = basePipelineDesc;
    uint16PipelineDesc.stripIndexFormat = rachis::IndexFormat::uint16;
    auto uint16Pipeline = device.createGraphicsPipeline(uint16PipelineDesc);
    REQUIRE(uint16Pipeline.has_value());
    auto uint32PipelineDesc = basePipelineDesc;
    uint32PipelineDesc.stripIndexFormat = rachis::IndexFormat::uint32;
    auto uint32Pipeline = device.createGraphicsPipeline(uint32PipelineDesc);
    REQUIRE(uint32Pipeline.has_value());
    auto linePipelineDesc = uint16PipelineDesc;
    linePipelineDesc.topology = rachis::PrimitiveTopology::lineStrip;
    auto linePipeline = device.createGraphicsPipeline(linePipelineDesc);
    REQUIRE(linePipeline.has_value());

    auto invalidListDesc = uint16PipelineDesc;
    invalidListDesc.topology = rachis::PrimitiveTopology::triangleList;
    CHECK_FALSE(device.createGraphicsPipeline(invalidListDesc).has_value());
    auto invalidFormatDesc = uint16PipelineDesc;
    invalidFormatDesc.stripIndexFormat = static_cast<rachis::IndexFormat>(255);
    CHECK_FALSE(device.createGraphicsPipeline(invalidFormatDesc).has_value());

    const std::array vertices{
        Vertex{.position = {-0.9F, -0.6F}, .color = {255, 0, 0, 255}},
        Vertex{.position = {-0.5F, 0.6F}, .color = {255, 0, 0, 255}},
        Vertex{.position = {-0.1F, -0.6F}, .color = {255, 0, 0, 255}},
        Vertex{.position = {0.1F, -0.6F}, .color = {0, 255, 0, 255}},
        Vertex{.position = {0.5F, 0.6F}, .color = {0, 255, 0, 255}},
        Vertex{.position = {0.9F, -0.6F}, .color = {0, 255, 0, 255}},
    };
    const std::array<std::uint16_t, 7> uint16Indices{
        0,
        1,
        2,
        std::numeric_limits<std::uint16_t>::max(),
        3,
        4,
        5,
    };
    const std::array<std::uint32_t, 7> uint32Indices{
        0,
        1,
        2,
        std::numeric_limits<std::uint32_t>::max(),
        3,
        4,
        5,
    };
    auto vertexBuffer = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(vertices),
        .usage = rachis::BufferUsage::vertex,
        .memoryType = rachis::MemoryType::upload,
    });
    auto uint16IndexBuffer = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(uint16Indices),
        .usage = rachis::BufferUsage::index,
        .memoryType = rachis::MemoryType::upload,
    });
    auto uint32IndexBuffer = device.createBuffer(rachis::BufferDesc{
        .size = sizeof(uint32Indices),
        .usage = rachis::BufferUsage::index,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(vertexBuffer.has_value());
    REQUIRE(uint16IndexBuffer.has_value());
    REQUIRE(uint32IndexBuffer.has_value());
    auto mappedVertices = device.mapBuffer(*vertexBuffer);
    auto mappedUint16 = device.mapBuffer(*uint16IndexBuffer);
    auto mappedUint32 = device.mapBuffer(*uint32IndexBuffer);
    REQUIRE(mappedVertices.has_value());
    REQUIRE(mappedUint16.has_value());
    REQUIRE(mappedUint32.has_value());
    std::memcpy(mappedVertices->data, vertices.data(), sizeof(vertices));
    std::memcpy(mappedUint16->data, uint16Indices.data(), sizeof(uint16Indices));
    std::memcpy(mappedUint32->data, uint32Indices.data(), sizeof(uint32Indices));
    REQUIRE(device.unmapBuffer(*vertexBuffer).has_value());
    REQUIRE(device.unmapBuffer(*uint16IndexBuffer).has_value());
    REQUIRE(device.unmapBuffer(*uint32IndexBuffer).has_value());

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
            .clearValue = {},
        },
    };
    REQUIRE(device.beginRendering(*command, rachis::RenderingDesc{.colorAttachments = attachments}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *uint16Pipeline).has_value());
    REQUIRE(device
                .setViewport(*command,
                             rachis::Viewport{
                                 .width = static_cast<float>(targetWidth),
                                 .height = static_cast<float>(targetHeight),
                             })
                .has_value());
    REQUIRE(device.setScissor(*command, rachis::ScissorRect{.width = targetWidth, .height = targetHeight}).has_value());
    REQUIRE(device.bindVertexBuffer(*command, 0, *vertexBuffer).has_value());
    CHECK_FALSE(device.bindIndexBuffer(*command, *uint32IndexBuffer, rachis::IndexFormat::uint32).has_value());
    CHECK_FALSE(
        device.bindIndexBuffer(*command, *uint16IndexBuffer, static_cast<rachis::IndexFormat>(255)).has_value());
    CHECK_FALSE(device.drawIndexed(*command, static_cast<std::uint32_t>(uint16Indices.size())).has_value());
    REQUIRE(device.bindIndexBuffer(*command, *uint16IndexBuffer, rachis::IndexFormat::uint16).has_value());
    REQUIRE(device.drawIndexed(*command, static_cast<std::uint32_t>(uint16Indices.size())).has_value());

    REQUIRE(device.bindGraphicsPipeline(*command, *uint32Pipeline).has_value());
    REQUIRE(device.bindVertexBuffer(*command, 0, *vertexBuffer).has_value());
    CHECK_FALSE(device.bindIndexBuffer(*command, *uint16IndexBuffer, rachis::IndexFormat::uint16).has_value());
    REQUIRE(device.bindIndexBuffer(*command, *uint32IndexBuffer, rachis::IndexFormat::uint32).has_value());
    REQUIRE(device.drawIndexed(*command, static_cast<std::uint32_t>(uint32Indices.size())).has_value());
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
    REQUIRE(device.destroyBuffer(*uint16IndexBuffer).has_value());
    REQUIRE(device.destroyBuffer(*uint32IndexBuffer).has_value());
    REQUIRE(device.destroyPipeline(*disabledPipeline).has_value());
    REQUIRE(device.destroyPipeline(*uint16Pipeline).has_value());
    REQUIRE(device.destroyPipeline(*uint32Pipeline).has_value());
    REQUIRE(device.destroyPipeline(*linePipeline).has_value());
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
