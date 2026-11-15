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
    constexpr std::uint64_t imageBytes = static_cast<std::uint64_t>(rowPitch) * targetHeight;
    constexpr std::uint64_t readbackBytes = 3 * imageBytes;

    struct DistancePushConstants final
    {
        float clipDistance = 1.0F;
        float cullDistance = 1.0F;
    };

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

    [[nodiscard]] std::size_t coloredPixelCount(const rachis::MappedMemory& mapped, std::uint64_t imageOffset)
    {
        std::size_t count = 0;
        for (std::uint32_t y = 0; y < targetHeight; ++y)
        {
            for (std::uint32_t x = 0; x < targetWidth; ++x)
            {
                const auto offset =
                    imageOffset + static_cast<std::uint64_t>(y) * rowPitch + static_cast<std::uint64_t>(x) * 4;
                count += mapped.data[offset] != std::byte{0} ? 1U : 0U;
            }
        }
        return count;
    }

} // namespace

TEST_CASE("shader clip and cull distances reject primitives on WARP",
          "[size:medium][rhi][graphics][shader-distance][warp]")
{
    auto adapters = rachis::enumerateAdapters();
    REQUIRE(adapters.has_value());
    const auto selected = std::ranges::find_if(
        *adapters, [](const auto& adapter) { return adapter.type == rachis::AdapterType::software; });
    REQUIRE(selected != adapters->end());
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .adapterId = selected->id,
        .validation = rachis::ValidationLevel::standard,
        .diagnosticCallback = &printDiagnostic,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    const auto limits = rachis::test::requireCapabilities(device).limits;

    const auto vertexBytecode = loadShader(RACHIS_TEST_SHADER_DISTANCE_VERTEX_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_SHADER_DISTANCE_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    auto vertexShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .label = "Shader-distance vertex shader",
        .shaderInterface = rachis::test::shaderInterface(
            {}, std::array{rachis::ShaderPushConstantRequirement{0, sizeof(DistancePushConstants)}}),
    });
    auto pixelShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .label = "Shader-distance pixel shader",
        .shaderInterface = {},
    });
    REQUIRE(vertexShader.has_value());
    REQUIRE(pixelShader.has_value());

    const std::array pushConstantRanges{
        rachis::PushConstantRange{
            .visibility = rachis::ShaderStage::vertex,
            .size = sizeof(DistancePushConstants),
        },
    };
    auto layout = device.createPipelineLayout(rachis::PipelineLayoutDesc{
        .pushConstantRanges = pushConstantRanges,
        .label = "Shader-distance pipeline layout",
    });
    REQUIRE(layout.has_value());
    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    const rachis::GraphicsPipelineDesc pipelineDesc{
        .layout = *layout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .clipDistanceCount = 1,
        .cullDistanceCount = 1,
        .label = "Shader-distance pipeline",
    };
    auto invalidClipDesc = pipelineDesc;
    invalidClipDesc.clipDistanceCount = limits.maxClipDistances + 1;
    auto invalidClip = device.createGraphicsPipeline(invalidClipDesc);
    REQUIRE_FALSE(invalidClip.has_value());
    CHECK(invalidClip.error().code == rachis::ErrorCode::invalidArgument);
    auto invalidCullDesc = pipelineDesc;
    invalidCullDesc.cullDistanceCount = limits.maxCullDistances + 1;
    auto invalidCull = device.createGraphicsPipeline(invalidCullDesc);
    REQUIRE_FALSE(invalidCull.has_value());
    CHECK(invalidCull.error().code == rachis::ErrorCode::invalidArgument);
    auto invalidCombinedDesc = pipelineDesc;
    invalidCombinedDesc.clipDistanceCount = limits.maxCombinedClipAndCullDistances;
    invalidCombinedDesc.cullDistanceCount = 1;
    auto invalidCombined = device.createGraphicsPipeline(invalidCombinedDesc);
    REQUIRE_FALSE(invalidCombined.has_value());
    CHECK(invalidCombined.error().code == rachis::ErrorCode::invalidArgument);

    auto pipeline = device.createGraphicsPipeline(pipelineDesc);
    REQUIRE(pipeline.has_value());
    auto target = device.createTexture(rachis::TextureDesc{
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
        .label = "Shader-distance target",
    });
    REQUIRE(target.has_value());
    auto targetView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
        .label = "Shader-distance target view",
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = readbackBytes,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Shader-distance readback",
    });
    REQUIRE(targetView.has_value());
    REQUIRE(readback.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    const auto recordDraw = [&](DistancePushConstants distances, std::uint64_t bufferOffset, bool firstDraw) {
        auto toRenderTarget = rachis::TextureBarrier{
            .texture = *target,
            .stagesAfter = rachis::PipelineStage::renderTarget,
            .accessAfter = rachis::Access::renderTarget,
            .layoutAfter = rachis::TextureLayout::renderTarget,
        };
        if (!firstDraw)
        {
            toRenderTarget.stagesBefore = rachis::PipelineStage::copy;
            toRenderTarget.accessBefore = rachis::Access::copySource;
            toRenderTarget.layoutBefore = rachis::TextureLayout::copySource;
        }
        REQUIRE(device.textureBarrier(*command, toRenderTarget).has_value());
        const std::array attachments{
            rachis::ColorAttachment{
                .view = *targetView,
                .loadOperation = rachis::LoadOperation::clear,
                .clearValue = {0.0F, 0.0F, 0.0F, 1.0F},
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
        REQUIRE(
            device.setScissor(*command, rachis::ScissorRect{.width = targetWidth, .height = targetHeight}).has_value());
        REQUIRE(device
                    .setGraphicsPushConstants(
                        *command, rachis::ShaderStage::vertex, 0, std::as_bytes(std::span(&distances, 1)))
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
    recordDraw({.clipDistance = 1.0F, .cullDistance = 1.0F}, 0, true);
    recordDraw({.clipDistance = -1.0F, .cullDistance = 1.0F}, imageBytes, false);
    recordDraw({.clipDistance = 1.0F, .cullDistance = -1.0F}, 2 * imageBytes, false);

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    REQUIRE(mapped->size == readbackBytes);
    CHECK(coloredPixelCount(*mapped, 0) > 0);
    CHECK(coloredPixelCount(*mapped, imageBytes) == 0);
    CHECK(coloredPixelCount(*mapped, 2 * imageBytes) == 0);
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
