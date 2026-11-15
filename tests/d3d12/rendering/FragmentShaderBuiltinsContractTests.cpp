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
#include <optional>
#include <string>
#include <vector>

namespace
{

    constexpr std::uint32_t targetWidth = 64;
    constexpr std::uint32_t targetHeight = 64;
    constexpr std::uint32_t rowPitch = 256;
    constexpr std::size_t readbackSize = static_cast<std::size_t>(rowPitch) * targetHeight;

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

    [[nodiscard]] rachis::Result<rachis::Device> createAdapterDevice(const rachis::AdapterInfo& adapter)
    {
        return rachis::createDevice(rachis::DeviceDesc{
            .adapterPreference = adapter.type == rachis::AdapterType::software
                                     ? rachis::AdapterPreference::warp
                                     : rachis::AdapterPreference::highPerformance,
            .adapterId = adapter.id,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = true,
            .diagnosticCallback = &printDiagnostic,
        });
    }

    void render(rachis::Device& device,
                rachis::PipelineHandle pipeline,
                std::uint32_t vertexCount,
                std::array<std::byte, readbackSize>& output)
    {
        auto target = device.createTexture(rachis::TextureDesc{
            .width = targetWidth,
            .height = targetHeight,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
            .label = "Fragment built-in target",
        });
        REQUIRE(target.has_value());
        auto targetView = device.createTextureView(rachis::TextureViewDesc{
            .texture = *target,
            .usage = rachis::TextureViewUsage::renderTarget,
            .label = "Fragment built-in target view",
        });
        auto readback = device.createBuffer(rachis::BufferDesc{
            .size = readbackSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Fragment built-in readback",
        });
        REQUIRE(targetView.has_value());
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
        REQUIRE(device.beginRendering(*command, rachis::RenderingDesc{.colorAttachments = attachments}).has_value());
        REQUIRE(device
                    .setViewport(*command,
                                 rachis::Viewport{
                                     .width = static_cast<float>(targetWidth),
                                     .height = static_cast<float>(targetHeight),
                                 })
                    .has_value());
        REQUIRE(
            device.setScissor(*command, rachis::ScissorRect{.width = targetWidth, .height = targetHeight}).has_value());
        REQUIRE(device.bindGraphicsPipeline(*command, pipeline).has_value());
        REQUIRE(device.draw(*command, vertexCount).has_value());
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
        REQUIRE(mapped->size == readbackSize);
        std::memcpy(output.data(), mapped->data, readbackSize);
        REQUIRE(device.unmapBuffer(*readback).has_value());

        REQUIRE(device.destroyBuffer(*readback).has_value());
        REQUIRE(device.destroyTextureView(*targetView).has_value());
        REQUIRE(device.destroyTexture(*target).has_value());
    }

    [[nodiscard]] const std::byte* pixel(const std::array<std::byte, readbackSize>& output,
                                         std::uint32_t x,
                                         std::uint32_t y) noexcept
    {
        return output.data() + static_cast<std::size_t>(y) * rowPitch + static_cast<std::size_t>(x) * 4;
    }

    void checkColor(const std::byte* actual, std::array<std::uint8_t, 4> expected)
    {
        CHECK(std::to_integer<std::uint8_t>(actual[0]) == expected[0]);
        CHECK(std::to_integer<std::uint8_t>(actual[1]) == expected[1]);
        CHECK(std::to_integer<std::uint8_t>(actual[2]) == expected[2]);
        CHECK(std::to_integer<std::uint8_t>(actual[3]) == expected[3]);
    }

} // namespace

TEST_CASE("fragment primitive IDs distinguish primitives in one draw",
          "[size:medium][rhi][graphics][fragment-builtins][warp]")
{
    auto adapters = rachis::enumerateAdapters();
    REQUIRE(adapters.has_value());
    const auto selected = std::ranges::find_if(
        *adapters, [](const auto& adapter) { return adapter.type == rachis::AdapterType::software; });
    REQUIRE(selected != adapters->end());
    auto created = createAdapterDevice(*selected);
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).fragmentPrimitiveId);

    const auto vertexBytecode = loadShader(RACHIS_TEST_FRAGMENT_PRIMITIVE_ID_VERTEX_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_FRAGMENT_PRIMITIVE_ID_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    auto vertexShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .entryPoint = "primitiveIdVertex",
        .label = "Fragment primitive ID vertex shader",
        .shaderInterface = {},
    });
    auto pixelShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "primitiveIdPixel",
        .label = "Fragment primitive ID pixel shader",
        .shaderInterface = {},
    });
    REQUIRE(vertexShader.has_value());
    REQUIRE(pixelShader.has_value());
    auto layout = device.createPipelineLayout({});
    REQUIRE(layout.has_value());
    const std::array targets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    auto pipeline = device.createGraphicsPipeline(rachis::GraphicsPipelineDesc{
        .layout = *layout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = targets,
        .label = "Fragment primitive ID pipeline",
    });
    REQUIRE(pipeline.has_value());

    std::array<std::byte, readbackSize> output{};
    render(device, *pipeline, 6, output);
    checkColor(pixel(output, 16, 32), {255, 0, 0, 255});
    checkColor(pixel(output, 48, 32), {0, 255, 0, 255});

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*layout).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("reported fragment barycentric coordinates identify all three vertex regions",
          "[size:medium][rhi][graphics][fragment-builtins][barycentrics]")
{
    auto adapters = rachis::enumerateAdapters();
    REQUIRE(adapters.has_value());
    std::optional<rachis::Device> selectedDevice;
    std::string selectedAdapter;
    for (const auto& adapter : *adapters)
    {
        auto created = createAdapterDevice(adapter);
        if (created && rachis::test::requireCapabilities(*created).fragmentBarycentricCoordinates)
        {
            selectedAdapter = adapter.name;
            selectedDevice.emplace(std::move(*created));
            break;
        }
    }
    if (!selectedDevice)
    {
        SKIP("No D3D12 adapter reports fragment barycentric coordinates.");
    }
    INFO("Selected adapter: " << selectedAdapter);
    auto& device = *selectedDevice;
    REQUIRE(rachis::test::requireCapabilities(device).shaderModel >= rachis::ApiVersion{6, 1});

    const auto vertexBytecode = loadShader(RACHIS_TEST_FRAGMENT_BARYCENTRIC_VERTEX_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_FRAGMENT_BARYCENTRIC_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    auto vertexShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .entryPoint = "barycentricVertex",
        .label = "Fragment barycentric vertex shader",
        .shaderInterface = {},
    });
    auto pixelShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "barycentricPixel",
        .label = "Fragment barycentric pixel shader",
        .shaderInterface = {},
    });
    REQUIRE(vertexShader.has_value());
    REQUIRE(pixelShader.has_value());
    auto layout = device.createPipelineLayout({});
    REQUIRE(layout.has_value());
    const std::array targets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    auto pipeline = device.createGraphicsPipeline(rachis::GraphicsPipelineDesc{
        .layout = *layout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = targets,
        .label = "Fragment barycentric pipeline",
    });
    REQUIRE(pipeline.has_value());

    std::array<std::byte, readbackSize> output{};
    render(device, *pipeline, 3, output);
    checkColor(pixel(output, 32, 12), {255, 0, 0, 255});
    checkColor(pixel(output, 50, 50), {0, 255, 0, 255});
    checkColor(pixel(output, 14, 50), {0, 0, 255, 255});

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*layout).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
