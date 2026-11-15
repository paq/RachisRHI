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
#include <vector>

namespace
{

    constexpr std::uint32_t targetSize = 64;
    constexpr std::uint32_t rowPitch = 256;
    constexpr std::uint64_t readbackSize = static_cast<std::uint64_t>(rowPitch) * targetSize;

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

} // namespace

TEST_CASE("tessellation pipelines retain paired stages and draw patch lists",
          "[size:medium][rhi][graphics][tessellation][warp]")
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
    REQUIRE(rachis::test::requireCapabilities(device).tessellationShaders);

    const auto vertexBytecode = loadShader(RACHIS_TEST_TESSELLATION_VERTEX_SHADER_PATH);
    const auto hullBytecode = loadShader(RACHIS_TEST_TESSELLATION_HULL_SHADER_PATH);
    const auto domainBytecode = loadShader(RACHIS_TEST_TESSELLATION_DOMAIN_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_TESSELLATION_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(hullBytecode.empty());
    REQUIRE_FALSE(domainBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());

    auto vertexShader = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .entryPoint = "tessellationVertex",
        .shaderInterface = {},
    });
    auto hullShader = device.createShaderModule({
        .stage = rachis::ShaderStage::hull,
        .bytecode = hullBytecode,
        .entryPoint = "tessellationHull",
        .shaderInterface = rachis::test::shaderInterface(
            std::array{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1}}),
    });
    auto domainShader = device.createShaderModule({
        .stage = rachis::ShaderStage::domain,
        .bytecode = domainBytecode,
        .entryPoint = "tessellationDomain",
        .shaderInterface =
            rachis::test::shaderInterface({}, std::array{rachis::ShaderPushConstantRequirement{0, 4 * sizeof(float)}}),
    });
    auto pixelShader = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "tessellationPixel",
        .shaderInterface = {},
    });
    REQUIRE(vertexShader.has_value());
    REQUIRE(hullShader.has_value());
    REQUIRE(domainShader.has_value());
    REQUIRE(pixelShader.has_value());

    auto constants = device.createBuffer({
        .size = 256,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Hull-stage tessellation constants",
    });
    REQUIRE(constants.has_value());
    auto mapped = device.mapBuffer(*constants);
    REQUIRE(mapped.has_value());
    std::memset(mapped->data, 0, mapped->size);
    const std::array hullConstants{0.0F, 1.0F, 0.0F, 1.0F, 4.0F, 0.0F, 0.0F, 0.0F};
    std::memcpy(mapped->data, hullConstants.data(), sizeof(hullConstants));
    REQUIRE(device.unmapBuffer(*constants).has_value());

    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::hull, 1},
    };
    auto bindingLayout = device.createBindingLayout({
        .entries = bindingEntries,
        .label = "Hull-only binding layout",
    });
    REQUIRE(bindingLayout.has_value());
    const std::array bindingWrites{
        rachis::BindingWrite{
            .binding = 0,
            .resource = rachis::BufferBinding{.buffer = *constants, .size = 256},
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device,
                                                     {
                                                         .layout = *bindingLayout,
                                                         .writes = bindingWrites,
                                                         .label = "Hull-only binding set",
                                                     });
    REQUIRE(bindingSet.has_value());

    const std::array bindingLayouts{*bindingLayout};
    const std::array pushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::domain, 0, 4 * sizeof(float)},
    };
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = pushConstantRanges,
        .label = "Tessellation contract pipeline layout",
    });
    REQUIRE(pipelineLayout.has_value());

    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    const rachis::GraphicsPipelineDesc pipelineDesc{
        .layout = *pipelineLayout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .hullShader = *hullShader,
        .domainShader = *domainShader,
        .topology = rachis::PrimitiveTopology::patchList,
        .patchControlPoints = 3,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .label = "Tessellation patch pipeline",
    };

    auto invalid = pipelineDesc;
    invalid.domainShader = {};
    CHECK_FALSE(device.createGraphicsPipeline(invalid).has_value());
    invalid = pipelineDesc;
    invalid.domainShader = *pixelShader;
    CHECK_FALSE(device.createGraphicsPipeline(invalid).has_value());
    invalid = pipelineDesc;
    invalid.topology = rachis::PrimitiveTopology::triangleList;
    CHECK_FALSE(device.createGraphicsPipeline(invalid).has_value());
    invalid = pipelineDesc;
    invalid.patchControlPoints = 0;
    CHECK_FALSE(device.createGraphicsPipeline(invalid).has_value());
    invalid.patchControlPoints = 33;
    CHECK_FALSE(device.createGraphicsPipeline(invalid).has_value());

    auto pipeline = device.createGraphicsPipeline(pipelineDesc);
    REQUIRE(pipeline.has_value());
    const auto staleHullShader = *hullShader;
    const auto staleDomainShader = *domainShader;
    REQUIRE(device.destroyShaderModule(*hullShader).has_value());
    REQUIRE(device.destroyShaderModule(*domainShader).has_value());
    REQUIRE_FALSE(device.createGraphicsPipeline(pipelineDesc).has_value());

    auto target = device.createTexture({
        .width = targetSize,
        .height = targetSize,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
        .label = "Tessellation contract target",
    });
    auto readback = device.createBuffer({
        .size = readbackSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Tessellation contract readback",
    });
    REQUIRE(target.has_value());
    REQUIRE(readback.has_value());
    auto targetView = device.createTextureView({
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
        .label = "Tessellation contract target view",
    });
    REQUIRE(targetView.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
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
            .clearValue = {0.0F, 0.0F, 1.0F, 1.0F},
        },
    };
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindGraphicsBindingSet(*command, 0, *bindingSet).has_value());
    REQUIRE(device
                .setViewport(*command,
                             {
                                 .width = static_cast<float>(targetSize),
                                 .height = static_cast<float>(targetSize),
                                 .maxDepth = 1.0F,
                             })
                .has_value());
    REQUIRE(device.setScissor(*command, {.width = targetSize, .height = targetSize}).has_value());
    const std::array domainControl{1.0F, 0.0F, 0.0F, 0.0F};
    REQUIRE(
        device
            .setGraphicsPushConstants(*command, rachis::ShaderStage::domain, 0, std::as_bytes(std::span(domainControl)))
            .has_value());
    REQUIRE(device.draw(*command, 3).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
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
                                     {
                                         .buffer = *readback,
                                         .bytesPerRow = rowPitch,
                                         .rowsPerImage = targetSize,
                                         .texture = *target,
                                         .extent = {targetSize, targetSize, 1},
                                     })
                .has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    const auto* corner = mapped->data;
    const auto* center = mapped->data + static_cast<std::uint64_t>(targetSize / 2) * rowPitch +
                         static_cast<std::uint64_t>(targetSize / 2) * 4;
    CHECK(corner[0] == std::byte{0});
    CHECK(corner[1] == std::byte{0});
    CHECK(corner[2] == std::byte{255});
    CHECK(corner[3] == std::byte{255});
    CHECK(center[0] == std::byte{0});
    CHECK(center[1] == std::byte{255});
    CHECK(center[2] == std::byte{0});
    CHECK(center[3] == std::byte{255});
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyBuffer(*constants).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
    REQUIRE_FALSE(device.destroyShaderModule(staleHullShader).has_value());
    REQUIRE_FALSE(device.destroyShaderModule(staleDomainShader).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
