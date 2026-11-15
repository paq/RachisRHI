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
#include <string_view>
#include <vector>

namespace
{

    constexpr std::uint32_t targetWidth = 1;
    constexpr std::uint32_t targetHeight = 1;
    constexpr std::uint32_t rowPitch = 256;
    constexpr std::uint64_t copyPlacementAlignment = 512;

    struct OperationCase final
    {
        rachis::LogicOperation operation;
        std::uint8_t truthTable;
        std::string_view name;
    };

    constexpr std::array operationCases{
        OperationCase{rachis::LogicOperation::clear, 0x0, "clear"},
        OperationCase{rachis::LogicOperation::andOperation, 0x8, "and"},
        OperationCase{rachis::LogicOperation::andReverse, 0x2, "and reverse"},
        OperationCase{rachis::LogicOperation::copy, 0xA, "copy"},
        OperationCase{rachis::LogicOperation::andInverted, 0x4, "and inverted"},
        OperationCase{rachis::LogicOperation::noOp, 0xC, "no-op"},
        OperationCase{rachis::LogicOperation::xorOperation, 0x6, "xor"},
        OperationCase{rachis::LogicOperation::orOperation, 0xE, "or"},
        OperationCase{rachis::LogicOperation::nor, 0x1, "nor"},
        OperationCase{rachis::LogicOperation::equivalent, 0x9, "equivalent"},
        OperationCase{rachis::LogicOperation::invert, 0x3, "invert"},
        OperationCase{rachis::LogicOperation::orReverse, 0xB, "or reverse"},
        OperationCase{rachis::LogicOperation::copyInverted, 0x5, "copy inverted"},
        OperationCase{rachis::LogicOperation::orInverted, 0xD, "or inverted"},
        OperationCase{rachis::LogicOperation::nand, 0x7, "nand"},
        OperationCase{rachis::LogicOperation::set, 0xF, "set"},
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

} // namespace

TEST_CASE("logic operations validate global output-merger state and execute all Boolean functions",
          "[size:medium][rhi][graphics][logic-operation][warp]")
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
    REQUIRE(rachis::test::requireCapabilities(device).logicOperations);

    const auto rgba8Properties = device.queryFormatProperties(rachis::Format::rgba8Unorm);
    REQUIRE_FALSE(hasAny(rgba8Properties.features, rachis::FormatFeature::logicOperation));
    const auto r32UintProperties = device.queryFormatProperties(rachis::Format::r32Uint);
    REQUIRE(hasAny(r32UintProperties.features, rachis::FormatFeature::logicOperation));
    const auto rgba16FloatProperties = device.queryFormatProperties(rachis::Format::rgba16Float);
    REQUIRE_FALSE(hasAny(rgba16FloatProperties.features, rachis::FormatFeature::logicOperation));

    const auto vertexBytecode = loadShader(RACHIS_TEST_GRAPHICS_DIRECT_VERTEX_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_LOGIC_OPERATION_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());

    auto vertexShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .entryPoint = "directVertex",
        .label = "Logic operation vertex shader",
        .shaderInterface = {},
    });
    auto pixelShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "pixelMain",
        .label = "Logic operation pixel shader",
        .shaderInterface = rachis::test::shaderInterface(
            {}, std::array{rachis::ShaderPushConstantRequirement{0, sizeof(std::uint32_t)}}),
    });
    REQUIRE(vertexShader.has_value());
    REQUIRE(pixelShader.has_value());

    const std::array pushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::pixel, 0, sizeof(std::uint32_t)},
    };
    auto layout = device.createPipelineLayout({
        .pushConstantRanges = pushConstantRanges,
        .label = "Logic operation pipeline layout",
    });
    REQUIRE(layout.has_value());

    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::r32Uint},
    };
    const auto basePipelineDesc = rachis::GraphicsPipelineDesc{
        .layout = *layout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
    };

    auto invalidOperationDesc = basePipelineDesc;
    invalidOperationDesc.logicOperation = static_cast<rachis::LogicOperation>(255);
    auto invalidOperation = device.createGraphicsPipeline(invalidOperationDesc);
    REQUIRE_FALSE(invalidOperation.has_value());
    CHECK(invalidOperation.error().code == rachis::ErrorCode::invalidArgument);

    auto noColorTargetDesc = basePipelineDesc;
    noColorTargetDesc.colorTargets = {};
    noColorTargetDesc.depthStencil.format = rachis::Format::d32Float;
    noColorTargetDesc.logicOperation = rachis::LogicOperation::copy;
    auto noColorTarget = device.createGraphicsPipeline(noColorTargetDesc);
    REQUIRE_FALSE(noColorTarget.has_value());
    CHECK(noColorTarget.error().code == rachis::ErrorCode::invalidArgument);

    auto blendedTarget = colorTargets;
    blendedTarget[0].blend.enabled = true;
    auto blendedDesc = basePipelineDesc;
    blendedDesc.colorTargets = blendedTarget;
    blendedDesc.logicOperation = rachis::LogicOperation::copy;
    auto blended = device.createGraphicsPipeline(blendedDesc);
    REQUIRE_FALSE(blended.has_value());
    CHECK(blended.error().code == rachis::ErrorCode::invalidArgument);

    const std::array mismatchedWriteMasks{
        rachis::ColorTargetState{
            .format = rachis::Format::r32Uint,
            .writeMask = rachis::ColorWriteMask::all,
        },
        rachis::ColorTargetState{
            .format = rachis::Format::r32Uint,
            .writeMask = rachis::ColorWriteMask::red,
        },
    };
    auto mismatchedWriteMaskDesc = basePipelineDesc;
    mismatchedWriteMaskDesc.colorTargets = mismatchedWriteMasks;
    mismatchedWriteMaskDesc.logicOperation = rachis::LogicOperation::copy;
    auto mismatchedWriteMask = device.createGraphicsPipeline(mismatchedWriteMaskDesc);
    REQUIRE_FALSE(mismatchedWriteMask.has_value());
    CHECK(mismatchedWriteMask.error().code == rachis::ErrorCode::invalidArgument);

    const std::array unsupportedTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba16Float},
    };
    auto unsupportedFormatDesc = basePipelineDesc;
    unsupportedFormatDesc.colorTargets = unsupportedTargets;
    unsupportedFormatDesc.logicOperation = rachis::LogicOperation::copy;
    auto unsupportedFormat = device.createGraphicsPipeline(unsupportedFormatDesc);
    REQUIRE_FALSE(unsupportedFormat.has_value());
    CHECK(unsupportedFormat.error().code == rachis::ErrorCode::unsupported);

    auto seedPipeline = device.createGraphicsPipeline(basePipelineDesc);
    REQUIRE(seedPipeline.has_value());

    std::array<rachis::PipelineHandle, operationCases.size()> pipelines{};
    for (std::size_t index = 0; index < operationCases.size(); ++index)
    {
        auto pipelineDesc = basePipelineDesc;
        pipelineDesc.logicOperation = operationCases[index].operation;
        pipelineDesc.label = operationCases[index].name;
        auto pipeline = device.createGraphicsPipeline(pipelineDesc);
        if (!pipeline)
        {
            INFO(pipeline.error().message);
        }
        REQUIRE(pipeline.has_value());
        pipelines[index] = *pipeline;
    }

    auto target = device.createTexture({
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::r32Uint,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
        .label = "Logic operation truth-table target",
    });
    REQUIRE(target.has_value());
    auto targetView = device.createTextureView({
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    REQUIRE(targetView.has_value());

    constexpr auto readbackBytes = copyPlacementAlignment * operationCases.size();
    auto readback = device.createBuffer({
        .size = readbackBytes,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Logic operation truth-table readback",
    });
    REQUIRE(readback.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    const std::array source{UINT32_C(0xAAAAAAAA)};
    const std::array destination{UINT32_C(0xCCCCCCCC)};
    const std::array attachments{
        rachis::ColorAttachment{
            .view = *targetView,
            .loadOperation = rachis::LoadOperation::discard,
        },
    };
    for (std::size_t index = 0; index < operationCases.size(); ++index)
    {
        REQUIRE(device
                    .textureBarrier(
                        *command,
                        {
                            .texture = *target,
                            .stagesBefore = index == 0 ? rachis::PipelineStage::all : rachis::PipelineStage::copy,
                            .stagesAfter = rachis::PipelineStage::renderTarget,
                            .accessBefore = index == 0 ? rachis::Access::none : rachis::Access::copySource,
                            .accessAfter = rachis::Access::renderTarget,
                            .layoutBefore =
                                index == 0 ? rachis::TextureLayout::undefined : rachis::TextureLayout::copySource,
                            .layoutAfter = rachis::TextureLayout::renderTarget,
                        })
                    .has_value());
        REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
        REQUIRE(device.bindGraphicsPipeline(*command, *seedPipeline).has_value());
        REQUIRE(device
                    .setViewport(*command,
                                 {
                                     .width = static_cast<float>(targetWidth),
                                     .height = static_cast<float>(targetHeight),
                                 })
                    .has_value());
        REQUIRE(device.setScissor(*command, {.width = targetWidth, .height = targetHeight}).has_value());
        REQUIRE(device
                    .setGraphicsPushConstants(
                        *command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(destination)))
                    .has_value());
        REQUIRE(device.draw(*command, 3).has_value());
        REQUIRE(device.bindGraphicsPipeline(*command, pipelines[index]).has_value());
        REQUIRE(
            device.setGraphicsPushConstants(*command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(source)))
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
                                             .bufferOffset = index * copyPlacementAlignment,
                                             .bytesPerRow = rowPitch,
                                             .rowsPerImage = targetHeight,
                                             .texture = *target,
                                             .extent = {targetWidth, targetHeight, 1},
                                         })
                    .has_value());
    }

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    for (std::size_t operationIndex = 0; operationIndex < operationCases.size(); ++operationIndex)
    {
        const auto* result = mapped->data + operationIndex * copyPlacementAlignment;
        std::uint32_t actual = 0;
        std::memcpy(&actual, result, sizeof(actual));
        const auto expected =
            static_cast<std::uint32_t>(operationCases[operationIndex].truthTable) * UINT32_C(0x11111111);
        CAPTURE(operationCases[operationIndex].name, operationIndex, actual, expected);
        CHECK(actual == expected);
    }
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    for (const auto pipeline : pipelines)
    {
        REQUIRE(device.destroyPipeline(pipeline).has_value());
    }
    REQUIRE(device.destroyPipeline(*seedPipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*layout).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
