#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace
{

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

    [[nodiscard]] bool isObservableState(rachis::PipelineCompilationState state) noexcept
    {
        return state == rachis::PipelineCompilationState::pending ||
               state == rachis::PipelineCompilationState::succeeded;
    }

} // namespace

TEST_CASE("pipeline compilation retains inputs and publishes each result once",
          "[size:medium][rhi][pipeline][compilation][warp]")
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

    const auto computeBytecode = loadShader(RACHIS_TEST_COMPUTE_SHADER_PATH);
    REQUIRE_FALSE(computeBytecode.empty());
    auto computeShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::compute,
        .bytecode = computeBytecode,
        .entryPoint = "main",
        .label = "Async compute shader",
        .shaderInterface = rachis::test::computeWriteShaderInterface(),
    });
    REQUIRE(computeShader.has_value());
    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = bindingEntries, .label = "Async binding layout"});
    REQUIRE(bindingLayout.has_value());
    const std::array bindingLayouts{*bindingLayout};
    const std::array pushConstants{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)},
    };
    auto computeLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = pushConstants,
        .label = "Async compute layout",
    });
    REQUIRE(computeLayout.has_value());

    std::string computeLabel = "Async compute pipeline";
    const rachis::ComputePipelineDesc computeDesc{
        .layout = *computeLayout,
        .computeShader = *computeShader,
        .label = computeLabel,
    };
    auto compilation = device.beginComputePipelineCompilation(computeDesc);
    auto abandoned = device.beginComputePipelineCompilation(computeDesc);
    REQUIRE(compilation.has_value());
    REQUIRE(abandoned.has_value());
    const auto observed = device.pipelineCompilationState(*compilation);
    REQUIRE(observed.has_value());
    CHECK(isObservableState(*observed));
    REQUIRE(device.destroyPipelineCompilation(*abandoned).has_value());
    CHECK_FALSE(device.destroyPipelineCompilation(*abandoned).has_value());

    computeLabel.assign("caller storage was changed");
    REQUIRE(device.destroyPipelineLayout(*computeLayout).has_value());
    REQUIRE(device.destroyShaderModule(*computeShader).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());

    auto computePipeline = device.waitPipelineCompilation(*compilation);
    REQUIRE(computePipeline.has_value());
    CHECK_FALSE(device.pipelineCompilationState(*compilation).has_value());
    CHECK_FALSE(device.waitPipelineCompilation(*compilation).has_value());
    REQUIRE(device.destroyPipeline(*computePipeline).has_value());

    auto wrongStageShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = computeBytecode,
        .entryPoint = "main",
        .shaderInterface = {},
    });
    auto failureLayout = device.createPipelineLayout({});
    REQUIRE(wrongStageShader.has_value());
    REQUIRE(failureLayout.has_value());
    auto failed = device.beginComputePipelineCompilation({
        .layout = *failureLayout,
        .computeShader = *wrongStageShader,
    });
    REQUIRE(failed.has_value());
    REQUIRE(device.destroyPipelineLayout(*failureLayout).has_value());
    REQUIRE(device.destroyShaderModule(*wrongStageShader).has_value());
    CHECK_FALSE(device.waitPipelineCompilation(*failed).has_value());
    CHECK_FALSE(device.pipelineCompilationState(*failed).has_value());

    const auto vertexBytecode = loadShader(RACHIS_TEST_GRAPHICS_DIRECT_VERTEX_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_GRAPHICS_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    auto vertexShader = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .entryPoint = "main",
        .shaderInterface = {},
    });
    auto pixelShader = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "main",
        .shaderInterface = rachis::test::graphicsPixelShaderInterface(),
    });
    const std::array graphicsPushConstants{
        rachis::PushConstantRange{rachis::ShaderStage::pixel, 0, 4 * sizeof(float)},
    };
    auto graphicsLayout = device.createPipelineLayout({
        .pushConstantRanges = graphicsPushConstants,
    });
    REQUIRE(vertexShader.has_value());
    REQUIRE(pixelShader.has_value());
    REQUIRE(graphicsLayout.has_value());

    std::vector<rachis::VertexBufferLayout> vertexBuffers;
    std::vector colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    std::string graphicsLabel = "Async graphics pipeline";
    auto graphicsCompilation = device.beginGraphicsPipelineCompilation({
        .layout = *graphicsLayout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .vertexBuffers = vertexBuffers,
        .colorTargets = colorTargets,
        .label = graphicsLabel,
    });
    REQUIRE(graphicsCompilation.has_value());
    colorTargets.clear();
    graphicsLabel.clear();
    REQUIRE(device.destroyPipelineLayout(*graphicsLayout).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());

    auto graphicsPipeline = device.waitPipelineCompilation(*graphicsCompilation);
    REQUIRE(graphicsPipeline.has_value());
    REQUIRE(device.destroyPipeline(*graphicsPipeline).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
