#include "DrawIndexContract.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <vector>

namespace
{

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

} // namespace

TEST_CASE("Vulkan indirect draws and mesh dispatches give each record its draw index",
          "[size:medium][rhi][vulkan][indirect][draw-index]")
{
    auto created = rachis::createDevice(rachis::test::vulkanDeviceDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    if (rachis::test::skipWithoutCapability(rachis::test::requireCapabilities(device).shaderDrawIndex,
                                            "shaderDrawParameters"))
    {
        return;
    }

    const std::filesystem::path directory(RACHIS_RENDERER_SPIRV_DIR);
    const auto vertex = loadBytes(directory / "DrawIndex.vs.spv");
    const auto plainVertex = loadBytes(directory / "DrawIndex.plain.vs.spv");
    const auto mesh = loadBytes(directory / "DrawIndex.ms.spv");
    const auto pixel = loadBytes(directory / "DrawIndex.ps.spv");
    REQUIRE_FALSE(vertex.empty());
    REQUIRE_FALSE(plainVertex.empty());
    REQUIRE_FALSE(mesh.empty());
    REQUIRE_FALSE(pixel.empty());

    rachis::test::draw_index::runContract(device,
                                          {
                                              .vertex = vertex,
                                              .plainVertex = plainVertex,
                                              .mesh = mesh,
                                              .pixel = pixel,
                                          });
}

TEST_CASE("Vulkan rejects task and mesh pipelines whose mesh shader reads the draw index",
          "[size:medium][rhi][vulkan][mesh][draw-index]")
{
    auto created = rachis::createDevice(rachis::test::vulkanDeviceDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (!capabilities.shaderDrawIndex || !capabilities.meshShaders || !capabilities.taskShaders)
    {
        SKIP("The Vulkan adapter does not advertise draw indices with task and mesh shaders.");
    }

    const std::filesystem::path directory(RACHIS_RENDERER_SPIRV_DIR);
    const auto taskBytecode = loadBytes(directory / "TaskMeshWorkload.as.spv");
    const auto meshBytecode = loadBytes(directory / "DrawIndex.ms.spv");
    const auto pixelBytecode = loadBytes(directory / "DrawIndex.ps.spv");
    REQUIRE_FALSE(taskBytecode.empty());
    REQUIRE_FALSE(meshBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    auto task = device.createShaderModule({
        .stage = rachis::ShaderStage::task,
        .bytecode = taskBytecode,
        .entryPoint = "taskMain",
        .shaderInterface = rachis::test::computeShaderInterface(
            {1, 1, 1},
            std::array{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1}}),
    });
    auto mesh = device.createShaderModule({
        .stage = rachis::ShaderStage::mesh,
        .bytecode = meshBytecode,
        .entryPoint = "drawIndexMesh",
        .shaderInterface = rachis::test::draw_index::meshInterface(),
    });
    auto pixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "recordPixel",
    });
    REQUIRE(task.has_value());
    REQUIRE(mesh.has_value());
    REQUIRE(pixel.has_value());
    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::task, 1},
    };
    auto bindingLayout = device.createBindingLayout({.entries = bindingEntries});
    REQUIRE(bindingLayout.has_value());
    const std::array bindingLayouts{*bindingLayout};
    auto layout = device.createPipelineLayout({.bindingLayouts = bindingLayouts, .drawIndex = true});
    REQUIRE(layout.has_value());
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};

    const auto rejected = device.createGraphicsPipeline({
        .layout = *layout,
        .pixelShader = *pixel,
        .taskShader = *task,
        .meshShader = *mesh,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
    });
    REQUIRE_FALSE(rejected.has_value());
    CHECK(rejected.error().code == rachis::ErrorCode::unsupported);

    auto meshOnly = device.createGraphicsPipeline({
        .layout = *layout,
        .pixelShader = *pixel,
        .meshShader = *mesh,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
    });
    if (!meshOnly)
    {
        UNSCOPED_INFO(meshOnly.error().message);
    }
    REQUIRE(meshOnly.has_value());

    REQUIRE(device.destroyPipeline(*meshOnly).has_value());
    REQUIRE(device.destroyPipelineLayout(*layout).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyShaderModule(*task).has_value());
    REQUIRE(device.destroyShaderModule(*mesh).has_value());
    REQUIRE(device.destroyShaderModule(*pixel).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
