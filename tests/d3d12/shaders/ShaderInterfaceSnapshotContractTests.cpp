#include "Rachis/Device.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <vector>

namespace
{

    [[nodiscard]] std::vector<std::byte> loadShader()
    {
        std::ifstream stream(RACHIS_TEST_COMPUTE_SHADER_PATH, std::ios::binary | std::ios::ate);
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

TEST_CASE("shader interface snapshots own portable metadata independently of module lifetime",
          "[size:medium][rhi][pipeline][shader-interface][snapshot][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .enableDred = true,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    const auto bytecode = loadShader();
    REQUIRE_FALSE(bytecode.empty());
    std::array bindings{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1},
        rachis::ShaderBindingRequirement{1, 3, rachis::ShaderResourceType::sampler, 2},
    };
    std::array pushConstants{
        rachis::ShaderPushConstantRequirement{0, sizeof(std::uint32_t)},
        rachis::ShaderPushConstantRequirement{8, 2 * sizeof(std::uint32_t)},
    };
    auto module = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .shaderInterface =
            rachis::ShaderInterfaceDesc{
                .bindings = bindings,
                .pushConstants = pushConstants,
                .workgroupSize = rachis::ShaderWorkgroupSize{64, 1, 1},
            },
    });
    REQUIRE(module.has_value());

    bindings[0].binding = 7;
    pushConstants[0].offset = 16;
    auto queried = device.shaderModuleInterface(*module);
    REQUIRE(queried.has_value());
    auto snapshot = std::move(*queried);
    CHECK(snapshot.stage == rachis::ShaderStage::compute);
    REQUIRE(snapshot.bindings.size() == 2);
    CHECK(snapshot.bindings[0].set == 0);
    CHECK(snapshot.bindings[0].binding == 0);
    CHECK(snapshot.bindings[0].type == rachis::ShaderResourceType::storageBuffer);
    CHECK(snapshot.bindings[0].count == 1);
    CHECK(snapshot.bindings[1].set == 1);
    CHECK(snapshot.bindings[1].binding == 3);
    CHECK(snapshot.bindings[1].type == rachis::ShaderResourceType::sampler);
    CHECK(snapshot.bindings[1].count == 2);
    REQUIRE(snapshot.pushConstants.size() == 2);
    CHECK(snapshot.pushConstants[0].offset == 0);
    CHECK(snapshot.pushConstants[0].size == sizeof(std::uint32_t));
    CHECK(snapshot.pushConstants[1].offset == 8);
    CHECK(snapshot.pushConstants[1].size == 2 * sizeof(std::uint32_t));
    CHECK(snapshot.vertexInputLocations.empty());
    REQUIRE(snapshot.workgroupSize.has_value());
    const auto workgroupSize = snapshot.workgroupSize.value_or(rachis::ShaderWorkgroupSize{});
    CHECK(workgroupSize.x == 64);
    CHECK(workgroupSize.y == 1);
    CHECK(workgroupSize.z == 1);
    CHECK_FALSE(snapshot.shaderRecordConstants.has_value());

    const auto staleModule = *module;
    REQUIRE(device.destroyShaderModule(*module).has_value());
    CHECK_FALSE(device.shaderModuleInterface(staleModule).has_value());
}
