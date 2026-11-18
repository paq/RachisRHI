#include "DrawIndexContract.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

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
