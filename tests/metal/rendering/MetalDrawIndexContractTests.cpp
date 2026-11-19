#include "DrawIndexContract.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <vector>

namespace
{

    [[nodiscard]] rachis::DeviceDesc metalDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
    }

    [[nodiscard]] std::vector<std::byte> loadPnms(std::string_view name)
    {
        std::ifstream stream(std::filesystem::path(RACHIS_METALLIB_DIR) / std::filesystem::path(name),
                             std::ios::binary | std::ios::ate);
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

TEST_CASE("Metal shaders read each multi-draw record's draw index", "[size:large][rhi][metal][indirect][draw-index]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).shaderDrawIndex);

    const auto vertex = loadPnms("DrawIndex.vs.pnms");
    const auto plainVertex = loadPnms("DrawIndex.plain.vs.pnms");
    const auto mesh = loadPnms("DrawIndex.ms.pnms");
    const auto pixel = loadPnms("DrawIndex.ps.pnms");
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
