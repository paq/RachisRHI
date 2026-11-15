#include "Rachis/Device.h"

#include <catch2/catch_test_macros.hpp>

#include <array>

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

} // namespace

TEST_CASE("Metal rejects buffer-backed acceleration structures and DXR shader-table pipelines",
          "[size:large][rhi][metal][ray]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    const auto storage = device.createBuffer(rachis::BufferDesc{
        .size = 256,
        .usage = rachis::BufferUsage::accelerationStructureStorage,
    });
    REQUIRE_FALSE(storage);
    CHECK(storage.error().code == rachis::ErrorCode::unsupported);

    const std::array acceleration{
        rachis::BindingLayoutEntry{0, rachis::BindingType::accelerationStructure, rachis::ShaderStage::compute, 1},
    };
    auto accelerationLayout = device.createBindingLayout({.entries = acceleration});
    REQUIRE_FALSE(accelerationLayout);
    CHECK(accelerationLayout.error().code == rachis::ErrorCode::unsupported);
}
