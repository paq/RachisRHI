#include "Rachis/Device.h"

#include <catch2/catch_test_macros.hpp>

#include <string>

TEST_CASE("Metal selection rejects WARP and DRED without fallback", "[size:small][rhi][metal][selection]")
{
    auto warp = rachis::createDevice(rachis::DeviceDesc{
        .backend = rachis::Backend::metal,
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .enableDred = false,
    });
    REQUIRE_FALSE(warp.has_value());
    CHECK(warp.error().code == rachis::ErrorCode::unsupported);
    CHECK(warp.error().message.contains("warp"));

    auto dred = rachis::createDevice(rachis::DeviceDesc{
        .backend = rachis::Backend::metal,
        .validation = rachis::ValidationLevel::standard,
        .enableDred = true,
    });
    REQUIRE_FALSE(dred.has_value());
    CHECK(dred.error().code == rachis::ErrorCode::unsupported);
    CHECK(dred.error().message.contains("enableDred"));
}

TEST_CASE("Metal enumeration is backend-local and never mixed with D3D12 or Vulkan",
          "[size:small][rhi][metal][selection]")
{
    auto adapters = rachis::enumerateAdapters(rachis::Backend::metal);
#if RACHIS_ENABLE_METAL
    REQUIRE(adapters.has_value());
    REQUIRE_FALSE(adapters->empty());
    for (const auto& adapter : *adapters)
    {
        CHECK(adapter.backend == rachis::Backend::metal);
    }

    auto missing = rachis::createDevice(rachis::DeviceDesc{
        .backend = rachis::Backend::metal,
        .adapterId = rachis::AdapterId{.value = 0xdeadbeefdeadbeefULL},
        .validation = rachis::ValidationLevel::standard,
        .enableDred = false,
    });
    REQUIRE_FALSE(missing.has_value());
    CHECK(missing.error().code == rachis::ErrorCode::adapterNotFound);

    auto created = rachis::createDevice(rachis::DeviceDesc{
        .backend = rachis::Backend::metal,
        .validation = rachis::ValidationLevel::standard,
        .enableDred = false,
    });
    REQUIRE(created.has_value());
    REQUIRE(created->isValid());
    REQUIRE(created->adapterInfo().has_value());
    const auto checkedAdapter = created->adapterInfo();
    REQUIRE(checkedAdapter.has_value());
    CHECK(checkedAdapter->backend == rachis::Backend::metal);
#else
    REQUIRE_FALSE(adapters.has_value());
    CHECK(adapters.error().code == rachis::ErrorCode::backendUnavailable);

    auto created = rachis::createDevice(rachis::DeviceDesc{
        .backend = rachis::Backend::metal,
        .validation = rachis::ValidationLevel::standard,
        .enableDred = false,
    });
    REQUIRE_FALSE(created.has_value());
    CHECK(created.error().code == rachis::ErrorCode::backendUnavailable);
#endif
}
