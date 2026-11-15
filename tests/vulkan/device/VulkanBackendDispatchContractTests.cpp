#include "Internal/VulkanBackendConfiguration.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <string>

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

TEST_CASE("Vulkan selection rejects WARP and DRED without opening D3D12", "[size:small][rhi][vulkan][selection]")
{
    auto warp = rachis::createDevice(rachis::DeviceDesc{
        .backend = rachis::Backend::vulkan,
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .enableDred = false,
    });
    REQUIRE_FALSE(warp.has_value());
    CHECK(warp.error().code == rachis::vulkanSelectionError(rachis::VulkanSelectionFailure::warpIsD3D12Only));
    CHECK(warp.error().message.contains("warp"));

    auto dred = rachis::createDevice(rachis::DeviceDesc{
        .backend = rachis::Backend::vulkan,
        .validation = rachis::ValidationLevel::standard,
        .enableDred = true,
    });
    REQUIRE_FALSE(dred.has_value());
    CHECK(dred.error().code == rachis::vulkanSelectionError(rachis::VulkanSelectionFailure::dredIsD3D12Only));
}

TEST_CASE("Vulkan enumeration is backend-local and never mixed with D3D12", "[size:medium][rhi][vulkan][selection]")
{
    auto vulkan = rachis::enumerateAdapters(rachis::Backend::vulkan);
    if (rachis::test::skipWithoutVulkanAdapterList(vulkan))
    {
        return;
    }

    for (const auto& adapter : *vulkan)
    {
        CHECK(adapter.backend == rachis::Backend::vulkan);
        CHECK_FALSE(adapter.name.contains("Dozen"));
    }

    auto missing = rachis::createDevice(rachis::DeviceDesc{
        .backend = rachis::Backend::vulkan,
        .adapterId = rachis::AdapterId{.value = 0xdeadbeefdeadbeefULL},
        .validation = rachis::ValidationLevel::standard,
        .enableDred = false,
    });
    REQUIRE_FALSE(missing.has_value());
    CHECK(missing.error().code == rachis::ErrorCode::adapterNotFound);

    auto created = rachis::createDevice(rachis::DeviceDesc{
        .backend = rachis::Backend::vulkan,
        .validation = rachis::ValidationLevel::standard,
        .enableDred = false,
    });
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    REQUIRE(created->adapterInfo().has_value());
    CHECK(rachis::test::requireAdapterInfo(*created).backend == rachis::Backend::vulkan);
    CHECK_FALSE(rachis::test::requireAdapterInfo(*created).name.contains("Dozen"));
}
