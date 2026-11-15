#include "Internal/VulkanLinuxProfile.h"
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

TEST_CASE("Linux surfaces reject the wrong window system without fallback",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][linux][surface]")
{
#ifdef __APPLE__
    SUCCEED("Linux WSI contracts are not the macOS MoltenVK investigation.");
    return;
#endif

    auto outcome = rachis::test::tryCreateDevice(rachis::test::vulkanDeviceDesc());
    if (outcome.classification != rachis::test::DeviceCreateClassification::created)
    {
        REQUIRE_FALSE(outcome.device.has_value());
        if (rachis::test::skipWithoutCreatedDevice(outcome.device))
        {
            return;
        }
    }

    auto device = std::move(*outcome.device);
    REQUIRE(rachis::test::requireAdapterInfo(device).backend == rachis::Backend::vulkan);

#ifdef _WIN32
    auto wayland = device.createSurface({
        .kind = rachis::SurfaceKind::wayland,
        .windowHandle = &device,
        .displayHandle = &device,
        .label = "Rejected Wayland",
    });
    REQUIRE_FALSE(wayland.has_value());
    CHECK(wayland.error().code == rachis::ErrorCode::unsupported);
    CHECK(wayland.error().message.contains("Linux-only"));

    auto xcb = device.createSurface({
        .kind = rachis::SurfaceKind::xcb,
        .windowHandle = &device,
        .displayHandle = &device,
        .label = "Rejected XCB",
    });
    REQUIRE_FALSE(xcb.has_value());
    CHECK(xcb.error().code == rachis::ErrorCode::unsupported);
    CHECK(xcb.error().message.contains("Linux-only"));
#else
    auto win32 = device.createSurface({
        .kind = rachis::SurfaceKind::win32,
        .windowHandle = &device,
        .label = "Rejected Win32",
    });
    REQUIRE_FALSE(win32.has_value());
    CHECK(win32.error().code == rachis::ErrorCode::unsupported);
    CHECK(win32.error().message.contains("Windows-only"));

    auto waylandMissingDisplay = device.createSurface({
        .kind = rachis::SurfaceKind::wayland,
        .windowHandle = &device,
        .label = "Wayland without display",
    });
    REQUIRE_FALSE(waylandMissingDisplay.has_value());
    CHECK(waylandMissingDisplay.error().code == rachis::ErrorCode::invalidArgument);

    auto xcbMissingWindow = device.createSurface({
        .kind = rachis::SurfaceKind::xcb,
        .displayHandle = &device,
        .label = "XCB without window",
    });
    REQUIRE_FALSE(xcbMissingWindow.has_value());
    CHECK(xcbMissingWindow.error().code == rachis::ErrorCode::invalidArgument);
#endif

    auto unknown = device.createSurface({
        .kind = static_cast<rachis::SurfaceKind>(255),
        .windowHandle = &device,
        .displayHandle = &device,
        .label = "Xlib is not a declared path",
    });
    REQUIRE_FALSE(unknown.has_value());
    CHECK(unknown.error().code == rachis::ErrorCode::unsupported);
    CHECK(unknown.error().message.contains("Xlib"));
}
