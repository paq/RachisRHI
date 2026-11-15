#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include <catch2/catch_test_macros.hpp>

#include <Windows.h>
#include <algorithm>
#include <cstdint>

namespace
{

    class TestWindow final
    {
    public:
        explicit TestWindow(bool visible)
            : window_(CreateWindowExW(0,
                                      L"STATIC",
                                      L"Rachis Vulkan surface contract",
                                      WS_OVERLAPPEDWINDOW,
                                      CW_USEDEFAULT,
                                      CW_USEDEFAULT,
                                      96,
                                      64,
                                      nullptr,
                                      nullptr,
                                      GetModuleHandleW(nullptr),
                                      nullptr))
        {
            if (visible && window_ != nullptr)
            {
                static_cast<void>(ShowWindow(window_, SW_SHOW));
            }
        }

        ~TestWindow()
        {
            if (window_ != nullptr)
            {
                static_cast<void>(DestroyWindow(window_));
            }
        }

        TestWindow(const TestWindow&) = delete;
        TestWindow& operator=(const TestWindow&) = delete;
        TestWindow(TestWindow&&) = delete;
        TestWindow& operator=(TestWindow&&) = delete;

        [[nodiscard]] HWND get() const noexcept
        {
            return window_;
        }

    private:
        HWND window_ = nullptr;
    };

    [[nodiscard]] rachis::DeviceDesc vulkanDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::vulkan,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
    }

} // namespace

TEST_CASE("Vulkan Win32 surfaces reject malformed windows and expose capability-selected swapchains",
          "[size:medium][rhi][vulkan][surface]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireAdapterInfo(device).backend == rachis::Backend::vulkan);
    REQUIRE(rachis::test::requireAdapterInfo(device).type != rachis::AdapterType::software);

    CHECK_FALSE(device.createSurface({.kind = rachis::SurfaceKind::win32}).has_value());

    TestWindow const window(false);
    REQUIRE(window.get() != nullptr);
    auto surface = device.createSurface({
        .kind = rachis::SurfaceKind::win32,
        .windowHandle = window.get(),
        .label = "Vulkan surface contract",
    });
    REQUIRE(surface);
    auto capabilities = device.querySurfaceCapabilities(*surface);
    REQUIRE(capabilities);
    CHECK(capabilities->minImageCount >= 1);
    CHECK(rachis::supportsPresentMode(capabilities->presentModes, rachis::PresentMode::fifo));
    CHECK_FALSE(capabilities->formats.empty());

    auto zero = device.createSwapchain({
        .surface = *surface,
        .width = 0,
        .height = 0,
        .imageCount = capabilities->minImageCount,
        .format = rachis::Format::bgra8Unorm,
    });
    REQUIRE_FALSE(zero.has_value());
    CHECK(zero.error().code == rachis::ErrorCode::invalidArgument);

    auto unsupported = device.createSwapchain({
        .surface = *surface,
        .width = (std::max)(capabilities->preferredWidth, capabilities->minWidth == 0 ? 64 : capabilities->minWidth),
        .height =
            (std::max)(capabilities->preferredHeight, capabilities->minHeight == 0 ? 64 : capabilities->minHeight),
        .imageCount = capabilities->minImageCount,
        .format = rachis::Format::d32Float,
    });
    REQUIRE_FALSE(unsupported.has_value());
    CHECK(unsupported.error().code == rachis::ErrorCode::unsupported);

    const auto width = capabilities->preferredWidth == 0 ? 64 : capabilities->preferredWidth;
    const auto height = capabilities->preferredHeight == 0 ? 64 : capabilities->preferredHeight;
    const auto format = capabilities->formats.front();
    auto swapchain = device.createSwapchain({
        .surface = *surface,
        .width = width,
        .height = height,
        .imageCount = (std::min)(capabilities->maxImageCount, (std::max)(capabilities->minImageCount, 2U)),
        .format = format.format,
        .colorSpace = format.colorSpace,
        .usage = rachis::TextureUsage::renderTarget,
        .presentMode = rachis::PresentMode::fifo,
        .label = "Vulkan swapchain contract",
    });
    if (!swapchain)
    {
        INFO(swapchain.error().message);
        REQUIRE(swapchain);
    }
    auto second = device.createSwapchain({
        .surface = *surface,
        .width = width,
        .height = height,
        .imageCount = capabilities->minImageCount,
        .format = rachis::Format::bgra8Unorm,
    });
    REQUIRE_FALSE(second.has_value());
    CHECK(second.error().code == rachis::ErrorCode::invalidState);

    auto acquired = device.acquireNextImage(*swapchain);
    REQUIRE(acquired);
    CHECK(acquired->width == width);
    CHECK(acquired->height == height);
    CHECK(acquired->generation != 0);
    CHECK_FALSE(device.destroyTexture(acquired->texture).has_value());
    REQUIRE(device.cancelSwapchainImage(*swapchain).has_value());
    REQUIRE(device.destroySwapchain(*swapchain).has_value());
    REQUIRE(device.destroySurface(*surface).has_value());
}

TEST_CASE("Vulkan Win32 windows accept one live swapchain across every surface on them",
          "[size:medium][rhi][vulkan][surface]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireAdapterInfo(device).backend == rachis::Backend::vulkan);
    REQUIRE(rachis::test::requireAdapterInfo(device).type != rachis::AdapterType::software);

    TestWindow const window(false);
    TestWindow const otherWindow(false);
    REQUIRE(window.get() != nullptr);
    REQUIRE(otherWindow.get() != nullptr);
    auto first = device.createSurface({.kind = rachis::SurfaceKind::win32, .windowHandle = window.get()});
    REQUIRE(first);
    auto second = device.createSurface({.kind = rachis::SurfaceKind::win32, .windowHandle = window.get()});
    REQUIRE(second);
    auto other = device.createSurface({.kind = rachis::SurfaceKind::win32, .windowHandle = otherWindow.get()});
    REQUIRE(other);
    auto capabilities = device.querySurfaceCapabilities(*first);
    REQUIRE(capabilities);
    REQUIRE_FALSE(capabilities->formats.empty());

    const auto format = capabilities->formats.front();
    const auto desc = [&](rachis::SurfaceHandle surface) {
        return rachis::SwapchainDesc{
            .surface = surface,
            .width = capabilities->preferredWidth == 0 ? 64 : capabilities->preferredWidth,
            .height = capabilities->preferredHeight == 0 ? 64 : capabilities->preferredHeight,
            .imageCount = (std::min)(capabilities->maxImageCount, (std::max)(capabilities->minImageCount, 2U)),
            .format = format.format,
            .colorSpace = format.colorSpace,
            .usage = rachis::TextureUsage::renderTarget,
            .presentMode = rachis::PresentMode::fifo,
            .label = "Vulkan one swapchain per window",
        };
    };

    auto live = device.createSwapchain(desc(*first));
    if (!live)
    {
        INFO(live.error().message);
        REQUIRE(live);
    }
    auto refused = device.createSwapchain(desc(*second));
    REQUIRE_FALSE(refused.has_value());
    CHECK(refused.error().code == rachis::ErrorCode::invalidState);
    CHECK(refused.error().message == "The native window already has a live swapchain.");
    auto otherSwapchain = device.createSwapchain(desc(*other));
    if (!otherSwapchain)
    {
        INFO(otherSwapchain.error().message);
        REQUIRE(otherSwapchain);
    }

    auto acquired = device.acquireNextImage(*live);
    REQUIRE(acquired);
    REQUIRE(device.cancelSwapchainImage(*live).has_value());
    const auto liveDesc = desc(*first);
    REQUIRE(device.resizeSwapchain(*live, liveDesc.width, liveDesc.height).has_value());
    auto stillRefused = device.createSwapchain(desc(*second));
    REQUIRE_FALSE(stillRefused.has_value());
    CHECK(stillRefused.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.destroySwapchain(*live).has_value());

    auto unsupportedDesc = desc(*second);
    unsupportedDesc.format = rachis::Format::d32Float;
    auto unsupported = device.createSwapchain(unsupportedDesc);
    REQUIRE_FALSE(unsupported.has_value());
    CHECK(unsupported.error().code == rachis::ErrorCode::unsupported);

    auto replacement = device.createSwapchain(desc(*second));
    if (!replacement)
    {
        INFO(replacement.error().message);
        REQUIRE(replacement);
    }
    auto replacementImage = device.acquireNextImage(*replacement);
    REQUIRE(replacementImage);
    REQUIRE(device.cancelSwapchainImage(*replacement).has_value());
    auto firstRefused = device.createSwapchain(desc(*first));
    REQUIRE_FALSE(firstRefused.has_value());
    CHECK(firstRefused.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.destroySwapchain(*replacement).has_value());

    auto recreated = device.createSwapchain(desc(*first));
    if (!recreated)
    {
        INFO(recreated.error().message);
        REQUIRE(recreated);
    }
    REQUIRE(device.destroySwapchain(*recreated).has_value());
    REQUIRE(device.destroySwapchain(*otherSwapchain).has_value());
    REQUIRE(device.destroySurface(*first).has_value());
    REQUIRE(device.destroySurface(*second).has_value());
    REQUIRE(device.destroySurface(*other).has_value());
}
