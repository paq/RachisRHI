#include "Internal/VulkanLinuxProfile.h"

#include <catch2/catch_test_macros.hpp>

#include <string_view>

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

namespace
{

    [[nodiscard]] rachis::VulkanInstanceReport linuxHeadlessInstance() noexcept
    {
        return {
            .apiMajor = 1,
            .apiMinor = 3,
            .khrSurface = false,
            .khrWin32Surface = false,
        };
    }

    [[nodiscard]] rachis::VulkanPhysicalDeviceReport linuxDesktopDevice() noexcept
    {
        return {
            .apiMajor = 1,
            .apiMinor = 3,
            .apiPatch = 296,
            .deviceName = "NVIDIA GeForce RTX 4080",
            .driverId = "NVIDIA_PROPRIETARY",
            .vendorId = 0x10de,
            .deviceId = 0x2704,
            .software = false,
            .translationLayer = false,
            .dynamicRendering = true,
            .synchronization2 = true,
            .timelineSemaphore = true,
            .independentBlend = true,
            .samplerAnisotropy = true,
            .textureCompressionBC = true,
            .shaderSampledImageArrayDynamicIndexing = true,
            .timestampComputeAndGraphics = true,
            .hostQueryReset = true,
            .khrSwapchain = false,
            .formatRgba8Unorm = true,
            .formatRgba8UnormSrgb = true,
            .formatBgra8Unorm = true,
            .formatBgra8UnormSrgb = true,
            .formatD32Float = true,
            .formatD32FloatS8Uint = true,
            .formatRgba16Float = true,
            .formatR32Uint = true,
            .maxImageDimension2D = 32768,
            .maxBoundDescriptorSets = 32,
            .maxPushConstantsSize = 256,
            .maxColorAttachments = 8,
            .maxComputeWorkgroupInvocations = 1024,
        };
    }

} // namespace

TEST_CASE("Linux headless profile accepts a complete device without WSI", "[size:small][rhi][vulkan][linux][profile]")
{
    const auto decision = rachis::evaluateLinuxVulkanProfile(linuxHeadlessInstance(), linuxDesktopDevice());
    REQUIRE(decision.accepted);
    CHECK(rachis::describeLinuxVulkanProfileRejection(decision).empty());
}

TEST_CASE("Linux headless profile rejects Dozen and implicit software", "[size:small][rhi][vulkan][linux][profile]")
{
    auto dozen = linuxDesktopDevice();
    dozen.driverId = "MESA_DOZEN";
    dozen.translationLayer = true;
    const auto translation = rachis::evaluateLinuxVulkanProfile(linuxHeadlessInstance(), dozen);
    REQUIRE_FALSE(translation.accepted);
    CHECK(translation.rejection == rachis::VulkanProfileRejection::translationLayer);

    auto software = linuxDesktopDevice();
    software.deviceName = "llvmpipe";
    software.driverId = "MESA_LLVMPIPE";
    software.software = true;
    const auto implicit = rachis::evaluateLinuxVulkanProfile(linuxHeadlessInstance(), software);
    REQUIRE_FALSE(implicit.accepted);
    CHECK(implicit.rejection == rachis::VulkanProfileRejection::softwareAdapterNotSelected);

    auto missingHostQueryReset = linuxDesktopDevice();
    missingHostQueryReset.hostQueryReset = false;
    const auto hostQueryReset = rachis::evaluateLinuxVulkanProfile(linuxHeadlessInstance(), missingHostQueryReset);
    REQUIRE_FALSE(hostQueryReset.accepted);
    CHECK(hostQueryReset.rejection == rachis::VulkanProfileRejection::missingRequiredFeature);
    CHECK(rachis::describeLinuxVulkanProfileRejection(hostQueryReset).contains("hostQueryReset"));
}
