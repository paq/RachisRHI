#include "Internal/VulkanLinuxProfile.h"

#include <catch2/catch_test_macros.hpp>

#include <string_view>

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

namespace
{

    [[nodiscard]] rachis::VulkanInstanceReport macosHeadlessInstance() noexcept
    {
        return {
            .apiMajor = 1,
            .apiMinor = 3,
            .khrSurface = false,
            .khrWin32Surface = false,
            .khrPortabilityEnumeration = true,
        };
    }

    [[nodiscard]] rachis::VulkanPhysicalDeviceReport moltenVkDevice() noexcept
    {
        return {
            .apiMajor = 1,
            .apiMinor = 3,
            .apiPatch = 275,
            .deviceName = "Apple M4",
            .driverId = "MOLTENVK",
            .vendorId = 0x106b,
            .deviceId = 0,
            .software = false,
            .translationLayer = false,
            .dynamicRendering = true,
            .synchronization2 = true,
            .timelineSemaphore = true,
            .independentBlend = true,
            .samplerAnisotropy = true,
            .textureCompressionBC = false,
            .shaderSampledImageArrayDynamicIndexing = true,
            .timestampComputeAndGraphics = true,
            .hostQueryReset = true,
            .khrSwapchain = true,
            .khrPortabilitySubset = true,
            .formatRgba8Unorm = true,
            .formatRgba8UnormSrgb = true,
            .formatBgra8Unorm = true,
            .formatBgra8UnormSrgb = true,
            .formatD32Float = true,
            .formatD32FloatS8Uint = true,
            .formatRgba16Float = true,
            .formatR32Uint = true,
            .maxImageDimension2D = 16384,
            .maxBoundDescriptorSets = 8,
            .maxPushConstantsSize = 128,
            .maxColorAttachments = 8,
            .maxComputeWorkgroupInvocations = 1024,
        };
    }

} // namespace

TEST_CASE("macOS MoltenVK profile accepts a complete device without BC or WSI",
          "[size:small][rhi][vulkan][macos][profile]")
{
    const auto decision = rachis::evaluateMacosVulkanProfile(macosHeadlessInstance(), moltenVkDevice());
    REQUIRE(decision.accepted);
    CHECK(rachis::describeMacosVulkanProfileRejection(decision).empty());
}

TEST_CASE("macOS MoltenVK profile rejects missing portability and Dozen", "[size:small][rhi][vulkan][macos][profile]")
{
    auto noEnumeration = macosHeadlessInstance();
    noEnumeration.khrPortabilityEnumeration = false;
    const auto enumeration = rachis::evaluateMacosVulkanProfile(noEnumeration, moltenVkDevice());
    REQUIRE_FALSE(enumeration.accepted);
    CHECK(enumeration.rejection == rachis::VulkanProfileRejection::missingRequiredExtension);

    auto noSubset = moltenVkDevice();
    noSubset.khrPortabilitySubset = false;
    const auto subset = rachis::evaluateMacosVulkanProfile(macosHeadlessInstance(), noSubset);
    REQUIRE_FALSE(subset.accepted);
    CHECK(subset.rejection == rachis::VulkanProfileRejection::missingRequiredExtension);

    auto dozen = moltenVkDevice();
    dozen.driverId = "MESA_DOZEN";
    dozen.translationLayer = true;
    const auto translation = rachis::evaluateMacosVulkanProfile(macosHeadlessInstance(), dozen);
    REQUIRE_FALSE(translation.accepted);
    CHECK(translation.rejection == rachis::VulkanProfileRejection::translationLayer);

    const auto warp = rachis::evaluateMacosVulkanProfile(
        macosHeadlessInstance(), moltenVkDevice(), rachis::VulkanCreateIntent{.warpPreference = true});
    REQUIRE_FALSE(warp.accepted);
    CHECK(warp.rejection == rachis::VulkanProfileRejection::warpPreferenceIsD3D12Only);

    auto missingHostQueryReset = moltenVkDevice();
    missingHostQueryReset.hostQueryReset = false;
    const auto hostQueryReset = rachis::evaluateMacosVulkanProfile(macosHeadlessInstance(), missingHostQueryReset);
    REQUIRE_FALSE(hostQueryReset.accepted);
    CHECK(hostQueryReset.rejection == rachis::VulkanProfileRejection::missingRequiredFeature);
    CHECK(rachis::describeMacosVulkanProfileRejection(hostQueryReset).contains("hostQueryReset"));
}
