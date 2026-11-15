#include "Internal/VulkanWindowsProfile.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <span>
#include <string_view>

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

namespace
{

    [[nodiscard]] rachis::VulkanInstanceReport windowsInstanceReport() noexcept
    {
        return {
            .apiMajor = 1,
            .apiMinor = 4,
            .khrSurface = true,
            .khrWin32Surface = true,
        };
    }

    [[nodiscard]] rachis::VulkanPhysicalDeviceReport nvidiaRtx4080Report() noexcept
    {
        return {
            .apiMajor = 1,
            .apiMinor = 4,
            .apiPatch = 325,
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
            .khrSwapchain = true,
            .khrSwapchainMaintenance1 = true,
            .extSwapchainMaintenance1 = true,
            .khrPresentId = true,
            .khrPresentWait = true,
            .extDeviceFault = true,
            .formatRgba8Unorm = true,
            .formatRgba8UnormSrgb = true,
            .formatBgra8Unorm = true,
            .formatBgra8UnormSrgb = true,
            .formatD32Float = true,
            .formatD24UnormS8Uint = true,
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

    [[nodiscard]] rachis::VulkanPhysicalDeviceReport dozenNvidiaReport() noexcept
    {
        return {
            .apiMajor = 1,
            .apiMinor = 2,
            .apiPatch = 359,
            .deviceName = "Microsoft Direct3D12 (NVIDIA GeForce RTX 4080)",
            .driverId = "MESA_DOZEN",
            .vendorId = 0x10de,
            .deviceId = 0x2704,
            .software = false,
            .translationLayer = true,
            .dynamicRendering = true,
            .synchronization2 = true,
            .timelineSemaphore = true,
            .independentBlend = true,
            .samplerAnisotropy = true,
            .textureCompressionBC = true,
            .shaderSampledImageArrayDynamicIndexing = true,
            .timestampComputeAndGraphics = true,
            .hostQueryReset = true,
            .khrSwapchain = true,
            .formatRgba8Unorm = true,
            .formatRgba8UnormSrgb = true,
            .formatBgra8Unorm = true,
            .formatBgra8UnormSrgb = true,
            .formatD32Float = true,
            .formatD24UnormS8Uint = true,
            .formatRgba16Float = true,
            .formatR32Uint = true,
            .maxImageDimension2D = 16384,
            .maxBoundDescriptorSets = 8,
            .maxPushConstantsSize = 128,
            .maxColorAttachments = 8,
            .maxComputeWorkgroupInvocations = 1024,
        };
    }

    [[nodiscard]] rachis::VulkanPhysicalDeviceReport dozenWarpReport() noexcept
    {
        auto report = dozenNvidiaReport();
        report.deviceName = "Microsoft Direct3D12 (Microsoft Basic Render Driver)";
        report.vendorId = 0x1414;
        report.deviceId = 0x008c;
        report.software = true;
        return report;
    }

    [[nodiscard]] rachis::VulkanPhysicalDeviceReport compliantDeviceWithoutPreferredPresent() noexcept
    {
        auto report = nvidiaRtx4080Report();
        report.apiMinor = 3;
        report.apiPatch = 0;
        report.khrSwapchainMaintenance1 = false;
        report.extSwapchainMaintenance1 = false;
        report.khrPresentId = false;
        report.khrPresentWait = false;
        report.extDeviceFault = false;
        return report;
    }

    [[nodiscard]] bool containsMissing(const rachis::VulkanProfileDecision& decision, std::string_view name)
    {
        const auto items = std::span(decision.missingItems).first(decision.missingItemCount);
        return std::ranges::find(items, name) != items.end();
    }

} // namespace

TEST_CASE("native RTX 4080 report satisfies the Windows Vulkan profile", "[size:small][rhi][vulkan][profile]")
{
    const auto decision = rachis::evaluateWindowsVulkanProfile(windowsInstanceReport(), nvidiaRtx4080Report());
    CHECK(decision.accepted);
    CHECK(decision.rejection == rachis::VulkanProfileRejection::none);
    CHECK(decision.presentCompletion == rachis::VulkanPresentCompletionStrategy::swapchainMaintenancePresentFence);
    CHECK(decision.missingItemCount == 0);
}

TEST_CASE("Dozen D3D12 translation ICDs are rejected even when features look complete",
          "[size:small][rhi][vulkan][profile]")
{
    const auto hardware = rachis::evaluateWindowsVulkanProfile(windowsInstanceReport(), dozenNvidiaReport());
    CHECK_FALSE(hardware.accepted);
    CHECK(hardware.rejection == rachis::VulkanProfileRejection::translationLayer);
    CHECK(containsMissing(hardware, "MESA_DOZEN"));
    CHECK(hardware.presentCompletion == rachis::VulkanPresentCompletionStrategy::none);

    const auto software = rachis::evaluateWindowsVulkanProfile(windowsInstanceReport(), dozenWarpReport());
    CHECK_FALSE(software.accepted);
    CHECK(software.rejection == rachis::VulkanProfileRejection::translationLayer);
}

TEST_CASE("software Vulkan adapters require an explicit test-device policy", "[size:small][rhi][vulkan][profile]")
{
    auto swiftShader = nvidiaRtx4080Report();
    swiftShader.deviceName = "SwiftShader Device";
    swiftShader.driverId = "GOOGLE_SWIFTSHADER";
    swiftShader.software = true;
    swiftShader.khrSwapchainMaintenance1 = false;
    swiftShader.extSwapchainMaintenance1 = false;
    swiftShader.khrPresentId = false;
    swiftShader.khrPresentWait = false;

    const auto implicit = rachis::evaluateWindowsVulkanProfile(windowsInstanceReport(), swiftShader);
    CHECK_FALSE(implicit.accepted);
    CHECK(implicit.rejection == rachis::VulkanProfileRejection::softwareAdapterNotSelected);

    const auto explicitSoftware =
        rachis::evaluateWindowsVulkanProfile(windowsInstanceReport(),
                                             swiftShader,
                                             rachis::VulkanCreateIntent{
                                                 .policy = rachis::VulkanAdapterSelectionPolicy::softwareTestDevice,
                                             });
    CHECK(explicitSoftware.accepted);
    CHECK(explicitSoftware.presentCompletion == rachis::VulkanPresentCompletionStrategy::imageLifetimeAssociation);
}

TEST_CASE("missing core features and formats produce named rejection diagnostics", "[size:small][rhi][vulkan][profile]")
{
    auto missingRendering = nvidiaRtx4080Report();
    missingRendering.dynamicRendering = false;
    const auto rendering = rachis::evaluateWindowsVulkanProfile(windowsInstanceReport(), missingRendering);
    CHECK_FALSE(rendering.accepted);
    CHECK(rendering.rejection == rachis::VulkanProfileRejection::missingRequiredFeature);
    CHECK(containsMissing(rendering, "dynamicRendering"));

    auto missingHostQueryReset = nvidiaRtx4080Report();
    missingHostQueryReset.hostQueryReset = false;
    const auto hostQueryReset = rachis::evaluateWindowsVulkanProfile(windowsInstanceReport(), missingHostQueryReset);
    CHECK_FALSE(hostQueryReset.accepted);
    CHECK(hostQueryReset.rejection == rachis::VulkanProfileRejection::missingRequiredFeature);
    CHECK(containsMissing(hostQueryReset, "hostQueryReset"));

    auto missingDepth = nvidiaRtx4080Report();
    missingDepth.formatD24UnormS8Uint = false;
    missingDepth.formatD32FloatS8Uint = false;
    const auto depth = rachis::evaluateWindowsVulkanProfile(windowsInstanceReport(), missingDepth);
    CHECK_FALSE(depth.accepted);
    CHECK(depth.rejection == rachis::VulkanProfileRejection::missingRequiredFormat);
    CHECK(containsMissing(depth, "packedDepthStencil"));

    auto smallPush = nvidiaRtx4080Report();
    smallPush.maxPushConstantsSize = 64;
    const auto limits = rachis::evaluateWindowsVulkanProfile(windowsInstanceReport(), smallPush);
    CHECK_FALSE(limits.accepted);
    CHECK(limits.rejection == rachis::VulkanProfileRejection::limitBelowMinimum);
    CHECK(containsMissing(limits, "maxPushConstantsSize"));
}

TEST_CASE("Vulkan 1.2 devices are rejected even if 1.3 extensions are present", "[size:small][rhi][vulkan][profile]")
{
    auto report = nvidiaRtx4080Report();
    report.apiMinor = 2;
    report.translationLayer = false;
    report.driverId = "NVIDIA_PROPRIETARY";
    const auto decision = rachis::evaluateWindowsVulkanProfile(windowsInstanceReport(), report);
    CHECK_FALSE(decision.accepted);
    CHECK(decision.rejection == rachis::VulkanProfileRejection::apiVersionTooLow);
    CHECK(containsMissing(decision, "vulkan1.3"));
}

TEST_CASE("present completion never depends on queue-submit completion alone", "[size:small][rhi][vulkan][profile]")
{
    const auto withFence = rachis::selectVulkanPresentCompletionStrategy(nvidiaRtx4080Report());
    CHECK(withFence == rachis::VulkanPresentCompletionStrategy::swapchainMaintenancePresentFence);

    auto presentWaitOnly = nvidiaRtx4080Report();
    presentWaitOnly.khrSwapchainMaintenance1 = false;
    presentWaitOnly.extSwapchainMaintenance1 = false;
    CHECK(rachis::selectVulkanPresentCompletionStrategy(presentWaitOnly) ==
          rachis::VulkanPresentCompletionStrategy::presentWait);

    auto missingPresentId = presentWaitOnly;
    missingPresentId.khrPresentId = false;
    CHECK(rachis::selectVulkanPresentCompletionStrategy(missingPresentId) ==
          rachis::VulkanPresentCompletionStrategy::imageLifetimeAssociation);

    const auto portable =
        rachis::evaluateWindowsVulkanProfile(windowsInstanceReport(), compliantDeviceWithoutPreferredPresent());
    CHECK(portable.accepted);
    CHECK(portable.presentCompletion == rachis::VulkanPresentCompletionStrategy::imageLifetimeAssociation);
    CHECK(portable.presentCompletion != rachis::VulkanPresentCompletionStrategy::none);
}

TEST_CASE("Linux and Android profile items stay follow-ups and are not Windows rejects",
          "[size:small][rhi][vulkan][profile]")
{
    auto instance = windowsInstanceReport();
    instance.khrWin32Surface = false;
    const auto missingWin32 = rachis::evaluateWindowsVulkanProfile(instance, nvidiaRtx4080Report());
    CHECK_FALSE(missingWin32.accepted);
    CHECK(missingWin32.rejection == rachis::VulkanProfileRejection::missingRequiredExtension);
    CHECK(containsMissing(missingWin32, "VK_KHR_win32_surface"));
}
