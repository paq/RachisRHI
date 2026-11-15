#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace rachis
{

    inline constexpr std::uint16_t vulkanWindowsProfileApiMajor = 1;
    inline constexpr std::uint16_t vulkanWindowsProfileApiMinor = 3;

    enum class VulkanProfileItemKind : std::uint8_t
    {
        apiVersion,
        feature,
        instanceExtension,
        deviceExtension,
        format,
        limitMinimum,
        driverExclusion,
        platformFollowUp,
    };

    enum class VulkanProfileDisposition : std::uint8_t
    {
        required,
        preferred,
        followUp,
    };

    enum class VulkanPresentCompletionStrategy : std::uint8_t
    {
        none,
        imageLifetimeAssociation,
        presentWait,
        swapchainMaintenancePresentFence,
    };

    enum class VulkanProfileRejection : std::uint8_t
    {
        none,
        warpPreferenceIsD3D12Only,
        apiVersionTooLow,
        translationLayer,
        softwareAdapterNotSelected,
        missingRequiredFeature,
        missingRequiredExtension,
        missingRequiredFormat,
        limitBelowMinimum,
    };

    enum class VulkanAdapterSelectionPolicy : std::uint8_t
    {
        hardwarePreferred,
        explicitAdapter,
        softwareTestDevice,
    };

    struct VulkanProfileItem final
    {
        std::string_view name;
        VulkanProfileItemKind kind = VulkanProfileItemKind::feature;
        VulkanProfileDisposition disposition = VulkanProfileDisposition::required;
        std::uint64_t minimum = 0;
        std::string_view justification;
    };

    struct VulkanInstanceReport final
    {
        std::uint16_t apiMajor = 0;
        std::uint16_t apiMinor = 0;
        bool khrSurface = false;
        bool khrWin32Surface = false;
        bool khrWaylandSurface = false;
        bool khrXcbSurface = false;
        bool khrGetSurfaceCapabilities2 = false;
        bool khrSurfaceMaintenance1 = false;
        bool extSurfaceMaintenance1 = false;
        bool khrPortabilityEnumeration = false;
        bool khrMetalSurface = false;
    };

    struct VulkanPhysicalDeviceReport final
    {
        std::uint16_t apiMajor = 0;
        std::uint16_t apiMinor = 0;
        std::uint16_t apiPatch = 0;
        std::string_view deviceName = {};
        std::string_view driverId = {};
        std::uint32_t vendorId = 0;
        std::uint32_t deviceId = 0;
        bool software = false;
        bool translationLayer = false;

        bool dynamicRendering = false;
        bool synchronization2 = false;
        bool timelineSemaphore = false;
        bool independentBlend = false;
        bool samplerAnisotropy = false;
        bool textureCompressionBC = false;
        bool shaderSampledImageArrayDynamicIndexing = false;
        bool timestampComputeAndGraphics = false;
        bool hostQueryReset = false;

        bool khrSwapchain = false;
        bool khrPortabilitySubset = false;
        bool khrSwapchainMaintenance1 = false;
        bool extSwapchainMaintenance1 = false;
        bool khrPresentId = false;
        bool khrPresentWait = false;
        bool extDeviceFault = false;

        bool formatRgba8Unorm = false;
        bool formatRgba8UnormSrgb = false;
        bool formatBgra8Unorm = false;
        bool formatBgra8UnormSrgb = false;
        bool formatD32Float = false;
        bool formatD24UnormS8Uint = false;
        bool formatD32FloatS8Uint = false;
        bool formatRgba16Float = false;
        bool formatR32Uint = false;

        std::uint32_t maxImageDimension2D = 0;
        std::uint32_t maxBoundDescriptorSets = 0;
        std::uint32_t maxPushConstantsSize = 0;
        std::uint32_t maxColorAttachments = 0;
        std::uint32_t maxComputeWorkgroupInvocations = 0;
    };

    struct VulkanCreateIntent final
    {
        VulkanAdapterSelectionPolicy policy = VulkanAdapterSelectionPolicy::hardwarePreferred;
        bool warpPreference = false;
    };

    struct VulkanProfileDecision final
    {
        bool accepted = false;
        VulkanProfileRejection rejection = VulkanProfileRejection::none;
        VulkanPresentCompletionStrategy presentCompletion = VulkanPresentCompletionStrategy::none;
        std::array<std::string_view, 16> missingItems{};
        std::uint32_t missingItemCount = 0;
    };

    [[nodiscard]] constexpr std::span<const VulkanProfileItem> vulkanWindowsProfileItems() noexcept;

    [[nodiscard]] constexpr VulkanPresentCompletionStrategy selectVulkanPresentCompletionStrategy(
        const VulkanPhysicalDeviceReport& device) noexcept
    {
        if (device.khrSwapchainMaintenance1 || device.extSwapchainMaintenance1)
        {
            return VulkanPresentCompletionStrategy::swapchainMaintenancePresentFence;
        }
        if (device.khrPresentId && device.khrPresentWait)
        {
            return VulkanPresentCompletionStrategy::presentWait;
        }
        return VulkanPresentCompletionStrategy::imageLifetimeAssociation;
    }

    [[nodiscard]] constexpr bool isRejectedVulkanTranslationDriver(std::string_view driverId) noexcept
    {
        return driverId == "MESA_DOZEN";
    }

    enum class VulkanValidationLayerPlan : std::uint8_t
    {
        disabled,
        enableKhronos,
        optionalMissing,
        requiredMissing,
    };

    [[nodiscard]] constexpr VulkanValidationLayerPlan planVulkanValidationLayers(bool requested,
                                                                                 bool requireValidation,
                                                                                 bool khronosPresent) noexcept
    {
        if (!requested)
        {
            return VulkanValidationLayerPlan::disabled;
        }
        if (khronosPresent)
        {
            return VulkanValidationLayerPlan::enableKhronos;
        }
        return requireValidation ? VulkanValidationLayerPlan::requiredMissing
                                 : VulkanValidationLayerPlan::optionalMissing;
    }

    struct VulkanInstanceCreatePolicy final
    {
        bool enablePortabilityEnumeration = false;
        bool enumeratePortabilityBit = false;
    };

    [[nodiscard]] constexpr VulkanInstanceCreatePolicy selectVulkanInstanceCreatePolicy(
        bool portabilityEnumerationAvailable) noexcept
    {
        return {
            .enablePortabilityEnumeration = portabilityEnumerationAvailable,
            .enumeratePortabilityBit = portabilityEnumerationAvailable,
        };
    }

    [[nodiscard]] constexpr bool hostRequiresPortabilityEnumeration() noexcept
    {
#ifdef __APPLE__
        return true;
#else
        return false;
#endif
    }

    namespace detail
    {

        [[nodiscard]] constexpr bool apiVersionAtLeast(std::uint16_t major,
                                                       std::uint16_t minor,
                                                       std::uint16_t requiredMajor,
                                                       std::uint16_t requiredMinor) noexcept
        {
            return major > requiredMajor || (major == requiredMajor && minor >= requiredMinor);
        }

        inline void appendMissing(VulkanProfileDecision& decision, std::string_view name) noexcept
        {
            if (decision.missingItemCount < decision.missingItems.size())
            {
                decision.missingItems[decision.missingItemCount++] = name;
            }
        }

        [[nodiscard]] constexpr bool reportHasFeature(const VulkanPhysicalDeviceReport& device,
                                                      std::string_view name) noexcept
        {
            if (name == "dynamicRendering")
            {
                return device.dynamicRendering;
            }
            if (name == "synchronization2")
            {
                return device.synchronization2;
            }
            if (name == "timelineSemaphore")
            {
                return device.timelineSemaphore;
            }
            if (name == "independentBlend")
            {
                return device.independentBlend;
            }
            if (name == "samplerAnisotropy")
            {
                return device.samplerAnisotropy;
            }
            if (name == "textureCompressionBC")
            {
                return device.textureCompressionBC;
            }
            if (name == "shaderSampledImageArrayDynamicIndexing")
            {
                return device.shaderSampledImageArrayDynamicIndexing;
            }
            if (name == "timestampComputeAndGraphics")
            {
                return device.timestampComputeAndGraphics;
            }
            if (name == "hostQueryReset")
            {
                return device.hostQueryReset;
            }
            return false;
        }

        [[nodiscard]] constexpr bool reportHasInstanceExtension(const VulkanInstanceReport& instance,
                                                                std::string_view name) noexcept
        {
            if (name == "VK_KHR_surface")
            {
                return instance.khrSurface;
            }
            if (name == "VK_KHR_win32_surface")
            {
                return instance.khrWin32Surface;
            }
            if (name == "VK_KHR_wayland_surface")
            {
                return instance.khrWaylandSurface;
            }
            if (name == "VK_KHR_xcb_surface")
            {
                return instance.khrXcbSurface;
            }
            if (name == "VK_KHR_portability_enumeration")
            {
                return instance.khrPortabilityEnumeration;
            }
            if (name == "VK_EXT_metal_surface")
            {
                return instance.khrMetalSurface;
            }
            return false;
        }

        [[nodiscard]] constexpr bool reportHasDeviceExtension(const VulkanPhysicalDeviceReport& device,
                                                              std::string_view name) noexcept
        {
            if (name == "VK_KHR_swapchain")
            {
                return device.khrSwapchain;
            }
            if (name == "VK_KHR_portability_subset")
            {
                return device.khrPortabilitySubset;
            }
            if (name == "VK_KHR_swapchain_maintenance1")
            {
                return device.khrSwapchainMaintenance1;
            }
            if (name == "VK_EXT_swapchain_maintenance1")
            {
                return device.extSwapchainMaintenance1;
            }
            if (name == "VK_KHR_present_id")
            {
                return device.khrPresentId;
            }
            if (name == "VK_KHR_present_wait")
            {
                return device.khrPresentWait;
            }
            if (name == "VK_EXT_device_fault")
            {
                return device.extDeviceFault;
            }
            return false;
        }

        [[nodiscard]] constexpr bool reportHasFormat(const VulkanPhysicalDeviceReport& device,
                                                     std::string_view name) noexcept
        {
            if (name == "rgba8Unorm")
            {
                return device.formatRgba8Unorm;
            }
            if (name == "rgba8UnormSrgb")
            {
                return device.formatRgba8UnormSrgb;
            }
            if (name == "bgra8Unorm")
            {
                return device.formatBgra8Unorm;
            }
            if (name == "bgra8UnormSrgb")
            {
                return device.formatBgra8UnormSrgb;
            }
            if (name == "d32Float")
            {
                return device.formatD32Float;
            }
            if (name == "packedDepthStencil")
            {
                return device.formatD24UnormS8Uint || device.formatD32FloatS8Uint;
            }
            if (name == "rgba16Float")
            {
                return device.formatRgba16Float;
            }
            if (name == "r32Uint")
            {
                return device.formatR32Uint;
            }
            return false;
        }

        [[nodiscard]] constexpr std::uint64_t reportLimit(const VulkanPhysicalDeviceReport& device,
                                                          std::string_view name) noexcept
        {
            if (name == "maxImageDimension2D")
            {
                return device.maxImageDimension2D;
            }
            if (name == "maxBoundDescriptorSets")
            {
                return device.maxBoundDescriptorSets;
            }
            if (name == "maxPushConstantsSize")
            {
                return device.maxPushConstantsSize;
            }
            if (name == "maxColorAttachments")
            {
                return device.maxColorAttachments;
            }
            if (name == "maxComputeWorkgroupInvocations")
            {
                return device.maxComputeWorkgroupInvocations;
            }
            return 0;
        }

    } // namespace detail

    [[nodiscard]] constexpr std::span<const VulkanProfileItem> vulkanWindowsProfileItems() noexcept
    {
        static constexpr std::array items{
            VulkanProfileItem{
                .name = "vulkan1.3",
                .kind = VulkanProfileItemKind::apiVersion,
                .disposition = VulkanProfileDisposition::required,
                .minimum = 1'003,
                .justification = "Dynamic rendering, synchronization2, and timeline semaphores are core.",
            },
            VulkanProfileItem{
                .name = "dynamicRendering",
                .kind = VulkanProfileItemKind::feature,
                .disposition = VulkanProfileDisposition::required,
                .justification = "The RHI has no render-pass object; attachments are scoped per recording.",
            },
            VulkanProfileItem{
                .name = "synchronization2",
                .kind = VulkanProfileItemKind::feature,
                .disposition = VulkanProfileDisposition::required,
                .justification = "Portable barriers map to stage/access/layout pairs, not D3D12 resource states.",
            },
            VulkanProfileItem{
                .name = "timelineSemaphore",
                .kind = VulkanProfileItemKind::feature,
                .disposition = VulkanProfileDisposition::required,
                .justification = "Queue frontiers and host waits are monotonic timeline points.",
            },
            VulkanProfileItem{
                .name = "independentBlend",
                .kind = VulkanProfileItemKind::feature,
                .disposition = VulkanProfileDisposition::required,
                .justification = "UI and offscreen targets use per-attachment blend state.",
            },
            VulkanProfileItem{
                .name = "samplerAnisotropy",
                .kind = VulkanProfileItemKind::feature,
                .disposition = VulkanProfileDisposition::required,
                .justification = "Existing sampler contracts expose anisotropy on Windows desktop devices.",
            },
            VulkanProfileItem{
                .name = "textureCompressionBC",
                .kind = VulkanProfileItemKind::feature,
                .disposition = VulkanProfileDisposition::required,
                .justification = "Windows desktop textures use BC1-BC7; ASTC is not required.",
            },
            VulkanProfileItem{
                .name = "shaderSampledImageArrayDynamicIndexing",
                .kind = VulkanProfileItemKind::feature,
                .disposition = VulkanProfileDisposition::required,
                .justification = "Uniform dynamic indexing of sampled images is required by core shaders.",
            },
            VulkanProfileItem{
                .name = "timestampComputeAndGraphics",
                .kind = VulkanProfileItemKind::feature,
                .disposition = VulkanProfileDisposition::required,
                .justification = "Diagnostics and later query work need timestamps on both queue families.",
            },
            VulkanProfileItem{
                .name = "hostQueryReset",
                .kind = VulkanProfileItemKind::feature,
                .disposition = VulkanProfileDisposition::required,
                .justification = "Vulkan 1.3 core feature required to initialize query pools in the reset state.",
            },
            VulkanProfileItem{
                .name = "VK_KHR_surface",
                .kind = VulkanProfileItemKind::instanceExtension,
                .disposition = VulkanProfileDisposition::required,
                .justification = "Windows presentation is a surface/swapchain layer above headless devices.",
            },
            VulkanProfileItem{
                .name = "VK_KHR_win32_surface",
                .kind = VulkanProfileItemKind::instanceExtension,
                .disposition = VulkanProfileDisposition::required,
                .justification = "The only Windows WSI path is Win32.",
            },
            VulkanProfileItem{
                .name = "VK_KHR_swapchain",
                .kind = VulkanProfileItemKind::deviceExtension,
                .disposition = VulkanProfileDisposition::required,
                .justification = "Presentation needs a swapchain; headless tests skip swapchain use.",
            },
            VulkanProfileItem{
                .name = "VK_KHR_swapchain_maintenance1",
                .kind = VulkanProfileItemKind::deviceExtension,
                .disposition = VulkanProfileDisposition::preferred,
                .justification = "Present fences retire swapchains without a device-wide idle.",
            },
            VulkanProfileItem{
                .name = "VK_EXT_swapchain_maintenance1",
                .kind = VulkanProfileItemKind::deviceExtension,
                .disposition = VulkanProfileDisposition::preferred,
                .justification = "Accepted as the pre-KHR alias of swapchain maintenance present fences.",
            },
            VulkanProfileItem{
                .name = "VK_KHR_present_id",
                .kind = VulkanProfileItemKind::deviceExtension,
                .disposition = VulkanProfileDisposition::preferred,
                .justification = "Attaches monotonic present IDs required by VK_KHR_present_wait.",
            },
            VulkanProfileItem{
                .name = "VK_KHR_present_wait",
                .kind = VulkanProfileItemKind::deviceExtension,
                .disposition = VulkanProfileDisposition::preferred,
                .justification = "Optional present-completion wait when present fences are unavailable.",
            },
            VulkanProfileItem{
                .name = "VK_EXT_device_fault",
                .kind = VulkanProfileItemKind::deviceExtension,
                .disposition = VulkanProfileDisposition::preferred,
                .justification = "Optional vendor fault dumps; DRED is D3D12-only and is never emulated.",
            },
            VulkanProfileItem{
                .name = "rgba8Unorm",
                .kind = VulkanProfileItemKind::format,
                .disposition = VulkanProfileDisposition::required,
                .justification = "Linear 8-bit color is used by UI and offscreen composition.",
            },
            VulkanProfileItem{
                .name = "rgba8UnormSrgb",
                .kind = VulkanProfileItemKind::format,
                .disposition = VulkanProfileDisposition::required,
                .justification = "sRGB 8-bit color is used by UI images and swapchain-compatible targets.",
            },
            VulkanProfileItem{
                .name = "bgra8Unorm",
                .kind = VulkanProfileItemKind::format,
                .disposition = VulkanProfileDisposition::required,
                .justification = "Win32 swapchains prefer BGRA8 UNORM.",
            },
            VulkanProfileItem{
                .name = "bgra8UnormSrgb",
                .kind = VulkanProfileItemKind::format,
                .disposition = VulkanProfileDisposition::required,
                .justification = "Win32 swapchains prefer BGRA8 sRGB when presenting UI.",
            },
            VulkanProfileItem{
                .name = "d32Float",
                .kind = VulkanProfileItemKind::format,
                .disposition = VulkanProfileDisposition::required,
                .justification = "Offscreen viewports use a 32-bit float depth target.",
            },
            VulkanProfileItem{
                .name = "packedDepthStencil",
                .kind = VulkanProfileItemKind::format,
                .disposition = VulkanProfileDisposition::required,
                .justification = "At least one of D24S8 or D32S8 must be usable as a depth-stencil target.",
            },
            VulkanProfileItem{
                .name = "rgba16Float",
                .kind = VulkanProfileItemKind::format,
                .disposition = VulkanProfileDisposition::required,
                .justification = "Higher-precision offscreen color is part of the current format set.",
            },
            VulkanProfileItem{
                .name = "r32Uint",
                .kind = VulkanProfileItemKind::format,
                .disposition = VulkanProfileDisposition::required,
                .justification = "Integer storage/attachment cases already exist in RHI contracts.",
            },
            VulkanProfileItem{
                .name = "maxImageDimension2D",
                .kind = VulkanProfileItemKind::limitMinimum,
                .disposition = VulkanProfileDisposition::required,
                .minimum = 8192,
                .justification = "Viewports and atlases need at least 8K 2D textures.",
            },
            VulkanProfileItem{
                .name = "maxBoundDescriptorSets",
                .kind = VulkanProfileItemKind::limitMinimum,
                .disposition = VulkanProfileDisposition::required,
                .minimum = 4,
                .justification = "Core renderer layouts use a small number of binding sets.",
            },
            VulkanProfileItem{
                .name = "maxPushConstantsSize",
                .kind = VulkanProfileItemKind::limitMinimum,
                .disposition = VulkanProfileDisposition::required,
                .minimum = 128,
                .justification = "UI and scene push-constant blocks fit in 128 bytes.",
            },
            VulkanProfileItem{
                .name = "maxColorAttachments",
                .kind = VulkanProfileItemKind::limitMinimum,
                .disposition = VulkanProfileDisposition::required,
                .minimum = 8,
                .justification = "The portable color-attachment limit is already eight.",
            },
            VulkanProfileItem{
                .name = "maxComputeWorkgroupInvocations",
                .kind = VulkanProfileItemKind::limitMinimum,
                .disposition = VulkanProfileDisposition::required,
                .minimum = 256,
                .justification = "Internal compute helpers use 256-wide workgroups.",
            },
            VulkanProfileItem{
                .name = "MESA_DOZEN",
                .kind = VulkanProfileItemKind::driverExclusion,
                .disposition = VulkanProfileDisposition::required,
                .justification = "Dozen is a D3D12 translation ICD and must never count as Vulkan support.",
            },
            VulkanProfileItem{
                .name = "VK_KHR_wayland_surface",
                .kind = VulkanProfileItemKind::platformFollowUp,
                .disposition = VulkanProfileDisposition::followUp,
                .justification = "Wayland WSI belongs to the Linux profile; Windows devices must not require it.",
            },
            VulkanProfileItem{
                .name = "VK_KHR_xcb_surface",
                .kind = VulkanProfileItemKind::platformFollowUp,
                .disposition = VulkanProfileDisposition::followUp,
                .justification = "X11 WSI belongs to the Linux profile and is not a Windows requirement.",
            },
            VulkanProfileItem{
                .name = "textureCompressionASTC_LDR",
                .kind = VulkanProfileItemKind::platformFollowUp,
                .disposition = VulkanProfileDisposition::followUp,
                .justification = "ASTC is a mobile texture format; Windows requires BC instead.",
            },
            VulkanProfileItem{
                .name = "XR_KHR_vulkan_enable2",
                .kind = VulkanProfileItemKind::platformFollowUp,
                .disposition = VulkanProfileDisposition::followUp,
                .justification = "OpenXR swapchains are outside the Windows profile.",
            },
        };
        return items;
    }

    [[nodiscard]] inline VulkanProfileDecision evaluateWindowsVulkanProfile(
        const VulkanInstanceReport& instance,
        const VulkanPhysicalDeviceReport& device,
        const VulkanCreateIntent& intent = {}) noexcept
    {
        VulkanProfileDecision decision{};

        if (intent.warpPreference)
        {
            decision.rejection = VulkanProfileRejection::warpPreferenceIsD3D12Only;
            detail::appendMissing(decision, "AdapterPreference::warp");
            return decision;
        }

        if (device.translationLayer || isRejectedVulkanTranslationDriver(device.driverId))
        {
            decision.rejection = VulkanProfileRejection::translationLayer;
            detail::appendMissing(decision, device.driverId.empty() ? "translationLayer" : device.driverId);
            return decision;
        }

        if (device.software && intent.policy != VulkanAdapterSelectionPolicy::softwareTestDevice)
        {
            decision.rejection = VulkanProfileRejection::softwareAdapterNotSelected;
            detail::appendMissing(decision, "softwareAdapter");
            return decision;
        }

        if (!detail::apiVersionAtLeast(
                instance.apiMajor, instance.apiMinor, vulkanWindowsProfileApiMajor, vulkanWindowsProfileApiMinor) ||
            !detail::apiVersionAtLeast(
                device.apiMajor, device.apiMinor, vulkanWindowsProfileApiMajor, vulkanWindowsProfileApiMinor))
        {
            decision.rejection = VulkanProfileRejection::apiVersionTooLow;
            detail::appendMissing(decision, "vulkan1.3");
            return decision;
        }

        for (const auto& item : vulkanWindowsProfileItems())
        {
            if (item.disposition != VulkanProfileDisposition::required)
            {
                continue;
            }

            switch (item.kind)
            {
            case VulkanProfileItemKind::apiVersion:
            case VulkanProfileItemKind::driverExclusion:
            case VulkanProfileItemKind::platformFollowUp:
                break;
            case VulkanProfileItemKind::feature:
                if (!detail::reportHasFeature(device, item.name))
                {
                    decision.rejection = VulkanProfileRejection::missingRequiredFeature;
                    detail::appendMissing(decision, item.name);
                }
                break;
            case VulkanProfileItemKind::instanceExtension:
                if (!detail::reportHasInstanceExtension(instance, item.name))
                {
                    decision.rejection = VulkanProfileRejection::missingRequiredExtension;
                    detail::appendMissing(decision, item.name);
                }
                break;
            case VulkanProfileItemKind::deviceExtension:
                if (!detail::reportHasDeviceExtension(device, item.name))
                {
                    decision.rejection = VulkanProfileRejection::missingRequiredExtension;
                    detail::appendMissing(decision, item.name);
                }
                break;
            case VulkanProfileItemKind::format:
                if (!detail::reportHasFormat(device, item.name))
                {
                    decision.rejection = VulkanProfileRejection::missingRequiredFormat;
                    detail::appendMissing(decision, item.name);
                }
                break;
            case VulkanProfileItemKind::limitMinimum:
                if (detail::reportLimit(device, item.name) < item.minimum)
                {
                    decision.rejection = VulkanProfileRejection::limitBelowMinimum;
                    detail::appendMissing(decision, item.name);
                }
                break;
            }
        }

        if (decision.rejection != VulkanProfileRejection::none)
        {
            return decision;
        }

        decision.accepted = true;
        decision.presentCompletion = selectVulkanPresentCompletionStrategy(device);
        return decision;
    }

    [[nodiscard]] inline std::string describeVulkanProfileRejection(const VulkanProfileDecision& decision)
    {
        std::string message = "Vulkan adapter failed the Windows profile";
        switch (decision.rejection)
        {
        case VulkanProfileRejection::none:
            return {};
        case VulkanProfileRejection::warpPreferenceIsD3D12Only:
            message += " because AdapterPreference::warp is D3D12-only";
            break;
        case VulkanProfileRejection::apiVersionTooLow:
            message += " because the API version is below Vulkan 1.3";
            break;
        case VulkanProfileRejection::translationLayer:
            message += " because the ICD is a D3D12 translation layer";
            break;
        case VulkanProfileRejection::softwareAdapterNotSelected:
            message += " because software (CPU) Vulkan implementations are not supported";
            break;
        case VulkanProfileRejection::missingRequiredFeature:
            message += " because a required feature is missing";
            break;
        case VulkanProfileRejection::missingRequiredExtension:
            message += " because a required extension is missing";
            break;
        case VulkanProfileRejection::missingRequiredFormat:
            message += " because a required format is missing";
            break;
        case VulkanProfileRejection::limitBelowMinimum:
            message += " because a required limit is below the Windows minimum";
            break;
        }
        if (decision.missingItemCount != 0)
        {
            message += " (";
            for (std::uint32_t i = 0; i < decision.missingItemCount; ++i)
            {
                if (i != 0)
                {
                    message += ", ";
                }
                message += decision.missingItems[i];
            }
            message += ")";
        }
        message += ". Selection did not fall back to D3D12.";
        return message;
    }

} // namespace rachis
