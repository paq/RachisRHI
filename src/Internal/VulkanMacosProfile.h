#pragma once

#include "Internal/VulkanWindowsProfile.h"

namespace rachis
{

    inline constexpr std::uint16_t vulkanMacosProfileApiMajor = 1;
    inline constexpr std::uint16_t vulkanMacosProfileApiMinor = 3;

    [[nodiscard]] constexpr std::span<const VulkanProfileItem> vulkanMacosProfileItems() noexcept
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
                .justification = "Portable barriers map to stage/access/layout pairs.",
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
                .justification = "Existing sampler contracts expose anisotropy on desktop devices.",
            },
            VulkanProfileItem{
                .name = "textureCompressionBC",
                .kind = VulkanProfileItemKind::feature,
                .disposition = VulkanProfileDisposition::preferred,
                .justification = "MoltenVK may advertise BC; the investigation profile does not require it. Offscreen "
                                 "RGBA contracts still run; BC cases fail closed.",
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
                .name = "VK_KHR_portability_enumeration",
                .kind = VulkanProfileItemKind::instanceExtension,
                .disposition = VulkanProfileDisposition::required,
                .justification = "MoltenVK is a portability ICD and is invisible without this instance extension.",
            },
            VulkanProfileItem{
                .name = "VK_KHR_portability_subset",
                .kind = VulkanProfileItemKind::deviceExtension,
                .disposition = VulkanProfileDisposition::required,
                .justification = "The Vulkan spec requires enabling portability_subset when the device advertises it.",
            },
            VulkanProfileItem{
                .name = "VK_KHR_surface",
                .kind = VulkanProfileItemKind::instanceExtension,
                .disposition = VulkanProfileDisposition::followUp,
                .justification = "Enabled when present. Headless devices do not require a surface.",
            },
            VulkanProfileItem{
                .name = "VK_EXT_metal_surface",
                .kind = VulkanProfileItemKind::platformFollowUp,
                .disposition = VulkanProfileDisposition::followUp,
                .justification = "AppKit/Metal presentation is a later host issue. Not required for headless load.",
            },
            VulkanProfileItem{
                .name = "VK_KHR_wayland_surface",
                .kind = VulkanProfileItemKind::platformFollowUp,
                .disposition = VulkanProfileDisposition::followUp,
                .justification = "Wayland is Linux-only and must not be required on macOS.",
            },
            VulkanProfileItem{
                .name = "VK_KHR_xcb_surface",
                .kind = VulkanProfileItemKind::platformFollowUp,
                .disposition = VulkanProfileDisposition::followUp,
                .justification = "XCB is Linux-only and must not be required on macOS.",
            },
            VulkanProfileItem{
                .name = "VK_KHR_win32_surface",
                .kind = VulkanProfileItemKind::platformFollowUp,
                .disposition = VulkanProfileDisposition::followUp,
                .justification = "Win32 WSI is Windows-only and must not be required on macOS.",
            },
            VulkanProfileItem{
                .name = "VK_KHR_swapchain",
                .kind = VulkanProfileItemKind::deviceExtension,
                .disposition = VulkanProfileDisposition::followUp,
                .justification = "Enabled when present. Offscreen contracts skip swapchain use.",
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
                .justification = "Optional vendor fault dumps; DRED is D3D12-only.",
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
                .justification = "sRGB 8-bit color is used by UI images and offscreen targets.",
            },
            VulkanProfileItem{
                .name = "bgra8Unorm",
                .kind = VulkanProfileItemKind::format,
                .disposition = VulkanProfileDisposition::required,
                .justification = "MoltenVK exposes BGRA8 as the Metal-native 8-bit color layout.",
            },
            VulkanProfileItem{
                .name = "bgra8UnormSrgb",
                .kind = VulkanProfileItemKind::format,
                .disposition = VulkanProfileDisposition::required,
                .justification = "MoltenVK exposes BGRA8 sRGB for UI-compatible targets.",
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
        };
        return items;
    }

    [[nodiscard]] inline VulkanProfileDecision evaluateMacosVulkanProfile(
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
                instance.apiMajor, instance.apiMinor, vulkanMacosProfileApiMajor, vulkanMacosProfileApiMinor) ||
            !detail::apiVersionAtLeast(
                device.apiMajor, device.apiMinor, vulkanMacosProfileApiMajor, vulkanMacosProfileApiMinor))
        {
            decision.rejection = VulkanProfileRejection::apiVersionTooLow;
            detail::appendMissing(decision, "vulkan1.3");
            return decision;
        }

        for (const auto& item : vulkanMacosProfileItems())
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

    [[nodiscard]] inline std::string describeMacosVulkanProfileRejection(const VulkanProfileDecision& decision)
    {
        std::string message = "Vulkan adapter failed the macOS MoltenVK investigation profile";
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
            message += " because a required limit is below the macOS minimum";
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
