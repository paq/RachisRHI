#include "Internal/BackendFactory.h"

#include "Internal/VulkanBackendConfiguration.h"
#include "Internal/VulkanBuildConfiguration.h"

#include <string>

#if RACHIS_ENABLE_D3D12
#include "Internal/D3D12Factory.h"
#endif
#if RACHIS_ENABLE_VULKAN
#include "Internal/VulkanFactory.h"
#endif
#if RACHIS_ENABLE_METAL
#include "Internal/MetalFactory.h"
#endif

namespace rachis::detail
{
    namespace
    {

        [[nodiscard]] Error selectionError(VulkanSelectionFailure failure, std::string message)
        {
            return Error{
                vulkanSelectionError(failure),
                0,
                std::move(message),
            };
        }

    } // namespace

    Result<std::vector<AdapterInfo>> enumerateBackendAdapters(Backend backend)
    {
        switch (backend)
        {
        case Backend::d3d12:
#if RACHIS_ENABLE_D3D12
            return d3d12::enumerateAdapters();
#else
            return std::unexpected(selectionError(VulkanSelectionFailure::loaderUnavailable,
                                                  "The D3D12 backend is not linked into this build."));
#endif
        case Backend::vulkan:
#if RACHIS_ENABLE_VULKAN
            return vulkan::enumerateAdapters();
#else
            return std::unexpected(selectionError(VulkanSelectionFailure::loaderUnavailable,
                                                  "The Vulkan backend is not linked into this build."));
#endif
        case Backend::metal:
#if RACHIS_ENABLE_METAL
            return metal::enumerateAdapters();
#else
            return std::unexpected(
                selectionError(VulkanSelectionFailure::loaderUnavailable,
                               "The Metal backend is not linked into this build. Selection did not fall back."));
#endif
        }

        return std::unexpected(
            selectionError(VulkanSelectionFailure::unsupportedBackend, "The requested renderer backend is unknown."));
    }

    Result<std::unique_ptr<BackendDevice>> createBackendDevice(const DeviceDesc& desc)
    {
        switch (desc.backend)
        {
        case Backend::d3d12:
#if RACHIS_ENABLE_D3D12
            return d3d12::createDevice(desc);
#else
            return std::unexpected(selectionError(VulkanSelectionFailure::loaderUnavailable,
                                                  "The D3D12 backend is not linked into this build."));
#endif
        case Backend::vulkan:
            if (desc.adapterPreference == AdapterPreference::warp)
            {
                return std::unexpected(
                    selectionError(VulkanSelectionFailure::warpIsD3D12Only,
                                   "AdapterPreference::warp is D3D12-only and is not a Vulkan ICD."));
            }
            if (desc.enableDred)
            {
                return std::unexpected(selectionError(
                    VulkanSelectionFailure::dredIsD3D12Only,
                    "DeviceDesc::enableDred is D3D12-only. Use validation layers or optional device-fault."));
            }
#if RACHIS_ENABLE_VULKAN
            return vulkan::createDevice(desc);
#else
            return std::unexpected(selectionError(VulkanSelectionFailure::loaderUnavailable,
                                                  "The Vulkan backend is not linked into this build."));
#endif
        case Backend::metal:
            if (desc.adapterPreference == AdapterPreference::warp)
            {
                return std::unexpected(selectionError(
                    VulkanSelectionFailure::warpIsD3D12Only,
                    "AdapterPreference::warp is D3D12-only and is not a Metal device. Selection did not fall back."));
            }
            if (desc.enableDred)
            {
                return std::unexpected(selectionError(
                    VulkanSelectionFailure::dredIsD3D12Only,
                    "DeviceDesc::enableDred is D3D12-only. Selection did not fall back to another backend."));
            }
#if RACHIS_ENABLE_METAL
            return metal::createDevice(desc);
#else
            return std::unexpected(
                selectionError(VulkanSelectionFailure::loaderUnavailable,
                               "The Metal backend is not linked into this build. Selection did not fall back."));
#endif
        }

        return std::unexpected(
            selectionError(VulkanSelectionFailure::unsupportedBackend,
                           "The requested renderer backend is unknown. Selection did not fall back."));
    }

} // namespace rachis::detail
