#pragma once

#include "Rachis/Error.h"

#include <array>
#include <cstdint>
#include <span>

namespace rachis
{

    enum class VulkanSelectionFailure : std::uint8_t
    {
        unsupportedBackend,
        loaderUnavailable,
        adapterNotFound,
        requiredFeatureMissing,
        warpIsD3D12Only,
        dredIsD3D12Only,
    };

    struct VulkanSelectionErrorMap final
    {
        VulkanSelectionFailure failure = VulkanSelectionFailure::unsupportedBackend;
        ErrorCode rhi = ErrorCode::unsupported;
    };

    [[nodiscard]] constexpr std::span<const VulkanSelectionErrorMap> vulkanSelectionErrors() noexcept
    {
        static constexpr std::array rows{
            VulkanSelectionErrorMap{
                .failure = VulkanSelectionFailure::unsupportedBackend,
                .rhi = ErrorCode::unsupported,
            },
            VulkanSelectionErrorMap{
                .failure = VulkanSelectionFailure::loaderUnavailable,
                .rhi = ErrorCode::backendUnavailable,
            },
            VulkanSelectionErrorMap{
                .failure = VulkanSelectionFailure::adapterNotFound,
                .rhi = ErrorCode::adapterNotFound,
            },
            VulkanSelectionErrorMap{
                .failure = VulkanSelectionFailure::requiredFeatureMissing,
                .rhi = ErrorCode::unsupported,
            },
            VulkanSelectionErrorMap{
                .failure = VulkanSelectionFailure::warpIsD3D12Only,
                .rhi = ErrorCode::unsupported,
            },
            VulkanSelectionErrorMap{
                .failure = VulkanSelectionFailure::dredIsD3D12Only,
                .rhi = ErrorCode::unsupported,
            },
        };
        return rows;
    }

    [[nodiscard]] constexpr ErrorCode vulkanSelectionError(VulkanSelectionFailure failure) noexcept
    {
        for (const auto& row : vulkanSelectionErrors())
        {
            if (row.failure == failure)
            {
                return row.rhi;
            }
        }
        return ErrorCode::unsupported;
    }

} // namespace rachis
