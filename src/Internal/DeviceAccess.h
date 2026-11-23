#pragma once

#include "Common/PresentTimingSink.h"
#include "Internal/BackendDevice.h"
#include "Rachis/Device.h"

#include <cstdint>
#include <memory>
#include <utility>

namespace rachis::detail
{

    struct DeviceAccess final
    {
        [[nodiscard]] static Device make(std::unique_ptr<BackendDevice> backend) noexcept
        {
            return Device(std::move(backend));
        }

        [[nodiscard]] static Result<SwapchainPresentTimings> querySwapchainPresentTimings(
            const Device& device,
            SwapchainHandle swapchain,
            std::uint64_t firstSequence,
            const PresentTimingSink& records)
        {
            if (device.backend_ == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Cannot query swapchain present timings on an invalid device.",
                });
            }
            return device.backend_->querySwapchainPresentTimings(swapchain, firstSequence, records);
        }
    };

} // namespace rachis::detail
