#pragma once

#include "Internal/BackendDevice.h"
#include "Rachis/Device.h"

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
    };

} // namespace rachis::detail
