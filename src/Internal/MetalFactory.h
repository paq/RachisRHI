#pragma once

#include "Internal/BackendDevice.h"

#include <memory>
#include <vector>

namespace rachis::metal
{

    [[nodiscard]] Result<std::vector<AdapterInfo>> enumerateAdapters();
    [[nodiscard]] Result<std::unique_ptr<detail::BackendDevice>> createDevice(const DeviceDesc& desc);

} // namespace rachis::metal
