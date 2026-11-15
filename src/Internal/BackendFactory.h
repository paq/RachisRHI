#pragma once

#include "Internal/BackendDevice.h"

#include <memory>
#include <vector>

namespace rachis::detail
{

    [[nodiscard]] Result<std::vector<AdapterInfo>> enumerateBackendAdapters(Backend backend);
    [[nodiscard]] Result<std::unique_ptr<BackendDevice>> createBackendDevice(const DeviceDesc& desc);

} // namespace rachis::detail
