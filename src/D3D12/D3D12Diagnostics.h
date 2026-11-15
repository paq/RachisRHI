#pragma once

#include "Rachis/Diagnostics.h"
#include "Rachis/Error.h"

#include <d3d12.h>

#include <string_view>

namespace rachis::d3d12
{

    [[nodiscard]] Result<DeviceFaultReport> captureDeviceFaultReport(ID3D12Device& device);
    void emitDeviceFaultReport(const DeviceFaultReport& report,
                               std::string_view context,
                               DiagnosticCallback callback,
                               void* userData) noexcept;

} // namespace rachis::d3d12
