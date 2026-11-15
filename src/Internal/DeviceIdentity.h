#pragma once

#include "Common/MonotonicIdentitySource.h"

namespace rachis::detail
{

    [[nodiscard]] MonotonicIdentitySource& deviceTokenSource() noexcept;

#ifdef RACHIS_TESTING
    void setDeviceTokenSourceOverrideForTesting(MonotonicIdentitySource* overrideSource) noexcept;
#endif

} // namespace rachis::detail
