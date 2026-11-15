#pragma once

namespace rachis::test
{

    [[nodiscard]] inline bool& forceVulkanRequestedFlag() noexcept
    {
        static bool s_requested = false;
        return s_requested;
    }

    inline void setForceVulkanRequested(bool requested) noexcept
    {
        forceVulkanRequestedFlag() = requested;
    }

    [[nodiscard]] inline bool forceVulkanRequested() noexcept
    {
        return forceVulkanRequestedFlag();
    }

} // namespace rachis::test
