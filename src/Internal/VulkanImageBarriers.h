#pragma once

#include <volk.h>

namespace rachis::vulkan
{

    inline constexpr VkImageLayout shaderImageDescriptorLayout = VK_IMAGE_LAYOUT_GENERAL;

    inline void ensureImageLayoutTransitionSourceScope(VkImageMemoryBarrier2& barrier) noexcept
    {
        if (barrier.oldLayout == barrier.newLayout)
        {
            return;
        }
        if (barrier.srcStageMask == VK_PIPELINE_STAGE_2_NONE && barrier.srcAccessMask == VK_ACCESS_2_NONE)
        {
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        }
        if (barrier.oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && barrier.srcStageMask != VK_PIPELINE_STAGE_2_NONE)
        {
            barrier.srcAccessMask |= VK_ACCESS_2_MEMORY_WRITE_BIT;
        }
    }

} // namespace rachis::vulkan
