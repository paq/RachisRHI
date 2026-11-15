#pragma once

#include <volk.h>

namespace rachis::vulkan
{

    [[nodiscard]] inline VkBufferMemoryBarrier2 copyWritesToHostReadBarrier(VkBuffer buffer, VkDeviceSize size) noexcept
    {
        return VkBufferMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .pNext = {},
            .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
            .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = buffer,
            .offset = 0,
            .size = size,
        };
    }

    inline void makeCopyWritesHostVisible(const VolkDeviceTable& table,
                                          VkCommandBuffer command,
                                          VkBuffer buffer,
                                          VkDeviceSize size) noexcept
    {
        const auto barrier = copyWritesToHostReadBarrier(buffer, size);
        const VkDependencyInfo dependency{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = {},
            .dependencyFlags = {},
            .memoryBarrierCount = {},
            .pMemoryBarriers = {},
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &barrier,
            .imageMemoryBarrierCount = {},
            .pImageMemoryBarriers = {},
        };
        table.vkCmdPipelineBarrier2(command, &dependency);
    }

} // namespace rachis::vulkan
