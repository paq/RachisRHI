#pragma once

#include "Internal/VulkanResources.h"
#include "Rachis/Device.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <volk.h>

namespace rachis::vulkan
{

    inline constexpr std::uint32_t countedDispatchWorkgroupSize = 64;
    inline constexpr std::uint64_t countedDispatchRecordSize = 12;

    struct VulkanCountedDispatchSource final
    {
        VkDeviceAddress arguments = 0;
        VkDeviceAddress count = 0;
        std::uint32_t stride = 0;
        std::uint32_t maxCount = 0;
    };

    struct VulkanCountedDispatchRecords final
    {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceAddress address = 0;
        std::shared_ptr<void> keepAlive;
    };

    class VulkanCountedDispatchPass final
    {
    public:
        VulkanCountedDispatchPass() = default;
        ~VulkanCountedDispatchPass();

        VulkanCountedDispatchPass(const VulkanCountedDispatchPass&) = delete;
        VulkanCountedDispatchPass& operator=(const VulkanCountedDispatchPass&) = delete;
        VulkanCountedDispatchPass(VulkanCountedDispatchPass&&) = delete;
        VulkanCountedDispatchPass& operator=(VulkanCountedDispatchPass&&) = delete;

        void initialize(VkDevice device,
                        const VolkDeviceTable* table,
                        VulkanResourceAllocator* resources,
                        std::uint32_t maxWorkgroupCountX) noexcept;
        void shutdown() noexcept;

        [[nodiscard]] Status prepare();
        [[nodiscard]] Result<VulkanCountedDispatchRecords> allocateRecords(std::uint32_t maxCount);
        void record(VkCommandBuffer command,
                    const VulkanCountedDispatchSource& source,
                    const VulkanCountedDispatchRecords& records) const noexcept;

    private:
        void destroyLocked() noexcept;

        VkDevice device_ = VK_NULL_HANDLE;
        const VolkDeviceTable* table_ = nullptr;
        VulkanResourceAllocator* resources_ = nullptr;
        std::uint32_t maxWorkgroupCountX_ = 65535;
        mutable std::mutex mutex_{};
        VkShaderModule module_ = VK_NULL_HANDLE;
        VkPipelineLayout layout_ = VK_NULL_HANDLE;
        VkPipeline pipeline_ = VK_NULL_HANDLE;
    };

} // namespace rachis::vulkan
