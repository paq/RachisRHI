#pragma once

#include "Internal/BackendCommandBuffer.h"
#include "Internal/VulkanCommands.h"
#include "Internal/VulkanRuntime.h"
#include "Rachis/Command.h"
#include "Rachis/Interop.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <thread>
#include <volk.h>

namespace rachis::vulkan
{

    inline constexpr std::uint32_t transientVulkanRetryLimit = 64;

    template<typename Operation>
    [[nodiscard]] bool retryTransientVulkanFailures(Operation&& operation)
    {
        for (std::uint32_t attempt = 0; attempt < transientVulkanRetryLimit; ++attempt)
        {
            const VkResult result = operation();
            if (result == VK_SUCCESS)
            {
                return true;
            }
            if (result != VK_ERROR_OUT_OF_HOST_MEMORY && result != VK_ERROR_OUT_OF_DEVICE_MEMORY)
            {
                return false;
            }
            std::this_thread::yield();
        }
        return false;
    }

    class VulkanQueueManager final
    {
    public:
        VulkanQueueManager();
        ~VulkanQueueManager();

        VulkanQueueManager(const VulkanQueueManager&) = delete;
        VulkanQueueManager& operator=(const VulkanQueueManager&) = delete;
        VulkanQueueManager(VulkanQueueManager&&) = delete;
        VulkanQueueManager& operator=(VulkanQueueManager&&) = delete;

        [[nodiscard]] Status initialize(VkDevice device,
                                        const VolkDeviceTable* table,
                                        const VulkanQueueHandles& queues,
                                        std::uint64_t deviceToken,
                                        bool exportableTimelines = false,
                                        const VulkanInstanceTable* instanceTable = nullptr) noexcept;
        void shutdown() noexcept;

        [[nodiscard]] Result<TimelinePoint> submit(
            std::span<std::unique_ptr<detail::BackendCommandBuffer>> commandBuffers,
            std::span<const TimelinePoint> waits);
        [[nodiscard]] Result<TimelinePointStatus> queryTimelinePoint(TimelinePoint point) const;
        [[nodiscard]] Result<TimelinePointStatus> wait(TimelinePoint point, std::chrono::nanoseconds timeout);
        [[nodiscard]] Status wait(TimelinePoint point);
        [[nodiscard]] Status waitIdle(QueueType queueType);
        [[nodiscard]] Status waitPresentIdle();
        [[nodiscard]] Status waitIdle();
        [[nodiscard]] std::uint64_t completedValue(QueueType queueType) const noexcept;
        void collectGarbage() noexcept;

        [[nodiscard]] Status validateSubmittedPoint(TimelinePoint point) const;
        [[nodiscard]] Result<SharedTimelinePoint> exportTimelinePoint(TimelinePoint point);
        [[nodiscard]] Result<TimelinePoint> enqueueSharedTimelineWait(QueueType queueType,
                                                                      SharedTimelinePointView point);

        struct PresentationSync final
        {
            std::optional<TimelinePoint> waitTimeline{};
            VkSemaphore waitBinary = VK_NULL_HANDLE;
            VkSemaphore signalBinary = VK_NULL_HANDLE;
        };

        [[nodiscard]] Result<TimelinePoint> submitGraphicsPresentationSync(const PresentationSync& sync);

        void drainGraphicsAcquire(VkSemaphore semaphore);
        [[nodiscard]] Status submitPresentPresentationSync(const PresentationSync& sync);
        struct PresentResult final
        {
            VkResult result = VK_SUCCESS;
            bool waitsEnqueued = false;
            bool injectedDeviceLoss = false;
        };

        [[nodiscard]] Result<PresentResult> presentOnPresentQueue(const VkPresentInfoKHR& info);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace rachis::vulkan
