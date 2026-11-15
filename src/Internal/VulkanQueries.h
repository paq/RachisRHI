#pragma once

#include "Common/HandleTable.h"
#include "Internal/VulkanCommands.h"
#include "Internal/VulkanResources.h"
#include "Internal/VulkanRuntime.h"
#include "Rachis/Query.h"

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <volk.h>

namespace rachis::vulkan
{

    struct PreparedRenderingTimestamps final
    {
        QueryPoolHandle queryPool;
        std::uint32_t beginningOfPassWriteIndex = timestampWriteUnused;
        std::uint32_t endOfPassWriteIndex = timestampWriteUnused;
        std::shared_ptr<void> owner;
        std::shared_ptr<VulkanSubmittedQueryPoolState> submittedState;
        VkQueryPool pool = VK_NULL_HANDLE;
    };

    class VulkanQueryManager final
    {
    public:
        enum class QueryPoolKind : std::uint8_t
        {
            timestamp,
            preciseOcclusion,
            binaryOcclusion,
            pipelineStatistics,
            meshPipelineStatistics,
        };

        VulkanQueryManager() = default;
        ~VulkanQueryManager();

        VulkanQueryManager(const VulkanQueryManager&) = delete;
        VulkanQueryManager& operator=(const VulkanQueryManager&) = delete;
        VulkanQueryManager(VulkanQueryManager&&) = delete;
        VulkanQueryManager& operator=(VulkanQueryManager&&) = delete;

        void initialize(VkDevice device,
                        const VolkDeviceTable* table,
                        const DeviceCapabilities* capabilities,
                        VulkanResourceAllocator* resources,
                        bool calibratedTimestamps,
                        const VulkanInstanceTable* instanceTable = nullptr,
                        std::array<bool, 3> queueTimestampSupport = {true, true, true}) noexcept;
        void shutdown() noexcept;

        [[nodiscard]] Result<QueueClockCalibration> calibrateQueueClock(VkPhysicalDevice physical,
                                                                        QueueType queueType,
                                                                        const VulkanPhysicalDeviceRecord& record) const;
        [[nodiscard]] Result<QueryPoolHandle> createTimestampQueryPool(const TimestampQueryPoolDesc& desc);
        [[nodiscard]] Result<QueryPoolHandle> createOcclusionQueryPool(const OcclusionQueryPoolDesc& desc);
        [[nodiscard]] Result<QueryPoolHandle> createPipelineStatisticsQueryPool(
            const PipelineStatisticsQueryPoolDesc& desc);
        [[nodiscard]] Status destroyQueryPool(QueryPoolHandle queryPool);
        [[nodiscard]] Status resetQueryPool(VulkanCommandBuffer& commandBuffer,
                                            QueryPoolHandle queryPool,
                                            std::uint32_t firstQuery,
                                            std::uint32_t queryCount);
        [[nodiscard]] Status writeTimestamp(VulkanCommandBuffer& commandBuffer,
                                            QueryPoolHandle queryPool,
                                            std::uint32_t queryIndex);
        [[nodiscard]] Status beginGpuProfileScope(VulkanCommandBuffer& commandBuffer, const GpuProfileScopeDesc& desc);
        [[nodiscard]] Status endGpuProfileScope(VulkanCommandBuffer& commandBuffer);
        [[nodiscard]] Status resolveTimestamps(VulkanCommandBuffer& commandBuffer,
                                               const TimestampQueryResolveDesc& desc);
        [[nodiscard]] Result<std::optional<PreparedRenderingTimestamps>> prepareRenderingTimestamps(
            VulkanCommandBuffer& commandBuffer, const std::optional<PassTimestampWrites>& timestampWrites);
        void beginRenderingTimestamps(VulkanCommandBuffer& commandBuffer, const PreparedRenderingTimestamps& prepared);
        void commitRenderingTimestamps(VulkanCommandBuffer& commandBuffer,
                                       PreparedRenderingTimestamps prepared) noexcept;
        [[nodiscard]] Status endRenderingTimestamps(VulkanCommandBuffer& commandBuffer);
        [[nodiscard]] Status beginOcclusionQuery(VulkanCommandBuffer& commandBuffer,
                                                 QueryPoolHandle queryPool,
                                                 std::uint32_t queryIndex);
        [[nodiscard]] Status endOcclusionQuery(VulkanCommandBuffer& commandBuffer);
        [[nodiscard]] Status resolveOcclusionQueries(VulkanCommandBuffer& commandBuffer,
                                                     const OcclusionQueryResolveDesc& desc);
        [[nodiscard]] Status beginPipelineStatisticsQuery(VulkanCommandBuffer& commandBuffer,
                                                          QueryPoolHandle queryPool,
                                                          std::uint32_t queryIndex);
        [[nodiscard]] Status endPipelineStatisticsQuery(VulkanCommandBuffer& commandBuffer);
        [[nodiscard]] Status resolvePipelineStatisticsQueries(VulkanCommandBuffer& commandBuffer,
                                                              const PipelineStatisticsQueryResolveDesc& desc);

    private:
        struct QueryPoolRecord final
        {
            VkDevice device = VK_NULL_HANDLE;
            const VolkDeviceTable* table = nullptr;
            VkQueryPool pool = VK_NULL_HANDLE;
            VkQueryPool primitivesPool = VK_NULL_HANDLE;
            std::shared_ptr<VulkanSubmittedQueryPoolState> submittedState;
            QueryPoolKind kind = QueryPoolKind::timestamp;
            QueueType queueType = QueueType::graphics;
            std::uint32_t queryCount = 0;
            std::string label;

            QueryPoolRecord() = default;
            QueryPoolRecord(const QueryPoolRecord&) = delete;
            QueryPoolRecord& operator=(const QueryPoolRecord&) = delete;
            QueryPoolRecord(QueryPoolRecord&& other) noexcept
            {
                *this = std::move(other);
            }
            QueryPoolRecord& operator=(QueryPoolRecord&& other) noexcept
            {
                if (this != &other)
                {
                    destroyNative();
                    device = other.device;
                    table = other.table;
                    pool = other.pool;
                    primitivesPool = other.primitivesPool;
                    submittedState = std::move(other.submittedState);
                    kind = other.kind;
                    queueType = other.queueType;
                    queryCount = other.queryCount;
                    label = std::move(other.label);
                    other.pool = VK_NULL_HANDLE;
                    other.primitivesPool = VK_NULL_HANDLE;
                    other.device = VK_NULL_HANDLE;
                    other.table = nullptr;
                }
                return *this;
            }
            ~QueryPoolRecord()
            {
                destroyNative();
            }

            void destroyNative() noexcept
            {
                if (table != nullptr && table->vkDestroyQueryPool != nullptr && device != VK_NULL_HANDLE)
                {
                    if (primitivesPool != VK_NULL_HANDLE)
                    {
                        table->vkDestroyQueryPool(device, primitivesPool, nullptr);
                    }
                    if (pool != VK_NULL_HANDLE)
                    {
                        table->vkDestroyQueryPool(device, pool, nullptr);
                    }
                }
                primitivesPool = VK_NULL_HANDLE;
                pool = VK_NULL_HANDLE;
            }
        };

        [[nodiscard]] Result<QueryPoolHandle> createPool(QueryPoolKind kind,
                                                         QueueType queueType,
                                                         std::uint32_t queryCount,
                                                         std::string_view label);
        [[nodiscard]] Result<std::shared_ptr<QueryPoolRecord>> queryPool(QueryPoolHandle handle);
        [[nodiscard]] Status retain(VulkanCommandBuffer& commandBuffer, const std::shared_ptr<void>& owner);
        void writeTimestampSlot(VkCommandBuffer command,
                                VkQueryPool pool,
                                std::uint32_t queryIndex,
                                VkPipelineStageFlags2 stage) const noexcept;
        void resetSlots(VkCommandBuffer command,
                        VkQueryPool pool,
                        std::uint32_t firstQuery,
                        std::uint32_t queryCount) const noexcept;
        [[nodiscard]] Result<VulkanResourceAllocator::ResolvedBindingBuffer> copyResults(
            VulkanCommandBuffer& commandBuffer,
            const std::shared_ptr<QueryPoolRecord>& pool,
            std::uint32_t firstQuery,
            std::uint32_t queryCount,
            BufferHandle destination,
            std::uint64_t destinationOffset,
            std::uint64_t stride,
            std::string_view operation);
        void recordResultCopyBarrier(VkCommandBuffer command,
                                     const VulkanResourceAllocator::ResolvedBindingBuffer& destination) const noexcept;

        VkDevice device_ = VK_NULL_HANDLE;
        const VolkDeviceTable* table_ = nullptr;
        const VulkanInstanceTable* instanceTable_ = nullptr;
        const DeviceCapabilities* capabilities_ = nullptr;
        VulkanResourceAllocator* resources_ = nullptr;
        bool calibratedTimestamps_ = false;
        std::array<bool, 3> queueTimestampSupport_{true, true, true};
        std::mutex mutex_{};
        detail::HandleTable<std::shared_ptr<QueryPoolRecord>, QueryPoolHandle> pools_{};
    };

} // namespace rachis::vulkan
