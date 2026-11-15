#pragma once

#include "D3D12/D3D12Queue.h"
#include "D3D12/D3D12Resource.h"
#include "Rachis/Query.h"

#include <d3d12.h>

#include <cstdint>
#include <memory>
#include <optional>

namespace rachis::d3d12
{

    class DebugMarkerManager;

    struct PreparedRenderingTimestamps final
    {
        QueryPoolHandle queryPool;
        std::uint32_t beginningOfPassWriteIndex = timestampWriteUnused;
        std::uint32_t endOfPassWriteIndex = timestampWriteUnused;
        std::shared_ptr<void> owner;
        ID3D12QueryHeap* heap = nullptr;
    };

    class QueryManager final
    {
    public:
        QueryManager();
        ~QueryManager();

        QueryManager(const QueryManager&) = delete;
        QueryManager& operator=(const QueryManager&) = delete;
        QueryManager(QueryManager&&) = delete;
        QueryManager& operator=(QueryManager&&) = delete;

        [[nodiscard]] Status initialize(ID3D12Device& device,
                                        QueueManager& queues,
                                        bool enhancedBarriers,
                                        bool meshPipelineStatisticsQueries);
        [[nodiscard]] QueueProperties queueProperties(QueueType queueType) const noexcept;
        [[nodiscard]] Result<QueueClockCalibration> calibrateQueueClock(const QueueManager& queues,
                                                                        QueueType queueType) const;
        [[nodiscard]] Status beginConditionalExecution(QueueManager& queues,
                                                       ResourceManager& resources,
                                                       detail::BackendCommandBuffer& commandBuffer,
                                                       const ConditionalExecutionDesc& desc);
        [[nodiscard]] Status endConditionalExecution(QueueManager& queues, detail::BackendCommandBuffer& commandBuffer);
        [[nodiscard]] Result<QueryPoolHandle> createTimestampQueryPool(const TimestampQueryPoolDesc& desc);
        [[nodiscard]] Result<QueryPoolHandle> createOcclusionQueryPool(const OcclusionQueryPoolDesc& desc);
        [[nodiscard]] Status destroyQueryPool(QueryPoolHandle queryPool);
        [[nodiscard]] Status resetQueryPool(QueueManager& queues,
                                            detail::BackendCommandBuffer& commandBuffer,
                                            QueryPoolHandle queryPool,
                                            std::uint32_t firstQuery,
                                            std::uint32_t queryCount);
        [[nodiscard]] Status writeTimestamp(QueueManager& queues,
                                            detail::BackendCommandBuffer& commandBuffer,
                                            QueryPoolHandle queryPool,
                                            std::uint32_t queryIndex);
        [[nodiscard]] Status beginGpuProfileScope(QueueManager& queues,
                                                  const DebugMarkerManager& debugMarkers,
                                                  detail::BackendCommandBuffer& commandBuffer,
                                                  const GpuProfileScopeDesc& desc);
        [[nodiscard]] Status endGpuProfileScope(QueueManager& queues,
                                                const DebugMarkerManager& debugMarkers,
                                                detail::BackendCommandBuffer& commandBuffer);
        [[nodiscard]] Status resolveTimestamps(QueueManager& queues,
                                               ResourceManager& resources,
                                               detail::BackendCommandBuffer& commandBuffer,
                                               const TimestampQueryResolveDesc& desc);
        [[nodiscard]] Result<std::optional<PreparedRenderingTimestamps>> prepareRenderingTimestamps(
            detail::BackendCommandBuffer& commandBuffer, const std::optional<PassTimestampWrites>& timestampWrites);
        void commitRenderingTimestamps(ID3D12GraphicsCommandList& commandList,
                                       detail::BackendCommandBuffer& commandBuffer,
                                       PreparedRenderingTimestamps prepared) noexcept;
        void endRenderingTimestamps(ID3D12GraphicsCommandList& commandList,
                                    detail::BackendCommandBuffer& commandBuffer) noexcept;
        [[nodiscard]] Status beginOcclusionQuery(QueueManager& queues,
                                                 detail::BackendCommandBuffer& commandBuffer,
                                                 QueryPoolHandle queryPool,
                                                 std::uint32_t queryIndex);
        [[nodiscard]] Status endOcclusionQuery(QueueManager& queues, detail::BackendCommandBuffer& commandBuffer);
        [[nodiscard]] Status resolveOcclusionQueries(QueueManager& queues,
                                                     ResourceManager& resources,
                                                     detail::BackendCommandBuffer& commandBuffer,
                                                     const OcclusionQueryResolveDesc& desc);
        [[nodiscard]] Result<QueryPoolHandle> createPipelineStatisticsQueryPool(
            const PipelineStatisticsQueryPoolDesc& desc);
        [[nodiscard]] Status beginPipelineStatisticsQuery(QueueManager& queues,
                                                          detail::BackendCommandBuffer& commandBuffer,
                                                          QueryPoolHandle queryPool,
                                                          std::uint32_t queryIndex);
        [[nodiscard]] Status endPipelineStatisticsQuery(QueueManager& queues,
                                                        detail::BackendCommandBuffer& commandBuffer);
        [[nodiscard]] Status resolvePipelineStatisticsQueries(QueueManager& queues,
                                                              ResourceManager& resources,
                                                              detail::BackendCommandBuffer& commandBuffer,
                                                              const PipelineStatisticsQueryResolveDesc& desc);
        [[nodiscard]] Result<QueryPoolHandle> createStreamOutputStatisticsQueryPool(
            const StreamOutputStatisticsQueryPoolDesc& desc);
        [[nodiscard]] Status beginStreamOutputStatisticsQuery(QueueManager& queues,
                                                              detail::BackendCommandBuffer& commandBuffer,
                                                              QueryPoolHandle queryPool,
                                                              std::uint32_t queryIndex);
        [[nodiscard]] Status endStreamOutputStatisticsQuery(QueueManager& queues,
                                                            detail::BackendCommandBuffer& commandBuffer,
                                                            std::uint32_t stream);
        [[nodiscard]] Status resolveStreamOutputStatisticsQueries(QueueManager& queues,
                                                                  ResourceManager& resources,
                                                                  detail::BackendCommandBuffer& commandBuffer,
                                                                  const StreamOutputStatisticsQueryResolveDesc& desc);
        [[nodiscard]] Result<QueryPoolHandle> createAccelerationStructureCompactedSizeQueryPool(
            const AccelerationStructureCompactedSizeQueryPoolDesc& desc);
        [[nodiscard]] Status writeAccelerationStructureCompactedSize(QueueManager& queues,
                                                                     ResourceManager& resources,
                                                                     detail::BackendCommandBuffer& commandBuffer,
                                                                     QueryPoolHandle queryPool,
                                                                     std::uint32_t queryIndex,
                                                                     AccelerationStructureHandle source);
        [[nodiscard]] Status resolveAccelerationStructureCompactedSizes(
            QueueManager& queues,
            ResourceManager& resources,
            detail::BackendCommandBuffer& commandBuffer,
            const AccelerationStructureCompactedSizeQueryResolveDesc& desc);
        [[nodiscard]] Result<QueryPoolHandle> createAccelerationStructureSerializationInfoQueryPool(
            const AccelerationStructureSerializationInfoQueryPoolDesc& desc);
        [[nodiscard]] Status writeAccelerationStructureSerializationInfo(QueueManager& queues,
                                                                         ResourceManager& resources,
                                                                         detail::BackendCommandBuffer& commandBuffer,
                                                                         QueryPoolHandle queryPool,
                                                                         std::uint32_t queryIndex,
                                                                         AccelerationStructureHandle source);
        [[nodiscard]] Status resolveAccelerationStructureSerializationInfo(
            QueueManager& queues,
            ResourceManager& resources,
            detail::BackendCommandBuffer& commandBuffer,
            const AccelerationStructureSerializationInfoQueryResolveDesc& desc);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace rachis::d3d12
