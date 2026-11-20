#include "D3D12/D3D12Query.h"

#include "Common/HandleTable.h"
#include "D3D12/D3D12DebugMarker.h"
#include "D3D12/D3D12NativeHelpers.h"
#include "D3D12/D3D12RuntimePolicies.h"
#include "D3D12/D3D12TestHooks.h"
#include "Internal/PortableValidation.h"

#include <wrl/client.h>

#include <Windows.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace rachis::d3d12
{
    namespace
    {

        using Microsoft::WRL::ComPtr;

        constexpr std::size_t queueCount = 3;

        [[nodiscard]] constexpr std::size_t queueIndex(QueueType queueType) noexcept
        {
            return static_cast<std::size_t>(queueType);
        }

        [[nodiscard]] constexpr bool validQueueType(QueueType queueType) noexcept
        {
            return queueIndex(queueType) < queueCount;
        }

        [[nodiscard]] constexpr std::uint64_t ticksToNanosecondsRoundedUp(std::uint64_t ticks,
                                                                          std::uint64_t frequency) noexcept
        {
            constexpr std::uint64_t nanosecondsPerSecond = 1'000'000'000ULL;
            const auto seconds = ticks / frequency;
            const auto remainder = ticks % frequency;
            return seconds * nanosecondsPerSecond + (remainder * nanosecondsPerSecond + frequency - 1) / frequency;
        }

        [[nodiscard]] constexpr std::optional<D3D12_PREDICATION_OP> nativePredicateOperation(
            PredicateOperation operation) noexcept
        {
            switch (operation)
            {
            case PredicateOperation::equalZero:
                return D3D12_PREDICATION_OP_NOT_EQUAL_ZERO;
            case PredicateOperation::notEqualZero:
                return D3D12_PREDICATION_OP_EQUAL_ZERO;
            }
            return std::nullopt;
        }

        enum class QueryPoolKind : std::uint8_t
        {
            timestamp,
            preciseOcclusion,
            binaryOcclusion,
            pipelineStatistics,
            meshPipelineStatistics,
            streamOutputStatistics0,
            streamOutputStatistics1,
            streamOutputStatistics2,
            streamOutputStatistics3,
            accelerationStructureCompactedSize,
            accelerationStructureSerializationInfo,
        };

        [[nodiscard]] constexpr D3D12_QUERY_HEAP_TYPE timestampHeapType(QueueType queueType) noexcept
        {
            return queueType == QueueType::copy ? D3D12_QUERY_HEAP_TYPE_COPY_QUEUE_TIMESTAMP
                                                : D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        }

        [[nodiscard]] constexpr D3D12_QUERY_TYPE nativeQueryType(QueryPoolKind kind) noexcept
        {
            switch (kind)
            {
            case QueryPoolKind::timestamp:
                return D3D12_QUERY_TYPE_TIMESTAMP;
            case QueryPoolKind::preciseOcclusion:
                return D3D12_QUERY_TYPE_OCCLUSION;
            case QueryPoolKind::binaryOcclusion:
                return D3D12_QUERY_TYPE_BINARY_OCCLUSION;
            case QueryPoolKind::pipelineStatistics:
                return D3D12_QUERY_TYPE_PIPELINE_STATISTICS;
            case QueryPoolKind::meshPipelineStatistics:
                return D3D12_QUERY_TYPE_PIPELINE_STATISTICS1;
            case QueryPoolKind::streamOutputStatistics0:
                return D3D12_QUERY_TYPE_SO_STATISTICS_STREAM0;
            case QueryPoolKind::streamOutputStatistics1:
                return D3D12_QUERY_TYPE_SO_STATISTICS_STREAM1;
            case QueryPoolKind::streamOutputStatistics2:
                return D3D12_QUERY_TYPE_SO_STATISTICS_STREAM2;
            case QueryPoolKind::streamOutputStatistics3:
                return D3D12_QUERY_TYPE_SO_STATISTICS_STREAM3;
            case QueryPoolKind::accelerationStructureCompactedSize:
            case QueryPoolKind::accelerationStructureSerializationInfo:
                break;
            }
            return D3D12_QUERY_TYPE_TIMESTAMP;
        }

        [[nodiscard]] constexpr D3D12_QUERY_HEAP_TYPE nativeHeapType(QueryPoolKind kind, QueueType queueType) noexcept
        {
            switch (kind)
            {
            case QueryPoolKind::timestamp:
                return timestampHeapType(queueType);
            case QueryPoolKind::preciseOcclusion:
            case QueryPoolKind::binaryOcclusion:
                return D3D12_QUERY_HEAP_TYPE_OCCLUSION;
            case QueryPoolKind::pipelineStatistics:
                return D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS;
            case QueryPoolKind::meshPipelineStatistics:
                return D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS1;
            case QueryPoolKind::streamOutputStatistics0:
            case QueryPoolKind::streamOutputStatistics1:
            case QueryPoolKind::streamOutputStatistics2:
            case QueryPoolKind::streamOutputStatistics3:
                return D3D12_QUERY_HEAP_TYPE_SO_STATISTICS;
            case QueryPoolKind::accelerationStructureCompactedSize:
            case QueryPoolKind::accelerationStructureSerializationInfo:
                break;
            }
            return D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        }

        [[nodiscard]] constexpr bool isOcclusion(QueryPoolKind kind) noexcept
        {
            return kind == QueryPoolKind::preciseOcclusion || kind == QueryPoolKind::binaryOcclusion;
        }

        [[nodiscard]] constexpr bool isPipelineStatistics(QueryPoolKind kind) noexcept
        {
            return kind == QueryPoolKind::pipelineStatistics || kind == QueryPoolKind::meshPipelineStatistics;
        }

        [[nodiscard]] constexpr bool isStreamOutputStatistics(QueryPoolKind kind) noexcept
        {
            return kind >= QueryPoolKind::streamOutputStatistics0 && kind <= QueryPoolKind::streamOutputStatistics3;
        }

        [[nodiscard]] constexpr std::uint32_t streamOutputStatisticsStream(QueryPoolKind kind) noexcept
        {
            return static_cast<std::uint32_t>(kind) -
                   static_cast<std::uint32_t>(QueryPoolKind::streamOutputStatistics0);
        }

        [[nodiscard]] bool hasActiveStreamOutputStatistics(const detail::QueryCommandState& state) noexcept
        {
            return std::ranges::any_of(state.activeStreamOutputStatistics,
                                       [](const auto& active) { return active.has_value(); });
        }

        [[nodiscard]] constexpr std::uint64_t pipelineStatisticsResultSize(QueryPoolKind kind) noexcept
        {
            return kind == QueryPoolKind::meshPipelineStatistics ? meshPipelineStatisticsQueryResultSize
                                                                 : pipelineStatisticsQueryResultSize;
        }

        [[nodiscard]] constexpr bool isAccelerationStructurePostbuild(QueryPoolKind kind) noexcept
        {
            return kind == QueryPoolKind::accelerationStructureCompactedSize ||
                   kind == QueryPoolKind::accelerationStructureSerializationInfo;
        }

        [[nodiscard]] constexpr std::uint64_t postbuildResultSize(QueryPoolKind kind) noexcept
        {
            return kind == QueryPoolKind::accelerationStructureSerializationInfo
                       ? accelerationStructureSerializationInfoQueryResultSize
                       : accelerationStructureCompactedSizeQueryResultSize;
        }

        struct QueryPoolRecord final
        {
            ComPtr<ID3D12QueryHeap> heap;
            ComPtr<ID3D12Resource> postbuildResults;
            QueryPoolKind kind = QueryPoolKind::timestamp;
            QueueType queueType = QueueType::graphics;
            std::uint32_t queryCount = 0;
            std::string label;
        };

        static_assert(sizeof(PipelineStatistics) == sizeof(D3D12_QUERY_DATA_PIPELINE_STATISTICS));
        static_assert(sizeof(StreamOutputStatistics) == sizeof(D3D12_QUERY_DATA_SO_STATISTICS));
        static_assert(alignof(StreamOutputStatistics) == alignof(D3D12_QUERY_DATA_SO_STATISTICS));
        static_assert(offsetof(StreamOutputStatistics, primitivesWritten) ==
                      offsetof(D3D12_QUERY_DATA_SO_STATISTICS, NumPrimitivesWritten));
        static_assert(offsetof(StreamOutputStatistics, primitivesStorageNeeded) ==
                      offsetof(D3D12_QUERY_DATA_SO_STATISTICS, PrimitivesStorageNeeded));
        static_assert(alignof(PipelineStatistics) == alignof(D3D12_QUERY_DATA_PIPELINE_STATISTICS));
        static_assert(offsetof(PipelineStatistics, inputAssemblyVertices) ==
                      offsetof(D3D12_QUERY_DATA_PIPELINE_STATISTICS, IAVertices));
        static_assert(offsetof(PipelineStatistics, inputAssemblyPrimitives) ==
                      offsetof(D3D12_QUERY_DATA_PIPELINE_STATISTICS, IAPrimitives));
        static_assert(offsetof(PipelineStatistics, vertexShaderInvocations) ==
                      offsetof(D3D12_QUERY_DATA_PIPELINE_STATISTICS, VSInvocations));
        static_assert(offsetof(PipelineStatistics, geometryShaderInvocations) ==
                      offsetof(D3D12_QUERY_DATA_PIPELINE_STATISTICS, GSInvocations));
        static_assert(offsetof(PipelineStatistics, geometryShaderPrimitives) ==
                      offsetof(D3D12_QUERY_DATA_PIPELINE_STATISTICS, GSPrimitives));
        static_assert(offsetof(PipelineStatistics, clippingInvocations) ==
                      offsetof(D3D12_QUERY_DATA_PIPELINE_STATISTICS, CInvocations));
        static_assert(offsetof(PipelineStatistics, clippingPrimitives) ==
                      offsetof(D3D12_QUERY_DATA_PIPELINE_STATISTICS, CPrimitives));
        static_assert(offsetof(PipelineStatistics, fragmentShaderInvocations) ==
                      offsetof(D3D12_QUERY_DATA_PIPELINE_STATISTICS, PSInvocations));
        static_assert(offsetof(PipelineStatistics, tessellationControlShaderPatches) ==
                      offsetof(D3D12_QUERY_DATA_PIPELINE_STATISTICS, HSInvocations));
        static_assert(offsetof(PipelineStatistics, tessellationEvaluationShaderInvocations) ==
                      offsetof(D3D12_QUERY_DATA_PIPELINE_STATISTICS, DSInvocations));
        static_assert(offsetof(PipelineStatistics, computeShaderInvocations) ==
                      offsetof(D3D12_QUERY_DATA_PIPELINE_STATISTICS, CSInvocations));
        static_assert(sizeof(MeshPipelineStatistics) == sizeof(D3D12_QUERY_DATA_PIPELINE_STATISTICS1));
        static_assert(alignof(MeshPipelineStatistics) == alignof(D3D12_QUERY_DATA_PIPELINE_STATISTICS1));
        static_assert(offsetof(MeshPipelineStatistics, base) == 0);
        static_assert(offsetof(MeshPipelineStatistics, taskShaderInvocations) ==
                      offsetof(D3D12_QUERY_DATA_PIPELINE_STATISTICS1, ASInvocations));
        static_assert(offsetof(MeshPipelineStatistics, meshShaderInvocations) ==
                      offsetof(D3D12_QUERY_DATA_PIPELINE_STATISTICS1, MSInvocations));
        static_assert(offsetof(MeshPipelineStatistics, meshShaderPrimitives) ==
                      offsetof(D3D12_QUERY_DATA_PIPELINE_STATISTICS1, MSPrimitives));
        static_assert(sizeof(AccelerationStructureSerializationInfo) ==
                      sizeof(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_SERIALIZATION_DESC));
        static_assert(alignof(AccelerationStructureSerializationInfo) ==
                      alignof(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_SERIALIZATION_DESC));
        static_assert(offsetof(AccelerationStructureSerializationInfo, serializedSize) ==
                      offsetof(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_SERIALIZATION_DESC,
                               SerializedSizeInBytes));
        static_assert(offsetof(AccelerationStructureSerializationInfo, bottomLevelPointerCount) ==
                      offsetof(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_SERIALIZATION_DESC,
                               NumBottomLevelAccelerationStructurePointers));

        [[nodiscard]] bool isPending(std::span<const detail::QueryWrite> writes,
                                     QueryPoolHandle queryPool,
                                     std::uint32_t queryIndex) noexcept
        {
            return std::ranges::any_of(writes, [&](const detail::QueryWrite& write) {
                return write.queryPool == queryPool && write.queryIndex == queryIndex && !write.resolved;
            });
        }

        [[nodiscard]] Status markPending(std::vector<detail::QueryWrite>& writes,
                                         QueryPoolHandle queryPool,
                                         std::uint32_t queryIndex)
        {
            const auto prior = std::ranges::find_if(writes, [&](const detail::QueryWrite& write) {
                return write.queryPool == queryPool && write.queryIndex == queryIndex;
            });
            if (prior != writes.end())
            {
                prior->resolved = false;
                return {};
            }
            try
            {
                writes.push_back(detail::QueryWrite{
                    .queryPool = queryPool,
                    .queryIndex = queryIndex,
                });
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Query completion bookkeeping could not be allocated.",
                });
            }
            return {};
        }

        [[nodiscard]] Status reservePending(std::vector<detail::QueryWrite>& writes,
                                            QueryPoolHandle queryPool,
                                            std::span<const std::uint32_t> queryIndices)
        {
            std::size_t additions = 0;
            for (const auto queryIndex : queryIndices)
            {
                if (queryIndex == timestampWriteUnused)
                {
                    continue;
                }
                const auto prior = std::ranges::find_if(writes, [&](const detail::QueryWrite& write) {
                    return write.queryPool == queryPool && write.queryIndex == queryIndex;
                });
                additions += prior == writes.end() ? 1 : 0;
            }
            try
            {
                writes.reserve(writes.size() + additions);
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Timestamp-scope bookkeeping could not be allocated.",
                });
            }
            return {};
        }

        void markPendingReserved(std::vector<detail::QueryWrite>& writes,
                                 QueryPoolHandle queryPool,
                                 std::uint32_t queryIndex) noexcept
        {
            const auto prior = std::ranges::find_if(writes, [&](const detail::QueryWrite& write) {
                return write.queryPool == queryPool && write.queryIndex == queryIndex;
            });
            if (prior != writes.end())
            {
                prior->resolved = false;
                return;
            }

            writes.push_back(detail::QueryWrite{
                .queryPool = queryPool,
                .queryIndex = queryIndex,
            });
        }

        void resetPendingRange(std::vector<detail::QueryWrite>& writes,
                               QueryPoolHandle queryPool,
                               std::uint32_t firstQuery,
                               std::uint32_t queryCount) noexcept
        {
            std::erase_if(writes, [&](const detail::QueryWrite& write) {
                return write.queryPool == queryPool && write.queryIndex >= firstQuery &&
                       write.queryIndex - firstQuery < queryCount;
            });
        }

        [[nodiscard]] Status retainQueryCommandDependencies(
            QueueManager& queues,
            detail::BackendCommandBuffer& commandBuffer,
            std::shared_ptr<void> poolOwner,
            std::shared_ptr<void> resourceOwner,
            std::span<const std::shared_ptr<ResidencyRecord>> residencies,
            AccelerationStructureExecutionLease lease = {})
        {
            const std::array dependencies{
                QueueManager::ResourceRetention{
                    .owner = std::move(resourceOwner),
                    .residencySet = residencies,
                },
                QueueManager::ResourceRetention{
                    .owner = std::move(poolOwner),
                },
            };
            return queues.retainAccelerationStructureDependencies(commandBuffer, dependencies, std::move(lease));
        }

        [[nodiscard]] Status retainQueryCommandDependencies(QueueManager& queues,
                                                            detail::BackendCommandBuffer& commandBuffer,
                                                            std::shared_ptr<void> poolOwner,
                                                            std::shared_ptr<void> resourceOwner,
                                                            const std::shared_ptr<ResidencyRecord>& residency)
        {
            const std::array residencies{residency};
            return retainQueryCommandDependencies(
                queues, commandBuffer, std::move(poolOwner), std::move(resourceOwner), residencies);
        }

        void transitionPostbuildResults(ID3D12GraphicsCommandList& commandList,
                                        ID3D12GraphicsCommandList7* enhancedCommandList,
                                        ID3D12Resource& resource,
                                        std::uint64_t size,
                                        bool toCopySource) noexcept
        {
            if (enhancedCommandList != nullptr)
            {
                const D3D12_BUFFER_BARRIER barrier{
                    .SyncBefore = toCopySource ? D3D12_BARRIER_SYNC_ALL : D3D12_BARRIER_SYNC_COPY,
                    .SyncAfter = toCopySource ? D3D12_BARRIER_SYNC_COPY : D3D12_BARRIER_SYNC_ALL,
                    .AccessBefore =
                        toCopySource ? D3D12_BARRIER_ACCESS_UNORDERED_ACCESS : D3D12_BARRIER_ACCESS_COPY_SOURCE,
                    .AccessAfter =
                        toCopySource ? D3D12_BARRIER_ACCESS_COPY_SOURCE : D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    .pResource = &resource,
                    .Offset = 0,
                    .Size = size,
                };
                const D3D12_BARRIER_GROUP group{
                    .Type = D3D12_BARRIER_TYPE_BUFFER,
                    .NumBarriers = 1,
                    .pBufferBarriers = &barrier,
                };
                enhancedCommandList->Barrier(1, &group);
            }
            else
            {
                const D3D12_RESOURCE_BARRIER barrier{
                    .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    .Transition =
                        {
                            .pResource = &resource,
                            .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                            .StateBefore =
                                toCopySource ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_COPY_SOURCE,
                            .StateAfter =
                                toCopySource ? D3D12_RESOURCE_STATE_COPY_SOURCE : D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        },
                };
                commandList.ResourceBarrier(1, &barrier);
            }
        }

        void serializePostbuildResultWrites(ID3D12GraphicsCommandList& commandList,
                                            ID3D12GraphicsCommandList7* enhancedCommandList,
                                            ID3D12Resource& resource,
                                            std::uint64_t size) noexcept
        {
            if (enhancedCommandList != nullptr)
            {
                const D3D12_BUFFER_BARRIER barrier{
                    .SyncBefore = D3D12_BARRIER_SYNC_ALL,
                    .SyncAfter = D3D12_BARRIER_SYNC_ALL,
                    .AccessBefore = D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    .AccessAfter = D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    .pResource = &resource,
                    .Offset = 0,
                    .Size = size,
                };
                const D3D12_BARRIER_GROUP group{
                    .Type = D3D12_BARRIER_TYPE_BUFFER,
                    .NumBarriers = 1,
                    .pBufferBarriers = &barrier,
                };
                enhancedCommandList->Barrier(1, &group);
            }
            else
            {
                const D3D12_RESOURCE_BARRIER barrier{
                    .Type = D3D12_RESOURCE_BARRIER_TYPE_UAV,
                    .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    .UAV = {.pResource = &resource},
                };
                commandList.ResourceBarrier(1, &barrier);
            }
            test::notePostbuildQueryResetBarrier();
        }

    } // namespace

    struct QueryManager::Impl final
    {
        ID3D12Device* device = nullptr;
        bool enhancedBarriers = false;
        bool meshPipelineStatisticsQueries = false;
        std::array<QueueProperties, queueCount> queueProperties;
        std::mutex mutex;
        detail::HandleTable<std::shared_ptr<QueryPoolRecord>, QueryPoolHandle> queryPools;

        [[nodiscard]] Result<QueryPoolHandle> createPool(QueryPoolKind kind,
                                                         QueueType queueType,
                                                         std::uint32_t queryCount,
                                                         std::string_view label)
        {
            auto record = std::make_shared<QueryPoolRecord>();
            HRESULT result = S_OK;
            if (isAccelerationStructurePostbuild(kind))
            {
                const D3D12_HEAP_PROPERTIES heapProperties{
                    .Type = D3D12_HEAP_TYPE_DEFAULT,
                    .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                    .MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,
                    .CreationNodeMask = 1,
                    .VisibleNodeMask = 1,
                };
                const D3D12_RESOURCE_DESC resourceDesc{
                    .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
                    .Alignment = 0,
                    .Width = static_cast<std::uint64_t>(queryCount) * postbuildResultSize(kind),
                    .Height = 1,
                    .DepthOrArraySize = 1,
                    .MipLevels = 1,
                    .Format = DXGI_FORMAT_UNKNOWN,
                    .SampleDesc = {.Count = 1, .Quality = 0},
                    .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    .Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                };
                result = device->CreateCommittedResource(&heapProperties,
                                                         D3D12_HEAP_FLAG_NONE,
                                                         &resourceDesc,
                                                         D3D12_RESOURCE_STATE_COMMON,
                                                         nullptr,
                                                         IID_PPV_ARGS(&record->postbuildResults));
            }
            else
            {
                const D3D12_QUERY_HEAP_DESC nativeDesc{
                    .Type = nativeHeapType(kind, queueType),
                    .Count = queryCount,
                    .NodeMask = 0,
                };
                result = device->CreateQueryHeap(&nativeDesc, IID_PPV_ARGS(&record->heap));
            }
            if (FAILED(result))
            {
                return std::unexpected(
                    nativeError(classifyD3D12ResultErrorCode(static_cast<std::int32_t>(result)),
                                result,
                                isAccelerationStructurePostbuild(kind)
                                    ? "ID3D12Device::CreateCommittedResource(acceleration-structure postbuild queries)"
                                    : "ID3D12Device::CreateQueryHeap"));
            }

            record->kind = kind;
            record->queueType = queueType;
            record->queryCount = queryCount;
            record->label = label;
            const auto nativeLabel = wideLabel(record->label);
            if (!nativeLabel.empty() && record->heap != nullptr)
            {
                record->heap->SetName(nativeLabel.c_str());
            }
            if (!nativeLabel.empty() && record->postbuildResults != nullptr)
            {
                record->postbuildResults->SetName(nativeLabel.c_str());
            }

            const std::scoped_lock lock(mutex);
            return queryPools.emplace(std::move(record));
        }

        [[nodiscard]] Result<std::shared_ptr<QueryPoolRecord>> queryPool(QueryPoolHandle handle)
        {
            const std::scoped_lock lock(mutex);
            const auto* stored = queryPools.find(handle);
            if (stored == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The query pool handle is invalid or stale.",
                });
            }
            return *stored;
        }
    };

    QueryManager::QueryManager() : impl_(std::make_unique<Impl>())
    {
    }

    QueryManager::~QueryManager() = default;

    Status QueryManager::initialize(ID3D12Device& device,
                                    QueueManager& queues,
                                    bool enhancedBarriers,
                                    bool meshPipelineStatisticsQueries)
    {
        impl_->device = &device;
        impl_->enhancedBarriers = enhancedBarriers;
        impl_->meshPipelineStatisticsQueries = meshPipelineStatisticsQueries;

        D3D12_FEATURE_DATA_D3D12_OPTIONS3 options{};
        const bool copySupported =
            SUCCEEDED(device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS3, &options, sizeof(options))) &&
            options.CopyQueueTimestampQueriesSupported != FALSE;
        constexpr std::array queueTypes{QueueType::graphics, QueueType::compute, QueueType::copy};
        for (const auto queueType : queueTypes)
        {
            auto& properties = impl_->queueProperties[queueIndex(queueType)];
            if (queueType == QueueType::copy && !copySupported)
            {
                continue;
            }

            auto queue = queues.nativeQueue(queueType);
            if (!queue)
            {
                continue;
            }
            std::uint64_t frequency = 0;
            if (SUCCEEDED((*queue)->GetTimestampFrequency(&frequency)) && frequency != 0)
            {
                properties.timestampQueries = true;
                properties.inlineTimestampWrites = true;
                properties.timestampValidBits = 64;
                properties.timestampFrequency = frequency;
            }
        }
        return {};
    }

    QueueProperties QueryManager::queueProperties(QueueType queueType) const noexcept
    {
        return validQueueType(queueType) ? impl_->queueProperties[queueIndex(queueType)] : QueueProperties{};
    }

    Result<QueueClockCalibration> QueryManager::calibrateQueueClock(const QueueManager& queues,
                                                                    QueueType queueType) const
    {
        if (!validQueueType(queueType))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Queue clock calibration requires a valid queue type.",
            });
        }
        if (!queueProperties(queueType).timestampQueries)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The selected queue does not expose a calibratable timestamp clock.",
            });
        }

        auto queue = queues.nativeQueue(queueType);
        if (!queue)
        {
            return std::unexpected(std::move(queue.error()));
        }

        std::uint64_t gpuFrequency = 0;
        HRESULT result = (*queue)->GetTimestampFrequency(&gpuFrequency);
        if (FAILED(result) || gpuFrequency == 0)
        {
            const auto code = isD3D12DeviceLossHresult(static_cast<std::int32_t>(result)) ? ErrorCode::deviceLost
                                                                                          : ErrorCode::nativeFailure;
            return std::unexpected(nativeError(code, result, "ID3D12CommandQueue::GetTimestampFrequency"));
        }

        LARGE_INTEGER cpuFrequency{};
        if (QueryPerformanceFrequency(&cpuFrequency) == FALSE || cpuFrequency.QuadPart <= 0)
        {
            return std::unexpected(Error{
                ErrorCode::nativeFailure,
                static_cast<std::int64_t>(GetLastError()),
                "QueryPerformanceFrequency failed during queue clock calibration.",
            });
        }

        std::uint64_t gpuTimestamp = 0;
        std::uint64_t cpuTimestamp = 0;
        LARGE_INTEGER before{};
        LARGE_INTEGER after{};
        static_cast<void>(QueryPerformanceCounter(&before));
        result = (*queue)->GetClockCalibration(&gpuTimestamp, &cpuTimestamp);
        static_cast<void>(QueryPerformanceCounter(&after));
        if (FAILED(result))
        {
            const auto code = isD3D12DeviceLossHresult(static_cast<std::int32_t>(result)) ? ErrorCode::deviceLost
                                                                                          : ErrorCode::nativeFailure;
            return std::unexpected(nativeError(code, result, "ID3D12CommandQueue::GetClockCalibration"));
        }

        const auto hostFrequency = static_cast<std::uint64_t>(cpuFrequency.QuadPart);
        const auto bracketTicks =
            (after.QuadPart > before.QuadPart ? static_cast<std::uint64_t>(after.QuadPart - before.QuadPart) : 0) + 1;
        return QueueClockCalibration{
            .gpuTimestamp = gpuTimestamp,
            .cpuTimestamp = cpuTimestamp,
            .gpuFrequency = gpuFrequency,
            .cpuFrequency = hostFrequency,
            .maxDeviationNanoseconds = ticksToNanosecondsRoundedUp(bracketTicks, hostFrequency),
            .cpuClock = CpuClock::queryPerformanceCounter,
        };
    }

    Status QueryManager::beginConditionalExecution(QueueManager& queues,
                                                   ResourceManager& resources,
                                                   detail::BackendCommandBuffer& commandBuffer,
                                                   const ConditionalExecutionDesc& desc)
    {
        const auto operation = nativePredicateOperation(desc.operation);
        if (commandBuffer.queueType() == QueueType::copy || !operation)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Conditional execution requires a graphics or compute command buffer and a valid predicate operation.",
            });
        }
        auto& active = commandBuffer.queryState().activeConditionalExecution;
        if (active.has_value())
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Conditional execution scopes cannot be nested.",
            });
        }

        auto source = resources.bufferBindingSource(desc.buffer);
        if (!source)
        {
            return std::unexpected(std::move(source.error()));
        }
        if (!hasAny(source->desc.usage, BufferUsage::predicate) || desc.offset % predicateValueAlignment != 0 ||
            desc.offset > source->desc.size || predicateValueSize > source->desc.size - desc.offset)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Conditional execution requires an aligned in-bounds uint64 value in a predicate buffer.",
            });
        }

        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto retained = queues.retainResource(commandBuffer, source->owner, source->residency);
        if (!retained)
        {
            return retained;
        }

        (*commandList)->SetPredication(source->resource, desc.offset, *operation);
        active = detail::ActiveConditionalExecution{.owner = std::move(source->owner)};
        return {};
    }

    Status QueryManager::endConditionalExecution(QueueManager& queues, detail::BackendCommandBuffer& commandBuffer)
    {
        auto& active = commandBuffer.queryState().activeConditionalExecution;
        if (commandBuffer.queueType() == QueueType::copy || !active.has_value())
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Ending conditional execution requires an active scope on a graphics or compute command buffer.",
            });
        }
        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }

        (*commandList)->SetPredication(nullptr, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
        active.reset();
        return {};
    }

    Result<QueryPoolHandle> QueryManager::createTimestampQueryPool(const TimestampQueryPoolDesc& desc)
    {
        if (!validQueueType(desc.queueType) || desc.queryCount == 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A timestamp query pool requires a valid queue type and a non-zero query count.",
            });
        }
        if (!queueProperties(desc.queueType).timestampQueries)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The selected queue does not support timestamp queries.",
            });
        }

        return impl_->createPool(QueryPoolKind::timestamp, desc.queueType, desc.queryCount, desc.label);
    }

    Result<QueryPoolHandle> QueryManager::createOcclusionQueryPool(const OcclusionQueryPoolDesc& desc)
    {
        if ((desc.mode != OcclusionQueryMode::precise && desc.mode != OcclusionQueryMode::binary) ||
            desc.queryCount == 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "An occlusion query pool requires a valid mode and a non-zero query count.",
            });
        }
        const auto kind =
            desc.mode == OcclusionQueryMode::precise ? QueryPoolKind::preciseOcclusion : QueryPoolKind::binaryOcclusion;
        return impl_->createPool(kind, QueueType::graphics, desc.queryCount, desc.label);
    }

    Result<QueryPoolHandle> QueryManager::createPipelineStatisticsQueryPool(const PipelineStatisticsQueryPoolDesc& desc)
    {
        if ((desc.mode != PipelineStatisticsQueryMode::base && desc.mode != PipelineStatisticsQueryMode::meshShader) ||
            desc.queryCount == 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A pipeline-statistics query pool requires a valid mode and a non-zero query count.",
            });
        }
        if (desc.mode == PipelineStatisticsQueryMode::meshShader && !impl_->meshPipelineStatisticsQueries)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Mesh pipeline statistics are unavailable on this adapter.",
            });
        }
        const auto kind = desc.mode == PipelineStatisticsQueryMode::meshShader ? QueryPoolKind::meshPipelineStatistics
                                                                               : QueryPoolKind::pipelineStatistics;
        return impl_->createPool(kind, QueueType::graphics, desc.queryCount, desc.label);
    }

    Result<QueryPoolHandle> QueryManager::createStreamOutputStatisticsQueryPool(
        const StreamOutputStatisticsQueryPoolDesc& desc)
    {
        if (desc.stream >= maxStreamOutputStreams || desc.queryCount == 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A stream-output-statistics query pool requires a stream in [0, 3] and a non-zero query count.",
            });
        }
        const auto kind = static_cast<QueryPoolKind>(
            static_cast<std::uint32_t>(QueryPoolKind::streamOutputStatistics0) + desc.stream);
        return impl_->createPool(kind, QueueType::graphics, desc.queryCount, desc.label);
    }

    Status QueryManager::destroyQueryPool(QueryPoolHandle queryPool)
    {
        const std::scoped_lock lock(impl_->mutex);
        if (!impl_->queryPools.erase(queryPool))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The query pool handle is invalid or stale.",
            });
        }
        return {};
    }

    Status QueryManager::resetQueryPool(QueueManager& queues,
                                        detail::BackendCommandBuffer& commandBuffer,
                                        QueryPoolHandle queryPool,
                                        std::uint32_t firstQuery,
                                        std::uint32_t queryCount)
    {
        auto pool = impl_->queryPool(queryPool);
        if (!pool)
        {
            return std::unexpected(std::move(pool.error()));
        }
        if ((*pool)->queueType != commandBuffer.queueType() || queryCount == 0 || firstQuery >= (*pool)->queryCount ||
            queryCount > (*pool)->queryCount - firstQuery)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A query-pool reset requires a matching queue and a non-empty in-bounds query range.",
            });
        }

        auto& state = commandBuffer.queryState();
        if (commandBuffer.graphicsState().renderingOwner != nullptr || state.activeOcclusion.has_value() ||
            state.activePipelineStatistics.has_value() || hasActiveStreamOutputStatistics(state) ||
            state.activePassTimestampWrites.has_value() || state.activeGpuProfileScopeDepth != 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A query-pool reset cannot be recorded while rendering or a query or GPU-profile scope is open.",
            });
        }

        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        const bool postbuild = isAccelerationStructurePostbuild((*pool)->kind);
        ID3D12GraphicsCommandList7* enhancedCommandList = nullptr;
        if (postbuild && impl_->enhancedBarriers)
        {
            auto enhanced = queues.commandList7(commandBuffer);
            if (!enhanced)
            {
                return std::unexpected(std::move(enhanced.error()));
            }
            enhancedCommandList = *enhanced;
        }
        auto retained = queues.retainResource(commandBuffer, *pool);
        if (!retained)
        {
            return retained;
        }

        if (postbuild)
        {
            serializePostbuildResultWrites(**commandList,
                                           enhancedCommandList,
                                           *(*pool)->postbuildResults.Get(),
                                           static_cast<std::uint64_t>((*pool)->queryCount) *
                                               postbuildResultSize((*pool)->kind));
        }
        resetPendingRange(state.timestampWrites, queryPool, firstQuery, queryCount);
        resetPendingRange(state.occlusionWrites, queryPool, firstQuery, queryCount);
        resetPendingRange(state.pipelineStatisticsWrites, queryPool, firstQuery, queryCount);
        resetPendingRange(state.streamOutputStatisticsWrites, queryPool, firstQuery, queryCount);
        resetPendingRange(state.accelerationStructureCompactedSizeWrites, queryPool, firstQuery, queryCount);
        resetPendingRange(state.accelerationStructureSerializationInfoWrites, queryPool, firstQuery, queryCount);
        return {};
    }

    Status QueryManager::writeTimestamp(QueueManager& queues,
                                        detail::BackendCommandBuffer& commandBuffer,
                                        QueryPoolHandle queryPool,
                                        std::uint32_t queryIndex)
    {
        auto pool = impl_->queryPool(queryPool);
        if (!pool)
        {
            return std::unexpected(std::move(pool.error()));
        }
        if ((*pool)->kind != QueryPoolKind::timestamp || (*pool)->queueType != commandBuffer.queueType() ||
            queryIndex >= (*pool)->queryCount)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A timestamp write requires a matching queue and an in-bounds query index.",
            });
        }
        if (!queueProperties(commandBuffer.queueType()).inlineTimestampWrites)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The command buffer queue does not support inline timestamp writes.",
            });
        }

        auto& state = commandBuffer.queryState();
        if (isPending(state.timestampWrites, queryPool, queryIndex))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A timestamp query cannot be written twice before its first value is resolved.",
            });
        }

        const std::array queryIndices{queryIndex};
        auto reserved = reservePending(state.timestampWrites, queryPool, queryIndices);
        if (!reserved)
        {
            return reserved;
        }

        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto retained = queues.retainResource(commandBuffer, *pool);
        if (!retained)
        {
            return retained;
        }

        markPendingReserved(state.timestampWrites, queryPool, queryIndex);
        (*commandList)->EndQuery((*pool)->heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, queryIndex);
        return {};
    }

    Status QueryManager::beginGpuProfileScope(QueueManager& queues,
                                              const DebugMarkerManager& debugMarkers,
                                              detail::BackendCommandBuffer& commandBuffer,
                                              const GpuProfileScopeDesc& desc)
    {
        auto pool = impl_->queryPool(desc.queryPool);
        if (!pool)
        {
            return std::unexpected(std::move(pool.error()));
        }
        if ((*pool)->kind != QueryPoolKind::timestamp || (*pool)->queueType != commandBuffer.queueType() ||
            desc.beginningQueryIndex >= (*pool)->queryCount || desc.endQueryIndex >= (*pool)->queryCount ||
            desc.beginningQueryIndex == desc.endQueryIndex)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A GPU profile scope requires a matching timestamp pool and two distinct in-bounds query indices.",
            });
        }
        if (!queueProperties(commandBuffer.queueType()).inlineTimestampWrites)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The command-buffer queue does not support inline GPU profile timestamps.",
            });
        }

        auto& state = commandBuffer.queryState();
        if (state.activeGpuProfileScopeDepth == maxDebugGroupDepth ||
            isPending(state.timestampWrites, desc.queryPool, desc.beginningQueryIndex) ||
            isPending(state.timestampWrites, desc.queryPool, desc.endQueryIndex))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "GPU profile scope nesting is exhausted or one of its timestamp indices is unresolved.",
            });
        }

        const DebugMarkerDesc marker{
            .label = desc.label,
            .color = desc.color,
        };
        auto validMarker = debugMarkers.validateProfileBegin(commandBuffer, marker);
        if (!validMarker)
        {
            return validMarker;
        }

        const std::array queryIndices{desc.beginningQueryIndex, desc.endQueryIndex};
        auto reserved = reservePending(state.timestampWrites, desc.queryPool, queryIndices);
        if (!reserved)
        {
            return reserved;
        }
        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto retained = queues.retainResource(commandBuffer, *pool);
        if (!retained)
        {
            return retained;
        }

        markPendingReserved(state.timestampWrites, desc.queryPool, desc.beginningQueryIndex);
        markPendingReserved(state.timestampWrites, desc.queryPool, desc.endQueryIndex);
        (*commandList)->EndQuery((*pool)->heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, desc.beginningQueryIndex);
        debugMarkers.commitProfileBegin(**commandList, commandBuffer, marker);
        state.activeGpuProfileScopes[state.activeGpuProfileScopeDepth] = detail::ActiveGpuProfileScope{
            .queryPool = desc.queryPool,
            .endQueryIndex = desc.endQueryIndex,
            .owner = *pool,
        };
        ++state.activeGpuProfileScopeDepth;
        return {};
    }

    Status QueryManager::endGpuProfileScope(QueueManager& queues,
                                            const DebugMarkerManager& debugMarkers,
                                            detail::BackendCommandBuffer& commandBuffer)
    {
        auto& state = commandBuffer.queryState();
        if (state.activeGpuProfileScopeDepth == 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Cannot end a GPU profile scope when no profile scope is open.",
            });
        }
        auto validMarker = debugMarkers.validateProfileEnd(commandBuffer);
        if (!validMarker)
        {
            return validMarker;
        }
        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }

        auto& active = state.activeGpuProfileScopes[state.activeGpuProfileScopeDepth - 1];
        const auto pool = std::static_pointer_cast<QueryPoolRecord>(active.owner);
        (*commandList)->EndQuery(pool->heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, active.endQueryIndex);
        debugMarkers.commitProfileEnd(**commandList, commandBuffer);
        active = {};
        --state.activeGpuProfileScopeDepth;
        return {};
    }

    Status QueryManager::resolveTimestamps(QueueManager& queues,
                                           ResourceManager& resources,
                                           detail::BackendCommandBuffer& commandBuffer,
                                           const TimestampQueryResolveDesc& desc)
    {
        auto pool = impl_->queryPool(desc.queryPool);
        if (!pool)
        {
            return std::unexpected(std::move(pool.error()));
        }
        if ((*pool)->kind != QueryPoolKind::timestamp || (*pool)->queueType != commandBuffer.queueType() ||
            desc.queryCount == 0 || desc.firstQuery >= (*pool)->queryCount ||
            desc.queryCount > (*pool)->queryCount - desc.firstQuery)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A timestamp resolve requires a matching queue and a non-empty in-bounds query range.",
            });
        }
        if (commandBuffer.graphicsState().renderingOwner != nullptr ||
            commandBuffer.queryState().activePassTimestampWrites.has_value() ||
            commandBuffer.queryState().activeGpuProfileScopeDepth != 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Timestamp results can be resolved only outside rendering and closed GPU profile scopes.",
            });
        }
        if (desc.destinationOffset % timestampQueryResultAlignment != 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A timestamp resolve destination offset must be aligned to eight bytes.",
            });
        }

        auto destination = resources.bufferBindingSource(desc.destination);
        if (!destination)
        {
            return std::unexpected(std::move(destination.error()));
        }
        if (!hasAny(destination->desc.usage, BufferUsage::copyDestination) ||
            desc.destinationOffset > destination->desc.size ||
            desc.queryCount > (destination->desc.size - desc.destinationOffset) / timestampQueryResultSize)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Timestamp results require an in-bounds copy-destination buffer range.",
            });
        }

        auto& state = commandBuffer.queryState();
        for (std::uint32_t offset = 0; offset < desc.queryCount; ++offset)
        {
            if (!isPending(state.timestampWrites, desc.queryPool, desc.firstQuery + offset))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Every timestamp query must be written in this command buffer before it is resolved.",
                });
            }
        }

        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto retained =
            retainQueryCommandDependencies(queues, commandBuffer, *pool, destination->owner, destination->residency);
        if (!retained)
        {
            return retained;
        }

        (*commandList)
            ->ResolveQueryData((*pool)->heap.Get(),
                               D3D12_QUERY_TYPE_TIMESTAMP,
                               desc.firstQuery,
                               desc.queryCount,
                               destination->resource,
                               desc.destinationOffset);
        for (auto& write : state.timestampWrites)
        {
            if (write.queryPool == desc.queryPool && write.queryIndex >= desc.firstQuery &&
                write.queryIndex - desc.firstQuery < desc.queryCount)
            {
                write.resolved = true;
            }
        }
        return {};
    }

    Result<std::optional<PreparedRenderingTimestamps>> QueryManager::prepareRenderingTimestamps(
        detail::BackendCommandBuffer& commandBuffer, const std::optional<PassTimestampWrites>& timestampWrites)
    {
        if (!timestampWrites.has_value())
        {
            return std::optional<PreparedRenderingTimestamps>{};
        }
        const auto& writes = *timestampWrites;
        auto pool = impl_->queryPool(writes.queryPool);
        if (!pool)
        {
            return std::unexpected(std::move(pool.error()));
        }
        const bool writesBeginning = writes.beginningOfPassWriteIndex != timestampWriteUnused;
        const bool writesEnd = writes.endOfPassWriteIndex != timestampWriteUnused;
        if ((*pool)->kind != QueryPoolKind::timestamp || (*pool)->queueType != QueueType::graphics ||
            commandBuffer.queueType() != QueueType::graphics || (!writesBeginning && !writesEnd) ||
            (writesBeginning && writes.beginningOfPassWriteIndex >= (*pool)->queryCount) ||
            (writesEnd && writes.endOfPassWriteIndex >= (*pool)->queryCount) ||
            (writesBeginning && writesEnd && writes.beginningOfPassWriteIndex == writes.endOfPassWriteIndex))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Render-pass timestamps require a graphics timestamp pool and one or two distinct in-bounds indices.",
            });
        }
        if (!queueProperties(QueueType::graphics).timestampQueries)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The graphics queue does not support render-pass timestamp writes.",
            });
        }

        auto& state = commandBuffer.queryState();
        if (state.activePassTimestampWrites.has_value() ||
            (writesBeginning && isPending(state.timestampWrites, writes.queryPool, writes.beginningOfPassWriteIndex)) ||
            (writesEnd && isPending(state.timestampWrites, writes.queryPool, writes.endOfPassWriteIndex)))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Render-pass timestamp indices cannot be active or unresolved when the rendering scope begins.",
            });
        }

        const std::array queryIndices{writes.beginningOfPassWriteIndex, writes.endOfPassWriteIndex};
        auto reserved = reservePending(state.timestampWrites, writes.queryPool, queryIndices);
        if (!reserved)
        {
            return std::unexpected(std::move(reserved.error()));
        }
        return PreparedRenderingTimestamps{
            .queryPool = writes.queryPool,
            .beginningOfPassWriteIndex = writes.beginningOfPassWriteIndex,
            .endOfPassWriteIndex = writes.endOfPassWriteIndex,
            .owner = *pool,
            .heap = (*pool)->heap.Get(),
        };
    }

    void QueryManager::commitRenderingTimestamps(ID3D12GraphicsCommandList& commandList,
                                                 detail::BackendCommandBuffer& commandBuffer,
                                                 PreparedRenderingTimestamps prepared) noexcept
    {
        auto& state = commandBuffer.queryState();
        if (prepared.beginningOfPassWriteIndex != timestampWriteUnused)
        {
            markPendingReserved(state.timestampWrites, prepared.queryPool, prepared.beginningOfPassWriteIndex);
            commandList.EndQuery(prepared.heap, D3D12_QUERY_TYPE_TIMESTAMP, prepared.beginningOfPassWriteIndex);
        }
        if (prepared.endOfPassWriteIndex != timestampWriteUnused)
        {
            markPendingReserved(state.timestampWrites, prepared.queryPool, prepared.endOfPassWriteIndex);
        }
        state.activePassTimestampWrites = detail::ActivePassTimestampWrites{
            .queryPool = prepared.queryPool,
            .endOfPassWriteIndex = prepared.endOfPassWriteIndex,
            .owner = std::move(prepared.owner),
        };
    }

    void QueryManager::endRenderingTimestamps(ID3D12GraphicsCommandList& commandList,
                                              detail::BackendCommandBuffer& commandBuffer) noexcept
    {
        auto& active = commandBuffer.queryState().activePassTimestampWrites;
        if (!active.has_value())
        {
            return;
        }
        if (active->endOfPassWriteIndex != timestampWriteUnused)
        {
            const auto pool = std::static_pointer_cast<QueryPoolRecord>(active->owner);
            commandList.EndQuery(pool->heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, active->endOfPassWriteIndex);
        }
        active.reset();
    }

    Status QueryManager::beginOcclusionQuery(QueueManager& queues,
                                             detail::BackendCommandBuffer& commandBuffer,
                                             QueryPoolHandle queryPool,
                                             std::uint32_t queryIndex)
    {
        auto pool = impl_->queryPool(queryPool);
        if (!pool)
        {
            return std::unexpected(std::move(pool.error()));
        }
        if (!isOcclusion((*pool)->kind) || queryIndex >= (*pool)->queryCount)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "An occlusion query begin requires an occlusion pool and an in-bounds query index.",
            });
        }
        if (commandBuffer.queueType() != QueueType::graphics || commandBuffer.graphicsState().renderingOwner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "An occlusion query can begin only inside a graphics rendering scope.",
            });
        }

        auto& state = commandBuffer.queryState();
        if (state.activeOcclusion.has_value())
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Only one occlusion query can be active on a command buffer.",
            });
        }
        if (isPending(state.occlusionWrites, queryPool, queryIndex))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "An occlusion query cannot be reused before its previous result is resolved.",
            });
        }

        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto retained = queues.retainResource(commandBuffer, *pool);
        if (!retained)
        {
            return retained;
        }

        (*commandList)->BeginQuery((*pool)->heap.Get(), nativeQueryType((*pool)->kind), queryIndex);
        state.activeOcclusion = detail::ActiveQuery{
            .queryPool = queryPool,
            .queryIndex = queryIndex,
            .owner = *pool,
            .begunInRendering = true,
        };
        return {};
    }

    Status QueryManager::endOcclusionQuery(QueueManager& queues, detail::BackendCommandBuffer& commandBuffer)
    {
        auto& state = commandBuffer.queryState();
        if (commandBuffer.queueType() != QueueType::graphics ||
            commandBuffer.graphicsState().renderingOwner == nullptr || !state.activeOcclusion.has_value())
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "An occlusion query can end only while one is active inside a graphics rendering scope.",
            });
        }

        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        const auto active = *state.activeOcclusion;
        const auto pool = std::static_pointer_cast<QueryPoolRecord>(active.owner);

        auto tracked = markPending(state.occlusionWrites, active.queryPool, active.queryIndex);
        if (!tracked)
        {
            return tracked;
        }

        (*commandList)->EndQuery(pool->heap.Get(), nativeQueryType(pool->kind), active.queryIndex);
        state.activeOcclusion.reset();
        return {};
    }

    Status QueryManager::resolveOcclusionQueries(QueueManager& queues,
                                                 ResourceManager& resources,
                                                 detail::BackendCommandBuffer& commandBuffer,
                                                 const OcclusionQueryResolveDesc& desc)
    {
        auto pool = impl_->queryPool(desc.queryPool);
        if (!pool)
        {
            return std::unexpected(std::move(pool.error()));
        }
        if (!isOcclusion((*pool)->kind) || commandBuffer.queueType() != QueueType::graphics || desc.queryCount == 0 ||
            desc.firstQuery >= (*pool)->queryCount || desc.queryCount > (*pool)->queryCount - desc.firstQuery)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "An occlusion resolve requires a graphics command buffer, an occlusion pool, and a non-empty in-bounds "
                "query range.",
            });
        }
        if (commandBuffer.graphicsState().renderingOwner != nullptr ||
            commandBuffer.queryState().activeOcclusion.has_value())
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Occlusion query results can be resolved only outside a rendering scope with no active query.",
            });
        }
        if (desc.destinationOffset % occlusionQueryResultAlignment != 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "An occlusion resolve destination offset must be aligned to eight bytes.",
            });
        }

        auto destination = resources.bufferBindingSource(desc.destination);
        if (!destination)
        {
            return std::unexpected(std::move(destination.error()));
        }
        if (!hasAny(destination->desc.usage, BufferUsage::copyDestination) ||
            desc.destinationOffset > destination->desc.size ||
            desc.queryCount > (destination->desc.size - desc.destinationOffset) / occlusionQueryResultSize)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Occlusion results require an in-bounds copy-destination buffer range.",
            });
        }

        auto& state = commandBuffer.queryState();
        for (std::uint32_t offset = 0; offset < desc.queryCount; ++offset)
        {
            if (!isPending(state.occlusionWrites, desc.queryPool, desc.firstQuery + offset))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Every occlusion query must end in this command buffer before it is resolved.",
                });
            }
        }

        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto retained =
            retainQueryCommandDependencies(queues, commandBuffer, *pool, destination->owner, destination->residency);
        if (!retained)
        {
            return retained;
        }

        (*commandList)
            ->ResolveQueryData((*pool)->heap.Get(),
                               nativeQueryType((*pool)->kind),
                               desc.firstQuery,
                               desc.queryCount,
                               destination->resource,
                               desc.destinationOffset);
        for (auto& write : state.occlusionWrites)
        {
            if (write.queryPool == desc.queryPool && write.queryIndex >= desc.firstQuery &&
                write.queryIndex - desc.firstQuery < desc.queryCount)
            {
                write.resolved = true;
            }
        }
        return {};
    }

    Status QueryManager::beginPipelineStatisticsQuery(QueueManager& queues,
                                                      detail::BackendCommandBuffer& commandBuffer,
                                                      QueryPoolHandle queryPool,
                                                      std::uint32_t queryIndex)
    {
        auto pool = impl_->queryPool(queryPool);
        if (!pool)
        {
            return std::unexpected(std::move(pool.error()));
        }
        if (!isPipelineStatistics((*pool)->kind) || commandBuffer.queueType() != QueueType::graphics ||
            queryIndex >= (*pool)->queryCount)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A pipeline-statistics query requires a graphics command buffer, a matching pool, and an in-bounds "
                "query index.",
            });
        }

        auto& state = commandBuffer.queryState();
        if (state.activePipelineStatistics.has_value())
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Only one pipeline-statistics query can be active on a command buffer.",
            });
        }
        if (isPending(state.pipelineStatisticsWrites, queryPool, queryIndex))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A pipeline-statistics query cannot be reused before its previous result is resolved.",
            });
        }

        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto retained = queues.retainResource(commandBuffer, *pool);
        if (!retained)
        {
            return retained;
        }

        (*commandList)->BeginQuery((*pool)->heap.Get(), nativeQueryType((*pool)->kind), queryIndex);
        state.activePipelineStatistics = detail::ActiveQuery{
            .queryPool = queryPool,
            .queryIndex = queryIndex,
            .owner = *pool,
            .begunInRendering = commandBuffer.graphicsState().renderingOwner != nullptr,
        };
        return {};
    }

    Status QueryManager::endPipelineStatisticsQuery(QueueManager& queues, detail::BackendCommandBuffer& commandBuffer)
    {
        auto& state = commandBuffer.queryState();
        if (commandBuffer.queueType() != QueueType::graphics || !state.activePipelineStatistics.has_value())
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A pipeline-statistics query can end only while one is active on a graphics command buffer.",
            });
        }
        if (auto sameScope = detail::validateQueryEndRenderingScope(
                commandBuffer,
                *state.activePipelineStatistics,
                "A pipeline-statistics query that began inside a rendering scope must end inside it, and one that "
                "began outside rendering must end outside rendering.");
            !sameScope)
        {
            return sameScope;
        }

        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        const auto active = *state.activePipelineStatistics;
        const auto pool = std::static_pointer_cast<QueryPoolRecord>(active.owner);
        auto tracked = markPending(state.pipelineStatisticsWrites, active.queryPool, active.queryIndex);
        if (!tracked)
        {
            return tracked;
        }

        (*commandList)->EndQuery(pool->heap.Get(), nativeQueryType(pool->kind), active.queryIndex);
        state.activePipelineStatistics.reset();
        return {};
    }

    Status QueryManager::resolvePipelineStatisticsQueries(QueueManager& queues,
                                                          ResourceManager& resources,
                                                          detail::BackendCommandBuffer& commandBuffer,
                                                          const PipelineStatisticsQueryResolveDesc& desc)
    {
        auto pool = impl_->queryPool(desc.queryPool);
        if (!pool)
        {
            return std::unexpected(std::move(pool.error()));
        }
        if (!isPipelineStatistics((*pool)->kind) || commandBuffer.queueType() != QueueType::graphics ||
            desc.queryCount == 0 || desc.firstQuery >= (*pool)->queryCount ||
            desc.queryCount > (*pool)->queryCount - desc.firstQuery)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A pipeline-statistics resolve requires a graphics command buffer, a matching pool, and a non-empty "
                "in-bounds query range.",
            });
        }
        if (commandBuffer.graphicsState().renderingOwner != nullptr ||
            commandBuffer.queryState().activePipelineStatistics.has_value())
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Pipeline-statistics results can be resolved only outside rendering with no active statistics query.",
            });
        }
        const auto resultSize = pipelineStatisticsResultSize((*pool)->kind);
        const auto resultAlignment = (*pool)->kind == QueryPoolKind::meshPipelineStatistics
                                         ? meshPipelineStatisticsQueryResultAlignment
                                         : pipelineStatisticsQueryResultAlignment;
        if (desc.destinationOffset % resultAlignment != 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A pipeline-statistics destination offset must be aligned to eight bytes.",
            });
        }

        auto destination = resources.bufferBindingSource(desc.destination);
        if (!destination)
        {
            return std::unexpected(std::move(destination.error()));
        }
        if (!hasAny(destination->desc.usage, BufferUsage::copyDestination) ||
            desc.destinationOffset > destination->desc.size ||
            desc.queryCount > (destination->desc.size - desc.destinationOffset) / resultSize)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Pipeline-statistics results require an in-bounds copy-destination buffer range.",
            });
        }

        auto& state = commandBuffer.queryState();
        for (std::uint32_t offset = 0; offset < desc.queryCount; ++offset)
        {
            if (!isPending(state.pipelineStatisticsWrites, desc.queryPool, desc.firstQuery + offset))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Every pipeline-statistics query must end in this command buffer before it is resolved.",
                });
            }
        }

        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto retained =
            retainQueryCommandDependencies(queues, commandBuffer, *pool, destination->owner, destination->residency);
        if (!retained)
        {
            return retained;
        }

        (*commandList)
            ->ResolveQueryData((*pool)->heap.Get(),
                               nativeQueryType((*pool)->kind),
                               desc.firstQuery,
                               desc.queryCount,
                               destination->resource,
                               desc.destinationOffset);
        for (auto& write : state.pipelineStatisticsWrites)
        {
            if (write.queryPool == desc.queryPool && write.queryIndex >= desc.firstQuery &&
                write.queryIndex - desc.firstQuery < desc.queryCount)
            {
                write.resolved = true;
            }
        }
        return {};
    }

    Status QueryManager::beginStreamOutputStatisticsQuery(QueueManager& queues,
                                                          detail::BackendCommandBuffer& commandBuffer,
                                                          QueryPoolHandle queryPool,
                                                          std::uint32_t queryIndex)
    {
        auto pool = impl_->queryPool(queryPool);
        if (!pool)
        {
            return std::unexpected(std::move(pool.error()));
        }
        if (!isStreamOutputStatistics((*pool)->kind) || commandBuffer.queueType() != QueueType::graphics ||
            queryIndex >= (*pool)->queryCount)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A stream-output-statistics query requires a graphics command buffer, matching pool, and in-bounds "
                "index.",
            });
        }
        const auto stream = streamOutputStatisticsStream((*pool)->kind);
        auto& state = commandBuffer.queryState();
        if (state.activeStreamOutputStatistics[stream].has_value() ||
            isPending(state.streamOutputStatisticsWrites, queryPool, queryIndex))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A stream-output-statistics stream may have one active query and cannot reuse an unresolved slot.",
            });
        }
        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto retained = queues.retainResource(commandBuffer, *pool);
        if (!retained)
        {
            return retained;
        }
        (*commandList)->BeginQuery((*pool)->heap.Get(), nativeQueryType((*pool)->kind), queryIndex);
        state.activeStreamOutputStatistics[stream] = detail::ActiveQuery{
            .queryPool = queryPool,
            .queryIndex = queryIndex,
            .owner = *pool,
            .begunInRendering = commandBuffer.graphicsState().renderingOwner != nullptr,
        };
        return {};
    }

    Status QueryManager::endStreamOutputStatisticsQuery(QueueManager& queues,
                                                        detail::BackendCommandBuffer& commandBuffer,
                                                        std::uint32_t stream)
    {
        auto& state = commandBuffer.queryState();
        if (commandBuffer.queueType() != QueueType::graphics || stream >= maxStreamOutputStreams ||
            !state.activeStreamOutputStatistics[stream].has_value())
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A stream-output-statistics query can end only while that stream has an active query.",
            });
        }
        if (auto sameScope = detail::validateQueryEndRenderingScope(
                commandBuffer,
                *state.activeStreamOutputStatistics[stream],
                "A stream-output-statistics query that began inside a rendering scope must end inside it, and one "
                "that began outside rendering must end outside rendering.");
            !sameScope)
        {
            return sameScope;
        }
        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        const auto active = *state.activeStreamOutputStatistics[stream];
        const auto pool = std::static_pointer_cast<QueryPoolRecord>(active.owner);
        auto tracked = markPending(state.streamOutputStatisticsWrites, active.queryPool, active.queryIndex);
        if (!tracked)
        {
            return tracked;
        }
        (*commandList)->EndQuery(pool->heap.Get(), nativeQueryType(pool->kind), active.queryIndex);
        state.activeStreamOutputStatistics[stream].reset();
        return {};
    }

    Status QueryManager::resolveStreamOutputStatisticsQueries(QueueManager& queues,
                                                              ResourceManager& resources,
                                                              detail::BackendCommandBuffer& commandBuffer,
                                                              const StreamOutputStatisticsQueryResolveDesc& desc)
    {
        auto pool = impl_->queryPool(desc.queryPool);
        if (!pool)
        {
            return std::unexpected(std::move(pool.error()));
        }
        if (!isStreamOutputStatistics((*pool)->kind) || commandBuffer.queueType() != QueueType::graphics ||
            desc.queryCount == 0 || desc.firstQuery >= (*pool)->queryCount ||
            desc.queryCount > (*pool)->queryCount - desc.firstQuery)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A stream-output-statistics resolve requires a matching pool and non-empty in-bounds range.",
            });
        }
        if (desc.destinationOffset % streamOutputStatisticsQueryResultAlignment != 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Stream-output-statistics results require an aligned destination offset.",
            });
        }
        if (commandBuffer.graphicsState().renderingOwner != nullptr ||
            hasActiveStreamOutputStatistics(commandBuffer.queryState()))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Stream-output-statistics results require no rendering scope or active stream query.",
            });
        }
        auto destination = resources.bufferBindingSource(desc.destination);
        if (!destination)
        {
            return std::unexpected(std::move(destination.error()));
        }
        if (!hasAny(destination->desc.usage, BufferUsage::copyDestination) ||
            desc.destinationOffset > destination->desc.size ||
            desc.queryCount > (destination->desc.size - desc.destinationOffset) / streamOutputStatisticsQueryResultSize)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Stream-output-statistics results require an in-bounds copy-destination buffer range.",
            });
        }
        auto& state = commandBuffer.queryState();
        for (std::uint32_t offset = 0; offset < desc.queryCount; ++offset)
        {
            if (!isPending(state.streamOutputStatisticsWrites, desc.queryPool, desc.firstQuery + offset))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Every stream-output-statistics query must end before it is resolved.",
                });
            }
        }
        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto retained =
            retainQueryCommandDependencies(queues, commandBuffer, *pool, destination->owner, destination->residency);
        if (!retained)
        {
            return retained;
        }
        (*commandList)
            ->ResolveQueryData((*pool)->heap.Get(),
                               nativeQueryType((*pool)->kind),
                               desc.firstQuery,
                               desc.queryCount,
                               destination->resource,
                               desc.destinationOffset);
        for (auto& write : state.streamOutputStatisticsWrites)
        {
            if (write.queryPool == desc.queryPool && write.queryIndex >= desc.firstQuery &&
                write.queryIndex - desc.firstQuery < desc.queryCount)
            {
                write.resolved = true;
            }
        }
        return {};
    }

    Result<QueryPoolHandle> QueryManager::createAccelerationStructureCompactedSizeQueryPool(
        const AccelerationStructureCompactedSizeQueryPoolDesc& desc)
    {
        if ((desc.queueType != QueueType::graphics && desc.queueType != QueueType::compute) || desc.queryCount == 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A compacted-size query pool requires a graphics or compute queue and a non-zero query count.",
            });
        }
        return impl_->createPool(
            QueryPoolKind::accelerationStructureCompactedSize, desc.queueType, desc.queryCount, desc.label);
    }

    Status QueryManager::writeAccelerationStructureCompactedSize(QueueManager& queues,
                                                                 ResourceManager& resources,
                                                                 detail::BackendCommandBuffer& commandBuffer,
                                                                 QueryPoolHandle queryPool,
                                                                 std::uint32_t queryIndex,
                                                                 AccelerationStructureHandle sourceHandle)
    {
        auto pool = impl_->queryPool(queryPool);
        if (!pool)
        {
            return std::unexpected(std::move(pool.error()));
        }
        if ((*pool)->kind != QueryPoolKind::accelerationStructureCompactedSize ||
            (*pool)->queueType != commandBuffer.queueType() || queryIndex >= (*pool)->queryCount)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A compacted-size write requires a matching queue, pool kind, and in-bounds query index.",
            });
        }
        if (commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Acceleration-structure compacted-size queries must be written outside rendering.",
            });
        }
        auto& state = commandBuffer.queryState();
        if (isPending(state.accelerationStructureCompactedSizeWrites, queryPool, queryIndex))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A compacted-size query cannot be rewritten before its first value is resolved.",
            });
        }
        auto recordingId = queues.recordingId(commandBuffer);
        if (!recordingId)
        {
            return std::unexpected(std::move(recordingId.error()));
        }
        auto source = resources.accelerationStructureCompactionSource(sourceHandle, *recordingId);
        if (!source)
        {
            return std::unexpected(std::move(source.error()));
        }
        const std::array queryIndices{queryIndex};
        auto reserved = reservePending(state.accelerationStructureCompactedSizeWrites, queryPool, queryIndices);
        if (!reserved)
        {
            return reserved;
        }
        auto commandList = queues.commandList4(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto retained = retainQueryCommandDependencies(
            queues, commandBuffer, *pool, source->owner, source->residencies, std::move(source->lease));
        if (!retained)
        {
            return retained;
        }
        markPendingReserved(state.accelerationStructureCompactedSizeWrites, queryPool, queryIndex);

        const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC native{
            .DestBuffer = (*pool)->postbuildResults->GetGPUVirtualAddress() +
                          static_cast<std::uint64_t>(queryIndex) * accelerationStructureCompactedSizeQueryResultSize,
            .InfoType = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE,
        };
        const std::array sources{source->address};
        (*commandList)->EmitRaytracingAccelerationStructurePostbuildInfo(&native, 1, sources.data());
        return {};
    }

    Status QueryManager::resolveAccelerationStructureCompactedSizes(
        QueueManager& queues,
        ResourceManager& resources,
        detail::BackendCommandBuffer& commandBuffer,
        const AccelerationStructureCompactedSizeQueryResolveDesc& desc)
    {
        auto pool = impl_->queryPool(desc.queryPool);
        if (!pool)
        {
            return std::unexpected(std::move(pool.error()));
        }
        if ((*pool)->kind != QueryPoolKind::accelerationStructureCompactedSize ||
            (*pool)->queueType != commandBuffer.queueType() || desc.queryCount == 0 ||
            desc.firstQuery >= (*pool)->queryCount || desc.queryCount > (*pool)->queryCount - desc.firstQuery)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A compacted-size resolve requires a matching queue and a non-empty in-bounds query range.",
            });
        }
        if (commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Acceleration-structure compacted sizes can be resolved only outside rendering.",
            });
        }
        if (desc.destinationOffset % accelerationStructureCompactedSizeQueryResultAlignment != 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A compacted-size resolve destination offset must be aligned to eight bytes.",
            });
        }

        auto destination = resources.bufferBindingSource(desc.destination);
        if (!destination)
        {
            return std::unexpected(std::move(destination.error()));
        }
        if (!hasAny(destination->desc.usage, BufferUsage::copyDestination) ||
            desc.destinationOffset > destination->desc.size ||
            desc.queryCount >
                (destination->desc.size - desc.destinationOffset) / accelerationStructureCompactedSizeQueryResultSize)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Compacted-size results require an in-bounds copy-destination buffer range.",
            });
        }

        auto& state = commandBuffer.queryState();
        for (std::uint32_t offset = 0; offset < desc.queryCount; ++offset)
        {
            if (!isPending(state.accelerationStructureCompactedSizeWrites, desc.queryPool, desc.firstQuery + offset))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Every compacted-size query must be written in this command buffer before it is resolved.",
                });
            }
        }

        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        ID3D12GraphicsCommandList7* enhancedCommandList = nullptr;
        if (impl_->enhancedBarriers)
        {
            auto enhanced = queues.commandList7(commandBuffer);
            if (!enhanced)
            {
                return std::unexpected(std::move(enhanced.error()));
            }
            enhancedCommandList = *enhanced;
        }
        auto retained =
            retainQueryCommandDependencies(queues, commandBuffer, *pool, destination->owner, destination->residency);
        if (!retained)
        {
            return retained;
        }

        const auto poolSize =
            static_cast<std::uint64_t>((*pool)->queryCount) * accelerationStructureCompactedSizeQueryResultSize;
        transitionPostbuildResults(
            **commandList, enhancedCommandList, *(*pool)->postbuildResults.Get(), poolSize, true);
        (*commandList)
            ->CopyBufferRegion(
                destination->resource,
                desc.destinationOffset,
                (*pool)->postbuildResults.Get(),
                static_cast<std::uint64_t>(desc.firstQuery) * accelerationStructureCompactedSizeQueryResultSize,
                static_cast<std::uint64_t>(desc.queryCount) * accelerationStructureCompactedSizeQueryResultSize);
        transitionPostbuildResults(
            **commandList, enhancedCommandList, *(*pool)->postbuildResults.Get(), poolSize, false);
        for (auto& write : state.accelerationStructureCompactedSizeWrites)
        {
            if (write.queryPool == desc.queryPool && write.queryIndex >= desc.firstQuery &&
                write.queryIndex - desc.firstQuery < desc.queryCount)
            {
                write.resolved = true;
            }
        }
        return {};
    }

    Result<QueryPoolHandle> QueryManager::createAccelerationStructureSerializationInfoQueryPool(
        const AccelerationStructureSerializationInfoQueryPoolDesc& desc)
    {
        if ((desc.queueType != QueueType::graphics && desc.queueType != QueueType::compute) || desc.queryCount == 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A serialization-info query pool requires a graphics or compute queue and a non-zero query count.",
            });
        }
        return impl_->createPool(
            QueryPoolKind::accelerationStructureSerializationInfo, desc.queueType, desc.queryCount, desc.label);
    }

    Status QueryManager::writeAccelerationStructureSerializationInfo(QueueManager& queues,
                                                                     ResourceManager& resources,
                                                                     detail::BackendCommandBuffer& commandBuffer,
                                                                     QueryPoolHandle queryPool,
                                                                     std::uint32_t queryIndex,
                                                                     AccelerationStructureHandle sourceHandle)
    {
        auto pool = impl_->queryPool(queryPool);
        if (!pool)
        {
            return std::unexpected(std::move(pool.error()));
        }
        if ((*pool)->kind != QueryPoolKind::accelerationStructureSerializationInfo ||
            (*pool)->queueType != commandBuffer.queueType() || queryIndex >= (*pool)->queryCount)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A serialization-info write requires a matching queue, pool kind, and in-bounds query index.",
            });
        }
        if (commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Acceleration-structure serialization-info queries must be written outside rendering.",
            });
        }
        auto& state = commandBuffer.queryState();
        if (isPending(state.accelerationStructureSerializationInfoWrites, queryPool, queryIndex))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A serialization-info query cannot be rewritten before its first value is resolved.",
            });
        }
        auto recordingId = queues.recordingId(commandBuffer);
        if (!recordingId)
        {
            return std::unexpected(std::move(recordingId.error()));
        }
        auto source = resources.accelerationStructureSerializationSource(sourceHandle, *recordingId);
        if (!source)
        {
            return std::unexpected(std::move(source.error()));
        }
        const std::array queryIndices{queryIndex};
        auto reserved = reservePending(state.accelerationStructureSerializationInfoWrites, queryPool, queryIndices);
        if (!reserved)
        {
            return reserved;
        }
        auto commandList = queues.commandList4(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto retained = retainQueryCommandDependencies(
            queues, commandBuffer, *pool, source->owner, source->residencies, std::move(source->lease));
        if (!retained)
        {
            return retained;
        }
        markPendingReserved(state.accelerationStructureSerializationInfoWrites, queryPool, queryIndex);

        const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC native{
            .DestBuffer =
                (*pool)->postbuildResults->GetGPUVirtualAddress() +
                static_cast<std::uint64_t>(queryIndex) * accelerationStructureSerializationInfoQueryResultSize,
            .InfoType = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_SERIALIZATION,
        };
        const std::array sources{source->address};
        (*commandList)->EmitRaytracingAccelerationStructurePostbuildInfo(&native, 1, sources.data());
        return {};
    }

    Status QueryManager::resolveAccelerationStructureSerializationInfo(
        QueueManager& queues,
        ResourceManager& resources,
        detail::BackendCommandBuffer& commandBuffer,
        const AccelerationStructureSerializationInfoQueryResolveDesc& desc)
    {
        auto pool = impl_->queryPool(desc.queryPool);
        if (!pool)
        {
            return std::unexpected(std::move(pool.error()));
        }
        if ((*pool)->kind != QueryPoolKind::accelerationStructureSerializationInfo ||
            (*pool)->queueType != commandBuffer.queueType() || desc.queryCount == 0 ||
            desc.firstQuery >= (*pool)->queryCount || desc.queryCount > (*pool)->queryCount - desc.firstQuery)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A serialization-info resolve requires a matching queue and a non-empty in-bounds query range.",
            });
        }
        if (commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Acceleration-structure serialization information can be resolved only outside rendering.",
            });
        }
        if (desc.destinationOffset % accelerationStructureSerializationInfoQueryResultAlignment != 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A serialization-info resolve destination offset must be aligned to eight bytes.",
            });
        }

        auto destination = resources.bufferBindingSource(desc.destination);
        if (!destination)
        {
            return std::unexpected(std::move(destination.error()));
        }
        if (!hasAny(destination->desc.usage, BufferUsage::copyDestination) ||
            desc.destinationOffset > destination->desc.size ||
            desc.queryCount > (destination->desc.size - desc.destinationOffset) /
                                  accelerationStructureSerializationInfoQueryResultSize)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Serialization-info results require an in-bounds copy-destination buffer range.",
            });
        }

        auto& state = commandBuffer.queryState();
        for (std::uint32_t offset = 0; offset < desc.queryCount; ++offset)
        {
            if (!isPending(
                    state.accelerationStructureSerializationInfoWrites, desc.queryPool, desc.firstQuery + offset))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Every serialization-info query must be written in this command buffer before it is resolved.",
                });
            }
        }

        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        ID3D12GraphicsCommandList7* enhancedCommandList = nullptr;
        if (impl_->enhancedBarriers)
        {
            auto enhanced = queues.commandList7(commandBuffer);
            if (!enhanced)
            {
                return std::unexpected(std::move(enhanced.error()));
            }
            enhancedCommandList = *enhanced;
        }
        auto retained =
            retainQueryCommandDependencies(queues, commandBuffer, *pool, destination->owner, destination->residency);
        if (!retained)
        {
            return retained;
        }

        const auto poolSize =
            static_cast<std::uint64_t>((*pool)->queryCount) * accelerationStructureSerializationInfoQueryResultSize;
        transitionPostbuildResults(
            **commandList, enhancedCommandList, *(*pool)->postbuildResults.Get(), poolSize, true);
        (*commandList)
            ->CopyBufferRegion(
                destination->resource,
                desc.destinationOffset,
                (*pool)->postbuildResults.Get(),
                static_cast<std::uint64_t>(desc.firstQuery) * accelerationStructureSerializationInfoQueryResultSize,
                static_cast<std::uint64_t>(desc.queryCount) * accelerationStructureSerializationInfoQueryResultSize);
        transitionPostbuildResults(
            **commandList, enhancedCommandList, *(*pool)->postbuildResults.Get(), poolSize, false);
        for (auto& write : state.accelerationStructureSerializationInfoWrites)
        {
            if (write.queryPool == desc.queryPool && write.queryIndex >= desc.firstQuery &&
                write.queryIndex - desc.firstQuery < desc.queryCount)
            {
                write.resolved = true;
            }
        }
        return {};
    }

} // namespace rachis::d3d12
