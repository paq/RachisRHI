#include "Internal/VulkanQueries.h"

#include "Common/VectorReserve.h"
#include "Internal/PortableValidation.h"
#include "Internal/VulkanBufferBarriers.h"
#include "Internal/VulkanDiagnostics.h"
#include "Internal/VulkanGraphics.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <string>
#include <thread>
#include <utility>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <ctime>
#endif

namespace rachis::vulkan
{
    namespace
    {

        constexpr VkQueryPipelineStatisticFlags basePipelineStatistics =
            VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT |
            VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT |
            VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT |
            VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_INVOCATIONS_BIT |
            VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_PRIMITIVES_BIT |
            VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT | VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT |
            VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT |
            VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT |
            VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_EVALUATION_SHADER_INVOCATIONS_BIT |
            VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT;

        constexpr VkQueryPipelineStatisticFlags meshInvocationPipelineStatistics =
            basePipelineStatistics | VK_QUERY_PIPELINE_STATISTIC_TASK_SHADER_INVOCATIONS_BIT_EXT |
            VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT;

        [[nodiscard]] constexpr bool isPipelineStatisticsKind(VulkanQueryManager::QueryPoolKind kind) noexcept
        {
            return kind == VulkanQueryManager::QueryPoolKind::pipelineStatistics ||
                   kind == VulkanQueryManager::QueryPoolKind::meshPipelineStatistics;
        }

        [[nodiscard]] constexpr bool validQueueType(QueueType queueType) noexcept
        {
            return queueType == QueueType::graphics || queueType == QueueType::compute || queueType == QueueType::copy;
        }

        [[nodiscard]] constexpr bool isOcclusion(VulkanQueryManager::QueryPoolKind kind) noexcept
        {
            return kind == VulkanQueryManager::QueryPoolKind::preciseOcclusion ||
                   kind == VulkanQueryManager::QueryPoolKind::binaryOcclusion;
        }

        struct HostClockSample final
        {
            std::uint64_t timestamp = 0;
            std::uint64_t frequency = 0;
        };

        [[nodiscard]] Result<HostClockSample> sampleHostClock()
        {
#ifdef _WIN32
            LARGE_INTEGER frequency{};
            LARGE_INTEGER counter{};
            if (QueryPerformanceFrequency(&frequency) == 0 || frequency.QuadPart <= 0 ||
                QueryPerformanceCounter(&counter) == 0)
            {
                return std::unexpected(Error{
                    ErrorCode::nativeFailure,
                    0,
                    "QueryPerformanceCounter failed while calibrating a Vulkan queue clock.",
                });
            }
            return HostClockSample{
                .timestamp = static_cast<std::uint64_t>(counter.QuadPart),
                .frequency = static_cast<std::uint64_t>(frequency.QuadPart),
            };
#else
#ifdef __APPLE__
            constexpr clockid_t hostClock = CLOCK_MONOTONIC_RAW;
            constexpr const char* failure =
                "clock_gettime(CLOCK_MONOTONIC_RAW) failed while calibrating a Vulkan queue clock.";
#else
            constexpr clockid_t hostClock = CLOCK_MONOTONIC;
            constexpr const char* failure =
                "clock_gettime(CLOCK_MONOTONIC) failed while calibrating a Vulkan queue clock.";
#endif
            timespec ts{};
            if (clock_gettime(hostClock, &ts) != 0 || ts.tv_sec < 0 || ts.tv_nsec < 0)
            {
                return std::unexpected(Error{
                    ErrorCode::nativeFailure,
                    errno,
                    failure,
                });
            }
            return HostClockSample{
                .timestamp =
                    static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ULL + static_cast<std::uint64_t>(ts.tv_nsec),
                .frequency = 1'000'000'000ULL,
            };
#endif
        }

#ifdef VK_EXT_calibrated_timestamps
        [[nodiscard]] constexpr VkTimeDomainEXT hostCalibratedTimeDomain() noexcept
        {
#ifdef _WIN32
            return VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT;
#elif defined(__APPLE__)
            return VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_EXT;
#else
            return VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT;
#endif
        }
#endif

        [[nodiscard]] std::uint32_t activeViewCount(const VulkanCommandBuffer& commandBuffer) noexcept
        {
            const auto* const scope =
                static_cast<const VulkanRenderingScope*>(commandBuffer.graphicsState().renderingOwner.get());
            if (scope == nullptr || scope->viewMask == 0)
            {
                return 1;
            }
            return static_cast<std::uint32_t>(std::popcount(scope->viewMask));
        }

        [[nodiscard]] Status rejectMultiviewQuery(const VulkanCommandBuffer& commandBuffer, const char* message)
        {
            if (activeViewCount(commandBuffer) > 1)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    message,
                });
            }
            return {};
        }

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
                detail::reserveAdditional(writes, additions);
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

        void markResolvedRange(std::vector<detail::QueryWrite>& writes,
                               QueryPoolHandle queryPool,
                               std::uint32_t firstQuery,
                               std::uint32_t queryCount) noexcept
        {
            for (auto& write : writes)
            {
                if (write.queryPool == queryPool && write.queryIndex >= firstQuery &&
                    write.queryIndex - firstQuery < queryCount)
                {
                    write.resolved = true;
                }
            }
        }

        [[nodiscard]] Status validateMarkerDesc(const DebugMarkerDesc& desc)
        {
            if (desc.label.empty())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A debug marker label cannot be empty.",
                });
            }
            if (desc.label.size() > maxDebugLabelBytes || desc.label.contains('\0'))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A GPU profile label must stay within the portable debug-label contract.",
                });
            }
            if (desc.color)
            {
                const std::array components{desc.color->red, desc.color->green, desc.color->blue, desc.color->alpha};
                if (!std::ranges::all_of(components, [](float component) {
                        return std::isfinite(component) && component >= 0.0F && component <= 1.0F;
                    }))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Debug marker color components must be finite values in the inclusive range [0, 1].",
                    });
                }
            }
            return {};
        }

        void emitProfileLabel(const VulkanInstanceTable* instanceTable,
                              VkCommandBuffer command,
                              const DebugMarkerDesc& desc) noexcept
        {
            if (command == VK_NULL_HANDLE || instanceTable == nullptr ||
                instanceTable->vkCmdBeginDebugUtilsLabelEXT == nullptr)
            {
                return;
            }
            std::string const name(desc.label);
            VkDebugUtilsLabelEXT label{
                .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
                .pNext = {},
                .pLabelName = name.c_str(),
                .color = {0.0F, 0.0F, 0.0F, 1.0F},
            };
            if (desc.color)
            {
                label.color[0] = desc.color->red;
                label.color[1] = desc.color->green;
                label.color[2] = desc.color->blue;
                label.color[3] = desc.color->alpha;
            }
            instanceTable->vkCmdBeginDebugUtilsLabelEXT(command, &label);
        }

    } // namespace

    VulkanQueryManager::~VulkanQueryManager()
    {
        shutdown();
    }

    void VulkanQueryManager::initialize(VkDevice device,
                                        const VolkDeviceTable* table,
                                        const DeviceCapabilities* capabilities,
                                        VulkanResourceAllocator* resources,
                                        bool calibratedTimestamps,
                                        const VulkanInstanceTable* instanceTable,
                                        std::array<bool, 3> queueTimestampSupport) noexcept
    {
        device_ = device;
        table_ = table;
        instanceTable_ = instanceTable;
        capabilities_ = capabilities;
        resources_ = resources;
        calibratedTimestamps_ = calibratedTimestamps;
        queueTimestampSupport_ = queueTimestampSupport;
    }

    void VulkanQueryManager::shutdown() noexcept
    {
        const std::scoped_lock lock(mutex_);
        pools_ = {};
        device_ = VK_NULL_HANDLE;
        table_ = nullptr;
        instanceTable_ = nullptr;
        capabilities_ = nullptr;
        resources_ = nullptr;
        calibratedTimestamps_ = false;
        queueTimestampSupport_ = {true, true, true};
    }

    Result<QueueClockCalibration> VulkanQueryManager::calibrateQueueClock(
        VkPhysicalDevice physical, QueueType queueType, const VulkanPhysicalDeviceRecord& record) const
    {
        const auto properties = publishVulkanQueueProperties(record, queueType);
        auto hostClock = sampleHostClock();
        if (!hostClock)
        {
            return std::unexpected(std::move(hostClock.error()));
        }
        QueueClockCalibration calibration{
            .gpuTimestamp = 0,
            .cpuTimestamp = hostClock->timestamp,
            .gpuFrequency = properties.timestampFrequency,
            .cpuFrequency = hostClock->frequency,
        };
#ifdef VK_EXT_calibrated_timestamps
        constexpr VkTimeDomainEXT hostDomain = hostCalibratedTimeDomain();
        auto* const getCalibrated = table_ != nullptr ? table_->vkGetCalibratedTimestampsEXT : nullptr;
        auto* const getCalibrateableDomains =
            instanceTable_ != nullptr ? instanceTable_->vkGetPhysicalDeviceCalibrateableTimeDomainsEXT : nullptr;
        if (calibratedTimestamps_ && getCalibrated != nullptr && properties.timestampQueries &&
            physical != VK_NULL_HANDLE && getCalibrateableDomains != nullptr)
        {
            std::uint32_t domainCount = 0;
            if (getCalibrateableDomains(physical, &domainCount, nullptr) == VK_SUCCESS && domainCount > 0)
            {
                std::vector<VkTimeDomainEXT> domains(domainCount);
                if (getCalibrateableDomains(physical, &domainCount, domains.data()) == VK_SUCCESS)
                {
                    domains.resize(domainCount);
                    const bool hasDevice = std::ranges::find(domains, VK_TIME_DOMAIN_DEVICE_EXT) != domains.end();
                    const bool hasHost = std::ranges::find(domains, hostDomain) != domains.end();
                    if (hasDevice && hasHost)
                    {
                        const std::array infos{
                            VkCalibratedTimestampInfoEXT{
                                .sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT,
                                .pNext = {},
                                .timeDomain = VK_TIME_DOMAIN_DEVICE_EXT,
                            },
                            VkCalibratedTimestampInfoEXT{
                                .sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT,
                                .pNext = {},
                                .timeDomain = hostDomain,
                            },
                        };
                        std::array<std::uint64_t, 2> timestamps{};
                        std::uint64_t maxDeviation = 0;
                        if (getCalibrated(device_, 2, infos.data(), timestamps.data(), &maxDeviation) == VK_SUCCESS)
                        {
                            calibration.gpuTimestamp = timestamps[0];
                            calibration.cpuTimestamp = timestamps[1];
                        }
                    }
                }
            }
        }
#else
        static_cast<void>(physical);
        static_cast<void>(calibratedTimestamps_);
#endif
        return calibration;
    }

    Result<QueryPoolHandle> VulkanQueryManager::createTimestampQueryPool(const TimestampQueryPoolDesc& desc)
    {
        if (!validQueueType(desc.queueType) || desc.queryCount == 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A timestamp query pool requires a valid queue type and a non-zero query count.",
            });
        }
        if (capabilities_ == nullptr)
        {
            return std::unexpected(
                Error{ErrorCode::invalidState, 0, "Timestamp query pools require a live Vulkan device."});
        }
        if (!queueTimestampSupport_[static_cast<std::size_t>(desc.queueType)])
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Timestamp queries are not available on the requested Vulkan queue.",
            });
        }
        return createPool(QueryPoolKind::timestamp, desc.queueType, desc.queryCount, desc.label);
    }

    Result<QueryPoolHandle> VulkanQueryManager::createOcclusionQueryPool(const OcclusionQueryPoolDesc& desc)
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
        if (capabilities_ == nullptr || !capabilities_->occlusionQueries)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Occlusion queries are not available on this device.",
            });
        }
        const auto kind =
            desc.mode == OcclusionQueryMode::precise ? QueryPoolKind::preciseOcclusion : QueryPoolKind::binaryOcclusion;
        return createPool(kind, QueueType::graphics, desc.queryCount, desc.label);
    }

    Result<QueryPoolHandle> VulkanQueryManager::createPipelineStatisticsQueryPool(
        const PipelineStatisticsQueryPoolDesc& desc)
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
        if (desc.mode == PipelineStatisticsQueryMode::meshShader)
        {
            if (capabilities_ == nullptr || !capabilities_->meshPipelineStatisticsQueries)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Mesh pipeline statistics are not available on this device.",
                });
            }
            return createPool(QueryPoolKind::meshPipelineStatistics, QueueType::graphics, desc.queryCount, desc.label);
        }
        if (capabilities_ == nullptr || !capabilities_->pipelineStatisticsQueries)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Pipeline-statistics queries are not available on this device.",
            });
        }
        return createPool(QueryPoolKind::pipelineStatistics, QueueType::graphics, desc.queryCount, desc.label);
    }

    Status VulkanQueryManager::destroyQueryPool(QueryPoolHandle queryPool)
    {
        const std::scoped_lock lock(mutex_);
        if (pools_.find(queryPool) == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The query pool handle is invalid or stale.",
            });
        }
        if (!pools_.erase(queryPool))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The query pool handle is invalid or stale.",
            });
        }
        return {};
    }

    Status VulkanQueryManager::resetQueryPool(VulkanCommandBuffer& commandBuffer,
                                              QueryPoolHandle queryPool,
                                              std::uint32_t firstQuery,
                                              std::uint32_t queryCount)
    {
        auto pool = this->queryPool(queryPool);
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
            state.activePipelineStatistics.has_value() || state.activePassTimestampWrites.has_value() ||
            state.activeGpuProfileScopeDepth != 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A query-pool reset cannot be recorded while rendering or a query or GPU-profile scope is open.",
            });
        }
        if (auto retained = retain(commandBuffer, *pool); !retained)
        {
            return retained;
        }
        if (auto reservedUse = commandBuffer.reserveQueryPoolUse((*pool)->submittedState, *pool); !reservedUse)
        {
            return reservedUse;
        }
        resetSlots(commandBuffer.native(), (*pool)->pool, firstQuery, queryCount);
        if ((*pool)->primitivesPool != VK_NULL_HANDLE)
        {
            resetSlots(commandBuffer.native(), (*pool)->primitivesPool, firstQuery, queryCount);
        }
        commandBuffer.recordQuerySlotReset((*pool)->submittedState, firstQuery, queryCount);
        resetPendingRange(state.timestampWrites, queryPool, firstQuery, queryCount);
        resetPendingRange(state.occlusionWrites, queryPool, firstQuery, queryCount);
        resetPendingRange(state.pipelineStatisticsWrites, queryPool, firstQuery, queryCount);
        return {};
    }

    Status VulkanQueryManager::writeTimestamp(VulkanCommandBuffer& commandBuffer,
                                              QueryPoolHandle queryPool,
                                              std::uint32_t queryIndex)
    {
        auto pool = this->queryPool(queryPool);
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
        if (auto singleView = rejectMultiviewQuery(
                commandBuffer,
                "Vulkan cannot write a timestamp inside a rendering scope that renders more than one view.");
            !singleView)
        {
            return singleView;
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
        if (auto reserved = reservePending(state.timestampWrites, queryPool, queryIndices); !reserved)
        {
            return reserved;
        }
        if (auto retained = retain(commandBuffer, *pool); !retained)
        {
            return retained;
        }
        if (auto reservedUse = commandBuffer.reserveQueryPoolUse((*pool)->submittedState, *pool); !reservedUse)
        {
            return reservedUse;
        }
        if (commandBuffer.graphicsState().renderingOwner == nullptr)
        {
            resetSlots(commandBuffer.native(), (*pool)->pool, queryIndex, 1);
            commandBuffer.recordQuerySlotWriteAfterLocalReset((*pool)->submittedState, queryIndex);
        }
        else
        {
            commandBuffer.recordQuerySlotWriteWithoutLocalReset((*pool)->submittedState, queryIndex);
        }
        markPendingReserved(state.timestampWrites, queryPool, queryIndex);
        writeTimestampSlot(commandBuffer.native(), (*pool)->pool, queryIndex, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
        return {};
    }

    Status VulkanQueryManager::beginGpuProfileScope(VulkanCommandBuffer& commandBuffer, const GpuProfileScopeDesc& desc)
    {
        auto pool = this->queryPool(desc.queryPool);
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
        const DebugMarkerDesc marker{
            .label = desc.label,
            .color = desc.color,
        };
        if (auto valid = validateMarkerDesc(marker); !valid)
        {
            return valid;
        }
        if (auto singleView = rejectMultiviewQuery(
                commandBuffer,
                "Vulkan cannot begin a GPU profile scope inside a rendering scope that renders more than one view.");
            !singleView)
        {
            return singleView;
        }
        auto& state = commandBuffer.queryState();
        auto& debug = commandBuffer.debugState();
        const auto currentThread = std::this_thread::get_id();
        if (state.activeGpuProfileScopeDepth == maxDebugGroupDepth || debug.groupDepth == maxDebugGroupDepth ||
            isPending(state.timestampWrites, desc.queryPool, desc.beginningQueryIndex) ||
            isPending(state.timestampWrites, desc.queryPool, desc.endQueryIndex))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "GPU profile scope nesting is exhausted or one of its timestamp indices is unresolved.",
            });
        }
        if (debug.groupDepth != 0 && debug.groupThread != currentThread)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Nested debug and GPU profile groups must be recorded on the thread that opened the outer group.",
            });
        }
        const std::array queryIndices{desc.beginningQueryIndex, desc.endQueryIndex};
        if (auto reserved = reservePending(state.timestampWrites, desc.queryPool, queryIndices); !reserved)
        {
            return reserved;
        }
        if (auto retained = retain(commandBuffer, *pool); !retained)
        {
            return retained;
        }
        if (auto reservedUse = commandBuffer.reserveQueryPoolUse((*pool)->submittedState, *pool); !reservedUse)
        {
            return reservedUse;
        }
        if (commandBuffer.graphicsState().renderingOwner == nullptr)
        {
            resetSlots(commandBuffer.native(), (*pool)->pool, desc.beginningQueryIndex, 1);
            resetSlots(commandBuffer.native(), (*pool)->pool, desc.endQueryIndex, 1);
            commandBuffer.recordQuerySlotWriteAfterLocalReset((*pool)->submittedState, desc.beginningQueryIndex);
            commandBuffer.recordQuerySlotWriteAfterLocalReset((*pool)->submittedState, desc.endQueryIndex);
        }
        else
        {
            commandBuffer.recordQuerySlotWriteWithoutLocalReset((*pool)->submittedState, desc.beginningQueryIndex);
            commandBuffer.recordQuerySlotWriteWithoutLocalReset((*pool)->submittedState, desc.endQueryIndex);
        }
        markPendingReserved(state.timestampWrites, desc.queryPool, desc.beginningQueryIndex);
        markPendingReserved(state.timestampWrites, desc.queryPool, desc.endQueryIndex);
        writeTimestampSlot(
            commandBuffer.native(), (*pool)->pool, desc.beginningQueryIndex, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT);
        emitProfileLabel(instanceTable_, commandBuffer.native(), marker);
        if (debug.groupDepth == 0)
        {
            debug.groupThread = currentThread;
        }
        debug.groupOwners[debug.groupDepth] = detail::DebugGroupOwner::gpuProfile;
        ++debug.groupDepth;
        state.activeGpuProfileScopes[state.activeGpuProfileScopeDepth] = detail::ActiveGpuProfileScope{
            .queryPool = desc.queryPool,
            .endQueryIndex = desc.endQueryIndex,
            .owner = *pool,
        };
        ++state.activeGpuProfileScopeDepth;
        return {};
    }

    Status VulkanQueryManager::endGpuProfileScope(VulkanCommandBuffer& commandBuffer)
    {
        auto& state = commandBuffer.queryState();
        auto& debug = commandBuffer.debugState();
        if (state.activeGpuProfileScopeDepth == 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Cannot end a GPU profile scope when no profile scope is open.",
            });
        }
        if (debug.groupDepth == 0 || debug.groupOwners[debug.groupDepth - 1] != detail::DebugGroupOwner::gpuProfile)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A GPU profile-scope end cannot close a user debug group.",
            });
        }
        if (debug.groupThread != std::this_thread::get_id())
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A debug or GPU profile group must end on the thread that opened it.",
            });
        }
        if (auto singleView = rejectMultiviewQuery(
                commandBuffer,
                "Vulkan cannot end a GPU profile scope inside a rendering scope that renders more than one view.");
            !singleView)
        {
            return singleView;
        }
        auto& active = state.activeGpuProfileScopes[state.activeGpuProfileScopeDepth - 1];
        const auto pool = std::static_pointer_cast<QueryPoolRecord>(active.owner);
        writeTimestampSlot(
            commandBuffer.native(), pool->pool, active.endQueryIndex, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);
        if (instanceTable_ != nullptr && instanceTable_->vkCmdEndDebugUtilsLabelEXT != nullptr)
        {
            instanceTable_->vkCmdEndDebugUtilsLabelEXT(commandBuffer.native());
        }
        --debug.groupDepth;
        active = {};
        --state.activeGpuProfileScopeDepth;
        return {};
    }

    Status VulkanQueryManager::resolveTimestamps(VulkanCommandBuffer& commandBuffer,
                                                 const TimestampQueryResolveDesc& desc)
    {
        auto pool = this->queryPool(desc.queryPool);
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
        if (auto reserved = commandBuffer.reserveQueryPoolUse((*pool)->submittedState, *pool); !reserved)
        {
            return reserved;
        }
        auto copied = copyResults(commandBuffer,
                                  *pool,
                                  desc.firstQuery,
                                  desc.queryCount,
                                  desc.destination,
                                  desc.destinationOffset,
                                  timestampQueryResultSize,
                                  "Timestamp results require an in-bounds copy-destination buffer range.");
        if (!copied)
        {
            return std::unexpected(std::move(copied.error()));
        }
        recordResultCopyBarrier(commandBuffer.native(), *copied);
        resetSlots(commandBuffer.native(), (*pool)->pool, desc.firstQuery, desc.queryCount);
        commandBuffer.recordQuerySlotReset((*pool)->submittedState, desc.firstQuery, desc.queryCount);
        markResolvedRange(state.timestampWrites, desc.queryPool, desc.firstQuery, desc.queryCount);
        return {};
    }

    Result<std::optional<PreparedRenderingTimestamps>> VulkanQueryManager::prepareRenderingTimestamps(
        VulkanCommandBuffer& commandBuffer, const std::optional<PassTimestampWrites>& timestampWrites)
    {
        if (!timestampWrites.has_value())
        {
            return std::optional<PreparedRenderingTimestamps>{};
        }
        const auto& writes = *timestampWrites;
        auto pool = this->queryPool(writes.queryPool);
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
        if (auto reserved = reservePending(state.timestampWrites, writes.queryPool, queryIndices); !reserved)
        {
            return std::unexpected(std::move(reserved.error()));
        }
        if (auto reservedPoolUse = commandBuffer.reserveQueryPoolUse((*pool)->submittedState, *pool); !reservedPoolUse)
        {
            return std::unexpected(std::move(reservedPoolUse.error()));
        }
        if (commandBuffer.owner() != nullptr &&
            !hasAny(commandBuffer.owner()->flags, CommandPoolFlag::unretainedResources))
        {
            try
            {
                detail::reserveAdditional(commandBuffer.retained(), 1U);
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Query command dependencies could not be retained.",
                });
            }
        }
        return PreparedRenderingTimestamps{
            .queryPool = writes.queryPool,
            .beginningOfPassWriteIndex = writes.beginningOfPassWriteIndex,
            .endOfPassWriteIndex = writes.endOfPassWriteIndex,
            .owner = *pool,
            .submittedState = (*pool)->submittedState,
            .pool = (*pool)->pool,
        };
    }

    void VulkanQueryManager::beginRenderingTimestamps(VulkanCommandBuffer& commandBuffer,
                                                      const PreparedRenderingTimestamps& prepared)
    {
        if (prepared.beginningOfPassWriteIndex != timestampWriteUnused)
        {
            resetSlots(commandBuffer.native(), prepared.pool, prepared.beginningOfPassWriteIndex, 1);
            commandBuffer.recordQuerySlotReset(prepared.submittedState, prepared.beginningOfPassWriteIndex, 1);
            writeTimestampSlot(commandBuffer.native(),
                               prepared.pool,
                               prepared.beginningOfPassWriteIndex,
                               VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT);
        }
        if (prepared.endOfPassWriteIndex != timestampWriteUnused)
        {
            resetSlots(commandBuffer.native(), prepared.pool, prepared.endOfPassWriteIndex, 1);
            commandBuffer.recordQuerySlotReset(prepared.submittedState, prepared.endOfPassWriteIndex, 1);
        }
    }

    void VulkanQueryManager::commitRenderingTimestamps(VulkanCommandBuffer& commandBuffer,
                                                       PreparedRenderingTimestamps prepared) noexcept
    {
        if (commandBuffer.owner() != nullptr &&
            !hasAny(commandBuffer.owner()->flags, CommandPoolFlag::unretainedResources))
        {
            commandBuffer.retained().push_back(prepared.owner);
        }
        auto& state = commandBuffer.queryState();
        if (prepared.beginningOfPassWriteIndex != timestampWriteUnused)
        {
            markPendingReserved(state.timestampWrites, prepared.queryPool, prepared.beginningOfPassWriteIndex);
            commandBuffer.recordQuerySlotWriteWithoutLocalReset(prepared.submittedState,
                                                                prepared.beginningOfPassWriteIndex);
        }
        if (prepared.endOfPassWriteIndex != timestampWriteUnused)
        {
            markPendingReserved(state.timestampWrites, prepared.queryPool, prepared.endOfPassWriteIndex);
            commandBuffer.recordQuerySlotWriteWithoutLocalReset(prepared.submittedState, prepared.endOfPassWriteIndex);
        }
        state.activePassTimestampWrites = detail::ActivePassTimestampWrites{
            .queryPool = prepared.queryPool,
            .endOfPassWriteIndex = prepared.endOfPassWriteIndex,
            .owner = std::move(prepared.owner),
        };
    }

    Status VulkanQueryManager::endRenderingTimestamps(VulkanCommandBuffer& commandBuffer)
    {
        auto& active = commandBuffer.queryState().activePassTimestampWrites;
        if (!active.has_value())
        {
            return {};
        }
        if (active->endOfPassWriteIndex != timestampWriteUnused)
        {
            const auto pool = std::static_pointer_cast<QueryPoolRecord>(active->owner);
            writeTimestampSlot(commandBuffer.native(),
                               pool->pool,
                               active->endOfPassWriteIndex,
                               VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);
        }
        active.reset();
        return {};
    }

    Status VulkanQueryManager::beginOcclusionQuery(VulkanCommandBuffer& commandBuffer,
                                                   QueryPoolHandle queryPool,
                                                   std::uint32_t queryIndex)
    {
        auto pool = this->queryPool(queryPool);
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
        if (auto singleView = rejectMultiviewQuery(
                commandBuffer,
                "Vulkan cannot begin an occlusion query inside a rendering scope that renders more than one view.");
            !singleView)
        {
            return singleView;
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
        if (auto retained = retain(commandBuffer, *pool); !retained)
        {
            return retained;
        }
        if (auto reservedUse = commandBuffer.reserveQueryPoolUse((*pool)->submittedState, *pool); !reservedUse)
        {
            return reservedUse;
        }
        const auto flags = (*pool)->kind == QueryPoolKind::preciseOcclusion
                               ? static_cast<VkQueryControlFlags>(VK_QUERY_CONTROL_PRECISE_BIT)
                               : static_cast<VkQueryControlFlags>(0);
        commandBuffer.recordQuerySlotWriteWithoutLocalReset((*pool)->submittedState, queryIndex);
        table_->vkCmdBeginQuery(commandBuffer.native(), (*pool)->pool, queryIndex, flags);
        state.activeOcclusion = detail::ActiveQuery{
            .queryPool = queryPool,
            .queryIndex = queryIndex,
            .owner = *pool,
            .begunInRendering = true,
        };
        return {};
    }

    Status VulkanQueryManager::endOcclusionQuery(VulkanCommandBuffer& commandBuffer)
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
        const auto active = *state.activeOcclusion;
        const auto pool = std::static_pointer_cast<QueryPoolRecord>(active.owner);
        if (auto tracked = markPending(state.occlusionWrites, active.queryPool, active.queryIndex); !tracked)
        {
            return tracked;
        }
        table_->vkCmdEndQuery(commandBuffer.native(), pool->pool, active.queryIndex);
        state.activeOcclusion.reset();
        return {};
    }

    Status VulkanQueryManager::resolveOcclusionQueries(VulkanCommandBuffer& commandBuffer,
                                                       const OcclusionQueryResolveDesc& desc)
    {
        auto pool = this->queryPool(desc.queryPool);
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
        if (auto reservedUse = commandBuffer.reserveQueryPoolUse((*pool)->submittedState, *pool); !reservedUse)
        {
            return reservedUse;
        }
        auto copied = copyResults(commandBuffer,
                                  *pool,
                                  desc.firstQuery,
                                  desc.queryCount,
                                  desc.destination,
                                  desc.destinationOffset,
                                  occlusionQueryResultSize,
                                  "Occlusion results require an in-bounds copy-destination buffer range.");
        if (!copied)
        {
            return std::unexpected(std::move(copied.error()));
        }
        recordResultCopyBarrier(commandBuffer.native(), *copied);
        resetSlots(commandBuffer.native(), (*pool)->pool, desc.firstQuery, desc.queryCount);
        commandBuffer.recordQuerySlotReset((*pool)->submittedState, desc.firstQuery, desc.queryCount);
        markResolvedRange(state.occlusionWrites, desc.queryPool, desc.firstQuery, desc.queryCount);
        return {};
    }

    Status VulkanQueryManager::beginPipelineStatisticsQuery(VulkanCommandBuffer& commandBuffer,
                                                            QueryPoolHandle queryPool,
                                                            std::uint32_t queryIndex)
    {
        auto pool = this->queryPool(queryPool);
        if (!pool)
        {
            return std::unexpected(std::move(pool.error()));
        }
        if (!isPipelineStatisticsKind((*pool)->kind) || commandBuffer.queueType() != QueueType::graphics ||
            queryIndex >= (*pool)->queryCount)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A pipeline-statistics query requires a graphics command buffer, a matching pool, and an in-bounds "
                "query index.",
            });
        }
        if (auto singleView = rejectMultiviewQuery(commandBuffer,
                                                   "Vulkan cannot begin a pipeline-statistics query inside a rendering "
                                                   "scope that renders more than one view.");
            !singleView)
        {
            return singleView;
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
        if (auto retained = retain(commandBuffer, *pool); !retained)
        {
            return retained;
        }
        if (auto reservedUse = commandBuffer.reserveQueryPoolUse((*pool)->submittedState, *pool); !reservedUse)
        {
            return reservedUse;
        }
        const bool inRendering = commandBuffer.graphicsState().renderingOwner != nullptr;
        if (!inRendering)
        {
            resetSlots(commandBuffer.native(), (*pool)->pool, queryIndex, 1);
            if ((*pool)->primitivesPool != VK_NULL_HANDLE)
            {
                resetSlots(commandBuffer.native(), (*pool)->primitivesPool, queryIndex, 1);
            }
            commandBuffer.recordQuerySlotWriteAfterLocalReset((*pool)->submittedState, queryIndex);
        }
        else
        {
            commandBuffer.recordQuerySlotWriteWithoutLocalReset((*pool)->submittedState, queryIndex);
        }
        table_->vkCmdBeginQuery(commandBuffer.native(), (*pool)->pool, queryIndex, 0);
        if ((*pool)->primitivesPool != VK_NULL_HANDLE)
        {
            table_->vkCmdBeginQuery(commandBuffer.native(), (*pool)->primitivesPool, queryIndex, 0);
        }
        state.activePipelineStatistics = detail::ActiveQuery{
            .queryPool = queryPool,
            .queryIndex = queryIndex,
            .owner = *pool,
            .begunInRendering = inRendering,
        };
        return {};
    }

    Status VulkanQueryManager::endPipelineStatisticsQuery(VulkanCommandBuffer& commandBuffer)
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
        const auto active = *state.activePipelineStatistics;
        if (auto sameScope = detail::validateQueryEndRenderingScope(
                commandBuffer,
                active,
                "A pipeline-statistics query that began inside a rendering scope must end inside it, and one that "
                "began outside rendering must end outside rendering.");
            !sameScope)
        {
            return sameScope;
        }
        const auto pool = std::static_pointer_cast<QueryPoolRecord>(active.owner);
        if (auto tracked = markPending(state.pipelineStatisticsWrites, active.queryPool, active.queryIndex); !tracked)
        {
            return tracked;
        }
        if (pool->primitivesPool != VK_NULL_HANDLE)
        {
            table_->vkCmdEndQuery(commandBuffer.native(), pool->primitivesPool, active.queryIndex);
        }
        table_->vkCmdEndQuery(commandBuffer.native(), pool->pool, active.queryIndex);
        state.activePipelineStatistics.reset();
        return {};
    }

    Status VulkanQueryManager::resolvePipelineStatisticsQueries(VulkanCommandBuffer& commandBuffer,
                                                                const PipelineStatisticsQueryResolveDesc& desc)
    {
        auto pool = this->queryPool(desc.queryPool);
        if (!pool)
        {
            return std::unexpected(std::move(pool.error()));
        }
        if (!isPipelineStatisticsKind((*pool)->kind) || commandBuffer.queueType() != QueueType::graphics ||
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
        const bool meshMode = (*pool)->kind == QueryPoolKind::meshPipelineStatistics;
        const auto alignment =
            meshMode ? meshPipelineStatisticsQueryResultAlignment : pipelineStatisticsQueryResultAlignment;
        const auto stride = meshMode ? meshPipelineStatisticsQueryResultSize : pipelineStatisticsQueryResultSize;
        if (desc.destinationOffset % alignment != 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A pipeline-statistics resolve destination offset must be aligned to the portable result record.",
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
        if (auto reservedUse = commandBuffer.reserveQueryPoolUse((*pool)->submittedState, *pool); !reservedUse)
        {
            return reservedUse;
        }
        auto copied = copyResults(commandBuffer,
                                  *pool,
                                  desc.firstQuery,
                                  desc.queryCount,
                                  desc.destination,
                                  desc.destinationOffset,
                                  stride,
                                  "Pipeline-statistics results require an in-bounds copy-destination buffer range.");
        if (!copied)
        {
            return std::unexpected(std::move(copied.error()));
        }
        if (meshMode && (*pool)->primitivesPool != VK_NULL_HANDLE)
        {
            for (std::uint32_t offset = 0; offset < desc.queryCount; ++offset)
            {
                table_->vkCmdCopyQueryPoolResults(commandBuffer.native(),
                                                  (*pool)->primitivesPool,
                                                  desc.firstQuery + offset,
                                                  1,
                                                  copied->buffer,
                                                  desc.destinationOffset + offset * stride +
                                                      offsetof(MeshPipelineStatistics, meshShaderPrimitives),
                                                  sizeof(std::uint64_t),
                                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            }
        }
        recordResultCopyBarrier(commandBuffer.native(), *copied);
        resetSlots(commandBuffer.native(), (*pool)->pool, desc.firstQuery, desc.queryCount);
        if ((*pool)->primitivesPool != VK_NULL_HANDLE)
        {
            resetSlots(commandBuffer.native(), (*pool)->primitivesPool, desc.firstQuery, desc.queryCount);
        }
        commandBuffer.recordQuerySlotReset((*pool)->submittedState, desc.firstQuery, desc.queryCount);
        markResolvedRange(state.pipelineStatisticsWrites, desc.queryPool, desc.firstQuery, desc.queryCount);
        return {};
    }

    Result<QueryPoolHandle> VulkanQueryManager::createPool(QueryPoolKind kind,
                                                           QueueType queueType,
                                                           std::uint32_t queryCount,
                                                           std::string_view label)
    {
        if (table_ == nullptr || table_->vkCreateQueryPool == nullptr || table_->vkResetQueryPool == nullptr ||
            device_ == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Query pools require a live Vulkan device table with hostQueryReset.",
            });
        }
        VkQueryPoolCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        info.queryCount = queryCount;

        switch (kind)
        {
        case QueryPoolKind::timestamp:
            info.queryType = VK_QUERY_TYPE_TIMESTAMP;
            break;
        case QueryPoolKind::preciseOcclusion:
        case QueryPoolKind::binaryOcclusion:
            info.queryType = VK_QUERY_TYPE_OCCLUSION;
            break;
        case QueryPoolKind::pipelineStatistics:
            info.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
            info.pipelineStatistics = basePipelineStatistics;
            break;
        case QueryPoolKind::meshPipelineStatistics:
            info.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
            info.pipelineStatistics = meshInvocationPipelineStatistics;
            break;
        }
        auto record = std::make_shared<QueryPoolRecord>();
        record->device = device_;
        record->table = table_;
        record->kind = kind;
        record->queueType = queueType;
        record->queryCount = queryCount;
        record->label = std::string(label);
        record->submittedState = std::make_shared<VulkanSubmittedQueryPoolState>();
        record->submittedState->queryCount = queryCount;
        record->submittedState->dirtySlots.assign(queryCount, 0);
        const auto created = table_->vkCreateQueryPool(device_, &info, nullptr, &record->pool);
        if (created != VK_SUCCESS)
        {
            record->pool = VK_NULL_HANDLE;
            return std::unexpected(makeVulkanError(vulkanResultErrorCode(created), created, "vkCreateQueryPool"));
        }
        if (!record->label.empty())
        {
            setVulkanObjectName(instanceTable_,
                                device_,
                                VK_OBJECT_TYPE_QUERY_POOL,
                                reinterpret_cast<std::uint64_t>(record->pool),
                                record->label.c_str());
        }
        table_->vkResetQueryPool(device_, record->pool, 0, queryCount);
        if (kind == QueryPoolKind::meshPipelineStatistics)
        {
            VkQueryPoolCreateInfo const primitivesInfo{
                .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                .pNext = {},
                .flags = {},
                .queryType = VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT,
                .queryCount = queryCount,
                .pipelineStatistics = {},
            };
            const auto primitivesCreated =
                table_->vkCreateQueryPool(device_, &primitivesInfo, nullptr, &record->primitivesPool);
            if (primitivesCreated != VK_SUCCESS)
            {
                record->primitivesPool = VK_NULL_HANDLE;
                return std::unexpected(
                    makeVulkanError(vulkanResultErrorCode(primitivesCreated), primitivesCreated, "vkCreateQueryPool"));
            }
            table_->vkResetQueryPool(device_, record->primitivesPool, 0, queryCount);
        }
        record->submittedState->pool = record->pool;
        record->submittedState->primitivesPool = record->primitivesPool;
        const std::scoped_lock lock(mutex_);
        auto handle = pools_.emplace(std::move(record));
        if (!handle)
        {
            return std::unexpected(std::move(handle.error()));
        }
        return handle;
    }

    Result<std::shared_ptr<VulkanQueryManager::QueryPoolRecord>> VulkanQueryManager::queryPool(QueryPoolHandle handle)
    {
        const std::scoped_lock lock(mutex_);
        auto* found = pools_.find(handle);
        if (found == nullptr || *found == nullptr || (*found)->pool == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The query pool handle is invalid or stale.",
            });
        }
        return *found;
    }

    Status VulkanQueryManager::retain(VulkanCommandBuffer& commandBuffer, const std::shared_ptr<void>& owner)
    {
        if (commandBuffer.owner() == nullptr ||
            hasAny(commandBuffer.owner()->flags, CommandPoolFlag::unretainedResources))
        {
            return {};
        }
        try
        {
            commandBuffer.retained().push_back(owner);
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Query command dependencies could not be retained.",
            });
        }
        return {};
    }

    void VulkanQueryManager::writeTimestampSlot(VkCommandBuffer command,
                                                VkQueryPool pool,
                                                std::uint32_t queryIndex,
                                                VkPipelineStageFlags2 stage) const noexcept
    {
        if (table_ == nullptr || table_->vkCmdWriteTimestamp2 == nullptr || command == VK_NULL_HANDLE ||
            pool == VK_NULL_HANDLE)
        {
            return;
        }
        table_->vkCmdWriteTimestamp2(command, stage, pool, queryIndex);
    }

    void VulkanQueryManager::resetSlots(VkCommandBuffer command,
                                        VkQueryPool pool,
                                        std::uint32_t firstQuery,
                                        std::uint32_t queryCount) const noexcept
    {
        if (table_ == nullptr || table_->vkCmdResetQueryPool == nullptr || command == VK_NULL_HANDLE ||
            pool == VK_NULL_HANDLE || queryCount == 0)
        {
            return;
        }
        table_->vkCmdResetQueryPool(command, pool, firstQuery, queryCount);
    }

    Result<VulkanResourceAllocator::ResolvedBindingBuffer> VulkanQueryManager::copyResults(
        VulkanCommandBuffer& commandBuffer,
        const std::shared_ptr<QueryPoolRecord>& pool,
        std::uint32_t firstQuery,
        std::uint32_t queryCount,
        BufferHandle destination,
        std::uint64_t destinationOffset,
        std::uint64_t stride,
        std::string_view operation)
    {
        if (resources_ == nullptr || table_ == nullptr || table_->vkCmdCopyQueryPoolResults == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Query resolve requires a live Vulkan resource allocator.",
            });
        }
        auto resolved = resources_->resolveBindingBuffer(destination);
        if (!resolved)
        {
            return std::unexpected(std::move(resolved.error()));
        }
        if (!hasAny(resolved->desc.usage, BufferUsage::copyDestination) || destinationOffset > resolved->desc.size ||
            queryCount > (resolved->desc.size - destinationOffset) / stride)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                std::string(operation),
            });
        }
        if (auto retained = retain(commandBuffer, pool); !retained)
        {
            return std::unexpected(std::move(retained.error()));
        }
        if (auto retained = retain(commandBuffer, resolved->keepAlive); !retained)
        {
            return std::unexpected(std::move(retained.error()));
        }
        table_->vkCmdCopyQueryPoolResults(commandBuffer.native(),
                                          pool->pool,
                                          firstQuery,
                                          queryCount,
                                          resolved->buffer,
                                          destinationOffset,
                                          stride,
                                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        return resolved;
    }

    void VulkanQueryManager::recordResultCopyBarrier(
        VkCommandBuffer command, const VulkanResourceAllocator::ResolvedBindingBuffer& destination) const noexcept
    {
        if (table_ == nullptr || table_->vkCmdPipelineBarrier2 == nullptr || command == VK_NULL_HANDLE)
        {
            return;
        }
        const VkMemoryBarrier2 resetOrder{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .pNext = {},
            .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .dstAccessMask = VK_ACCESS_2_NONE,
        };
        const bool hostRead = destination.desc.memoryType == MemoryType::readback;
        const auto hostReadBarrier = copyWritesToHostReadBarrier(destination.buffer, destination.desc.size);
        const VkDependencyInfo dependency{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = {},
            .dependencyFlags = {},
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &resetOrder,
            .bufferMemoryBarrierCount = hostRead ? 1U : 0U,
            .pBufferMemoryBarriers = hostRead ? &hostReadBarrier : nullptr,
            .imageMemoryBarrierCount = {},
            .pImageMemoryBarriers = {},
        };
        table_->vkCmdPipelineBarrier2(command, &dependency);
    }

} // namespace rachis::vulkan
