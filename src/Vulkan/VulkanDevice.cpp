#include "Internal/VulkanDevice.h"

#include "Internal/VulkanDiagnostics.h"

#ifdef RACHIS_TESTING
#include "Vulkan/VulkanTestHooks.h"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <iterator>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <utility>
#include <vector>

namespace rachis::vulkan
{
    namespace
    {

#ifdef RACHIS_TESTING
        std::mutex boundDeviceMutex;
        VulkanDevice* boundDevice = nullptr;
        std::atomic<VkResult> injectedCreateResult{VK_SUCCESS};

        struct RealCreateEntryPoints final
        {
            const VulkanDevice* device = nullptr;
            PFN_vkCreatePipelineCache pipelineCache = nullptr;
            PFN_vkCreateComputePipelines computePipelines = nullptr;
            PFN_vkCreateGraphicsPipelines graphicsPipelines = nullptr;
            PFN_vkCreateRayTracingPipelinesKHR rayTracingPipelines = nullptr;
            PFN_vkCreateAccelerationStructureKHR accelerationStructure = nullptr;
        };

        RealCreateEntryPoints realCreateEntryPoints{};

        VKAPI_ATTR VkResult VKAPI_CALL injectedCreatePipelineCache(VkDevice,
                                                                   const VkPipelineCacheCreateInfo*,
                                                                   const VkAllocationCallbacks*,
                                                                   VkPipelineCache*)
        {
            return injectedCreateResult.load(std::memory_order_relaxed);
        }

        VKAPI_ATTR VkResult VKAPI_CALL injectedCreateComputePipelines(VkDevice,
                                                                      VkPipelineCache,
                                                                      std::uint32_t createInfoCount,
                                                                      const VkComputePipelineCreateInfo*,
                                                                      const VkAllocationCallbacks*,
                                                                      VkPipeline* pipelines)
        {
            std::fill_n(pipelines, createInfoCount, VK_NULL_HANDLE);
            return injectedCreateResult.load(std::memory_order_relaxed);
        }

        VKAPI_ATTR VkResult VKAPI_CALL injectedCreateGraphicsPipelines(VkDevice,
                                                                       VkPipelineCache,
                                                                       std::uint32_t createInfoCount,
                                                                       const VkGraphicsPipelineCreateInfo*,
                                                                       const VkAllocationCallbacks*,
                                                                       VkPipeline* pipelines)
        {
            std::fill_n(pipelines, createInfoCount, VK_NULL_HANDLE);
            return injectedCreateResult.load(std::memory_order_relaxed);
        }

        VKAPI_ATTR VkResult VKAPI_CALL injectedCreateRayTracingPipelines(VkDevice,
                                                                         VkDeferredOperationKHR,
                                                                         VkPipelineCache,
                                                                         std::uint32_t createInfoCount,
                                                                         const VkRayTracingPipelineCreateInfoKHR*,
                                                                         const VkAllocationCallbacks*,
                                                                         VkPipeline* pipelines)
        {
            std::fill_n(pipelines, createInfoCount, VK_NULL_HANDLE);
            return injectedCreateResult.load(std::memory_order_relaxed);
        }

        VKAPI_ATTR VkResult VKAPI_CALL injectedCreateAccelerationStructure(VkDevice,
                                                                           const VkAccelerationStructureCreateInfoKHR*,
                                                                           const VkAllocationCallbacks*,
                                                                           VkAccelerationStructureKHR*)
        {
            return injectedCreateResult.load(std::memory_order_relaxed);
        }
#endif

        [[nodiscard]] VulkanCommandBuffer* asOwnedVulkanCommand(detail::BackendCommandBuffer& commandBuffer,
                                                                std::uint64_t deviceToken) noexcept
        {
            if (commandBuffer.kind() != detail::BackendCommandBufferKind::vulkan || deviceToken == 0 ||
                commandBuffer.deviceToken() != deviceToken)
            {
                return nullptr;
            }
            return static_cast<VulkanCommandBuffer*>(&commandBuffer);
        }

        [[nodiscard]] std::vector<std::shared_ptr<void>>* retainSink(VulkanCommandBuffer& native) noexcept
        {
            if (native.owner() == nullptr || hasAny(native.owner()->flags, CommandPoolFlag::unretainedResources))
            {
                return nullptr;
            }
            return &native.retained();
        }

        struct LockedVulkanCommand final
        {
            VulkanCommandBuffer* command = nullptr;
            std::unique_lock<std::mutex> poolLock{};

            [[nodiscard]] VulkanCommandBuffer& operator*() const noexcept
            {
                return *command;
            }
            [[nodiscard]] VulkanCommandBuffer* operator->() const noexcept
            {
                return command;
            }
        };

        [[nodiscard]] Result<LockedVulkanCommand> lockLiveVulkanCommand(detail::BackendCommandBuffer& commandBuffer,
                                                                        std::uint64_t deviceToken,
                                                                        const char* message)
        {
            auto* native = asOwnedVulkanCommand(commandBuffer, deviceToken);
            if (native == nullptr || native->owner() == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    message,
                });
            }
            auto* const owner = native->owner();
            std::unique_lock<std::mutex> poolLock(owner->mutex);
            if (owner->pool == VK_NULL_HANDLE || native->native() == VK_NULL_HANDLE)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    message,
                });
            }
            return LockedVulkanCommand{
                .command = native,
                .poolLock = std::move(poolLock),
            };
        }

        [[nodiscard]] bool waitsOnAllCommandsBeforeBarriers(const VulkanPhysicalDeviceRecord& record) noexcept
        {
#ifdef RACHIS_TESTING
            if (test::forceAllCommandsBarrierSources().load(std::memory_order_relaxed))
            {
                return true;
            }
#endif
            return record.report.khrPortabilitySubset;
        }

        [[nodiscard]] bool isolatesDynamicBufferShaders(const VulkanPhysicalDeviceRecord& record) noexcept
        {
#ifdef RACHIS_TESTING
            if (test::forceIsolatedDynamicBufferShaders().load(std::memory_order_relaxed))
            {
                return true;
            }
#endif
            return record.report.khrPortabilitySubset;
        }

        [[nodiscard]] bool clipsScissorsToRenderArea(const VulkanPhysicalDeviceRecord& record) noexcept
        {
#ifdef RACHIS_TESTING
            if (test::forceScissorClipping().load(std::memory_order_relaxed))
            {
                return true;
            }
#endif
            return record.report.khrPortabilitySubset;
        }

        [[nodiscard]] std::unique_lock<std::mutex> lockCommandPoolIfPresent(
            VulkanCommandBuffer& native, std::shared_ptr<VulkanCommandPoolState>& owner)
        {
            owner = native.ownerShared();
            if (owner == nullptr)
            {
                return {};
            }
            return std::unique_lock<std::mutex>(owner->mutex);
        }

    } // namespace

    VulkanDevice::VulkanDevice(VulkanPhysicalDeviceRecord record,
                               VkDevice device,
                               VolkDeviceTable table,
                               VulkanQueueHandles queues,
                               DeviceCapabilities capabilities,
                               std::uint64_t deviceToken) noexcept
        : record_(std::move(record)), device_(device), table_(table), queues_(queues), capabilities_(capabilities),
          deviceToken_(deviceToken)
    {
        record_.report.deviceName = record_.deviceName;
        record_.report.driverId = record_.driverId;
        commands_.initialize(device_, &table_, queues_, deviceToken_, &resources_, &instance_.table);
#ifdef RACHIS_TESTING
        bindVulkanValidationTestDevice(device_, &table_);
        bindVulkanCommandAllocator(&commands_);
        bindVulkanDevice(this);
#endif
    }

    void VulkanDevice::adoptInstance(VulkanInstanceInfo instance) noexcept
    {
        instance_ = std::move(instance);
    }

    Status VulkanDevice::initializeResources()
    {
        auto timelines = submission_.initialize(
            device_, &table_, queues_, deviceToken_, capabilities_.sharedTimelineHandles, &instance_.table);
        if (!timelines)
        {
            return timelines;
        }
        const auto initialized = resources_.initialize(instance_.instance,
                                                       &instance_.table,
                                                       record_.physical,
                                                       device_,
                                                       &table_,
                                                       record_.memory,
                                                       queues_.families,
                                                       queues_.copy,
                                                       capabilities_.mappedMemory.nonCoherentAtomSize,
                                                       capabilities_.samplerMinMaxReduction,
                                                       capabilities_.textureViewComponentSwizzle,
                                                       capabilities_.sharedBufferHandles,
                                                       record_.memoryBudget,
                                                       record_.memoryPriority,
                                                       capabilities_.variableRateShadingTier2,
                                                       record_.bufferDeviceAddress || record_.accelerationStructures,
                                                       record_.accelerationStructures,
                                                       record_.rayTracingPipeline,
                                                       record_.rayTracingLimits.shaderGroupBaseAlignment,
                                                       record_.minAccelerationStructureScratchOffsetAlignment,
                                                       enabledShaderStages(capabilities_));
#ifdef RACHIS_TESTING
        if (initialized)
        {
            bindVulkanResourceAllocator(&resources_);
        }
#endif
        if (initialized)
        {
            resources_.waitOnAllCommandsBeforeBarriers(waitsOnAllCommandsBeforeBarriers(record_));
            const auto& nativeLimits = record_.properties.limits;
            const VulkanBufferBindingLimits bufferLimits{
                .minUniformBufferOffsetAlignment = nativeLimits.minUniformBufferOffsetAlignment != 0
                                                       ? nativeLimits.minUniformBufferOffsetAlignment
                                                       : 256,
                .minStorageBufferOffsetAlignment = nativeLimits.minStorageBufferOffsetAlignment != 0
                                                       ? nativeLimits.minStorageBufferOffsetAlignment
                                                       : 16,
                .maxUniformBufferRange =
                    nativeLimits.maxUniformBufferRange != 0 ? nativeLimits.maxUniformBufferRange : 65536,
                .maxStorageBufferRange =
                    nativeLimits.maxStorageBufferRange != 0 ? nativeLimits.maxStorageBufferRange : (1ULL << 30),
            };
            const VulkanDescriptorLimits descriptorLimits{
                .maxPerStageDescriptorSamplers = nativeLimits.maxPerStageDescriptorSamplers,
                .maxPerStageDescriptorUniformBuffers = nativeLimits.maxPerStageDescriptorUniformBuffers,
                .maxPerStageDescriptorStorageBuffers = nativeLimits.maxPerStageDescriptorStorageBuffers,
                .maxPerStageDescriptorSampledImages = nativeLimits.maxPerStageDescriptorSampledImages,
                .maxPerStageDescriptorStorageImages = nativeLimits.maxPerStageDescriptorStorageImages,
                .maxPerStageDescriptorAccelerationStructures = record_.maxPerStageDescriptorAccelerationStructures,
                .maxPerStageResources = nativeLimits.maxPerStageResources,
                .maxDescriptorSetSamplers = nativeLimits.maxDescriptorSetSamplers,
                .maxDescriptorSetUniformBuffers = nativeLimits.maxDescriptorSetUniformBuffers,
                .maxDescriptorSetUniformBuffersDynamic = nativeLimits.maxDescriptorSetUniformBuffersDynamic,
                .maxDescriptorSetStorageBuffers = nativeLimits.maxDescriptorSetStorageBuffers,
                .maxDescriptorSetStorageBuffersDynamic = nativeLimits.maxDescriptorSetStorageBuffersDynamic,
                .maxDescriptorSetSampledImages = nativeLimits.maxDescriptorSetSampledImages,
                .maxDescriptorSetStorageImages = nativeLimits.maxDescriptorSetStorageImages,
                .maxDescriptorSetAccelerationStructures = record_.maxDescriptorSetAccelerationStructures,
            };
            const std::uint32_t maxDrawIndirectCount = nativeLimits.maxDrawIndirectCount != 0
                                                           ? nativeLimits.maxDrawIndirectCount
                                                           : (record_.multiDrawIndirect ? 65535U : 1U);
            const bool isolatesDynamicBuffers = isolatesDynamicBufferShaders(record_);
            shaders_.initialize(device_,
                                &table_,
                                capabilities_,
                                &instance_.table,
                                queues_.maintenance4Enabled,
                                queues_.shaderFeaturesEnabled,
                                isolatesDynamicBuffers);
            bindings_.initialize(device_, &table_, capabilities_, &instance_.table, bufferLimits, descriptorLimits);
            pipelines_.initialize(device_,
                                  &table_,
                                  record_.physical,
                                  capabilities_,
                                  &shaders_,
                                  &bindings_,
                                  &instance_.table,
                                  record_.depthBiasClamp,
                                  &resources_.formatCache(),
                                  queues_.maintenance5Enabled,
                                  queues_.multiviewStagesEnabled,
                                  record_.minVertexInputBindingStrideAlignment,
                                  isolatesDynamicBuffers);
            graphics_.initialize(device_,
                                 &table_,
                                 &capabilities_,
                                 &resources_,
                                 &pipelines_,
                                 &bindings_,
                                 record_.depthBiasClamp,
                                 bufferLimits,
                                 maxDrawIndirectCount,
                                 clipsScissorsToRenderArea(record_));
            acceleration_.initialize(device_,
                                     &table_,
                                     &resources_,
                                     &capabilities_,
                                     record_.minAccelerationStructureScratchOffsetAlignment,
                                     &instance_.table);
            bindings_.setAccelerationManager(&acceleration_);
            const std::array queueTimestamps{
                publishVulkanQueueProperties(record_, QueueType::graphics).timestampQueries,
                publishVulkanQueueProperties(record_, QueueType::compute).timestampQueries,
                publishVulkanQueueProperties(record_, QueueType::copy).timestampQueries,
            };
            queries_.initialize(device_,
                                &table_,
                                &capabilities_,
                                &resources_,
                                record_.calibratedTimestamps,
                                &instance_.table,
                                queueTimestamps);
            images_.initialize(&record_, &capabilities_, &resources_, &shaders_, &bindings_, &pipelines_);
            swapchains_.initialize(instance_.instance,
                                   &instance_.table,
                                   record_.physical,
                                   device_,
                                   &table_,
                                   &resources_,
                                   &submission_,
                                   queues_.families.graphics,
                                   queues_.graphics,
                                   queues_.presentCompletion,
                                   record_.report.khrSwapchainMaintenance1,
                                   queues_.presentWaitEnabled);
        }
        return initialized;
    }

    VulkanDevice::~VulkanDevice()
    {
#ifdef RACHIS_TESTING
        unbindVulkanDevice(this);
        unbindVulkanValidationTestDevice(device_);
        unbindVulkanResourceAllocator(&resources_);
        unbindVulkanCommandAllocator(&commands_);
#endif
        const auto lost = operationalState_.load(std::memory_order_acquire) == DeviceOperationalState::lost;

        if (!lost || lossInjected_)
        {
            swapchains_.flushDeferredAcquireWaits();
            static_cast<void>(submission_.waitIdle());
        }
        submission_.collectGarbage();
        submission_.shutdown();
        {
            std::vector<std::weak_ptr<VulkanCommandPoolState>> pools;
            {
                const std::scoped_lock lock(acquiredPoolsMutex_);
                pools = std::move(acquiredPools_);
            }
            for (const auto& weakPool : pools)
            {
                auto pool = weakPool.lock();
                if (pool == nullptr)
                {
                    continue;
                }
                std::size_t entryCount = 0;
                {
                    const std::scoped_lock poolLock(pool->mutex);
                    for (const auto& entry : pool->entries)
                    {
                        if (entry != nullptr && entry->lease != nullptr)
                        {
                            entry->lease->clearRecordingState();
                        }
                    }
                    pool->available.clear();
                    pool->quarantined.clear();
                    entryCount = pool->entries.size();
                }
                for (std::size_t index = 0; index < entryCount; ++index)
                {
                    std::vector<std::shared_ptr<void>> retained;
                    {
                        const std::scoped_lock poolLock(pool->mutex);
                        if (index >= pool->entries.size())
                        {
                            break;
                        }
                        const auto& entry = pool->entries[index];
                        if (entry != nullptr && entry->lease != nullptr)
                        {
                            retained = entry->lease->takeRetained();
                        }
                    }
                    retained.clear();
                }
                pool->destroyNative();
                const std::scoped_lock poolLock(pool->mutex);
                pool->device = VK_NULL_HANDLE;
                pool->table = nullptr;
            }
        }
        commands_.shutdown();
        swapchains_.shutdown();
        images_.shutdown();
        queries_.shutdown();
        pipelines_.shutdown();
        bindings_.shutdown();
        acceleration_.shutdown();
        shaders_.shutdown();
        resources_.shutdown();
        if (device_ != VK_NULL_HANDLE)
        {
            if (table_.vkDestroyDevice != nullptr)
            {
                table_.vkDestroyDevice(device_, nullptr);
            }
            device_ = VK_NULL_HANDLE;
        }
    }

    const AdapterInfo& VulkanDevice::adapterInfo() const noexcept
    {
        return record_.adapter;
    }

    const DeviceCapabilities& VulkanDevice::capabilities() const noexcept
    {
        return capabilities_;
    }

    DeviceOperationalState VulkanDevice::operationalState() const noexcept
    {
        return operationalState_.load(std::memory_order_acquire);
    }

    std::optional<Error> VulkanDevice::deviceLossError() const
    {
        const std::scoped_lock lock(lossMutex_);
        return lossError_ != nullptr ? std::optional<Error>{*lossError_} : std::nullopt;
    }

    Status VulkanDevice::requireActive() const
    {
        if (operationalState_.load(std::memory_order_acquire) != DeviceOperationalState::lost)
        {
            return {};
        }
        const std::scoped_lock lock(lossMutex_);
        return std::unexpected(lossError_ != nullptr ? *lossError_
                                                     : Error{
                                                           ErrorCode::deviceLost,
                                                           VK_ERROR_DEVICE_LOST,
                                                           "The Vulkan device is lost.",
                                                       });
    }

    void VulkanDevice::publishLoss(const Error& error, DeviceFaultStatus status, bool injected) const
    {
        DeviceFaultReport captured{};
        Error publishedError = error;
        bool publish = false;
        {
            const std::scoped_lock lock(lossMutex_);
            if (lossError_ != nullptr)
            {
                return;
            }
            captured.status = status;
            captured.nativeReason =
                error.nativeCode != 0 ? error.nativeCode : static_cast<std::int64_t>(VK_ERROR_DEVICE_LOST);
            captured.dredAvailable = false;
            if (injected)
            {
                captured.breadcrumbs.push_back(DeviceFaultBreadcrumb{
                    .queueLabel = "RACHIS_TESTING",
                    .commandListLabel = "injected device loss",
                });
            }
            else if (queues_.deviceFaultEnabled && device_ != VK_NULL_HANDLE &&
                     table_.vkGetDeviceFaultInfoEXT != nullptr)
            {
                std::string faultSummary{};
                if (captureVulkanDeviceFaultInfo(device_, table_, captured, faultSummary) && !faultSummary.empty())
                {
                    if (!publishedError.message.empty())
                    {
                        publishedError.message += " ";
                    }
                    publishedError.message += faultSummary;
                }
            }
            lossError_ = std::make_shared<const Error>(publishedError);
            lossReport_ = std::make_shared<const DeviceFaultReport>(captured);
            lossInjected_ = injected;
            operationalState_.store(DeviceOperationalState::lost, std::memory_order_release);
            publish = true;
        }
        if (publish && instance_.sink != nullptr)
        {
            instance_.sink->publish(
                DiagnosticSeverity::error, static_cast<std::uint32_t>(captured.nativeReason), publishedError.message);
        }
    }

#ifdef RACHIS_TESTING
    Status VulkanDevice::injectDeviceLoss(DeviceFaultStatus status)
    {
        const Error injected{
            ErrorCode::deviceLost,
            VK_ERROR_DEVICE_LOST,
            "Injected Vulkan device loss for RACHIS_TESTING.",
        };
        publishLoss(injected, status, true);
        return requireActive();
    }

    void VulkanDevice::injectCreateResult(VkResult result) noexcept
    {
        if (realCreateEntryPoints.device == this)
        {
            table_.vkCreatePipelineCache = realCreateEntryPoints.pipelineCache;
            table_.vkCreateComputePipelines = realCreateEntryPoints.computePipelines;
            table_.vkCreateGraphicsPipelines = realCreateEntryPoints.graphicsPipelines;
            table_.vkCreateRayTracingPipelinesKHR = realCreateEntryPoints.rayTracingPipelines;
            table_.vkCreateAccelerationStructureKHR = realCreateEntryPoints.accelerationStructure;
            realCreateEntryPoints = {};
        }
        if (result == VK_SUCCESS)
        {
            return;
        }
        injectedCreateResult.store(result, std::memory_order_relaxed);
        realCreateEntryPoints = RealCreateEntryPoints{
            .device = this,
            .pipelineCache = table_.vkCreatePipelineCache,
            .computePipelines = table_.vkCreateComputePipelines,
            .graphicsPipelines = table_.vkCreateGraphicsPipelines,
            .rayTracingPipelines = table_.vkCreateRayTracingPipelinesKHR,
            .accelerationStructure = table_.vkCreateAccelerationStructureKHR,
        };
        table_.vkCreatePipelineCache = &injectedCreatePipelineCache;
        table_.vkCreateComputePipelines = &injectedCreateComputePipelines;
        table_.vkCreateGraphicsPipelines = &injectedCreateGraphicsPipelines;
        if (table_.vkCreateRayTracingPipelinesKHR != nullptr)
        {
            table_.vkCreateRayTracingPipelinesKHR = &injectedCreateRayTracingPipelines;
        }
        if (table_.vkCreateAccelerationStructureKHR != nullptr)
        {
            table_.vkCreateAccelerationStructureKHR = &injectedCreateAccelerationStructure;
        }
    }

    Result<VulkanPresentationSyncSnapshot> VulkanDevice::presentationSyncSnapshot(SwapchainHandle swapchain) const
    {
        return swapchains_.presentationSyncSnapshot(swapchain);
    }

    std::vector<VkTimeDomainEXT> VulkanDevice::calibrateableTimeDomains() const
    {
        auto* const getCalibrateableDomains = instance_.table.vkGetPhysicalDeviceCalibrateableTimeDomainsEXT;
        if (!record_.calibratedTimestamps || getCalibrateableDomains == nullptr || record_.physical == VK_NULL_HANDLE)
        {
            return {};
        }
        std::uint32_t domainCount = 0;
        if (getCalibrateableDomains(record_.physical, &domainCount, nullptr) != VK_SUCCESS || domainCount == 0)
        {
            return {};
        }
        std::vector<VkTimeDomainEXT> domains(domainCount);
        if (getCalibrateableDomains(record_.physical, &domainCount, domains.data()) != VK_SUCCESS)
        {
            return {};
        }
        domains.resize(domainCount);
        return domains;
    }
#endif

    Result<std::vector<MemoryBudget>> VulkanDevice::queryMemoryBudgets(std::uint32_t nodeIndex) const
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (nodeIndex >= capabilities_.nodeCount)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Memory-budget nodeIndex is outside the Vulkan device node count.",
            });
        }
        return observe(resources_.queryMemoryBudgets());
    }

    Result<ResourceMemorySnapshot> VulkanDevice::captureResourceMemorySnapshot() const
    {
        return resources_.captureResourceMemorySnapshot();
    }

    Result<DeviceFaultReport> VulkanDevice::captureDeviceFaultReport() const
    {
        {
            const std::scoped_lock lock(lossMutex_);
            if (lossReport_ != nullptr)
            {
                return *lossReport_;
            }
        }
        DeviceFaultReport report{};
        report.dredAvailable = false;

        return report;
    }

    BackendAllocationSnapshot VulkanDevice::captureBackendAllocationSnapshot() const noexcept
    {
        return resources_.captureBackendAllocationSnapshot();
    }

    FormatProperties VulkanDevice::queryFormatProperties(Format format) const noexcept
    {
        auto properties = resources_.queryFormatProperties(format);
        if (hasAny(properties.features, FormatFeature::depthStencilAttachment) &&
            (supportsSampleCount(properties.sampleCounts, 2) || supportsSampleCount(properties.sampleCounts, 4) ||
             supportsSampleCount(properties.sampleCounts, 8)))
        {
            properties.features |= FormatFeature::multisampleResolve;
            auto mask = ResolveModeMask::none;
            if ((record_.depthResolveModes & VK_RESOLVE_MODE_MIN_BIT) != 0)
            {
                mask = mask | ResolveModeMask::minimum;
            }
            if ((record_.depthResolveModes & VK_RESOLVE_MODE_MAX_BIT) != 0)
            {
                mask = mask | ResolveModeMask::maximum;
            }
            properties.depthResolveModes = mask;
            if (format == Format::d24UnormS8Uint)
            {
                auto stencil = ResolveModeMask::none;
                if ((record_.stencilResolveModes & VK_RESOLVE_MODE_MIN_BIT) != 0)
                {
                    stencil = stencil | ResolveModeMask::minimum;
                }
                if ((record_.stencilResolveModes & VK_RESOLVE_MODE_MAX_BIT) != 0)
                {
                    stencil = stencil | ResolveModeMask::maximum;
                }
                properties.stencilResolveModes = stencil;
            }
        }
        return properties;
    }

    QueueProperties VulkanDevice::queueProperties(QueueType queueType) const noexcept
    {
        return publishVulkanQueueProperties(record_, queueType);
    }

    Result<QueueClockCalibration> VulkanDevice::calibrateQueueClock(QueueType queueType) const
    {
        if (queueFor(queueType) == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "calibrateQueueClock requested a queue that was not created.",
            });
        }
        return queries_.calibrateQueueClock(record_.physical, queueType, record_);
    }

    DiagnosticCounts VulkanDevice::diagnosticCounts() const noexcept
    {
        return instance_.sink != nullptr ? instance_.sink->counts() : DiagnosticCounts{};
    }

    BindingUpdateStatistics VulkanDevice::bindingUpdateStatistics() const noexcept
    {
        return bindings_.statistics();
    }

    Result<TimelinePoint> VulkanDevice::submit(std::span<std::unique_ptr<detail::BackendCommandBuffer>> commandBuffers,
                                               std::span<const TimelinePoint> waits)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(submission_.submit(commandBuffers, waits));
    }

    Result<SharedTimelinePoint> VulkanDevice::exportTimelinePoint(TimelinePoint point)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(submission_.exportTimelinePoint(point));
    }

    Result<TimelinePoint> VulkanDevice::enqueueSharedTimelineWait(QueueType queueType, SharedTimelinePointView point)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(submission_.enqueueSharedTimelineWait(queueType, point));
    }

    Result<TimelinePointStatus> VulkanDevice::queryTimelinePoint(TimelinePoint point) const
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(submission_.queryTimelinePoint(point));
    }

    Result<TimelinePointStatus> VulkanDevice::wait(TimelinePoint point, std::chrono::nanoseconds timeout)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(submission_.wait(point, timeout));
    }

    Status VulkanDevice::wait(TimelinePoint point)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        return observe(submission_.wait(point));
    }

    Status VulkanDevice::waitIdle(QueueType queueType)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        return observe(submission_.waitIdle(queueType));
    }

    Status VulkanDevice::waitIdle()
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        return observe(submission_.waitIdle());
    }

    std::uint64_t VulkanDevice::completedValue(QueueType queueType) const noexcept
    {
        return submission_.completedValue(queueType);
    }

    void VulkanDevice::collectGarbage() noexcept
    {
        submission_.collectGarbage();
    }

    Result<ResourceMemoryRequirements> VulkanDevice::queryBufferMemoryRequirements(const BufferDesc& desc) const
    {
        return resources_.queryBufferMemoryRequirements(desc);
    }

    Result<ResourceHeapHandle> VulkanDevice::createResourceHeap(const ResourceHeapDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(resources_.createResourceHeap(desc));
    }

    Status VulkanDevice::destroyResourceHeap(ResourceHeapHandle heap)
    {
        return resources_.destroyResourceHeap(heap);
    }

    Result<BufferHandle> VulkanDevice::createBuffer(const BufferDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(resources_.createBuffer(desc));
    }

    Result<AccelerationStructureBuildSizes> VulkanDevice::queryBottomLevelAccelerationStructureBuildSizes(
        const BottomLevelAccelerationStructureBuildInput& input) const
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(acceleration_.queryBottomLevelBuildSizes(input));
    }

    Result<AccelerationStructureBuildSizes> VulkanDevice::queryTopLevelAccelerationStructureBuildSizes(
        const TopLevelAccelerationStructureBuildInput& input) const
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(acceleration_.queryTopLevelBuildSizes(input));
    }

    Result<AccelerationStructureHandle> VulkanDevice::createAccelerationStructure(const AccelerationStructureDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(acceleration_.createAccelerationStructure(desc));
    }

    Status VulkanDevice::destroyAccelerationStructure(AccelerationStructureHandle accelerationStructure)
    {
        return acceleration_.destroyAccelerationStructure(accelerationStructure);
    }

    Status VulkanDevice::buildBottomLevelAccelerationStructure(detail::BackendCommandBuffer& commandBuffer,
                                                               const BottomLevelAccelerationStructureBuildDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto* native = asOwnedVulkanCommand(commandBuffer, deviceToken_);
        if (native == nullptr)
        {
            return std::unexpected(
                Error{ErrorCode::invalidArgument, 0, "The command buffer belongs to another device or backend."});
        }
        std::shared_ptr<VulkanCommandPoolState> owner{};
        const auto poolLock = lockCommandPoolIfPresent(*native, owner);
        return observe(acceleration_.buildBottomLevel(*native, desc));
    }

    Status VulkanDevice::buildTopLevelAccelerationStructure(detail::BackendCommandBuffer& commandBuffer,
                                                            const TopLevelAccelerationStructureBuildDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto* native = asOwnedVulkanCommand(commandBuffer, deviceToken_);
        if (native == nullptr)
        {
            return std::unexpected(
                Error{ErrorCode::invalidArgument, 0, "The command buffer belongs to another device or backend."});
        }
        std::shared_ptr<VulkanCommandPoolState> owner{};
        const auto poolLock = lockCommandPoolIfPresent(*native, owner);
        return observe(acceleration_.buildTopLevel(*native, desc));
    }

    Status VulkanDevice::updateBottomLevelAccelerationStructure(detail::BackendCommandBuffer& commandBuffer,
                                                                const BottomLevelAccelerationStructureUpdateDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto* native = asOwnedVulkanCommand(commandBuffer, deviceToken_);
        if (native == nullptr)
        {
            return std::unexpected(
                Error{ErrorCode::invalidArgument, 0, "The command buffer belongs to another device or backend."});
        }
        std::shared_ptr<VulkanCommandPoolState> owner{};
        const auto poolLock = lockCommandPoolIfPresent(*native, owner);
        return observe(acceleration_.updateBottomLevel(*native, desc));
    }

    Status VulkanDevice::updateTopLevelAccelerationStructure(detail::BackendCommandBuffer& commandBuffer,
                                                             const TopLevelAccelerationStructureUpdateDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto* native = asOwnedVulkanCommand(commandBuffer, deviceToken_);
        if (native == nullptr)
        {
            return std::unexpected(
                Error{ErrorCode::invalidArgument, 0, "The command buffer belongs to another device or backend."});
        }
        std::shared_ptr<VulkanCommandPoolState> owner{};
        const auto poolLock = lockCommandPoolIfPresent(*native, owner);
        return observe(acceleration_.updateTopLevel(*native, desc));
    }

    Status VulkanDevice::updateBottomLevelAccelerationStructureInPlace(
        detail::BackendCommandBuffer& commandBuffer, const BottomLevelAccelerationStructureInPlaceUpdateDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto* native = asOwnedVulkanCommand(commandBuffer, deviceToken_);
        if (native == nullptr)
        {
            return std::unexpected(
                Error{ErrorCode::invalidArgument, 0, "The command buffer belongs to another device or backend."});
        }
        std::shared_ptr<VulkanCommandPoolState> owner{};
        const auto poolLock = lockCommandPoolIfPresent(*native, owner);
        return observe(acceleration_.updateBottomLevelInPlace(*native, desc));
    }

    Status VulkanDevice::updateTopLevelAccelerationStructureInPlace(
        detail::BackendCommandBuffer& commandBuffer, const TopLevelAccelerationStructureInPlaceUpdateDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto* native = asOwnedVulkanCommand(commandBuffer, deviceToken_);
        if (native == nullptr)
        {
            return std::unexpected(
                Error{ErrorCode::invalidArgument, 0, "The command buffer belongs to another device or backend."});
        }
        std::shared_ptr<VulkanCommandPoolState> owner{};
        const auto poolLock = lockCommandPoolIfPresent(*native, owner);
        return observe(acceleration_.updateTopLevelInPlace(*native, desc));
    }

    Status VulkanDevice::compactAccelerationStructure(detail::BackendCommandBuffer& commandBuffer,
                                                      const AccelerationStructureCompactionDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto* native = asOwnedVulkanCommand(commandBuffer, deviceToken_);
        if (native == nullptr)
        {
            return std::unexpected(
                Error{ErrorCode::invalidArgument, 0, "The command buffer belongs to another device or backend."});
        }
        std::shared_ptr<VulkanCommandPoolState> owner{};
        const auto poolLock = lockCommandPoolIfPresent(*native, owner);
        return observe(acceleration_.compact(*native, desc));
    }

    Result<BufferHandle> VulkanDevice::createSharedBuffer(const BufferDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(resources_.createSharedBuffer(desc));
    }

    Result<SharedHandle> VulkanDevice::exportSharedBuffer(BufferHandle buffer)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(resources_.exportSharedBuffer(buffer));
    }

    Result<BufferHandle> VulkanDevice::importSharedBuffer(SharedHandleView handle, const BufferDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(resources_.importSharedBuffer(handle, desc));
    }

    Result<BufferHandle> VulkanDevice::createPlacedBuffer(const ResourcePlacement& placement, const BufferDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(resources_.createPlacedBuffer(placement, desc));
    }

    Status VulkanDevice::destroyBuffer(BufferHandle buffer)
    {
        return resources_.destroyBuffer(buffer);
    }

    Result<MappedMemory> VulkanDevice::mapBuffer(BufferHandle buffer, std::uint64_t offset, std::uint64_t size)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(resources_.mapBuffer(buffer, offset, size));
    }

    Status VulkanDevice::unmapBuffer(BufferHandle buffer, std::uint64_t offset, std::uint64_t size)
    {
        auto unmapped = resources_.unmapBuffer(buffer, offset, size, requireActive().has_value());
        if (!unmapped && unmapped.error().code == ErrorCode::deviceLost)
        {
            publishLoss(unmapped.error(), DeviceFaultStatus::unknown, false);
            return resources_.unmapBuffer(buffer, offset, size, false);
        }
        return unmapped;
    }

    Status VulkanDevice::flushMappedMemoryRanges(BufferHandle buffer, std::span<const MappedMemoryRange> ranges)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        return observe(resources_.flushMappedMemoryRanges(buffer, ranges));
    }

    Status VulkanDevice::invalidateMappedMemoryRanges(BufferHandle buffer, std::span<const MappedMemoryRange> ranges)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        return observe(resources_.invalidateMappedMemoryRanges(buffer, ranges));
    }

    Status VulkanDevice::writeBuffer(BufferHandle buffer, std::uint64_t offset, std::span<const std::byte> data)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        return observe(resources_.writeBuffer(buffer, offset, data));
    }

    Status VulkanDevice::readBuffer(BufferHandle buffer, std::uint64_t offset, std::span<std::byte> destination)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        return observe(resources_.readBuffer(buffer, offset, destination));
    }

    Result<ResourceMemoryRequirements> VulkanDevice::queryTextureMemoryRequirements(const TextureDesc& desc) const
    {
        return resources_.queryTextureMemoryRequirements(desc);
    }

    Result<TextureHandle> VulkanDevice::createTexture(const TextureDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(resources_.createTexture(desc));
    }

    Result<TextureHandle> VulkanDevice::createSharedTexture(const TextureDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(resources_.createSharedTexture(desc));
    }

    Result<SharedHandle> VulkanDevice::exportSharedTexture(TextureHandle texture)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(resources_.exportSharedTexture(texture));
    }

    Result<TextureHandle> VulkanDevice::importSharedTexture(SharedHandleView handle, const TextureDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(resources_.importSharedTexture(handle, desc));
    }

    Result<TextureHandle> VulkanDevice::createPlacedTexture(const ResourcePlacement& placement, const TextureDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(resources_.createPlacedTexture(placement, desc));
    }

    Status VulkanDevice::destroyTexture(TextureHandle texture)
    {
        return resources_.destroyTexture(texture);
    }

    Result<TextureViewHandle> VulkanDevice::createTextureView(const TextureViewDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(resources_.createTextureView(desc));
    }

    Status VulkanDevice::destroyTextureView(TextureViewHandle view)
    {
        return resources_.destroyTextureView(view);
    }

    Result<SamplerHandle> VulkanDevice::createSampler(const SamplerDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(resources_.createSampler(desc));
    }

    Status VulkanDevice::destroySampler(SamplerHandle sampler)
    {
        return resources_.destroySampler(sampler);
    }

    Result<CommandPoolHandle> VulkanDevice::createCommandPool(const CommandPoolDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(commands_.createCommandPool(desc));
    }

    Status VulkanDevice::destroyCommandPool(CommandPoolHandle commandPool)
    {
        return commands_.destroyCommandPool(commandPool);
    }

    Status VulkanDevice::resetCommandPool(CommandPoolHandle commandPool)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        submission_.collectGarbage();
        return observe(commands_.resetCommandPool(commandPool));
    }

    Result<std::unique_ptr<detail::BackendCommandBuffer>> VulkanDevice::acquireCommandBuffer(
        CommandPoolHandle commandPool)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        submission_.collectGarbage();
        auto acquired = observe(commands_.acquireCommandBuffer(commandPool));
        if (acquired && *acquired != nullptr)
        {
            const auto* native = (*acquired)->kind() == detail::BackendCommandBufferKind::vulkan
                                     ? static_cast<const VulkanCommandBuffer*>(acquired->get())
                                     : nullptr;
            if (native != nullptr && native->ownerShared() != nullptr)
            {
                const auto& owner = native->ownerShared();
                if (!owner->trackedForShutdown.load(std::memory_order_acquire))
                {
                    const std::scoped_lock lock(acquiredPoolsMutex_);
                    if (!owner->trackedForShutdown.load(std::memory_order_relaxed))
                    {
                        if (acquiredPools_.size() >= nextAcquiredPoolPruneThreshold_)
                        {
                            static_cast<void>(
                                std::erase_if(acquiredPools_, [](const auto& existing) { return existing.expired(); }));
                            nextAcquiredPoolPruneThreshold_ = std::max<std::size_t>(64, acquiredPools_.size() * 2);
                        }
                        try
                        {
                            acquiredPools_.push_back(owner);
                        }
                        catch (const std::bad_alloc&)
                        {
                            return std::unexpected(
                                Error{ErrorCode::outOfMemory, 0, "Out of memory while tracking Vulkan command pool."});
                        }
                        owner->trackedForShutdown.store(true, std::memory_order_release);
                    }
                }
            }
        }
        return acquired;
    }

    Status VulkanDevice::resetCommandBuffer(detail::BackendCommandBuffer& commandBuffer)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        return observe(commands_.resetCommandBuffer(commandBuffer));
    }

    Status VulkanDevice::beginDebugGroup(detail::BackendCommandBuffer& commandBuffer, const DebugMarkerDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        return observe(commands_.beginDebugGroup(commandBuffer, desc));
    }

    Status VulkanDevice::endDebugGroup(detail::BackendCommandBuffer& commandBuffer)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        return observe(commands_.endDebugGroup(commandBuffer));
    }

    Status VulkanDevice::insertDebugMarker(detail::BackendCommandBuffer& commandBuffer, const DebugMarkerDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        return observe(commands_.insertDebugMarker(commandBuffer, desc));
    }

    Status VulkanDevice::bufferBarrier(detail::BackendCommandBuffer& commandBuffer, const BufferBarrier& barrier)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto locked = lockLiveVulkanCommand(
            commandBuffer, deviceToken_, "A buffer barrier requires a live Vulkan command-buffer lease.");
        if (!locked)
        {
            return std::unexpected(std::move(locked.error()));
        }
        return observe(
            resources_.bufferBarrier((*locked)->native(), (*locked)->queueType(), barrier, retainSink(**locked)));
    }

    Status VulkanDevice::activateBufferAlias(detail::BackendCommandBuffer& commandBuffer,
                                             const BufferAliasActivation& activation)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto locked = lockLiveVulkanCommand(
            commandBuffer, deviceToken_, "Buffer alias activation requires a live Vulkan command-buffer lease.");
        if (!locked)
        {
            return std::unexpected(std::move(locked.error()));
        }
        return observe(resources_.activateBufferAlias(
            (*locked)->native(), (*locked)->queueType(), activation, retainSink(**locked)));
    }

    Status VulkanDevice::textureBarrier(detail::BackendCommandBuffer& commandBuffer, const TextureBarrier& barrier)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto locked = lockLiveVulkanCommand(
            commandBuffer, deviceToken_, "A texture barrier requires a live Vulkan command-buffer lease.");
        if (!locked)
        {
            return std::unexpected(std::move(locked.error()));
        }
        return observe(
            resources_.textureBarrier((*locked)->native(), (*locked)->queueType(), barrier, retainSink(**locked)));
    }

    Status VulkanDevice::activateTextureAlias(detail::BackendCommandBuffer& commandBuffer,
                                              const TextureAliasActivation& activation)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto locked = lockLiveVulkanCommand(
            commandBuffer, deviceToken_, "Texture alias activation requires a live Vulkan command-buffer lease.");
        if (!locked)
        {
            return std::unexpected(std::move(locked.error()));
        }
        return observe(resources_.activateTextureAlias(
            (*locked)->native(), (*locked)->queueType(), activation, retainSink(**locked)));
    }

    Status VulkanDevice::resourceBarriers(detail::BackendCommandBuffer& commandBuffer,
                                          const ResourceBarrierBatch& barriers)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto locked = lockLiveVulkanCommand(
            commandBuffer, deviceToken_, "A resource-barrier batch requires a live Vulkan command-buffer lease.");
        if (!locked)
        {
            return std::unexpected(std::move(locked.error()));
        }
        return observe(
            resources_.resourceBarriers((*locked)->native(), (*locked)->queueType(), barriers, retainSink(**locked)));
    }

    Status VulkanDevice::copyBufferRegions(detail::BackendCommandBuffer& commandBuffer,
                                           BufferHandle source,
                                           BufferHandle destination,
                                           std::span<const BufferCopyRegion> regions)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto locked = lockLiveVulkanCommand(
            commandBuffer, deviceToken_, "A buffer copy requires a live Vulkan command-buffer lease.");
        if (!locked)
        {
            return std::unexpected(std::move(locked.error()));
        }
        if (commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Buffer copies cannot be recorded inside a rendering scope.",
            });
        }
        return observe(
            resources_.copyBufferRegions((*locked)->native(), source, destination, regions, retainSink(**locked)));
    }

    Status VulkanDevice::fillBuffer(detail::BackendCommandBuffer& commandBuffer,
                                    BufferHandle buffer,
                                    std::uint64_t offset,
                                    std::uint64_t size,
                                    std::uint32_t value)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto locked = lockLiveVulkanCommand(
            commandBuffer, deviceToken_, "A buffer fill requires a live Vulkan command-buffer lease.");
        if (!locked)
        {
            return std::unexpected(std::move(locked.error()));
        }
        return observe(resources_.fillBuffer((*locked)->native(), buffer, offset, size, value, retainSink(**locked)));
    }

    Result<TextureTransferLayout> VulkanDevice::queryTextureTransferLayout(TextureHandle texture,
                                                                           const BufferTextureCopyRegion& region) const
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(resources_.queryTextureTransferLayout(texture, region));
    }

    Status VulkanDevice::copyBufferToTextureRegions(detail::BackendCommandBuffer& commandBuffer,
                                                    BufferHandle source,
                                                    TextureHandle destination,
                                                    std::span<const BufferTextureCopyRegion> regions)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto locked = lockLiveVulkanCommand(
            commandBuffer, deviceToken_, "A buffer-to-texture copy requires a live Vulkan command-buffer lease.");
        if (!locked)
        {
            return std::unexpected(std::move(locked.error()));
        }
        if (commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Buffer-to-texture copies cannot be recorded inside a rendering scope.",
            });
        }
        return observe(resources_.copyBufferToTextureRegions(
            (*locked)->native(), (*locked)->queueType(), source, destination, regions, retainSink(**locked)));
    }

    Status VulkanDevice::copyTextureToBufferRegions(detail::BackendCommandBuffer& commandBuffer,
                                                    TextureHandle source,
                                                    BufferHandle destination,
                                                    std::span<const BufferTextureCopyRegion> regions)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto locked = lockLiveVulkanCommand(
            commandBuffer, deviceToken_, "A texture-to-buffer copy requires a live Vulkan command-buffer lease.");
        if (!locked)
        {
            return std::unexpected(std::move(locked.error()));
        }
        if (commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Texture-to-buffer copies cannot be recorded inside a rendering scope.",
            });
        }
        return observe(resources_.copyTextureToBufferRegions(
            (*locked)->native(), (*locked)->queueType(), source, destination, regions, retainSink(**locked)));
    }

    Status VulkanDevice::copyTextureRegions(detail::BackendCommandBuffer& commandBuffer,
                                            TextureHandle source,
                                            TextureHandle destination,
                                            std::span<const TextureCopyRegion> regions)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto locked = lockLiveVulkanCommand(
            commandBuffer, deviceToken_, "A texture copy requires a live Vulkan command-buffer lease.");
        if (!locked)
        {
            return std::unexpected(std::move(locked.error()));
        }
        if (commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Texture copies cannot be recorded inside a rendering scope.",
            });
        }
        return observe(
            resources_.copyTextureRegions((*locked)->native(), source, destination, regions, retainSink(**locked)));
    }

    Status VulkanDevice::clearStorageTexture(detail::BackendCommandBuffer& commandBuffer,
                                             TextureViewHandle view,
                                             const StorageTextureClearValue& value)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto locked = lockLiveVulkanCommand(
            commandBuffer, deviceToken_, "A storage-texture clear requires a live Vulkan command-buffer lease.");
        if (!locked)
        {
            return std::unexpected(std::move(locked.error()));
        }
        return observe(resources_.clearStorageTexture((*locked)->native(), view, value, retainSink(**locked)));
    }

    Status VulkanDevice::resolveTexture(detail::BackendCommandBuffer& commandBuffer, const TextureResolve& resolve)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto locked = lockLiveVulkanCommand(
            commandBuffer, deviceToken_, "A texture resolve requires a live Vulkan command-buffer lease.");
        if (!locked)
        {
            return std::unexpected(std::move(locked.error()));
        }
        return observe(images_.resolveTexture(**locked, resolve));
    }

    Status VulkanDevice::generateMipmaps(detail::BackendCommandBuffer& commandBuffer,
                                         const TextureMipmapGeneration& generation)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto locked = lockLiveVulkanCommand(
            commandBuffer, deviceToken_, "Mipmap generation requires a live Vulkan command-buffer lease.");
        if (!locked)
        {
            return std::unexpected(std::move(locked.error()));
        }
        return observe(images_.generateMipmaps(**locked, generation));
    }

    Result<BindingLayoutHandle> VulkanDevice::createBindingLayout(const BindingLayoutDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(bindings_.createBindingLayout(desc));
    }

    Status VulkanDevice::destroyBindingLayout(BindingLayoutHandle layout)
    {
        return bindings_.destroyBindingLayout(layout);
    }

    Result<BindingPoolHandle> VulkanDevice::createBindingPool(const BindingPoolDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(bindings_.createBindingPool(desc));
    }

    Status VulkanDevice::destroyBindingPool(BindingPoolHandle pool)
    {
        return bindings_.destroyBindingPool(pool);
    }

    Result<BindingSetHandle> VulkanDevice::createBindingSet(const BindingSetDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(bindings_.createBindingSet(resources_, desc));
    }

    Status VulkanDevice::updateBindingSet(BindingSetHandle set, std::span<const BindingWrite> writes)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        return observe(bindings_.updateBindingSet(resources_, set, writes));
    }

    Status VulkanDevice::destroyBindingSet(BindingSetHandle set)
    {
        return bindings_.destroyBindingSet(set);
    }

    Result<ShaderLibraryHandle> VulkanDevice::createShaderLibrary(const ShaderLibraryDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(shaders_.createShaderLibrary(desc));
    }

    Status VulkanDevice::destroyShaderLibrary(ShaderLibraryHandle shaderLibrary)
    {
        return shaders_.destroyShaderLibrary(shaderLibrary);
    }

    Result<ShaderModuleHandle> VulkanDevice::createShaderModule(const ShaderModuleDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(shaders_.createShaderModule(desc));
    }

    Result<ShaderInterfaceSnapshot> VulkanDevice::shaderModuleInterface(ShaderModuleHandle shaderModule) const
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return shaders_.shaderModuleInterface(shaderModule);
    }

    Status VulkanDevice::destroyShaderModule(ShaderModuleHandle shaderModule)
    {
        return shaders_.destroyShaderModule(shaderModule);
    }

    Result<PipelineLayoutHandle> VulkanDevice::createPipelineLayout(const PipelineLayoutDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(bindings_.createPipelineLayout(desc));
    }

    Status VulkanDevice::destroyPipelineLayout(PipelineLayoutHandle pipelineLayout)
    {
        return bindings_.destroyPipelineLayout(pipelineLayout);
    }

    Result<PipelineCacheHandle> VulkanDevice::createPipelineCache(const PipelineCacheDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(pipelines_.createPipelineCache(desc));
    }

    Status VulkanDevice::destroyPipelineCache(PipelineCacheHandle pipelineCache)
    {
        return pipelines_.destroyPipelineCache(pipelineCache);
    }

    Result<std::vector<std::byte>> VulkanDevice::pipelineCacheData(PipelineCacheHandle pipelineCache)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(pipelines_.pipelineCacheData(pipelineCache));
    }

    Result<PipelineHandle> VulkanDevice::createComputePipeline(const ComputePipelineDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(pipelines_.createComputePipeline(desc));
    }

    Result<PipelineHandle> VulkanDevice::createGraphicsPipeline(const GraphicsPipelineDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(pipelines_.createGraphicsPipeline(desc));
    }

    Result<PipelineCompilationHandle> VulkanDevice::beginComputePipelineCompilation(const ComputePipelineDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(pipelines_.beginComputePipelineCompilation(desc));
    }

    Result<PipelineCompilationHandle> VulkanDevice::beginGraphicsPipelineCompilation(const GraphicsPipelineDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(pipelines_.beginGraphicsPipelineCompilation(desc));
    }

    Result<PipelineCompilationState> VulkanDevice::pipelineCompilationState(PipelineCompilationHandle compilation)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return pipelines_.pipelineCompilationState(compilation);
    }

    Result<PipelineHandle> VulkanDevice::waitPipelineCompilation(PipelineCompilationHandle compilation)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(pipelines_.waitPipelineCompilation(compilation));
    }

    Status VulkanDevice::destroyPipelineCompilation(PipelineCompilationHandle compilation)
    {
        return pipelines_.destroyPipelineCompilation(compilation);
    }

    Status VulkanDevice::destroyPipeline(PipelineHandle pipeline)
    {
        return pipelines_.destroyPipeline(pipeline);
    }

    Result<RayTracingPipelineLibraryHandle> VulkanDevice::createRayTracingPipelineLibrary(
        const RayTracingPipelineLibraryDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(pipelines_.createRayTracingPipelineLibrary(desc));
    }

    Status VulkanDevice::destroyRayTracingPipelineLibrary(RayTracingPipelineLibraryHandle library)
    {
        return pipelines_.destroyRayTracingPipelineLibrary(library);
    }

    Result<PipelineHandle> VulkanDevice::createRayTracingPipeline(const RayTracingPipelineDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(pipelines_.createRayTracingPipeline(desc));
    }

    Status VulkanDevice::getRayTracingShaderGroupHandles(PipelineHandle pipeline,
                                                         std::uint32_t firstGroup,
                                                         std::uint32_t groupCount,
                                                         std::span<std::byte> destination)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        return observe(pipelines_.getRayTracingShaderGroupHandles(pipeline, firstGroup, groupCount, destination));
    }

    Result<std::uint32_t> VulkanDevice::rayTracingShaderGroupRecordSize(PipelineHandle pipeline,
                                                                        std::uint32_t groupIndex)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(pipelines_.rayTracingShaderGroupRecordSize(pipeline, groupIndex));
    }

    Status VulkanDevice::writeRayTracingShaderGroupRecord(PipelineHandle pipeline,
                                                          std::uint32_t groupIndex,
                                                          std::span<const std::byte> localConstants,
                                                          std::span<std::byte> destination)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        return observe(pipelines_.writeRayTracingShaderGroupRecord(pipeline, groupIndex, localConstants, destination));
    }

    namespace
    {
        [[nodiscard]] Result<LockedVulkanCommand> requireVulkanCommand(VulkanDevice& device,
                                                                       detail::BackendCommandBuffer& commandBuffer,
                                                                       std::uint64_t deviceToken);
    }

    Status VulkanDevice::bindRayTracingPipeline(detail::BackendCommandBuffer& commandBuffer, PipelineHandle pipeline)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.bindRayTracingPipeline(**native, pipeline));
    }

    Status VulkanDevice::bindRayTracingBindingSet(detail::BackendCommandBuffer& commandBuffer,
                                                  std::uint32_t setIndex,
                                                  BindingSetHandle bindingSet,
                                                  std::span<const std::uint32_t> dynamicOffsets)
    {
        const std::array sets{bindingSet};
        return bindRayTracingBindingSets(commandBuffer, setIndex, sets, dynamicOffsets);
    }

    Status VulkanDevice::bindRayTracingBindingSets(detail::BackendCommandBuffer& commandBuffer,
                                                   std::uint32_t firstSet,
                                                   std::span<const BindingSetHandle> bindingSets,
                                                   std::span<const std::uint32_t> dynamicOffsets)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.bindRayTracingBindingSets(**native, firstSet, bindingSets, dynamicOffsets));
    }

    Status VulkanDevice::setRayTracingPushConstants(detail::BackendCommandBuffer& commandBuffer,
                                                    std::uint32_t offset,
                                                    std::span<const std::byte> data)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.setRayTracingPushConstants(**native, offset, data));
    }

    Status VulkanDevice::dispatchRays(detail::BackendCommandBuffer& commandBuffer, const RayTracingDispatchDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.dispatchRays(**native, desc));
    }

    Status VulkanDevice::dispatchRaysIndirect(detail::BackendCommandBuffer& commandBuffer,
                                              const RayTracingIndirectDispatchDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.dispatchRaysIndirect(**native, desc));
    }

    namespace
    {

        [[nodiscard]] Result<LockedVulkanCommand> requireVulkanCommand(VulkanDevice& device,
                                                                       detail::BackendCommandBuffer& commandBuffer,
                                                                       std::uint64_t deviceToken)
        {
            static_cast<void>(device);
            return lockLiveVulkanCommand(
                commandBuffer, deviceToken, "A graphics command requires a live Vulkan command-buffer lease.");
        }

    } // namespace

    Status VulkanDevice::beginRendering(detail::BackendCommandBuffer& commandBuffer, const RenderingDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        auto prepared = queries_.prepareRenderingTimestamps(**native, desc.timestampWrites);
        if (!prepared)
        {
            return observe(Status{std::unexpect, std::move(prepared.error())});
        }
        auto begun = graphics_.beginRendering(**native, desc, [&]() {
            if (prepared->has_value())
            {
                queries_.beginRenderingTimestamps(**native, **prepared);
            }
        });
        if (!begun)
        {
            return observe(std::move(begun));
        }
        if (prepared->has_value())
        {
            queries_.commitRenderingTimestamps(**native, std::move(**prepared));
        }
        return {};
    }

    Status VulkanDevice::endRendering(detail::BackendCommandBuffer& commandBuffer)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        auto ended = graphics_.endRendering(**native);
        if (!ended)
        {
            return observe(std::move(ended));
        }
        return observe(queries_.endRenderingTimestamps(**native));
    }

    Status VulkanDevice::clearColorAttachment(detail::BackendCommandBuffer& commandBuffer,
                                              std::uint32_t attachmentIndex,
                                              const ClearColor& clearValue,
                                              std::span<const ScissorRect> rectangles)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.clearColorAttachment(**native, attachmentIndex, clearValue, rectangles));
    }

    Status VulkanDevice::clearDepthStencilAttachment(detail::BackendCommandBuffer& commandBuffer,
                                                     DepthStencilClearAspect aspects,
                                                     float depth,
                                                     std::uint32_t stencil,
                                                     std::span<const ScissorRect> rectangles)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.clearDepthStencilAttachment(**native, aspects, depth, stencil, rectangles));
    }

    Status VulkanDevice::bindGraphicsPipeline(detail::BackendCommandBuffer& commandBuffer, PipelineHandle pipeline)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.bindGraphicsPipeline(**native, pipeline));
    }

    Status VulkanDevice::bindGraphicsBindingSet(detail::BackendCommandBuffer& commandBuffer,
                                                std::uint32_t setIndex,
                                                BindingSetHandle bindingSet,
                                                std::span<const std::uint32_t> dynamicOffsets)
    {
        const std::array sets{bindingSet};
        return bindGraphicsBindingSets(commandBuffer, setIndex, sets, dynamicOffsets);
    }

    Status VulkanDevice::bindGraphicsBindingSets(detail::BackendCommandBuffer& commandBuffer,
                                                 std::uint32_t firstSet,
                                                 std::span<const BindingSetHandle> bindingSets,
                                                 std::span<const std::uint32_t> dynamicOffsets)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.bindGraphicsBindingSets(**native, firstSet, bindingSets, dynamicOffsets));
    }

    Status VulkanDevice::setGraphicsPushConstants(detail::BackendCommandBuffer& commandBuffer,
                                                  ShaderStage visibility,
                                                  std::uint32_t offset,
                                                  std::span<const std::byte> data)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.setGraphicsPushConstants(**native, visibility, offset, data));
    }

    Status VulkanDevice::setViewports(detail::BackendCommandBuffer& commandBuffer, std::span<const Viewport> viewports)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.setViewports(**native, viewports));
    }

    Status VulkanDevice::setScissors(detail::BackendCommandBuffer& commandBuffer, std::span<const ScissorRect> scissors)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.setScissors(**native, scissors));
    }

    Status VulkanDevice::setBlendConstants(detail::BackendCommandBuffer& commandBuffer, const BlendConstants& constants)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.setBlendConstants(**native, constants));
    }

    Status VulkanDevice::setStencilReference(detail::BackendCommandBuffer& commandBuffer,
                                             StencilFace face,
                                             std::uint32_t reference)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.setStencilReference(**native, face, reference));
    }

    Status VulkanDevice::setDepthBias(detail::BackendCommandBuffer& commandBuffer, const DepthBias& bias)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.setDepthBias(**native, bias));
    }

    Status VulkanDevice::setDepthBounds(detail::BackendCommandBuffer& commandBuffer, const DepthBounds& bounds)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.setDepthBounds(**native, bounds));
    }

    Status VulkanDevice::bindVertexBuffers(detail::BackendCommandBuffer& commandBuffer,
                                           std::uint32_t firstSlot,
                                           std::span<const VertexBufferBinding> bindings)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.bindVertexBuffers(**native, firstSlot, bindings));
    }

    Status VulkanDevice::bindIndexBuffer(detail::BackendCommandBuffer& commandBuffer,
                                         BufferHandle buffer,
                                         IndexFormat format,
                                         std::uint64_t offset)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.bindIndexBuffer(**native, buffer, format, offset));
    }

    Status VulkanDevice::draw(detail::BackendCommandBuffer& commandBuffer,
                              std::uint32_t vertexCount,
                              std::uint32_t instanceCount,
                              std::uint32_t firstVertex,
                              std::uint32_t firstInstance)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.draw(**native, vertexCount, instanceCount, firstVertex, firstInstance));
    }

    Status VulkanDevice::drawIndexed(detail::BackendCommandBuffer& commandBuffer,
                                     std::uint32_t indexCount,
                                     std::uint32_t instanceCount,
                                     std::uint32_t firstIndex,
                                     std::int32_t baseVertex,
                                     std::uint32_t firstInstance)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(
            graphics_.drawIndexed(**native, indexCount, instanceCount, firstIndex, baseVertex, firstInstance));
    }

    Status VulkanDevice::drawIndirect(detail::BackendCommandBuffer& commandBuffer, const IndirectDrawDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.drawIndirect(**native, desc));
    }

    Status VulkanDevice::drawIndexedIndirect(detail::BackendCommandBuffer& commandBuffer, const IndirectDrawDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.drawIndexedIndirect(**native, desc));
    }

    Status VulkanDevice::setShadingRate(detail::BackendCommandBuffer& commandBuffer,
                                        ShadingRate rate,
                                        const ShadingRateCombiners& combiners)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.setShadingRate(**native, rate, combiners));
    }

    Status VulkanDevice::setShadingRateImage(detail::BackendCommandBuffer& commandBuffer, TextureHandle image)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.setShadingRateImage(**native, image));
    }

    Status VulkanDevice::dispatchMesh(detail::BackendCommandBuffer& commandBuffer,
                                      std::uint32_t groupCountX,
                                      std::uint32_t groupCountY,
                                      std::uint32_t groupCountZ)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.dispatchMesh(**native, groupCountX, groupCountY, groupCountZ));
    }

    Status VulkanDevice::dispatchMeshIndirect(detail::BackendCommandBuffer& commandBuffer,
                                              const IndirectMeshDispatchDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.dispatchMeshIndirect(**native, desc));
    }

    Status VulkanDevice::bindComputePipeline(detail::BackendCommandBuffer& commandBuffer, PipelineHandle pipeline)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.bindComputePipeline(**native, pipeline));
    }

    Status VulkanDevice::bindComputeBindingSet(detail::BackendCommandBuffer& commandBuffer,
                                               std::uint32_t setIndex,
                                               BindingSetHandle bindingSet,
                                               std::span<const std::uint32_t> dynamicOffsets)
    {
        const std::array sets{bindingSet};
        return bindComputeBindingSets(commandBuffer, setIndex, sets, dynamicOffsets);
    }

    Status VulkanDevice::bindComputeBindingSets(detail::BackendCommandBuffer& commandBuffer,
                                                std::uint32_t firstSet,
                                                std::span<const BindingSetHandle> bindingSets,
                                                std::span<const std::uint32_t> dynamicOffsets)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.bindComputeBindingSets(**native, firstSet, bindingSets, dynamicOffsets));
    }

    Status VulkanDevice::setComputePushConstants(detail::BackendCommandBuffer& commandBuffer,
                                                 std::uint32_t offset,
                                                 std::span<const std::byte> data)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.setComputePushConstants(**native, offset, data));
    }

    Status VulkanDevice::dispatch(detail::BackendCommandBuffer& commandBuffer,
                                  std::uint32_t groupCountX,
                                  std::uint32_t groupCountY,
                                  std::uint32_t groupCountZ)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.dispatch(**native, groupCountX, groupCountY, groupCountZ));
    }

    Status VulkanDevice::dispatchIndirect(detail::BackendCommandBuffer& commandBuffer, const IndirectDispatchDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(graphics_.dispatchIndirect(**native, desc));
    }

    Result<SurfaceHandle> VulkanDevice::createSurface(const SurfaceDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(swapchains_.createSurface(desc));
    }

    Result<SurfaceCapabilities> VulkanDevice::querySurfaceCapabilities(SurfaceHandle surface) const
    {
        return swapchains_.querySurfaceCapabilities(surface);
    }

    Status VulkanDevice::destroySurface(SurfaceHandle surface)
    {
        return swapchains_.destroySurface(surface);
    }

    Result<SwapchainHandle> VulkanDevice::createSwapchain(const SwapchainDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(swapchains_.createSwapchain(desc));
    }

    Status VulkanDevice::destroySwapchain(SwapchainHandle swapchain)
    {
        bool lost = false;
        bool injected = false;
        if (operationalState_.load(std::memory_order_acquire) == DeviceOperationalState::lost)
        {
            const std::scoped_lock lock(lossMutex_);
            lost = true;
            injected = lossInjected_;
        }
        auto destroyed = swapchains_.destroySwapchain(swapchain, lost, !lost || injected);
        if (!destroyed && destroyed.error().code == ErrorCode::deviceLost)
        {
            publishLoss(destroyed.error(), DeviceFaultStatus::unknown, false);
            destroyed = swapchains_.destroySwapchain(swapchain, true, false);
        }
        return destroyed;
    }

    Status VulkanDevice::resizeSwapchain(SwapchainHandle swapchain, std::uint32_t width, std::uint32_t height)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        return observe(swapchains_.resizeSwapchain(swapchain, width, height));
    }

    Result<AcquiredSwapchainImage> VulkanDevice::acquireNextImage(SwapchainHandle swapchain)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(swapchains_.acquireNextImage(swapchain));
    }

    bool VulkanDevice::hasAcquiredSwapchainImage(SwapchainHandle swapchain) const noexcept
    {
        return swapchains_.hasAcquiredImage(swapchain);
    }

    Status VulkanDevice::cancelSwapchainImage(SwapchainHandle swapchain)
    {
        return swapchains_.cancelSwapchainImage(swapchain);
    }

    Result<PresentStatus> VulkanDevice::present(SwapchainHandle swapchain, TimelinePoint renderComplete)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        bool injectedDeviceLoss = false;
        auto result = swapchains_.present(swapchain, renderComplete, injectedDeviceLoss);
        if (!result && injectedDeviceLoss)
        {
            publishLoss(result.error(), DeviceFaultStatus::removed, true);
        }
        return observe(std::move(result));
    }

    Result<SwapchainPresentationInfo> VulkanDevice::querySwapchainPresentation(SwapchainHandle swapchain) const
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return swapchains_.queryPresentation(swapchain);
    }

    Result<SwapchainFrameTiming> VulkanDevice::querySwapchainFrameTiming(SwapchainHandle swapchain) const
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return swapchains_.queryFrameTiming(swapchain);
    }

    Result<QueryPoolHandle> VulkanDevice::createTimestampQueryPool(const TimestampQueryPoolDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (queueFor(desc.queueType) == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A timestamp query pool requires a created queue.",
            });
        }
        if (!publishVulkanQueueProperties(record_, desc.queueType).timestampQueries)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Timestamp queries are not available on this queue.",
            });
        }
        return observe(queries_.createTimestampQueryPool(desc));
    }

    Status VulkanDevice::destroyQueryPool(QueryPoolHandle queryPool)
    {
        return queries_.destroyQueryPool(queryPool);
    }

    Status VulkanDevice::resetQueryPool(detail::BackendCommandBuffer& commandBuffer,
                                        QueryPoolHandle queryPool,
                                        std::uint32_t firstQuery,
                                        std::uint32_t queryCount)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(queries_.resetQueryPool(**native, queryPool, firstQuery, queryCount));
    }

    Status VulkanDevice::writeTimestamp(detail::BackendCommandBuffer& commandBuffer,
                                        QueryPoolHandle queryPool,
                                        std::uint32_t queryIndex)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(queries_.writeTimestamp(**native, queryPool, queryIndex));
    }

    Status VulkanDevice::beginGpuProfileScope(detail::BackendCommandBuffer& commandBuffer,
                                              const GpuProfileScopeDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(queries_.beginGpuProfileScope(**native, desc));
    }

    Status VulkanDevice::endGpuProfileScope(detail::BackendCommandBuffer& commandBuffer)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(queries_.endGpuProfileScope(**native));
    }

    Status VulkanDevice::resolveTimestamps(detail::BackendCommandBuffer& commandBuffer,
                                           const TimestampQueryResolveDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(queries_.resolveTimestamps(**native, desc));
    }

    Result<QueryPoolHandle> VulkanDevice::createOcclusionQueryPool(const OcclusionQueryPoolDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(queries_.createOcclusionQueryPool(desc));
    }

    Status VulkanDevice::beginOcclusionQuery(detail::BackendCommandBuffer& commandBuffer,
                                             QueryPoolHandle queryPool,
                                             std::uint32_t queryIndex)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(queries_.beginOcclusionQuery(**native, queryPool, queryIndex));
    }

    Status VulkanDevice::endOcclusionQuery(detail::BackendCommandBuffer& commandBuffer)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(queries_.endOcclusionQuery(**native));
    }

    Status VulkanDevice::resolveOcclusionQueries(detail::BackendCommandBuffer& commandBuffer,
                                                 const OcclusionQueryResolveDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(queries_.resolveOcclusionQueries(**native, desc));
    }

    Result<QueryPoolHandle> VulkanDevice::createPipelineStatisticsQueryPool(const PipelineStatisticsQueryPoolDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        return observe(queries_.createPipelineStatisticsQueryPool(desc));
    }

    Status VulkanDevice::beginPipelineStatisticsQuery(detail::BackendCommandBuffer& commandBuffer,
                                                      QueryPoolHandle queryPool,
                                                      std::uint32_t queryIndex)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(queries_.beginPipelineStatisticsQuery(**native, queryPool, queryIndex));
    }

    Status VulkanDevice::endPipelineStatisticsQuery(detail::BackendCommandBuffer& commandBuffer)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(queries_.endPipelineStatisticsQuery(**native));
    }

    Status VulkanDevice::resolvePipelineStatisticsQueries(detail::BackendCommandBuffer& commandBuffer,
                                                          const PipelineStatisticsQueryResolveDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto native = requireVulkanCommand(*this, commandBuffer, deviceToken_);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        return observe(queries_.resolvePipelineStatisticsQueries(**native, desc));
    }

#ifdef RACHIS_TESTING
    void bindVulkanDevice(VulkanDevice* device) noexcept
    {
        const std::scoped_lock lock(boundDeviceMutex);
        boundDevice = device;
    }

    void unbindVulkanDevice(VulkanDevice* device) noexcept
    {
        const std::scoped_lock lock(boundDeviceMutex);
        if (boundDevice == device)
        {
            boundDevice = nullptr;
        }
        if (realCreateEntryPoints.device == device)
        {
            realCreateEntryPoints = {};
        }
    }

    Status injectVulkanDeviceLoss(DeviceFaultStatus status)
    {
        const std::scoped_lock lock(boundDeviceMutex);
        if (boundDevice == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "No Vulkan device is bound for loss injection.",
            });
        }
        return boundDevice->injectDeviceLoss(status);
    }

    Result<VulkanPresentationSyncSnapshot> queryVulkanPresentationSync(SwapchainHandle swapchain)
    {
        const std::scoped_lock lock(boundDeviceMutex);
        if (boundDevice == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "No Vulkan device is bound for presentation-sync inspection.",
            });
        }
        return boundDevice->presentationSyncSnapshot(swapchain);
    }

    std::size_t liveVulkanMipmapBindingPoolCount() noexcept
    {
        const std::scoped_lock lock(boundDeviceMutex);
        return boundDevice != nullptr ? boundDevice->liveMipmapBindingPoolCount() : 0;
    }

    Status test::injectCreateResult(std::int32_t vkResult)
    {
        const std::scoped_lock lock(boundDeviceMutex);
        if (boundDevice == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "No Vulkan device is bound for create-result injection.",
            });
        }
        boundDevice->injectCreateResult(static_cast<VkResult>(vkResult));
        return {};
    }

    std::vector<VkTimeDomainEXT> queryVulkanCalibrateableTimeDomains()
    {
        const std::scoped_lock lock(boundDeviceMutex);
        return boundDevice != nullptr ? boundDevice->calibrateableTimeDomains() : std::vector<VkTimeDomainEXT>{};
    }
#endif

    VkQueue VulkanDevice::queueFor(QueueType queueType) const noexcept
    {
        switch (queueType)
        {
        case QueueType::graphics:
            return queues_.graphics;
        case QueueType::compute:
            return queues_.compute;
        case QueueType::copy:
            return queues_.copy;
        }
        return VK_NULL_HANDLE;
    }

} // namespace rachis::vulkan
