#include "Metal/MetalInternal.h"

#include <mach/mach_time.h>

namespace rachis::metal
{
    namespace
    {

        [[nodiscard]] Status metalReserveStagedQueryRange(std::vector<MetalStagedQuerySlot>& staged,
                                                          QueryPoolHandle queryPool,
                                                          std::uint32_t firstQuery,
                                                          std::uint32_t queryCount)
        {
            std::size_t additions = 0;
            for (std::uint32_t offset = 0; offset < queryCount; ++offset)
            {
                const auto queryIndex = firstQuery + offset;
                const auto prior = std::ranges::find_if(staged, [&](const MetalStagedQuerySlot& slot) {
                    return slot.queryPool == queryPool && slot.queryIndex == queryIndex;
                });
                additions += prior == staged.end() ? 1 : 0;
            }
            try
            {
                staged.reserve(staged.size() + additions);
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::outOfMemory, "Timestamp-scope bookkeeping could not be allocated."));
            }
            return {};
        }

        void metalStageTimestampResolveRange(std::vector<MetalStagedQuerySlot>& staged,
                                             QueryPoolHandle queryPool,
                                             const std::shared_ptr<MetalQueryPoolRecord>& pool,
                                             std::uint32_t firstQuery,
                                             std::uint32_t queryCount) noexcept
        {
            for (std::uint32_t offset = 0; offset < queryCount; ++offset)
            {
                const auto queryIndex = firstQuery + offset;
                const auto prior = std::ranges::find_if(staged, [&](const MetalStagedQuerySlot& slot) {
                    return slot.queryPool == queryPool && slot.queryIndex == queryIndex;
                });
                if (prior != staged.end())
                {
                    prior->pendingAfter = false;
                    continue;
                }
                staged.push_back(MetalStagedQuerySlot{
                    .queryPool = queryPool,
                    .pool = pool,
                    .queryIndex = queryIndex,
                    .precondition = MetalQuerySlotPrecondition::mustBePending,
                    .pendingAfter = false,
                });
            }
        }

        void metalStageTimestampResetRange(std::vector<MetalStagedQuerySlot>& staged,
                                           QueryPoolHandle queryPool,
                                           const std::shared_ptr<MetalQueryPoolRecord>& pool,
                                           std::uint32_t firstQuery,
                                           std::uint32_t queryCount) noexcept
        {
            for (std::uint32_t offset = 0; offset < queryCount; ++offset)
            {
                const auto queryIndex = firstQuery + offset;
                const auto prior = std::ranges::find_if(staged, [&](const MetalStagedQuerySlot& slot) {
                    return slot.queryPool == queryPool && slot.queryIndex == queryIndex;
                });
                if (prior != staged.end())
                {
                    prior->pendingAfter = false;
                    continue;
                }
                staged.push_back(MetalStagedQuerySlot{
                    .queryPool = queryPool,
                    .pool = pool,
                    .queryIndex = queryIndex,
                    .precondition = MetalQuerySlotPrecondition::none,
                    .pendingAfter = false,
                });
            }
        }

        constexpr std::uint64_t metalSampledTimestampFrequency = 1'000'000'000ULL;

        [[nodiscard]] constexpr std::uint64_t scaleRoundedUp(std::uint64_t value,
                                                             std::uint32_t multiplier,
                                                             std::uint32_t divisor) noexcept
        {
            const std::uint64_t quotient = value / divisor;
            const std::uint64_t remainder = value % divisor;
            return quotient * multiplier + (remainder * multiplier + divisor - 1U) / divisor;
        }

        [[nodiscard]] constexpr std::uint64_t metalSampledTimestampToTicks(MTLTimestamp sampled,
                                                                           std::uint64_t ticksPerSecond) noexcept
        {
            const auto seconds = static_cast<std::uint64_t>(sampled) / metalSampledTimestampFrequency;
            const auto remainder = static_cast<std::uint64_t>(sampled) % metalSampledTimestampFrequency;
            return seconds * ticksPerSecond + remainder * ticksPerSecond / metalSampledTimestampFrequency;
        }

        [[nodiscard]] bool metalIsTimestampSlotPendingLocked(const MetalCommandBuffer& command,
                                                             QueryPoolHandle queryPool,
                                                             const MetalQueryPoolRecord& pool,
                                                             std::uint32_t queryIndex) noexcept
        {
            const auto& staged = command.stagedQuerySlots();
            const auto prior = std::ranges::find_if(staged, [&](const MetalStagedQuerySlot& slot) {
                return slot.queryPool == queryPool && slot.queryIndex == queryIndex;
            });
            if (prior != staged.end())
            {
                return prior->pendingAfter;
            }
            return queryIndex < pool.pendingSlots.size() && pool.pendingSlots[queryIndex] != 0;
        }

    } // namespace

    [[nodiscard]] bool metalIsTimestampSlotPending(const MetalCommandBuffer& command,
                                                   QueryPoolHandle queryPool,
                                                   const MetalQueryPoolRecord& pool,
                                                   std::uint32_t queryIndex) noexcept
    {
        const std::scoped_lock lock(pool.stateMutex);
        return metalIsTimestampSlotPendingLocked(command, queryPool, pool, queryIndex);
    }

    [[nodiscard]] Status metalReserveStagedQuerySlots(std::vector<MetalStagedQuerySlot>& staged,
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
            const auto prior = std::ranges::find_if(staged, [&](const MetalStagedQuerySlot& slot) {
                return slot.queryPool == queryPool && slot.queryIndex == queryIndex;
            });
            additions += prior == staged.end() ? 1 : 0;
        }
        try
        {
            staged.reserve(staged.size() + additions);
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::outOfMemory, "Timestamp-scope bookkeeping could not be allocated."));
        }
        return {};
    }

    void metalStageTimestampWrite(std::vector<MetalStagedQuerySlot>& staged,
                                  QueryPoolHandle queryPool,
                                  const std::shared_ptr<MetalQueryPoolRecord>& pool,
                                  std::uint32_t queryIndex) noexcept
    {
        const auto prior = std::ranges::find_if(staged, [&](const MetalStagedQuerySlot& slot) {
            return slot.queryPool == queryPool && slot.queryIndex == queryIndex;
        });
        if (prior != staged.end())
        {
            prior->pendingAfter = true;
            return;
        }
        staged.push_back(MetalStagedQuerySlot{
            .queryPool = queryPool,
            .pool = pool,
            .queryIndex = queryIndex,
            .precondition = MetalQuerySlotPrecondition::mustNotBePending,
            .pendingAfter = true,
        });
    }

    [[nodiscard]] Status metalEncodeTimestamp(MetalCommandEntry& entry,
                                              const std::shared_ptr<MetalQueryPoolRecord>& pool,
                                              std::uint32_t queryIndex,
                                              MTLRenderStages renderStage)
    {
        if (entry.command == nil || !entry.begun || pool == nullptr || pool->heap == nil)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState,
                                    "Timestamp writes require a begun Metal command buffer and a live counter heap."));
        }
        if (entry.renderEncoder != nil)
        {
            [entry.renderEncoder writeTimestampWithGranularity:MTL4TimestampGranularityPrecise
                                                    afterStage:renderStage
                                                      intoHeap:pool->heap
                                                       atIndex:queryIndex];
#ifdef RACHIS_TESTING
            recordMetalRenderTimestampStage(renderStage);
#endif
            return {};
        }
        if (entry.computeEncoder != nil)
        {
            [entry.computeEncoder writeTimestampWithGranularity:MTL4TimestampGranularityPrecise
                                                       intoHeap:pool->heap
                                                        atIndex:queryIndex];
            return {};
        }
        if (auto flushed = flushMetalDeferredBarriers(entry); !flushed)
        {
            return flushed;
        }
        [entry.command writeTimestampIntoHeap:pool->heap atIndex:queryIndex];
        return {};
    }

    Result<QueueClockCalibration> MetalDevice::calibrateQueueClock(QueueType queueType) const
    {
        if (queueType != QueueType::graphics && queueType != QueueType::compute && queueType != QueueType::copy)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "Queue clock calibration requires a valid queue type."));
        }
        const auto properties = queueProperties(queueType);
        if (!properties.timestampQueries || properties.timestampFrequency == 0)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::unsupported, "The selected queue does not expose a calibratable timestamp clock."));
        }
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || native_->device == nil)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal calibrateQueueClock requires an active device."));
        }
        mach_timebase_info_data_t timebase{};
        if (mach_timebase_info(&timebase) != KERN_SUCCESS || timebase.numer == 0 || timebase.denom == 0)
        {
            return std::unexpected(metalSelectionError(ErrorCode::nativeFailure,
                                                       "mach_timebase_info failed during queue clock calibration."));
        }
        MTLTimestamp cpuTimestamp = 0;
        MTLTimestamp gpuTimestamp = 0;
        const auto before = mach_absolute_time();
        [native_->device sampleTimestamps:&cpuTimestamp gpuTimestamp:&gpuTimestamp];
        const auto after = mach_absolute_time();
        const auto cpuTicks =
            std::clamp(scaleRoundedUp(cpuTimestamp, timebase.denom, timebase.numer), before, std::max(before, after));
        const auto ticksPerSecond = metalSampledTimestampFrequency * timebase.denom / timebase.numer;
        return QueueClockCalibration{
            .gpuTimestamp = metalSampledTimestampToTicks(gpuTimestamp, properties.timestampFrequency),
            .cpuTimestamp = cpuTicks,
            .gpuFrequency = properties.timestampFrequency,
            .cpuFrequency = ticksPerSecond,
            .maxDeviationNanoseconds =
                scaleRoundedUp(std::max(before, after) - before + 1U, timebase.numer, timebase.denom),
            .cpuClock = CpuClock::machAbsoluteTime,
        };
    }

    Result<QueryPoolHandle> MetalDevice::createTimestampQueryPool(const TimestampQueryPoolDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if ((desc.queueType != QueueType::graphics && desc.queueType != QueueType::compute &&
             desc.queueType != QueueType::copy) ||
            desc.queryCount == 0)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument,
                                    "A timestamp query pool requires a valid queue type and a non-zero query count."));
        }
        if (!queueProperties(desc.queueType).timestampQueries)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::unsupported, "The selected queue does not support timestamp queries."));
        }
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || native_->device == nil)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                       "Metal createTimestampQueryPool requires an active device."));
        }
        @autoreleasepool
        {
            MTL4CounterHeapDescriptor* const heapDesc = [MTL4CounterHeapDescriptor new];
            heapDesc.type = MTL4CounterHeapTypeTimestamp;
            heapDesc.count = desc.queryCount;
            NSError* error = nil;
            id<MTL4CounterHeap> const heap = [native_->device newCounterHeapWithDescriptor:heapDesc error:&error];
            if (heap == nil)
            {
                return std::unexpected(metalNsError(ErrorCode::outOfMemory, error, "newCounterHeapWithDescriptor"));
            }
            if (!desc.label.empty())
            {
                heap.label = metalNsString(desc.label);
            }
            auto record = std::make_shared<MetalQueryPoolRecord>();
            record->kind = MetalQueryPoolKind::timestamp;
            record->queueType = desc.queueType;
            record->queryCount = desc.queryCount;
            record->entrySize = [native_->device sizeOfCounterHeapEntry:MTL4CounterHeapTypeTimestamp];
            record->label = std::string(desc.label);
            record->heap = heap;
            record->pendingSlots.assign(desc.queryCount, 0U);
            if (record->entrySize != timestampQueryResultSize)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::unsupported, "Metal timestamp heap entries are not 8-byte portable timestamp ticks."));
            }
            return native_->queryPools.emplace(std::move(record));
        }
    }

    Status MetalDevice::destroyQueryPool(QueryPoolHandle queryPool)
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal destroyQueryPool requires an active device."));
        }
        if (native_->queryPools.find(queryPool) == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The query pool handle is not a live Metal timestamp pool."));
        }
        if (!native_->queryPools.erase(queryPool))
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The query pool handle is not a live Metal timestamp pool."));
        }
        return {};
    }

    Status MetalDevice::resetQueryPool(detail::BackendCommandBuffer& commandBuffer,
                                       QueryPoolHandle queryPool,
                                       std::uint32_t firstQuery,
                                       std::uint32_t queryCount)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (queryCount == 0)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "Query-pool reset requires a non-empty in-bounds range."));
        }
        std::uint64_t deviceToken = 0;
        std::shared_ptr<MetalQueryPoolRecord> pool;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal resetQueryPool requires an active device."));
            }
            deviceToken = native_->deviceToken;
            auto* found = native_->queryPools.find(queryPool);
            pool = found == nullptr ? nullptr : *found;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        if (pool == nullptr || pool->kind != MetalQueryPoolKind::timestamp || pool->queueType != native->queueType() ||
            firstQuery >= pool->queryCount || queryCount > pool->queryCount - firstQuery)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument, "Query-pool reset requires a matching queue and an in-bounds range."));
        }
        auto owner = metalRecordingPool(*native);
        if (!owner)
        {
            return std::unexpected(std::move(owner.error()));
        }
        const std::scoped_lock lock((*owner)->mutex);
        if (auto ready = metalRecordingReady(*native); !ready)
        {
            return ready;
        }
        if (native->graphicsState().renderingOwner != nullptr ||
            native->queryState().activePassTimestampWrites.has_value() ||
            native->queryState().activeGpuProfileScopeDepth != 0)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState,
                "Query-pool reset cannot run inside a rendering scope or an open GPU profile scope."));
        }
        auto reserved = metalReserveStagedQueryRange(native->stagedQuerySlots(), queryPool, firstQuery, queryCount);
        if (!reserved)
        {
            return reserved;
        }
        auto kept = retainMetalApplicationResource(*native, pool);
        if (!kept)
        {
            return kept;
        }
        metalStageTimestampResetRange(native->stagedQuerySlots(), queryPool, pool, firstQuery, queryCount);
        return {};
    }

    Status MetalDevice::writeTimestamp(detail::BackendCommandBuffer& commandBuffer,
                                       QueryPoolHandle queryPool,
                                       std::uint32_t queryIndex)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        std::uint64_t deviceToken = 0;
        std::shared_ptr<MetalQueryPoolRecord> pool;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal writeTimestamp requires an active device."));
            }
            deviceToken = native_->deviceToken;
            auto* found = native_->queryPools.find(queryPool);
            pool = found == nullptr ? nullptr : *found;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        if (pool == nullptr || pool->kind != MetalQueryPoolKind::timestamp || pool->queueType != native->queueType() ||
            queryIndex >= pool->queryCount)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument,
                                    "A timestamp write requires a matching queue and an in-bounds query index."));
        }
        if (!queueProperties(native->queueType()).inlineTimestampWrites)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::unsupported, "The command buffer queue does not support inline timestamp writes."));
        }
        auto owner = metalRecordingPool(*native);
        if (!owner)
        {
            return std::unexpected(std::move(owner.error()));
        }
        const std::scoped_lock lock(pool->stateMutex, (*owner)->mutex);
        if (auto ready = metalRecordingReady(*native); !ready)
        {
            return ready;
        }
        if (metalIsTimestampSlotPendingLocked(*native, queryPool, *pool, queryIndex))
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState,
                                    "A timestamp query cannot be written twice before its first value is resolved."));
        }
        const std::array queryIndices{queryIndex};
        auto reserved = metalReserveStagedQuerySlots(native->stagedQuerySlots(), queryPool, queryIndices);
        if (!reserved)
        {
            return reserved;
        }
        auto kept = retainMetalApplicationResource(*native, pool);
        if (!kept)
        {
            return kept;
        }
        auto encoded = metalEncodeTimestamp(*native->entry(), pool, queryIndex, MTLRenderStageFragment);
        if (!encoded)
        {
            return encoded;
        }
        metalStageTimestampWrite(native->stagedQuerySlots(), queryPool, pool, queryIndex);
        return {};
    }

    Status MetalDevice::beginGpuProfileScope(detail::BackendCommandBuffer& commandBuffer,
                                             const GpuProfileScopeDesc& desc)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        auto marker = validateMetalMarkerDesc({.label = desc.label, .color = desc.color});
        if (!marker)
        {
            return marker;
        }
        std::uint64_t deviceToken = 0;
        std::shared_ptr<MetalQueryPoolRecord> pool;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                           "Metal beginGpuProfileScope requires an active device."));
            }
            deviceToken = native_->deviceToken;
            auto* found = native_->queryPools.find(desc.queryPool);
            pool = found == nullptr ? nullptr : *found;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        if (pool == nullptr || pool->kind != MetalQueryPoolKind::timestamp || pool->queueType != native->queueType() ||
            desc.beginningQueryIndex >= pool->queryCount || desc.endQueryIndex >= pool->queryCount ||
            desc.beginningQueryIndex == desc.endQueryIndex)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument,
                "A GPU profile scope requires a matching timestamp pool and two distinct in-bounds query indices."));
        }
        auto owner = metalRecordingPool(*native);
        if (!owner)
        {
            return std::unexpected(std::move(owner.error()));
        }
        const std::scoped_lock lock(pool->stateMutex, (*owner)->mutex);
        if (auto ready = metalRecordingReady(*native); !ready)
        {
            return ready;
        }
        auto& queryState = native->queryState();
        auto& debugState = native->debugState();
        if (queryState.activeGpuProfileScopeDepth == maxDebugGroupDepth ||
            debugState.groupDepth == maxDebugGroupDepth ||
            metalIsTimestampSlotPendingLocked(*native, desc.queryPool, *pool, desc.beginningQueryIndex) ||
            metalIsTimestampSlotPendingLocked(*native, desc.queryPool, *pool, desc.endQueryIndex))
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState,
                "GPU profile scope nesting is exhausted or one of its timestamp indices is unresolved."));
        }
        if (debugState.groupDepth != 0 && debugState.groupThread != std::this_thread::get_id())
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState,
                "Nested debug and GPU profile groups must be recorded on the thread that opened the outer group."));
        }
        const std::array queryIndices{desc.beginningQueryIndex, desc.endQueryIndex};
        auto reserved = metalReserveStagedQuerySlots(native->stagedQuerySlots(), desc.queryPool, queryIndices);
        if (!reserved)
        {
            return reserved;
        }
        auto kept = retainMetalApplicationResource(*native, pool);
        if (!kept)
        {
            return kept;
        }
        auto* entry = native->entry();
        auto encoded = metalEncodeTimestamp(*entry, pool, desc.beginningQueryIndex, MTLRenderStageVertex);
        if (!encoded)
        {
            return encoded;
        }
        [entry->command pushDebugGroup:metalNsString(desc.label)];
        metalStageTimestampWrite(native->stagedQuerySlots(), desc.queryPool, pool, desc.beginningQueryIndex);
        metalStageTimestampWrite(native->stagedQuerySlots(), desc.queryPool, pool, desc.endQueryIndex);
        if (debugState.groupDepth == 0)
        {
            debugState.groupThread = std::this_thread::get_id();
        }
        debugState.groupOwners[debugState.groupDepth] = detail::DebugGroupOwner::gpuProfile;
        ++debugState.groupDepth;
        queryState.activeGpuProfileScopes[queryState.activeGpuProfileScopeDepth] = detail::ActiveGpuProfileScope{
            .queryPool = desc.queryPool,
            .endQueryIndex = desc.endQueryIndex,
            .owner = pool,
        };
        ++queryState.activeGpuProfileScopeDepth;
        return {};
    }

    Status MetalDevice::endGpuProfileScope(detail::BackendCommandBuffer& commandBuffer)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        std::uint64_t deviceToken = 0;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                           "Metal endGpuProfileScope requires an active device."));
            }
            deviceToken = native_->deviceToken;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        auto owner = metalRecordingPool(*native);
        if (!owner)
        {
            return std::unexpected(std::move(owner.error()));
        }
        const std::scoped_lock lock((*owner)->mutex);
        if (auto ready = metalRecordingReady(*native); !ready)
        {
            return ready;
        }
        auto& queryState = native->queryState();
        auto& debugState = native->debugState();
        if (queryState.activeGpuProfileScopeDepth == 0 || debugState.groupDepth == 0 ||
            debugState.groupOwners[debugState.groupDepth - 1] != detail::DebugGroupOwner::gpuProfile)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState, "Ending a GPU profile scope requires a matching open profile group."));
        }
        if (debugState.groupThread != std::this_thread::get_id())
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState, "A debug or GPU profile group must end on the thread that opened it."));
        }
        auto* entry = native->entry();
        auto& active = queryState.activeGpuProfileScopes[queryState.activeGpuProfileScopeDepth - 1];
        const auto pool = std::static_pointer_cast<MetalQueryPoolRecord>(active.owner);
        auto encoded = metalEncodeTimestamp(*entry, pool, active.endQueryIndex, MTLRenderStageFragment);
        if (!encoded)
        {
            return encoded;
        }
        [entry->command popDebugGroup];
        --debugState.groupDepth;
        active = {};
        --queryState.activeGpuProfileScopeDepth;
        return {};
    }

    namespace
    {

        struct MetalTimestampResolveTarget final
        {
            std::shared_ptr<MetalQueryPoolRecord> pool;
            __strong id<MTLBuffer> destination = nil;
            std::uint64_t destinationSize = 0;
            std::shared_ptr<void> destinationOwner;
        };

        [[nodiscard]] Status lookupMetalTimestampResolveTarget(MetalDeviceNative& native,
                                                               const TimestampQueryResolveDesc& desc,
                                                               MetalTimestampResolveTarget& target)
        {
            auto* found = native.queryPools.find(desc.queryPool);
            target.pool = found == nullptr ? nullptr : *found;
            auto* buffer = native.buffers.find(desc.destination);
            if (buffer == nullptr || buffer->buffer == nil || !hasAny(buffer->desc.usage, BufferUsage::copyDestination))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "Timestamp resolve requires a copy-destination buffer large enough for the query range."));
            }
            target.destination = buffer->buffer;
            target.destinationSize = buffer->desc.size;
            target.destinationOwner = buffer->owner;
            return {};
        }

        [[nodiscard]] Result<std::uint64_t> validateMetalTimestampResolveRange(
            const MetalCommandBuffer& native,
            const TimestampQueryResolveDesc& desc,
            const MetalTimestampResolveTarget& target)
        {
            const auto& pool = target.pool;
            if (pool == nullptr || pool->kind != MetalQueryPoolKind::timestamp ||
                pool->queueType != native.queueType() || desc.firstQuery >= pool->queryCount ||
                desc.queryCount > pool->queryCount - desc.firstQuery)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "Timestamp resolve requires a matching queue and an in-bounds query range."));
            }
            const auto requiredBytes = static_cast<std::uint64_t>(desc.queryCount) * pool->entrySize;
            if (desc.destinationOffset > target.destinationSize ||
                requiredBytes > target.destinationSize - desc.destinationOffset)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "Timestamp resolve requires a copy-destination buffer large enough for the query range."));
            }
            return requiredBytes;
        }

        [[nodiscard]] Status validateMetalTimestampResolveState(const MetalCommandBuffer& native,
                                                                const TimestampQueryResolveDesc& desc,
                                                                const MetalQueryPoolRecord& pool)
        {
            if (native.graphicsState().renderingOwner != nullptr ||
                native.queryState().activePassTimestampWrites.has_value() ||
                native.queryState().activeGpuProfileScopeDepth != 0)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidState,
                    "Timestamp resolve cannot run inside a rendering scope or an open GPU profile scope."));
            }
            for (std::uint32_t index = 0; index < desc.queryCount; ++index)
            {
                if (!metalIsTimestampSlotPendingLocked(native, desc.queryPool, pool, desc.firstQuery + index))
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidState,
                        "Timestamp resolve requires every slot to have been written after its last reset."));
                }
            }
            return {};
        }

        [[nodiscard]] Status retainMetalTimestampResolveResources(MetalCommandBuffer& native,
                                                                  const TimestampQueryResolveDesc& desc,
                                                                  const MetalTimestampResolveTarget& target)
        {
            auto reserved = metalReserveStagedQueryRange(
                native.stagedQuerySlots(), desc.queryPool, desc.firstQuery, desc.queryCount);
            if (!reserved)
            {
                return reserved;
            }
            auto keptPool = retainMetalApplicationResource(native, target.pool);
            if (!keptPool)
            {
                return keptPool;
            }
            auto keptDestination = retainMetalApplicationResource(native, target.destinationOwner);
            if (!keptDestination)
            {
                return keptDestination;
            }
            return {};
        }

        [[nodiscard]] Status encodeMetalTimestampResolve(MetalCommandBuffer& native,
                                                         const TimestampQueryResolveDesc& desc,
                                                         const MetalTimestampResolveTarget& target,
                                                         std::uint64_t requiredBytes)
        {
            auto* entry = native.entry();
            if (entry == nullptr || entry->command == nil || !entry->begun)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                           "Timestamp resolve requires a begun Metal command buffer."));
            }
            native.endEncoders();
            auto prepared = native.ensureCopyEncoder();
            if (!prepared)
            {
                return prepared;
            }
            auto hazard = native.encodeStageBarrier(MTLStageAll, MTLStageBlit);
            if (!hazard)
            {
                return hazard;
            }
            native.endEncoders();
            metalRetainResidency(*entry, target.destination);
            [entry->command
                resolveCounterHeap:target.pool->heap
                         withRange:NSMakeRange(desc.firstQuery, desc.queryCount)
                        intoBuffer:MTL4BufferRangeMake(target.destination.gpuAddress + desc.destinationOffset,
                                                       requiredBytes)
                         waitFence:nil
                       updateFence:nil];
            metalStageTimestampResolveRange(
                native.stagedQuerySlots(), desc.queryPool, target.pool, desc.firstQuery, desc.queryCount);
            return {};
        }

    } // namespace

    Status MetalDevice::resolveTimestamps(detail::BackendCommandBuffer& commandBuffer,
                                          const TimestampQueryResolveDesc& desc)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (desc.queryCount == 0 || desc.destinationOffset % timestampQueryResultAlignment != 0)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument,
                                    "Timestamp resolve requires a non-empty range and an aligned destination offset."));
        }
        std::uint64_t deviceToken = 0;
        MetalTimestampResolveTarget target;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal resolveTimestamps requires an active device."));
            }
            deviceToken = native_->deviceToken;
            if (auto found = lookupMetalTimestampResolveTarget(*native_, desc, target); !found)
            {
                return found;
            }
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        auto requiredBytes = validateMetalTimestampResolveRange(*native, desc, target);
        if (!requiredBytes)
        {
            return std::unexpected(std::move(requiredBytes.error()));
        }
        auto owner = metalRecordingPool(*native);
        if (!owner)
        {
            return std::unexpected(std::move(owner.error()));
        }
        const std::scoped_lock lock(target.pool->stateMutex, (*owner)->mutex);
        if (auto ready = metalRecordingReady(*native); !ready)
        {
            return ready;
        }
        if (auto valid = validateMetalTimestampResolveState(*native, desc, *target.pool); !valid)
        {
            return valid;
        }
        if (auto retained = retainMetalTimestampResolveResources(*native, desc, target); !retained)
        {
            return retained;
        }
        return encodeMetalTimestampResolve(*native, desc, target, *requiredBytes);
    }

} // namespace rachis::metal
