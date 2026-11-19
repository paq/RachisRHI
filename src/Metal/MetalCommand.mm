#include "Metal/MetalInternal.h"

#include "Common/VectorReserve.h"
#include "Internal/PortableValidation.h"
#ifdef RACHIS_TESTING
#include "Internal/CommandPoolRetirementTestHook.h"
#endif

namespace rachis::metal
{
    namespace
    {

        struct MetalRecordingRetentions final
        {
            __strong id<MTLResidencySet> residency = nil;
            std::vector<MetalWaitedDrawable> waitedDrawables;
            std::vector<std::shared_ptr<MetalBindingSetVersion>> boundComputeVersions;
            std::vector<std::shared_ptr<MetalBindingSetVersion>> boundGraphicsVersions;
            std::vector<std::shared_ptr<MetalArgumentBufferContents>> foldedArgumentContents;
            std::shared_ptr<MetalPipelineRecord> boundComputePipeline;
            std::shared_ptr<MetalPipelineRecord> boundGraphicsPipeline;
        };

        [[nodiscard]] MetalRecordingRetentions takeMetalRecordingRetentions(MetalCommandEntry& entry) noexcept
        {
            MetalRecordingRetentions retentions{
                .residency = entry.residency,
                .waitedDrawables = std::move(entry.waitedDrawables),
                .boundComputeVersions = std::move(entry.boundComputeVersions),
                .boundGraphicsVersions = std::move(entry.boundGraphicsVersions),
                .foldedArgumentContents = std::move(entry.foldedArgumentContents),
                .boundComputePipeline = std::move(entry.boundComputePipeline),
                .boundGraphicsPipeline = std::move(entry.boundGraphicsPipeline),
            };
            entry.residency = nil;
            entry.waitedDrawables.clear();
            entry.boundComputeVersions.clear();
            entry.boundGraphicsVersions.clear();
            entry.foldedArgumentContents.clear();
            return retentions;
        }

        [[nodiscard]] std::uint64_t metalWaitTimeoutMs(std::chrono::nanoseconds timeout) noexcept
        {
            if (timeout <= std::chrono::nanoseconds::zero())
            {
                return 0;
            }
            if (timeout == std::chrono::nanoseconds::max())
            {
                return std::numeric_limits<std::uint64_t>::max();
            }
            const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(timeout);
            if (milliseconds.count() <= 0)
            {
                return 1;
            }
            return static_cast<std::uint64_t>(milliseconds.count());
        }

        void resolveMetalDeferredStoreActions(MetalCommandEntry& entry) noexcept
        {
            auto& stores = entry.renderStoreActions;
            if (entry.renderEncoder != nil && stores.pending)
            {
                for (std::uint32_t index = 0; index < stores.colorCount; ++index)
                {
                    [entry.renderEncoder setColorStoreAction:stores.colors[index] atIndex:index];
                }
                if (stores.hasDepth)
                {
                    [entry.renderEncoder setDepthStoreAction:stores.depth];
                }
                if (stores.hasStencil)
                {
                    [entry.renderEncoder setStencilStoreAction:stores.stencil];
                }
            }
            stores = {};
        }

        void restoreMetalQuarantinedEntries(MetalCommandPoolState& pool) noexcept
        {
            const auto restorable = [](const MetalCommandEntry* entry) noexcept {
                return entry != nullptr && entry->allocator != nil && entry->command != nil &&
                       entry->lease == nullptr && !entry->inFlight;
            };
            for (auto* entry : pool.quarantined)
            {
                if (restorable(entry))
                {
                    pool.available.push_back(entry);
                }
            }
            std::erase_if(pool.quarantined, restorable);
        }

#ifdef RACHIS_TESTING
        [[nodiscard]] bool consumeInjectedMetalResidencySetFailure() noexcept
        {
            std::scoped_lock const testLock(metalTestingState().mutex);
            return std::exchange(metalTestingState().failNextResidencySetCreate, false);
        }
#endif

        void clearMetalStageArgumentTables(MetalCommandEntry& entry) noexcept
        {
            for (const auto& stage : entry.stageArgumentTables)
            {
                if (stage.table == nil)
                {
                    continue;
                }
                for (std::uint32_t i = 0; i < stage.maxBuffers; ++i)
                {
                    [stage.table setAddress:0 atIndex:i];
                }
                for (std::uint32_t i = 0; i < stage.maxTextures; ++i)
                {
                    [stage.table setTexture:MTLResourceID {} atIndex:i];
                }
                for (std::uint32_t i = 0; i < stage.maxSamplers; ++i)
                {
                    [stage.table setSamplerState:MTLResourceID {} atIndex:i];
                }
            }
            entry.supersededArgumentTables.clear();
        }

    } // namespace

    [[nodiscard]] rachis::Status validateMetalMarkerDesc(const rachis::DebugMarkerDesc& desc)
    {
        return detail::validateDebugMarkerDesc(desc);
    }

    void flushMetalComputePendingBarriers(MetalCommandEntry& entry) noexcept
    {
        if (entry.computeEncoder == nil || entry.pendingBarrierBeforeStages == static_cast<MTLStages>(0))
        {
            return;
        }
        const auto after = entry.pendingBarrierAfterStages == static_cast<MTLStages>(0)
                               ? MTLStageAll
                               : entry.pendingBarrierAfterStages;
        const auto before = entry.pendingBarrierBeforeStages;
        const auto beforeCompute = static_cast<MTLStages>(before & kMetalComputeEncoderStages);
        if (beforeCompute != static_cast<MTLStages>(0))
        {
            [entry.computeEncoder barrierAfterQueueStages:after
                                             beforeStages:beforeCompute
                                        visibilityOptions:MTL4VisibilityOptionDevice];
        }
        [entry.computeEncoder barrierAfterStages:after
                               beforeQueueStages:before
                               visibilityOptions:MTL4VisibilityOptionDevice];
        entry.pendingBarrierAfterStages = static_cast<MTLStages>(0);
        entry.pendingBarrierBeforeStages = static_cast<MTLStages>(0);
    }

    void flushMetalRenderPendingBarriers(MetalCommandEntry& entry) noexcept
    {
        if (entry.renderEncoder == nil || entry.pendingBarrierBeforeStages == static_cast<MTLStages>(0))
        {
            return;
        }
        const auto after = entry.pendingBarrierAfterStages == static_cast<MTLStages>(0)
                               ? MTLStageAll
                               : entry.pendingBarrierAfterStages;
        const auto before = entry.pendingBarrierBeforeStages;
        const auto beforeRender = static_cast<MTLStages>(before & kMetalRenderEncoderStages);
        if (beforeRender != static_cast<MTLStages>(0))
        {
            [entry.renderEncoder barrierAfterQueueStages:after
                                            beforeStages:beforeRender
                                       visibilityOptions:MTL4VisibilityOptionDevice];
        }
        [entry.renderEncoder barrierAfterStages:after
                              beforeQueueStages:before
                              visibilityOptions:MTL4VisibilityOptionDevice];
        entry.pendingBarrierAfterStages = static_cast<MTLStages>(0);
        entry.pendingBarrierBeforeStages = static_cast<MTLStages>(0);
    }

    [[nodiscard]] Status flushMetalDeferredBarriers(MetalCommandEntry& entry)
    {
        if (!entry.begun || entry.command == nil || entry.pendingBarrierBeforeStages == static_cast<MTLStages>(0))
        {
            return {};
        }
        if (entry.computeEncoder != nil)
        {
            flushMetalComputePendingBarriers(entry);
            return {};
        }
        if (entry.renderEncoder != nil)
        {
            flushMetalRenderPendingBarriers(entry);
            return {};
        }
        id<MTL4ComputeCommandEncoder> const barrierEncoder = [entry.command computeCommandEncoder];
        if (barrierEncoder == nil)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::nativeFailure, "Metal could not create a compute encoder to flush deferred barriers."));
        }
        barrierEncoder.label = @"RHI.Metal.DeferredBarrier";
        const auto after = entry.pendingBarrierAfterStages == static_cast<MTLStages>(0)
                               ? MTLStageAll
                               : entry.pendingBarrierAfterStages;
        [barrierEncoder barrierAfterStages:after
                         beforeQueueStages:entry.pendingBarrierBeforeStages
                         visibilityOptions:MTL4VisibilityOptionDevice];
        [barrierEncoder endEncoding];
        entry.pendingBarrierAfterStages = static_cast<MTLStages>(0);
        entry.pendingBarrierBeforeStages = static_cast<MTLStages>(0);
        return {};
    }

    void endMetalEncoders(MetalCommandEntry& entry) noexcept
    {
        if (entry.computeEncoder != nil)
        {
            [entry.computeEncoder endEncoding];
            entry.computeEncoder = nil;
        }
        resolveMetalDeferredStoreActions(entry);
        if (entry.renderEncoder != nil)
        {
            [entry.renderEncoder endEncoding];
            entry.renderEncoder = nil;
        }
        entry.encoder = MetalEncoderKind::none;
        entry.computeTableDirty = true;
        entry.vertexTableDirty = true;
        entry.fragmentTableDirty = true;
        entry.activeComputeTable = nil;
        entry.activeVertexTable = nil;
        entry.activeFragmentTable = nil;
        entry.activeComputePipeline = nullptr;
        entry.activeGraphicsPipeline = nullptr;
    }

    void endMetalRecording(MetalCommandEntry& entry) noexcept
    {
        endMetalEncoders(entry);
        if (entry.begun && entry.command != nil)
        {
            [entry.command endCommandBuffer];
        }
        entry.begun = false;
    }

    Status resetMetalRecording(MetalCommandEntry& entry)
    {
        const MetalAutoreleasePool autoreleasePool;
        endMetalRecording(entry);
        entry.residencyBound = false;
        entry.residencyDirty = false;
        entry.residency = nil;
        entry.residentAllocations.clear();
        entry.usedResidencySets.clear();
        entry.argumentResidencySetCount = 0;
        entry.foldedArgumentSets.clear();
        entry.foldedArgumentContents.clear();
        entry.pendingBarrierAfterStages = static_cast<MTLStages>(0);
        entry.pendingBarrierBeforeStages = static_cast<MTLStages>(0);
        entry.waitedDrawables.clear();
        entry.boundComputeVersions.clear();
        entry.boundGraphicsVersions.clear();
        clearMetalStageArgumentTables(entry);
        entry.computePushConstants = {};
        entry.computePushConstantBytes = 0;
        entry.computePushConstantsDirty = true;
        entry.activeComputePushSlice = {};
        entry.graphicsPushConstants = {};
        entry.graphicsPushConstantBytes = 0;
        entry.graphicsPushConstantsDirty = true;
        entry.activeGraphicsPushSlice = {};
        entry.pushConstantPageIndex = 0;
        entry.indirectScratchBuffers.clear();
        entry.drawIndex = 0;
        entry.drawIndexAddress = 0;
        entry.pushConstantPageOffset = 0;
        entry.boundComputePipeline.reset();
        entry.boundGraphicsPipeline.reset();
        entry.vertexBuffers = {};
        entry.indexBuffer = {};
        entry.indexFormat = IndexFormat::uint16;
        entry.viewports = {};
        entry.scissors = {};
        entry.scissorsEmpty = false;
        entry.blendConstants = {};
        entry.stencilReference = 0;
        if (entry.allocator != nil)
        {
            [entry.allocator reset];
        }
        return {};
    }

    Status beginMetalRecording(MetalCommandPoolState& pool, MetalCommandEntry& entry)
    {
        if (entry.command == nil || entry.allocator == nil)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState, "Metal command recording requires a live allocator and command buffer."));
        }
        if (pool.logState != nil)
        {
            MTL4CommandBufferOptions* const options = [MTL4CommandBufferOptions new];
            options.logState = pool.logState;
            [entry.command beginCommandBufferWithAllocator:entry.allocator options:options];
        }
        else
        {
            [entry.command beginCommandBufferWithAllocator:entry.allocator];
        }
        if (!pool.label.empty())
        {
            entry.command.label = metalNsString(pool.label);
        }
        entry.begun = true;
        entry.encoder = MetalEncoderKind::none;
        entry.computeEncoder = nil;
        entry.renderEncoder = nil;
        entry.renderStoreActions = {};
        MTLResidencySetDescriptor* const residencyDesc = [MTLResidencySetDescriptor new];
        residencyDesc.initialCapacity = 16;
        NSError* residencyError = nil;
        entry.residency = [pool.device newResidencySetWithDescriptor:residencyDesc error:&residencyError];
#ifdef RACHIS_TESTING
        if (consumeInjectedMetalResidencySetFailure())
        {
            entry.residency = nil;
        }
#endif
        entry.residencyBound = false;
        entry.residencyDirty = false;
        entry.residentAllocations.clear();
        entry.usedResidencySets.clear();
        entry.argumentResidencySetCount = 0;
        entry.foldedArgumentSets.clear();
        if (entry.residency == nil)
        {
            endMetalRecording(entry);
            return std::unexpected(
                metalNsError(ErrorCode::nativeFailure, residencyError, "newResidencySetWithDescriptor"));
        }
        return {};
    }

    MetalCommandBuffer* asMetalCommand(detail::BackendCommandBuffer& commandBuffer, std::uint64_t deviceToken) noexcept
    {
        if (commandBuffer.kind() != detail::BackendCommandBufferKind::metal || deviceToken == 0 ||
            commandBuffer.deviceToken() != deviceToken)
        {
            return nullptr;
        }
        auto* native = static_cast<MetalCommandBuffer*>(&commandBuffer);
        if (native->owner() == nullptr || native->entry() == nullptr)
        {
            return nullptr;
        }
        return native;
    }

    [[nodiscard]] Status retainMetalApplicationResource(MetalCommandBuffer& command, std::shared_ptr<void> resource)
    {
        if (resource == nullptr)
        {
            return {};
        }
        if (command.owner() != nullptr && hasAny(command.owner()->flags, CommandPoolFlag::unretainedResources))
        {
            return {};
        }
        try
        {
            command.retained().push_back(std::move(resource));
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::outOfMemory, "Failed to retain a Metal command-buffer resource."));
        }
        return {};
    }

    [[nodiscard]] Status retainMetalUtilityResource(MetalCommandBuffer& command, std::shared_ptr<void> resource)
    {
        if (resource == nullptr)
        {
            return {};
        }
        try
        {
            command.retained().push_back(std::move(resource));
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::outOfMemory, "Failed to retain a Metal image-utility resource."));
        }
        return {};
    }

    [[nodiscard]] Status metalRecordingReady(const MetalCommandBuffer& native)
    {
        auto* entry = native.entry();
        if (entry == nullptr || native.owner() == nullptr || entry->lease != &native)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument, "The command buffer does not hold a live command-pool recording lease."));
        }
        if (entry->inFlight)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                       "A command buffer cannot be submitted while it is in flight."));
        }
        if (!entry->begun || entry->command == nil)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState, "A command buffer cannot be submitted before recording has begun."));
        }
        return {};
    }

    Result<std::shared_ptr<MetalCommandPoolState>> metalRecordingPool(const MetalCommandBuffer& native)
    {
        auto owner = native.ownerShared();
        if (owner == nullptr)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument, "The command buffer does not hold a live command-pool recording lease."));
        }
        return owner;
    }

    void metalRetainResidency(MetalCommandEntry& entry, id<MTLAllocation> allocation) noexcept
    {
        if (entry.residency == nil || allocation == nil)
        {
            return;
        }
        const void* const raw = (__bridge const void*)allocation;
        if (!entry.residentAllocations.insertIfAbsent(raw))
        {
#ifdef RACHIS_TESTING
            recordMetalResidencySkip();
#endif
            return;
        }
        [entry.residency addAllocation:allocation];
        entry.residencyDirty = true;
#ifdef RACHIS_TESTING
        recordMetalResidencyAdd();
#endif
        if (!entry.residencyBound && entry.command != nil)
        {
            [entry.command useResidencySet:entry.residency];
            entry.residencyBound = true;
        }
    }

    void metalRetainBindingSetResidency(MetalCommandEntry& entry, const MetalBindingSetVersion& version) noexcept
    {
        const auto& argument = version.argumentNative;
        if (argument != nullptr && argument->residency != nil && entry.command != nil &&
            !entry.usedResidencySets.contains((__bridge const void*)argument->residency))
        {
            if (entry.argumentResidencySetCount < kMetalMaxCommandArgumentResidencySets)
            {
                static_cast<void>(entry.usedResidencySets.insertIfAbsent((__bridge const void*)argument->residency));
                [entry.command useResidencySet:argument->residency];
                ++entry.argumentResidencySetCount;
            }
            else if (version.argumentContents != nullptr &&
                     entry.foldedArgumentSets.insertIfAbsent(version.argumentContents.get()))
            {
                metalRetainResidency(entry, argument->buffer);
                if (entry.foldedArgumentContents.size() < entry.foldedArgumentContents.capacity())
                {
                    entry.foldedArgumentContents.push_back(version.argumentContents);
                }
            }
        }
        for (const auto& slot : version.slots)
        {
            if (slot.buffer != nil)
            {
                metalRetainResidency(entry, slot.buffer);
            }
            if (slot.texture != nil)
            {
                metalRetainResidency(entry, slot.texture);
            }
        }
    }

    Status metalReserveBindingSetResidency(MetalCommandEntry& entry, std::size_t versionCount)
    {
#ifdef RACHIS_TESTING
        if (consumeInjectedMetalBindingResidencyReserveFailure())
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::outOfMemory, "Injected Metal binding-array residency failure."));
        }
#endif
        try
        {
            detail::reserveAdditional(entry.foldedArgumentContents, versionCount);
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::outOfMemory, "Failed to reserve Metal binding-array residency."));
        }
        return {};
    }

    void metalFoldArgumentResidency(MetalCommandEntry& entry) noexcept
    {
        for (const auto& contents : entry.foldedArgumentContents)
        {
            const std::scoped_lock lock(*contents->mutex);
            for (const auto& [key, count] : contents->residentElements)
            {
                if (count != 0)
                {
                    metalRetainResidency(entry, (__bridge id<MTLAllocation>)const_cast<void*>(key));
                }
            }
        }
    }

    void commitMetalResidency(MetalCommandEntry& entry) noexcept
    {
        if (entry.residency != nil && entry.residencyDirty)
        {
            [entry.residency commit];
            entry.residencyDirty = false;
#ifdef RACHIS_TESTING
            recordMetalResidencyCommit();
#endif
        }
    }

    Result<id<MTL4ArgumentTable>> acquireMetalArgumentTable(id<MTLDevice> device,
                                                            MetalCommandEntry& entry,
                                                            MetalArgumentTableStage stage,
                                                            std::uint32_t maxBuffers,
                                                            std::uint32_t maxTextures,
                                                            std::uint32_t maxSamplers)
    {
        auto& current = entry.stageArgumentTables[static_cast<std::size_t>(stage)];
        if (current.table != nil && current.maxBuffers >= maxBuffers && current.maxTextures >= maxTextures &&
            current.maxSamplers >= maxSamplers)
        {
#ifdef RACHIS_TESTING
            recordMetalArgumentTablePoolReuse();
#endif
            return current.table;
        }

        const MetalStageArgumentTable grown{
            .maxBuffers = (std::max)(current.maxBuffers, maxBuffers),
            .maxTextures = (std::max)(current.maxTextures, maxTextures),
            .maxSamplers = (std::max)(current.maxSamplers, maxSamplers),
        };
        MTL4ArgumentTableDescriptor* const tableDesc = [MTL4ArgumentTableDescriptor new];
        tableDesc.initializeBindings = YES;
        tableDesc.maxBufferBindCount = grown.maxBuffers;
        tableDesc.maxTextureBindCount = grown.maxTextures;
        tableDesc.maxSamplerStateBindCount = grown.maxSamplers;
        NSError* tableError = nil;
        id<MTL4ArgumentTable> const created = [device newArgumentTableWithDescriptor:tableDesc error:&tableError];
        if (created == nil)
        {
            return std::unexpected(
                metalNsError(ErrorCode::nativeFailure, tableError, "newArgumentTableWithDescriptor"));
        }
        if (current.table != nil)
        {
            try
            {
                entry.supersededArgumentTables.push_back(current.table);
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::outOfMemory, "Failed to retire a Metal argument table."));
            }
        }
        current = grown;
        current.table = created;
#ifdef RACHIS_TESTING
        recordMetalArgumentTableCreate();
#endif
        return created;
    }

    Result<MetalPushConstantSlice> metalCapturePushConstantSlice(id<MTLDevice> device,
                                                                 MetalCommandEntry& entry,
                                                                 std::span<const std::byte> sourceBytes,
                                                                 std::uint32_t copyBytes)
    {
#ifdef RACHIS_TESTING
        if (consumeInjectedMetalPushConstantSliceFailure())
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::outOfMemory, "Injected Metal push-constant slice failure."));
        }
#endif
        const auto bytes = (std::min)(copyBytes, maxPushConstantBytes);
        const auto alignedBytes = metalAlignUp(std::max<std::uint32_t>(bytes, 1U), kMetalPushConstantSliceAlignment);
        if (entry.pushConstantPageIndex < entry.pushConstantPages.size() &&
            entry.pushConstantPageOffset + alignedBytes > kMetalPushConstantPageBytes)
        {
            ++entry.pushConstantPageIndex;
            entry.pushConstantPageOffset = 0;
        }

        id<MTLBuffer> page = nil;
        if (entry.pushConstantPageIndex < entry.pushConstantPages.size())
        {
            page = entry.pushConstantPages[entry.pushConstantPageIndex];
        }
        else
        {
            page = [device newBufferWithLength:kMetalPushConstantPageBytes
                                       options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined |
                                               MTLResourceHazardTrackingModeTracked];
            if (page == nil)
            {
                return std::unexpected(metalSelectionError(ErrorCode::outOfMemory,
                                                           "Metal could not allocate a push-constant slab buffer."));
            }
            page.label = @"RHI.Metal.PushConstants";
            try
            {
                entry.pushConstantPages.push_back(page);
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::outOfMemory, "Failed to record a Metal push-constant slab page."));
            }
#ifdef RACHIS_TESTING
            recordMetalPushConstantPageAllocate();
#endif
        }

        const std::uint32_t sliceOffset = entry.pushConstantPageOffset;
        entry.pushConstantPageOffset += alignedBytes;
#ifdef RACHIS_TESTING
        recordMetalPushConstantSliceSuballocate();
#endif
        auto* const destination = static_cast<std::byte*>(page.contents) + sliceOffset;
        const auto populated = (std::min)(sourceBytes.size(), static_cast<std::size_t>(bytes));
        if (populated != 0)
        {
            std::memcpy(destination, sourceBytes.data(), populated);
        }
        if (populated < bytes)
        {
            std::memset(destination + populated, 0, bytes - populated);
        }
        return MetalPushConstantSlice{
            .buffer = page,
            .gpuAddress = page.gpuAddress + sliceOffset,
            .byteLength = bytes,
        };
    }

    void collectMetalQueue(MetalQueueTimeline& state) noexcept
    {
        const std::uint64_t completed =
            state.feedback.completedValue(state.event != nil ? state.event.signaledValue : 0);
        while (true)
        {
            MetalPendingEntry retired;
            MetalRecordingRetentions retentions;
            {
                const std::scoped_lock lock(state.retirementMutex);
                if (state.pending.empty() || state.pending.front().timelineValue > completed)
                {
                    break;
                }
                retired = std::move(state.pending.front());
                state.pending.pop_front();
                if (retired.entry != nullptr && retired.pool != nullptr)
                {
                    retentions = takeMetalRecordingRetentions(*retired.entry);
                    auto reset = resetMetalRecording(*retired.entry);
                    const std::scoped_lock poolLock(retired.pool->mutex);
                    retired.entry->inFlight = false;
                    retired.entry->lease = nullptr;
                    if (reset)
                    {
                        retired.pool->available.push_back(retired.entry);
                    }
                    else
                    {
                        retired.pool->quarantined.push_back(retired.entry);
                    }
                }
            }
#ifdef RACHIS_TESTING
            rachis::detail::test::observeCommandPoolRetirement();
#endif
        }
    }

    void retireMetalPresents(MetalDeviceNative& native) noexcept
    {
        const auto& timeline = native.timelines[metalQueueIndex(QueueType::graphics)];
        const auto completed =
            timeline.feedback.completedValue(timeline.event != nil ? timeline.event.signaledValue : 0);
        native.swapchains.forEach([&](SwapchainHandle, std::shared_ptr<MetalSwapchainRecord>& record) {
            if (record == nullptr)
            {
                return;
            }
            auto& pending = record->inFlight;
            std::erase_if(pending, [&](const MetalInFlightPresent& present) {
                if (present.timelineValue > completed)
                {
                    return false;
                }
                ++record->presentsCompleted;
                return true;
            });
            record->inFlightPresents = static_cast<std::uint32_t>(pending.size());
        });
    }

    [[nodiscard]] Status checkMetalSubmission(const MetalDeviceNative& native)
    {
        if (!native.nativeLost.load(std::memory_order_acquire))
        {
            return {};
        }
        const std::scoped_lock lock(native.lossMutex);
        return std::unexpected(native.lossError != nullptr
                                   ? *native.lossError
                                   : metalSelectionError(ErrorCode::deviceLost, "The Metal device is lost."));
    }

    void publishMetalSubmissionFailure(MetalDeviceNative& native,
                                       NSError* error,
                                       std::string_view label,
                                       std::string_view queueLabel)
    {
        if (error == nil)
        {
            return;
        }
        const auto mapped = metalNsError(ErrorCode::deviceLost, error, "MTL4CommandQueue commit", label);
        publishMetalDeviceLoss(native, mapped, metalFaultStatusFromError(error), false, queueLabel, label);
    }

    [[nodiscard]] Result<TimelinePointStatus> queryMetalValue(const MetalDeviceNative& native,
                                                              MetalQueueTimeline& state,
                                                              std::uint64_t value)
    {
        if (auto checked = checkMetalSubmission(native); !checked)
        {
            return std::unexpected(std::move(checked.error()));
        }
        if (state.event == nil)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal timeline query requires a live shared event."));
        }
        const auto completed = state.feedback.completedValue(state.event.signaledValue);

        if (auto checked = checkMetalSubmission(native); !checked)
        {
            return std::unexpected(std::move(checked.error()));
        }
        if (completed < value)
        {
            return TimelinePointStatus::pending;
        }
        collectMetalQueue(state);
        return TimelinePointStatus::complete;
    }

    [[nodiscard]] Result<TimelinePointStatus> waitForMetalValue(const MetalDeviceNative& native,
                                                                MetalQueueTimeline& state,
                                                                std::uint64_t value,
                                                                std::uint64_t timeoutMs)
    {
        const auto started = std::chrono::steady_clock::now();
        auto remainingMs = timeoutMs;
        for (;;)
        {
            auto status = queryMetalValue(native, state, value);
            if (!status || *status == TimelinePointStatus::complete || remainingMs == 0)
            {
                return status;
            }

            const auto slice = std::min<std::uint64_t>(remainingMs, 50);
            if (state.event.signaledValue < value)
            {
                static_cast<void>([state.event waitUntilSignaledValue:value timeoutMS:slice]);
            }
            else
            {
                state.feedback.wait(value, std::chrono::milliseconds(slice));
            }
            if (timeoutMs != std::numeric_limits<std::uint64_t>::max())
            {
                const auto elapsedMs = static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started)
                        .count());
                remainingMs = elapsedMs >= timeoutMs ? 0 : timeoutMs - elapsedMs;
            }
        }
    }

    [[nodiscard]] Status drainMetalQueue(const MetalDeviceNative& native, MetalQueueTimeline& state)
    {
        const auto submitted = state.lastSubmitted.load(std::memory_order_acquire);
        if (submitted != 0)
        {
            auto waited = waitForMetalValue(native, state, submitted, std::numeric_limits<std::uint64_t>::max());
            if (!waited)
            {
                return std::unexpected(std::move(waited.error()));
            }
        }
        collectMetalQueue(state);
        return {};
    }

    Status MetalCommandBuffer::validateReset(bool requireIndividualOptIn) const
    {
        if (owner_ == nullptr || entry_ == nullptr || entry_->lease != this)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument, "The command buffer does not hold a live command-pool recording lease."));
        }
        if (requireIndividualOptIn && !hasAny(owner_->flags, CommandPoolFlag::allowIndividualReset))
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState, "The command pool did not opt into individual command-buffer reset."));
        }
        if (entry_->inFlight)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                       "A command buffer cannot be reset while it is in flight."));
        }
        if (auto ready = detail::validateCommandBufferReadyForReset(*this); !ready)
        {
            return ready;
        }
        return {};
    }

    Status MetalCommandBuffer::resetRecording()
    {
        auto reset = resetMetalRecording(*entry_);
        if (!reset)
        {
            return reset;
        }
        clearRecordingState();
        return beginMetalRecording(*owner_, *entry_);
    }

    void MetalCommandBuffer::clearRecordingState() noexcept
    {
        retained_.clear();
        stagedQuerySlots_.clear();
        layoutState_.clear();
        resetState();
    }

    Status MetalCommandBuffer::ensureCopyEncoder()
    {
        if (entry_ == nullptr || entry_->command == nil || !entry_->begun)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Copy encoding requires a begun Metal command buffer."));
        }
        if (graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Copy encoding cannot split an active rendering scope."));
        }
        if (entry_->encoder == MetalEncoderKind::copy && entry_->computeEncoder != nil)
        {
            flushMetalComputePendingBarriers(*entry_);
            return {};
        }
        endEncoders();
        entry_->computeEncoder = [entry_->command computeCommandEncoder];
        if (entry_->computeEncoder == nil)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::nativeFailure, "Metal could not create a compute encoder for copies."));
        }
        entry_->computeEncoder.label = @"RHI.Metal.Copy";
        entry_->encoder = MetalEncoderKind::copy;
        flushMetalComputePendingBarriers(*entry_);
        return {};
    }

    Status MetalCommandBuffer::ensureComputeEncoder()
    {
        if (entry_ == nullptr || entry_->command == nil || !entry_->begun)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                       "Compute encoding requires a begun Metal command buffer."));
        }
        if (graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                       "Compute encoding cannot split an active rendering scope."));
        }
        if (entry_->encoder == MetalEncoderKind::compute && entry_->computeEncoder != nil)
        {
            flushMetalComputePendingBarriers(*entry_);
            return {};
        }
        endEncoders();
        entry_->computeEncoder = [entry_->command computeCommandEncoder];
        if (entry_->computeEncoder == nil)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::nativeFailure, "Metal could not create a compute encoder."));
        }
        entry_->computeEncoder.label = @"RHI.Metal.Compute";
        entry_->encoder = MetalEncoderKind::compute;
        flushMetalComputePendingBarriers(*entry_);
        return {};
    }

    Status MetalCommandBuffer::encodeStageBarrier(MTLStages afterStages, MTLStages beforeStages)
    {
        if (entry_ == nullptr || entry_->command == nil || !entry_->begun)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                       "Metal resource barriers require a begun command buffer."));
        }

        const auto after = afterStages == static_cast<MTLStages>(0) ? MTLStageAll : afterStages;
        const auto before = beforeStages == static_cast<MTLStages>(0) ? MTLStageAll : beforeStages;
        const auto beforeCompute = static_cast<MTLStages>(before & kMetalComputeEncoderStages);
        const auto beforeRender = static_cast<MTLStages>(before & kMetalRenderEncoderStages);
        constexpr auto visibility = MTL4VisibilityOptionDevice;

        if (entry_->encoder == MetalEncoderKind::none)
        {
            entry_->pendingBarrierAfterStages |= after;
            entry_->pendingBarrierBeforeStages |= before;
            return {};
        }

        if (entry_->computeEncoder != nil)
        {
            const auto afterCompute = static_cast<MTLStages>(after & kMetalComputeEncoderStages);
            if (beforeCompute != static_cast<MTLStages>(0))
            {
                [entry_->computeEncoder barrierAfterQueueStages:after
                                                   beforeStages:beforeCompute
                                              visibilityOptions:visibility];
            }
            [entry_->computeEncoder barrierAfterStages:after beforeQueueStages:before visibilityOptions:visibility];
            if (afterCompute != static_cast<MTLStages>(0) && beforeCompute != static_cast<MTLStages>(0))
            {
                [entry_->computeEncoder barrierAfterEncoderStages:afterCompute
                                              beforeEncoderStages:beforeCompute
                                                visibilityOptions:visibility];
            }
            return {};
        }

        if (entry_->renderEncoder != nil)
        {
            const auto afterRender = static_cast<MTLStages>(after & kMetalRenderEncoderStages);
            if (beforeRender != static_cast<MTLStages>(0))
            {
                [entry_->renderEncoder barrierAfterQueueStages:after
                                                  beforeStages:beforeRender
                                             visibilityOptions:visibility];
            }
            [entry_->renderEncoder barrierAfterStages:after beforeQueueStages:before visibilityOptions:visibility];
            if (afterRender != static_cast<MTLStages>(0) && beforeRender != static_cast<MTLStages>(0))
            {
                [entry_->renderEncoder barrierAfterEncoderStages:afterRender
                                             beforeEncoderStages:beforeRender
                                               visibilityOptions:visibility];
            }
            return {};
        }

        return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                   "Metal resource barriers require a valid command-buffer state."));
    }

    void MetalCommandBuffer::endEncoders() noexcept
    {
        if (entry_ != nullptr)
        {
            endMetalEncoders(*entry_);
        }
    }

    void MetalCommandBuffer::detachForShutdown() noexcept
    {
        if (entry_ != nullptr)
        {
            while (debugState().groupDepth != 0 && entry_->command != nil && entry_->begun)
            {
                [entry_->command popDebugGroup];
                --debugState().groupDepth;
            }
            static_cast<void>(resetMetalRecording(*entry_));
            entry_->lease = nullptr;
            entry_->inFlight = false;
            entry_ = nullptr;
        }
        clearRecordingState();
        owner_.reset();
    }

    MetalCommandEntry* MetalCommandBuffer::release() noexcept
    {
        if (entry_ != nullptr)
        {
            entry_->lease = nullptr;
        }
        owner_.reset();
        return std::exchange(entry_, nullptr);
    }

    void MetalCommandBuffer::abandon() noexcept
    {
        if (owner_ == nullptr || entry_ == nullptr)
        {
            return;
        }
        retained_.clear();
        stagedQuerySlots_.clear();
        layoutState_.clear();
        auto owner = std::move(owner_);
        auto* entry = std::exchange(entry_, nullptr);
        while (debugState().groupDepth != 0 && entry->command != nil && entry->begun)
        {
            [entry->command popDebugGroup];
            --debugState().groupDepth;
        }
        auto reset = resetMetalRecording(*entry);
        const std::scoped_lock lock(owner->mutex);
        entry->lease = nullptr;
        entry->inFlight = false;
        if (reset)
        {
            owner->available.push_back(entry);
        }
        else
        {
            owner->quarantined.push_back(entry);
        }
    }

    Result<CommandPoolHandle> MetalDevice::createCommandPool(const CommandPoolDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (!isValidMetalQueueType(desc.queueType) || desc.maxCommandBuffers == 0 ||
            desc.label.size() > maxDebugLabelBytes || desc.label.contains('\0') ||
            !isValidMetalCommandPoolFlags(desc.flags))
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument,
                "A command pool requires a valid queue, non-zero capacity, bounded label, and known flags."));
        }
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || native_->device == nil)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal createCommandPool requires an active device."));
        }

        auto pool = std::make_shared<MetalCommandPoolState>();
        pool->device = native_->device;
        pool->logState = native_->logState;
        pool->queueType = desc.queueType;
        pool->flags = desc.flags;
        pool->label = std::string(desc.label);
        pool->entries.reserve(desc.maxCommandBuffers);
        pool->available.reserve(desc.maxCommandBuffers);
        pool->quarantined.reserve(desc.maxCommandBuffers);

        @autoreleasepool
        {
            for (std::uint32_t index = 0; index < desc.maxCommandBuffers; ++index)
            {
                auto entry = std::make_unique<MetalCommandEntry>();
                MTL4CommandAllocatorDescriptor* const allocatorDesc = [MTL4CommandAllocatorDescriptor new];
                if (!pool->label.empty())
                {
                    allocatorDesc.label = metalNsString(pool->label);
                }
                NSError* allocatorError = nil;
                entry->allocator = [native_->device newCommandAllocatorWithDescriptor:allocatorDesc
                                                                                error:&allocatorError];
                entry->command = [native_->device newCommandBuffer];
                if (entry->allocator == nil || entry->command == nil)
                {
                    return std::unexpected(
                        metalNsError(ErrorCode::outOfMemory, allocatorError, "newCommandAllocatorWithDescriptor"));
                }
                pool->available.push_back(entry.get());
                pool->entries.push_back(std::move(entry));
            }
        }

        if (native_->allCommandPools.size() >= native_->nextCommandPoolPruneThreshold)
        {
            static_cast<void>(
                std::erase_if(native_->allCommandPools, [](const auto& existing) { return existing.expired(); }));
            native_->nextCommandPoolPruneThreshold = std::max<std::size_t>(64, native_->allCommandPools.size() * 2);
        }
        if (native_->allCommandPools.size() == native_->allCommandPools.capacity())
        {
            try
            {
                const auto currentCapacity = native_->allCommandPools.capacity();
                const auto grownCapacity = currentCapacity == 0
                                               ? std::size_t{16}
                                               : (currentCapacity > std::numeric_limits<std::size_t>::max() / 2
                                                      ? native_->allCommandPools.size() + 1
                                                      : currentCapacity * 2);
                native_->allCommandPools.reserve(grownCapacity);
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::outOfMemory, "Failed to grow the Metal command-pool registry."));
            }
        }
        auto handle = native_->commandPools.emplace(pool);
        if (!handle)
        {
            return std::unexpected(std::move(handle.error()));
        }
        native_->allCommandPools.emplace_back(std::move(pool));
        return handle;
    }

    Status MetalDevice::destroyCommandPool(CommandPoolHandle commandPool)
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || !native_->commandPools.erase(commandPool))
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The command-pool handle is invalid or stale."));
        }
        return {};
    }

    Status MetalDevice::resetCommandPool(CommandPoolHandle commandPool)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        std::shared_ptr<MetalCommandPoolState> pool;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal resetCommandPool requires an active device."));
            }
            const auto* found = native_->commandPools.find(commandPool);
            if (found == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, "The command-pool handle is invalid or stale."));
            }
            pool = *found;
        }
        collectMetalQueue(native_->timelines[metalQueueIndex(pool->queueType)]);

        const std::scoped_lock lock(pool->mutex);
#ifdef RACHIS_TESTING
        holdMetalCommandPoolResetForTesting();
#endif
        std::size_t recordingCount = 0;
        for (const auto& entry : pool->entries)
        {
            if (entry != nullptr && entry->lease != nullptr)
            {
                ++recordingCount;
            }
            if (entry != nullptr && entry->inFlight)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidState, "A command pool cannot be reset while one of its entries is in flight."));
            }
        }
        if (pool->available.size() + pool->quarantined.size() + recordingCount != pool->entries.size())
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState, "A command pool cannot be reset while one of its entries is in flight."));
        }
        for (const auto& entry : pool->entries)
        {
            if (entry != nullptr && entry->lease != nullptr)
            {
                auto valid = entry->lease->validateReset(false);
                if (!valid)
                {
                    return valid;
                }
            }
        }
        for (const auto& entry : pool->entries)
        {
            if (entry == nullptr)
            {
                continue;
            }
            auto reset = resetMetalRecording(*entry);
            if (!reset)
            {
                return reset;
            }
            if (entry->lease != nullptr)
            {
                auto began = beginMetalRecording(*pool, *entry);
                if (!began)
                {
                    return began;
                }
                entry->lease->clearRecordingState();
            }
        }
        restoreMetalQuarantinedEntries(*pool);
        return {};
    }

    Result<std::unique_ptr<detail::BackendCommandBuffer>> MetalDevice::acquireCommandBuffer(
        CommandPoolHandle commandPool)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        {
            std::scoped_lock const lock(mutex_);
            if (native_ != nullptr)
            {
                for (auto& timeline : native_->timelines)
                {
                    collectMetalQueue(timeline);
                }
            }
        }
        std::shared_ptr<MetalCommandPoolState> pool;
        std::uint64_t deviceToken = 0;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                           "Metal acquireCommandBuffer requires an active device."));
            }
            const auto* found = native_->commandPools.find(commandPool);
            if (found == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, "The command-pool handle is invalid or stale."));
            }
            pool = *found;
            deviceToken = native_->deviceToken;
        }

        const std::scoped_lock lock(pool->mutex);
        if (pool->available.empty())
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::resourceExhausted, "The bounded command pool is exhausted."));
        }
        auto* entry = pool->available.back();
        pool->available.pop_back();
        auto reset = resetMetalRecording(*entry);
        if (!reset)
        {
            pool->quarantined.push_back(entry);
            return std::unexpected(std::move(reset.error()));
        }
        auto began = beginMetalRecording(*pool, *entry);
        if (!began)
        {
            pool->quarantined.push_back(entry);
            return std::unexpected(std::move(began.error()));
        }

        auto* allocation = new (std::nothrow) MetalCommandBuffer(pool, *entry, deviceToken);
        if (allocation == nullptr)
        {
            static_cast<void>(resetMetalRecording(*entry));
            pool->available.push_back(entry);
            return std::unexpected(
                metalSelectionError(ErrorCode::outOfMemory, "Failed to allocate a command-buffer lease."));
        }
        entry->lease = allocation;
        return std::unique_ptr<detail::BackendCommandBuffer>(allocation);
    }

    Status MetalDevice::resetCommandBuffer(detail::BackendCommandBuffer& commandBuffer)
    {
        const MetalAutoreleasePool autoreleasePool;
        std::uint64_t deviceToken = 0;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                           "Metal resetCommandBuffer requires an active device."));
            }
            deviceToken = native_->deviceToken;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        const auto owner = native->ownerShared();
        const std::scoped_lock lock(owner->mutex);
        auto valid = native->validateReset(true);
        if (!valid)
        {
            return valid;
        }
        return native->resetRecording();
    }

    namespace
    {

        [[nodiscard]] Result<MetalQueueTimeline*> collectMetalSubmissionCommands(
            MetalDeviceNative& deviceNative,
            std::span<const std::unique_ptr<detail::BackendCommandBuffer>> commandBuffers,
            std::vector<MetalCommandBuffer*>& nativeCommands)
        {
            try
            {
                nativeCommands.reserve(commandBuffers.size());
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::outOfMemory, "Failed to allocate Metal submission command buffer batch."));
            }
            MetalQueueTimeline* queueState = nullptr;
            for (const auto& commandBuffer : commandBuffers)
            {
                auto* native =
                    commandBuffer != nullptr ? asMetalCommand(*commandBuffer, deviceNative.deviceToken) : nullptr;
                if (native == nullptr || native->entry()->command == nil)
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument,
                        "A command buffer in the submission batch belongs to another device or backend."));
                }
                if (queueState == nullptr)
                {
                    if (!isValidMetalQueueType(native->queueType()))
                    {
                        return std::unexpected(metalSelectionError(
                            ErrorCode::invalidArgument, "A Metal submission batch targeted an invalid queue type."));
                    }
                    queueState = &deviceNative.timelines[metalQueueIndex(native->queueType())];
                }
                else if (queueState != &deviceNative.timelines[metalQueueIndex(native->queueType())])
                {
                    return std::unexpected(
                        metalSelectionError(ErrorCode::invalidArgument,
                                            "Every command buffer in a submission batch must target the same queue."));
                }
                if (std::ranges::any_of(nativeCommands,
                                        [&](const auto* other) { return other->entry() == native->entry(); }))
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument,
                        "A Metal submission batch cannot contain the same command-buffer lease more than once."));
                }
                auto ready = metalRecordingReady(*native);
                if (!ready)
                {
                    return std::unexpected(std::move(ready.error()));
                }
                nativeCommands.push_back(native);
            }
            return queueState;
        }

        [[nodiscard]] Status validateMetalSubmissionWaits(const MetalDeviceNative& native,
                                                          std::span<const TimelinePoint> waits)
        {
            for (const auto point : waits)
            {
                if (!point.isValid() || !isValidMetalQueueType(point.queueType()) ||
                    detail::TimelinePointAccess::deviceToken(point) != native.deviceToken ||
                    point.value() > native.timelines[metalQueueIndex(point.queueType())].lastSubmitted.load(
                                        std::memory_order_acquire))
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument,
                        "A queue wait references an invalid, foreign, or unsubmitted timeline point."));
                }
            }
            return {};
        }

        struct MetalSubmissionBatch final
        {
            std::vector<std::shared_ptr<MetalCommandPoolState>> pools;
            std::vector<MetalCommandEntry*> entries;
            std::vector<id<MTL4CommandBuffer>> nativeBuffers;
        };

        struct MetalSubmissionQuerySlots final
        {
            std::vector<std::shared_ptr<MetalQueryPoolRecord>> touchedQueryPools;
            std::vector<std::unique_lock<std::mutex>> queryPoolLocks;
            std::vector<std::vector<std::uint8_t>> validatedPoolSlots;
        };

        struct MetalSubmissionDrawableWaits final
        {
            std::vector<id<CAMetalDrawable>> waitedDrawables;
            std::vector<std::shared_ptr<MetalSwapchainRecord>> waitedSwapchains;
        };

        void collectMetalBatchEntries(const std::vector<MetalCommandBuffer*>& nativeCommands,
                                      MetalSubmissionBatch& batch)
        {
            batch.pools.reserve(nativeCommands.size());
            batch.entries.reserve(nativeCommands.size());
            batch.nativeBuffers.reserve(nativeCommands.size());
            for (auto* native : nativeCommands)
            {
                batch.pools.push_back(native->ownerShared());
                batch.entries.push_back(native->entry());
                batch.nativeBuffers.push_back(native->entry()->command);
            }
        }

        void lockAndSnapshotMetalTouchedQueryPools(const std::vector<MetalCommandBuffer*>& nativeCommands,
                                                   MetalSubmissionQuerySlots& querySlots)
        {
            for (auto* native : nativeCommands)
            {
                for (const auto& staged : native->stagedQuerySlots())
                {
                    if (staged.pool != nullptr)
                    {
                        querySlots.touchedQueryPools.push_back(staged.pool);
                    }
                }
            }
            std::ranges::sort(querySlots.touchedQueryPools,
                              [](const auto& lhs, const auto& rhs) noexcept { return lhs.get() < rhs.get(); });
            const auto uniquePools =
                std::ranges::unique(querySlots.touchedQueryPools,
                                    [](const auto& lhs, const auto& rhs) noexcept { return lhs.get() == rhs.get(); });
            querySlots.touchedQueryPools.erase(uniquePools.begin(), uniquePools.end());

            querySlots.queryPoolLocks.reserve(querySlots.touchedQueryPools.size());
            querySlots.validatedPoolSlots.reserve(querySlots.touchedQueryPools.size());
            for (const auto& pool : querySlots.touchedQueryPools)
            {
                querySlots.queryPoolLocks.emplace_back(pool->stateMutex);
                querySlots.validatedPoolSlots.push_back(pool->pendingSlots);
            }
        }

        [[nodiscard]] Status appendMetalWaitedDrawable(const MetalWaitedDrawable& candidate,
                                                       MetalSubmissionDrawableWaits& drawableWaits)
        {
            if (candidate.drawable == nil)
            {
                return {};
            }
            if (candidate.swapchain == nullptr ||
                candidate.swapchain->currentAcquireId.load(std::memory_order_acquire) != candidate.acquireId)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidState,
                    "A command buffer recorded a swapchain image that was cancelled, presented, or released before "
                    "submission."));
            }
            const void* const drawableToken = (__bridge const void*)candidate.drawable;
            if (candidate.swapchain->waitedDrawableToken.load(std::memory_order_acquire) == drawableToken)
            {
                return {};
            }
            bool seen = false;
            for (const auto& drawable : drawableWaits.waitedDrawables)
            {
                if (drawable == candidate.drawable)
                {
                    seen = true;
                    break;
                }
            }
            if (seen)
            {
                return {};
            }
            drawableWaits.waitedDrawables.push_back(candidate.drawable);
            drawableWaits.waitedSwapchains.push_back(candidate.swapchain);
            return {};
        }

        [[nodiscard]] Status collectMetalWaitedDrawables(const std::vector<MetalCommandEntry*>& entries,
                                                         MetalSubmissionDrawableWaits& drawableWaits)
        {
            drawableWaits.waitedDrawables.reserve(entries.size());
            drawableWaits.waitedSwapchains.reserve(entries.size());
            for (auto* entry : entries)
            {
                if (entry == nullptr)
                {
                    continue;
                }
                for (const auto& candidate : entry->waitedDrawables)
                {
                    if (auto appended = appendMetalWaitedDrawable(candidate, drawableWaits); !appended)
                    {
                        return appended;
                    }
                }
            }
            return {};
        }

        [[nodiscard]] Status gatherMetalSubmissionStorage(const std::vector<MetalCommandBuffer*>& nativeCommands,
                                                          MetalSubmissionBatch& batch,
                                                          MetalSubmissionQuerySlots& querySlots,
                                                          MetalSubmissionDrawableWaits& drawableWaits)
        {
            try
            {
                collectMetalBatchEntries(nativeCommands, batch);
                lockAndSnapshotMetalTouchedQueryPools(nativeCommands, querySlots);
                return collectMetalWaitedDrawables(batch.entries, drawableWaits);
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::outOfMemory, "Failed to allocate Metal submission batch storage."));
            }
        }

        [[nodiscard]] std::vector<std::uint8_t>* findMetalPoolSlotState(MetalSubmissionQuerySlots& querySlots,
                                                                        const MetalQueryPoolRecord* pool) noexcept
        {
            for (std::size_t index = 0; index < querySlots.touchedQueryPools.size(); ++index)
            {
                if (querySlots.touchedQueryPools[index].get() == pool)
                {
                    return &querySlots.validatedPoolSlots[index];
                }
            }
            return nullptr;
        }

        [[nodiscard]] Status applyMetalStagedQuerySlots(const std::vector<MetalCommandBuffer*>& nativeCommands,
                                                        MetalSubmissionQuerySlots& querySlots)
        {
            for (auto* native : nativeCommands)
            {
                for (const auto& staged : native->stagedQuerySlots())
                {
                    auto* slots = findMetalPoolSlotState(querySlots, staged.pool.get());
                    if (slots == nullptr || staged.queryIndex >= slots->size())
                    {
                        return std::unexpected(metalSelectionError(
                            ErrorCode::invalidState, "A submitted timestamp query slot is out of bounds."));
                    }
                    const bool currentlyPending = (*slots)[staged.queryIndex] != 0;
                    if (staged.precondition == MetalQuerySlotPrecondition::mustNotBePending && currentlyPending)
                    {
                        return std::unexpected(metalSelectionError(
                            ErrorCode::invalidState,
                            "A timestamp query cannot be written twice before its first value is resolved."));
                    }
                    if (staged.precondition == MetalQuerySlotPrecondition::mustBePending && !currentlyPending)
                    {
                        return std::unexpected(metalSelectionError(
                            ErrorCode::invalidState,
                            "Timestamp resolve requires every slot to have been written after its last reset."));
                    }
                    (*slots)[staged.queryIndex] = staged.pendingAfter ? 1U : 0U;
                }
            }
            return {};
        }

        [[nodiscard]] Status flushMetalBatchDeferredBarriers(const MetalSubmissionBatch& batch)
        {
            for (std::size_t index = 0; index < batch.entries.size(); ++index)
            {
                if (batch.entries[index] == nullptr)
                {
                    continue;
                }
                const std::scoped_lock poolLock(batch.pools[index]->mutex);
                if (auto flushed = flushMetalDeferredBarriers(*batch.entries[index]); !flushed)
                {
                    return std::unexpected(std::move(flushed.error()));
                }
            }
            return {};
        }

        void reserveMetalPoolQuarantineSlots(const std::vector<std::shared_ptr<MetalCommandPoolState>>& pools,
                                             std::size_t index)
        {
            if (pools[index] == nullptr)
            {
                return;
            }
            bool alreadyReserved = false;
            for (std::size_t prior = 0; prior < index; ++prior)
            {
                if (pools[prior].get() == pools[index].get())
                {
                    alreadyReserved = true;
                    break;
                }
            }
            if (alreadyReserved)
            {
                return;
            }
            std::size_t countForPool = 0;
            for (std::size_t other = index; other < pools.size(); ++other)
            {
                if (pools[other].get() == pools[index].get())
                {
                    ++countForPool;
                }
            }
            const std::scoped_lock poolLock(pools[index]->mutex);
            pools[index]->quarantined.reserve(
                std::max(pools[index]->entries.size(), pools[index]->quarantined.size() + countForPool));
        }

        [[nodiscard]] Status reserveMetalBatchRetirement(MetalQueueTimeline& state,
                                                         const MetalSubmissionBatch& batch,
                                                         std::uint64_t submittedValue)
        {
            std::size_t reservedPendingCount = 0;
            try
            {
                for (std::size_t index = 0; index < batch.entries.size(); ++index)
                {
                    state.pending.push_back(MetalPendingEntry{
                        .timelineValue = std::numeric_limits<std::uint64_t>::max(),
                        .pool = batch.pools[index],
                        .entry = batch.entries[index],
                    });
                    ++reservedPendingCount;
                }
                for (std::size_t index = 0; index < batch.pools.size(); ++index)
                {
                    reserveMetalPoolQuarantineSlots(batch.pools, index);
                }
                state.feedback.begin(submittedValue);
            }
            catch (const std::bad_alloc&)
            {
                while (reservedPendingCount != 0)
                {
                    state.pending.pop_back();
                    --reservedPendingCount;
                }
                return std::unexpected(
                    metalSelectionError(ErrorCode::outOfMemory, "Failed to reserve Metal batch retirement records."));
            }
            return {};
        }

        void consumeMetalCommandLeases(MetalQueueTimeline& state,
                                       const MetalSubmissionBatch& batch,
                                       const std::vector<MetalCommandBuffer*>& nativeCommands,
                                       std::span<std::unique_ptr<detail::BackendCommandBuffer>> commandBuffers)
        {
            for (std::size_t index = 0; index < nativeCommands.size(); ++index)
            {
                const auto pendingIndex = state.pending.size() - batch.entries.size() + index;
                state.pending[pendingIndex].retained = nativeCommands[index]->takeRetained();
                nativeCommands[index]->clearRecordingState();
                static_cast<void>(nativeCommands[index]->release());
                commandBuffers[index].reset();
            }
        }

#ifdef RACHIS_TESTING
        [[nodiscard]] Status rollBackInjectedMetalSubmitCloseFailure(MetalQueueTimeline& state,
                                                                     const MetalSubmissionBatch& batch,
                                                                     std::uint64_t submittedValue)
        {
            if (!rachis::metal::test::nextSubmitCloseFailure().exchange(false, std::memory_order_relaxed))
            {
                return {};
            }
            state.feedback.complete(submittedValue);
            for (std::size_t index = 0; index < batch.entries.size(); ++index)
            {
                state.pending.pop_back();
                const std::scoped_lock poolLock(batch.pools[index]->mutex);
                batch.pools[index]->quarantined.push_back(batch.entries[index]);
            }
            return std::unexpected(
                metalSelectionError(ErrorCode::validationError, "Injected post-consume Metal submit close failure."));
        }

        void applyMetalCommitFeedbackTestHooks(const std::shared_ptr<MetalCommitFeedbackGate>& feedbackGate,
                                               bool injectFailure,
                                               NSError* __strong& error)
        {
            if (feedbackGate != nullptr)
            {
                feedbackGate->waitUntilReleased();
            }
            if (injectFailure && error == nil)
            {
                if (feedbackGate == nullptr)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
                error = [NSError errorWithDomain:MTL4CommandQueueErrorDomain
                                            code:MTL4CommandQueueErrorInternal
                                        userInfo:nil];
            }
        }
#endif

        void addMetalSubmissionFeedbackHandler(MTL4CommitOptions* commitOptions,
                                               const std::shared_ptr<MetalDeviceNative>& native,
                                               QueueType queueType,
                                               std::uint64_t submittedValue)
        {
            const auto backend = native;
            const auto submittedQueue = metalQueueIndex(queueType);
            const char* const submittedQueueLabel = metalQueueTimelineLabel(queueType);
#ifdef RACHIS_TESTING
            const bool injectFailure = native->failNextCommitFeedback.exchange(false, std::memory_order_acq_rel);
            std::shared_ptr<MetalCommitFeedbackGate> feedbackGate;
            {
                const std::scoped_lock gateLock(native->feedbackGateMutex);
                feedbackGate = std::move(native->nextCommitFeedbackGate);
            }
#endif
            [commitOptions addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
              @autoreleasepool
              {
                  NSError* error = feedback.error;
#ifdef RACHIS_TESTING
                  applyMetalCommitFeedbackTestHooks(feedbackGate, injectFailure, error);
#endif
                  publishMetalSubmissionFailure(*backend, error, "RHI.Metal.Submit", submittedQueueLabel);
              }
              backend->timelines[submittedQueue].feedback.complete(submittedValue);
            }];
        }

        void encodeMetalQueueWaits(MetalDeviceNative& native,
                                   std::span<const TimelinePoint> waits,
                                   const MetalSubmissionDrawableWaits& drawableWaits)
        {
            for (const auto point : waits)
            {
                auto& waited = native.timelines[metalQueueIndex(point.queueType())];
                if (waited.event != nil)
                {
                    [native.commandQueue waitForEvent:waited.event value:point.value()];
                }
            }

            for (std::size_t index = 0; index < drawableWaits.waitedDrawables.size(); ++index)
            {
                [native.commandQueue waitForDrawable:drawableWaits.waitedDrawables[index]];
#ifdef RACHIS_TESTING
                rachis::metal::test::submittedDrawableWaitCount().fetch_add(1, std::memory_order_relaxed);
#endif
                if (drawableWaits.waitedSwapchains[index] != nullptr)
                {
                    drawableWaits.waitedSwapchains[index]->waitedDrawableToken.store(
                        (__bridge const void*)drawableWaits.waitedDrawables[index], std::memory_order_release);
                }
            }
        }

        void publishMetalSubmittedBatch(MetalQueueTimeline& state,
                                        const MetalSubmissionBatch& batch,
                                        MetalSubmissionQuerySlots& querySlots,
                                        std::uint64_t submittedValue)
        {
            for (std::size_t index = 0; index < querySlots.touchedQueryPools.size(); ++index)
            {
                querySlots.touchedQueryPools[index]->pendingSlots = std::move(querySlots.validatedPoolSlots[index]);
            }
            ++state.nextValue;
            state.lastSubmitted.store(submittedValue, std::memory_order_release);
            for (std::size_t index = 0; index < batch.entries.size(); ++index)
            {
                batch.entries[index]->inFlight = true;
                state.pending[state.pending.size() - batch.entries.size() + index].timelineValue = submittedValue;
            }
        }

    } // namespace

    Result<TimelinePoint> MetalDevice::submit(std::span<std::unique_ptr<detail::BackendCommandBuffer>> commandBuffers,
                                              std::span<const TimelinePoint> waits)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (native_ == nullptr || native_->commandQueue == nil || native_->queueGate == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal submit requires an active device."));
        }
        if (commandBuffers.empty())
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "A submission batch must contain at least one command buffer."));
        }

        std::vector<MetalCommandBuffer*> nativeCommands;
        auto queueState = collectMetalSubmissionCommands(*native_, commandBuffers, nativeCommands);
        if (!queueState)
        {
            return std::unexpected(std::move(queueState.error()));
        }

        auto& state = **queueState;
        if (auto valid = validateMetalSubmissionWaits(*native_, waits); !valid)
        {
            return std::unexpected(std::move(valid.error()));
        }

        const std::scoped_lock submissionLock(native_->queueGate->mutex);
        if (!hasUnreservedMetalTimelineSlot(state.nextValue) || state.event == nil)
        {
            return std::unexpected(metalSelectionError(ErrorCode::resourceExhausted,
                                                       "The queue timeline is exhausted and cannot be reused safely."));
        }

        MetalSubmissionBatch batch;
        MetalSubmissionQuerySlots querySlots;
        MetalSubmissionDrawableWaits drawableWaits;
        if (auto gathered = gatherMetalSubmissionStorage(nativeCommands, batch, querySlots, drawableWaits); !gathered)
        {
            return std::unexpected(std::move(gathered.error()));
        }
        if (auto applied = applyMetalStagedQuerySlots(nativeCommands, querySlots); !applied)
        {
            return std::unexpected(std::move(applied.error()));
        }
        auto layouts = metalPlanBatchLayouts(nativeCommands);
        if (!layouts)
        {
            return std::unexpected(std::move(layouts.error()));
        }
        if (auto flushed = flushMetalBatchDeferredBarriers(batch); !flushed)
        {
            return std::unexpected(std::move(flushed.error()));
        }

        const std::unique_lock retirementLock(state.retirementMutex);
        const auto submittedValue = state.nextValue;
        if (auto reserved = reserveMetalBatchRetirement(state, batch, submittedValue); !reserved)
        {
            return std::unexpected(std::move(reserved.error()));
        }

        consumeMetalCommandLeases(state, batch, nativeCommands, commandBuffers);
        for (auto* entry : batch.entries)
        {
            metalFoldArgumentResidency(*entry);
            commitMetalResidency(*entry);
            endMetalRecording(*entry);
        }
#ifdef RACHIS_TESTING
        if (auto closed = rollBackInjectedMetalSubmitCloseFailure(state, batch, submittedValue); !closed)
        {
            return std::unexpected(std::move(closed.error()));
        }
#endif

        @autoreleasepool
        {
            MTL4CommitOptions* const commitOptions = [MTL4CommitOptions new];
            addMetalSubmissionFeedbackHandler(commitOptions, native_, state.type, submittedValue);
            encodeMetalQueueWaits(*native_, waits, drawableWaits);
            [native_->commandQueue commit:batch.nativeBuffers.data()
                                    count:batch.nativeBuffers.size()
                                  options:commitOptions];
            [native_->commandQueue signalEvent:state.event value:submittedValue];
            publishMetalSubmittedBatch(state, batch, querySlots, submittedValue);
            metalPublishBatchLayouts(*layouts);
            return detail::TimelinePointAccess::make(state.type, submittedValue, native_->deviceToken);
        }
    }

    Result<TimelinePointStatus> MetalDevice::queryTimelinePoint(TimelinePoint point) const
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (native_ == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Cannot query a timeline point on an invalid device."));
        }
        if (!isValidMetalQueueType(point.queueType()) || !point.isValid() ||
            detail::TimelinePointAccess::deviceToken(point) != native_->deviceToken ||
            point.value() >
                native_->timelines[metalQueueIndex(point.queueType())].lastSubmitted.load(std::memory_order_acquire))
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The timeline point is invalid, foreign, or unpublished."));
        }
        return queryMetalValue(*native_, native_->timelines[metalQueueIndex(point.queueType())], point.value());
    }

    Result<TimelinePointStatus> MetalDevice::wait(TimelinePoint point, std::chrono::nanoseconds timeout)
    {
        if (timeout < std::chrono::nanoseconds::zero())
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "A bounded timeline wait requires a non-negative timeout."));
        }
        auto status = queryTimelinePoint(point);
        if (!status || *status == TimelinePointStatus::complete || timeout == std::chrono::nanoseconds::zero())
        {
            return status;
        }
        auto& state = native_->timelines[metalQueueIndex(point.queueType())];
        return waitForMetalValue(*native_, state, point.value(), metalWaitTimeoutMs(timeout));
    }

    Status MetalDevice::wait(TimelinePoint point)
    {
        auto status = queryTimelinePoint(point);
        if (!status)
        {
            return std::unexpected(std::move(status.error()));
        }
        if (*status == TimelinePointStatus::complete)
        {
            return {};
        }
        auto waited = waitForMetalValue(*native_,
                                        native_->timelines[metalQueueIndex(point.queueType())],
                                        point.value(),
                                        std::numeric_limits<std::uint64_t>::max());
        if (!waited)
        {
            return std::unexpected(std::move(waited.error()));
        }
        return {};
    }

    Status MetalDevice::beginDebugGroup(detail::BackendCommandBuffer& commandBuffer, const DebugMarkerDesc& desc)
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
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal beginDebugGroup requires an active device."));
            }
            deviceToken = native_->deviceToken;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        const auto owner = native->ownerShared();
        const std::scoped_lock lock(owner->mutex);
        auto& state = native->debugState();
        if (auto valid = detail::validateDebugGroupBegin(state, desc); !valid)
        {
            return valid;
        }
        const auto currentThread = std::this_thread::get_id();
        auto* entry = native->entry();
        if (entry == nullptr || entry->command == nil || !entry->begun)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Debug groups require a begun Metal command buffer."));
        }
        [entry->command pushDebugGroup:metalNsString(desc.label)];
        if (state.groupDepth == 0)
        {
            state.groupThread = currentThread;
        }
        state.groupOwners[state.groupDepth] = detail::DebugGroupOwner::user;
        ++state.groupDepth;
        return {};
    }

    Status MetalDevice::endDebugGroup(detail::BackendCommandBuffer& commandBuffer)
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
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal endDebugGroup requires an active device."));
            }
            deviceToken = native_->deviceToken;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        const auto owner = native->ownerShared();
        const std::scoped_lock lock(owner->mutex);
        auto& state = native->debugState();
        if (auto valid = detail::validateDebugGroupEnd(state); !valid)
        {
            return valid;
        }
        auto* entry = native->entry();
        if (entry == nullptr || entry->command == nil || !entry->begun)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Debug groups require a begun Metal command buffer."));
        }
        [entry->command popDebugGroup];
        --state.groupDepth;
        return {};
    }

    Status MetalDevice::insertDebugMarker(detail::BackendCommandBuffer& commandBuffer, const DebugMarkerDesc& desc)
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
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal insertDebugMarker requires an active device."));
            }
            deviceToken = native_->deviceToken;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        auto valid = detail::validateDebugMarkerDesc(desc);
        if (!valid)
        {
            return valid;
        }
        auto* entry = native->entry();
        if (entry == nullptr || entry->command == nil || !entry->begun)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Debug markers require a begun Metal command buffer."));
        }
        if (entry->computeEncoder != nil)
        {
            [entry->computeEncoder insertDebugSignpost:metalNsString(desc.label)];
        }
        else if (entry->renderEncoder != nil)
        {
            [entry->renderEncoder insertDebugSignpost:metalNsString(desc.label)];
        }
        else
        {
            [entry->command pushDebugGroup:metalNsString(desc.label)];
            [entry->command popDebugGroup];
        }
        return {};
    }

} // namespace rachis::metal
