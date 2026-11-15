#pragma once

#include "Internal/BackendCommandBuffer.h"
#include "Rachis/Device.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace rachis::detail
{

    [[nodiscard]] inline Error unsupportedBackendOperation(std::string_view backend, std::string_view operation)
    {
        if (backend == "Metal")
        {
            return Error{
                ErrorCode::unsupported,
                0,
                std::string(operation) +
                    " is not implemented on the Metal backend. Selection did not fall back to D3D12 or Vulkan.",
            };
        }
        return Error{
            ErrorCode::unsupported,
            0,
            std::string(operation) + " is not implemented on the " + std::string(backend) + " backend yet.",
        };
    }

    class BackendDevice
    {
    public:
        virtual ~BackendDevice() = default;

        BackendDevice(const BackendDevice&) = delete;
        BackendDevice& operator=(const BackendDevice&) = delete;
        BackendDevice(BackendDevice&&) = delete;
        BackendDevice& operator=(BackendDevice&&) = delete;

        [[nodiscard]] virtual std::string_view backendName() const noexcept = 0;
        [[nodiscard]] virtual std::uint64_t deviceToken() const noexcept = 0;
        [[nodiscard]] virtual const AdapterInfo& adapterInfo() const noexcept = 0;
        [[nodiscard]] virtual const DeviceCapabilities& capabilities() const noexcept = 0;
        [[nodiscard]] virtual DeviceOperationalState operationalState() const noexcept = 0;
        [[nodiscard]] virtual std::optional<Error> deviceLossError() const = 0;
        [[nodiscard]] virtual Result<std::vector<MemoryBudget>> queryMemoryBudgets(std::uint32_t nodeIndex) const = 0;
        [[nodiscard]] virtual Result<ResourceMemorySnapshot> captureResourceMemorySnapshot() const = 0;
        [[nodiscard]] virtual Result<DeviceFaultReport> captureDeviceFaultReport() const = 0;
        [[nodiscard]] virtual BackendAllocationSnapshot captureBackendAllocationSnapshot() const noexcept = 0;
        [[nodiscard]] virtual FormatProperties queryFormatProperties(Format format) const noexcept = 0;
        [[nodiscard]] virtual QueueProperties queueProperties(QueueType queueType) const noexcept = 0;
        [[nodiscard]] virtual Result<QueueClockCalibration> calibrateQueueClock(QueueType queueType) const = 0;
        [[nodiscard]] virtual DiagnosticCounts diagnosticCounts() const noexcept = 0;
        [[nodiscard]] virtual BindingUpdateStatistics bindingUpdateStatistics() const noexcept = 0;
        [[nodiscard]] virtual Result<CommandPoolHandle> createCommandPool(const CommandPoolDesc& desc) = 0;
        [[nodiscard]] virtual Status destroyCommandPool(CommandPoolHandle commandPool) = 0;
        [[nodiscard]] virtual Status resetCommandPool(CommandPoolHandle commandPool) = 0;
        [[nodiscard]] virtual Result<std::unique_ptr<BackendCommandBuffer>> acquireCommandBuffer(
            CommandPoolHandle commandPool) = 0;
        [[nodiscard]] virtual Status resetCommandBuffer(BackendCommandBuffer& commandBuffer) = 0;
        [[nodiscard]] virtual Result<TimelinePoint> submit(
            std::span<std::unique_ptr<BackendCommandBuffer>> commandBuffers, std::span<const TimelinePoint> waits) = 0;
        [[nodiscard]] virtual Result<TimelinePointStatus> queryTimelinePoint(TimelinePoint point) const = 0;
        [[nodiscard]] virtual Result<TimelinePointStatus> wait(TimelinePoint point,
                                                               std::chrono::nanoseconds timeout) = 0;
        [[nodiscard]] virtual Status wait(TimelinePoint point) = 0;
        [[nodiscard]] virtual Result<SharedTimelinePoint> exportTimelinePoint(TimelinePoint)
        {
            return std::unexpected(unsupportedOperation("exportTimelinePoint"));
        }
        [[nodiscard]] virtual Result<TimelinePoint> enqueueSharedTimelineWait(QueueType, SharedTimelinePointView)
        {
            return std::unexpected(unsupportedOperation("enqueueSharedTimelineWait"));
        }
        [[nodiscard]] virtual Status waitIdle(QueueType queueType) = 0;
        [[nodiscard]] virtual Status waitIdle() = 0;
        [[nodiscard]] virtual std::uint64_t completedValue(QueueType queueType) const noexcept = 0;
        virtual void collectGarbage() noexcept = 0;
        [[nodiscard]] virtual Status beginDebugGroup(BackendCommandBuffer& commandBuffer,
                                                     const DebugMarkerDesc& desc) = 0;
        [[nodiscard]] virtual Status endDebugGroup(BackendCommandBuffer& commandBuffer) = 0;
        [[nodiscard]] virtual Status insertDebugMarker(BackendCommandBuffer& commandBuffer,
                                                       const DebugMarkerDesc& desc) = 0;
        [[nodiscard]] virtual Result<ResourceHeapHandle> createResourceHeap(const ResourceHeapDesc&)
        {
            return std::unexpected(unsupportedOperation("createResourceHeap"));
        }
        [[nodiscard]] virtual Status destroyResourceHeap(ResourceHeapHandle)
        {
            return std::unexpected(unsupportedOperation("destroyResourceHeap"));
        }
        [[nodiscard]] virtual Result<ResidencyState> queryResidency(ResourceHeapHandle) const
        {
            return std::unexpected(unsupportedOperation("queryResidency"));
        }
        [[nodiscard]] virtual Result<ResourceMemoryRequirements> queryBufferMemoryRequirements(
            const BufferDesc& desc) const = 0;
        [[nodiscard]] virtual Result<BufferHandle> createBuffer(const BufferDesc& desc) = 0;
        [[nodiscard]] virtual Result<BufferHandle> createPlacedBuffer(const ResourcePlacement&, const BufferDesc&)
        {
            return std::unexpected(unsupportedOperation("createPlacedBuffer"));
        }
        [[nodiscard]] virtual Result<BufferHandle> createSparseBuffer(const BufferDesc&)
        {
            return std::unexpected(unsupportedOperation("createSparseBuffer"));
        }
        [[nodiscard]] virtual Result<SparseBufferProperties> querySparseBufferProperties(BufferHandle) const
        {
            return std::unexpected(unsupportedOperation("querySparseBufferProperties"));
        }
        [[nodiscard]] virtual Result<TimelinePoint> enqueueSparseBufferMappings(
            QueueType, std::span<const SparseBufferTileMapping>)
        {
            return std::unexpected(unsupportedOperation("enqueueSparseBufferMappings"));
        }
        [[nodiscard]] virtual Result<BufferHandle> createSharedBuffer(const BufferDesc&)
        {
            return std::unexpected(unsupportedOperation("createSharedBuffer"));
        }
        [[nodiscard]] virtual Result<SharedHandle> exportSharedBuffer(BufferHandle)
        {
            return std::unexpected(unsupportedOperation("exportSharedBuffer"));
        }
        [[nodiscard]] virtual Result<BufferHandle> importSharedBuffer(SharedHandleView, const BufferDesc&)
        {
            return std::unexpected(unsupportedOperation("importSharedBuffer"));
        }
        [[nodiscard]] virtual Status destroyBuffer(BufferHandle buffer) = 0;
        [[nodiscard]] virtual Result<ResidencyState> queryResidency(BufferHandle) const
        {
            return std::unexpected(unsupportedOperation("queryResidency"));
        }
        [[nodiscard]] virtual Result<MappedMemory> mapBuffer(BufferHandle buffer,
                                                             std::uint64_t offset,
                                                             std::uint64_t size) = 0;
        [[nodiscard]] virtual Status unmapBuffer(BufferHandle buffer, std::uint64_t offset, std::uint64_t size) = 0;
        [[nodiscard]] virtual Status flushMappedMemoryRanges(BufferHandle buffer,
                                                             std::span<const MappedMemoryRange> ranges) = 0;
        [[nodiscard]] virtual Status invalidateMappedMemoryRanges(BufferHandle buffer,
                                                                  std::span<const MappedMemoryRange> ranges) = 0;
        [[nodiscard]] virtual Status writeBuffer(BufferHandle buffer,
                                                 std::uint64_t offset,
                                                 std::span<const std::byte> data) = 0;
        [[nodiscard]] virtual Status readBuffer(BufferHandle buffer,
                                                std::uint64_t offset,
                                                std::span<std::byte> destination) = 0;
        [[nodiscard]] virtual Status copyBufferRegions(BackendCommandBuffer& commandBuffer,
                                                       BufferHandle source,
                                                       BufferHandle destination,
                                                       std::span<const BufferCopyRegion> regions) = 0;
        [[nodiscard]] virtual Status bufferBarrier(BackendCommandBuffer& commandBuffer,
                                                   const BufferBarrier& barrier) = 0;
        [[nodiscard]] virtual Status samplerFeedbackBarrier(BackendCommandBuffer&, const SamplerFeedbackBarrier&)
        {
            return std::unexpected(unsupportedOperation("samplerFeedbackBarrier"));
        }
        [[nodiscard]] virtual Status activateBufferAlias(BackendCommandBuffer&, const BufferAliasActivation&)
        {
            return std::unexpected(unsupportedOperation("activateBufferAlias"));
        }
        [[nodiscard]] virtual Status resourceBarriers(BackendCommandBuffer& commandBuffer,
                                                      const ResourceBarrierBatch& barriers) = 0;
        [[nodiscard]] virtual Result<AccelerationStructureBuildSizes> queryBottomLevelAccelerationStructureBuildSizes(
            const BottomLevelAccelerationStructureBuildInput&) const
        {
            return std::unexpected(unsupportedOperation("queryBottomLevelAccelerationStructureBuildSizes"));
        }
        [[nodiscard]] virtual Result<AccelerationStructureBuildSizes> queryTopLevelAccelerationStructureBuildSizes(
            const TopLevelAccelerationStructureBuildInput&) const
        {
            return std::unexpected(unsupportedOperation("queryTopLevelAccelerationStructureBuildSizes"));
        }
        [[nodiscard]] virtual Result<AccelerationStructureHandle> createAccelerationStructure(
            const AccelerationStructureDesc&)
        {
            return std::unexpected(unsupportedOperation("createAccelerationStructure"));
        }
        [[nodiscard]] virtual Status destroyAccelerationStructure(AccelerationStructureHandle)
        {
            return std::unexpected(unsupportedOperation("destroyAccelerationStructure"));
        }
        [[nodiscard]] virtual Status buildBottomLevelAccelerationStructure(
            BackendCommandBuffer&, const BottomLevelAccelerationStructureBuildDesc&)
        {
            return std::unexpected(unsupportedOperation("buildBottomLevelAccelerationStructure"));
        }
        [[nodiscard]] virtual Status buildTopLevelAccelerationStructure(BackendCommandBuffer&,
                                                                        const TopLevelAccelerationStructureBuildDesc&)
        {
            return std::unexpected(unsupportedOperation("buildTopLevelAccelerationStructure"));
        }
        [[nodiscard]] virtual Status updateBottomLevelAccelerationStructure(
            BackendCommandBuffer&, const BottomLevelAccelerationStructureUpdateDesc&)
        {
            return std::unexpected(unsupportedOperation("updateBottomLevelAccelerationStructure"));
        }
        [[nodiscard]] virtual Status updateTopLevelAccelerationStructure(BackendCommandBuffer&,
                                                                         const TopLevelAccelerationStructureUpdateDesc&)
        {
            return std::unexpected(unsupportedOperation("updateTopLevelAccelerationStructure"));
        }
        [[nodiscard]] virtual Status updateBottomLevelAccelerationStructureInPlace(
            BackendCommandBuffer&, const BottomLevelAccelerationStructureInPlaceUpdateDesc&)
        {
            return std::unexpected(unsupportedOperation("updateBottomLevelAccelerationStructureInPlace"));
        }
        [[nodiscard]] virtual Status updateTopLevelAccelerationStructureInPlace(
            BackendCommandBuffer&, const TopLevelAccelerationStructureInPlaceUpdateDesc&)
        {
            return std::unexpected(unsupportedOperation("updateTopLevelAccelerationStructureInPlace"));
        }
        [[nodiscard]] virtual Status compactAccelerationStructure(BackendCommandBuffer&,
                                                                  const AccelerationStructureCompactionDesc&)
        {
            return std::unexpected(unsupportedOperation("compactAccelerationStructure"));
        }
        [[nodiscard]] virtual Status serializeAccelerationStructure(BackendCommandBuffer&,
                                                                    const AccelerationStructureSerializationDesc&)
        {
            return std::unexpected(unsupportedOperation("serializeAccelerationStructure"));
        }
        [[nodiscard]] virtual Result<AccelerationStructureArchiveInfo> inspectAccelerationStructureArchive(
            std::span<const std::byte>) const
        {
            return std::unexpected(unsupportedOperation("inspectAccelerationStructureArchive"));
        }
        [[nodiscard]] virtual Status deserializeAccelerationStructure(BackendCommandBuffer&,
                                                                      const AccelerationStructureDeserializationDesc&)
        {
            return std::unexpected(unsupportedOperation("deserializeAccelerationStructure"));
        }
        [[nodiscard]] virtual Result<ResourceMemoryRequirements> queryTextureMemoryRequirements(
            const TextureDesc& desc) const = 0;
        [[nodiscard]] virtual Result<TextureHandle> createTexture(const TextureDesc& desc) = 0;
        [[nodiscard]] virtual Result<TextureHandle> createPlacedTexture(const ResourcePlacement&, const TextureDesc&)
        {
            return std::unexpected(unsupportedOperation("createPlacedTexture"));
        }
        [[nodiscard]] virtual Result<TextureHandle> createSparseTexture(const TextureDesc&)
        {
            return std::unexpected(unsupportedOperation("createSparseTexture"));
        }
        [[nodiscard]] virtual Result<SparseTextureProperties> querySparseTextureProperties(TextureHandle) const
        {
            return std::unexpected(unsupportedOperation("querySparseTextureProperties"));
        }
        [[nodiscard]] virtual Result<TimelinePoint> enqueueSparseTextureMappings(QueueType,
                                                                                 const SparseTextureMappingBatch&)
        {
            return std::unexpected(unsupportedOperation("enqueueSparseTextureMappings"));
        }
        [[nodiscard]] virtual Result<SamplerFeedbackMapHandle> createMinMipFeedbackMap(const MinMipFeedbackMapDesc&)
        {
            return std::unexpected(unsupportedOperation("createMinMipFeedbackMap"));
        }
        [[nodiscard]] virtual Result<MinMipFeedbackMapProperties> queryMinMipFeedbackMapProperties(
            SamplerFeedbackMapHandle) const
        {
            return std::unexpected(unsupportedOperation("queryMinMipFeedbackMapProperties"));
        }
        [[nodiscard]] virtual Result<SamplerFeedbackMapHandle> createMipRegionUsedFeedbackMap(
            const MipRegionUsedFeedbackMapDesc&)
        {
            return std::unexpected(unsupportedOperation("createMipRegionUsedFeedbackMap"));
        }
        [[nodiscard]] virtual Result<MipRegionUsedFeedbackMapProperties> queryMipRegionUsedFeedbackMapProperties(
            SamplerFeedbackMapHandle) const
        {
            return std::unexpected(unsupportedOperation("queryMipRegionUsedFeedbackMapProperties"));
        }
        [[nodiscard]] virtual Status destroySamplerFeedbackMap(SamplerFeedbackMapHandle)
        {
            return std::unexpected(unsupportedOperation("destroySamplerFeedbackMap"));
        }
        [[nodiscard]] virtual Status resetSamplerFeedbackMap(BackendCommandBuffer&, SamplerFeedbackMapHandle)
        {
            return std::unexpected(unsupportedOperation("resetSamplerFeedbackMap"));
        }
        [[nodiscard]] virtual Status decodeMinMipFeedback(BackendCommandBuffer&, const MinMipFeedbackDecode&)
        {
            return std::unexpected(unsupportedOperation("decodeMinMipFeedback"));
        }
        [[nodiscard]] virtual Status decodeMipRegionUsedFeedback(BackendCommandBuffer&,
                                                                 const MipRegionUsedFeedbackDecode&)
        {
            return std::unexpected(unsupportedOperation("decodeMipRegionUsedFeedback"));
        }
        [[nodiscard]] virtual Status encodeMinMipFeedback(BackendCommandBuffer&, const MinMipFeedbackEncode&)
        {
            return std::unexpected(unsupportedOperation("encodeMinMipFeedback"));
        }
        [[nodiscard]] virtual Status encodeMipRegionUsedFeedback(BackendCommandBuffer&,
                                                                 const MipRegionUsedFeedbackEncode&)
        {
            return std::unexpected(unsupportedOperation("encodeMipRegionUsedFeedback"));
        }
        [[nodiscard]] virtual Result<TextureHandle> createSharedTexture(const TextureDesc&)
        {
            return std::unexpected(unsupportedOperation("createSharedTexture"));
        }
        [[nodiscard]] virtual Result<SharedHandle> exportSharedTexture(TextureHandle)
        {
            return std::unexpected(unsupportedOperation("exportSharedTexture"));
        }
        [[nodiscard]] virtual Result<TextureHandle> importSharedTexture(SharedHandleView, const TextureDesc&)
        {
            return std::unexpected(unsupportedOperation("importSharedTexture"));
        }
        [[nodiscard]] virtual Status destroyTexture(TextureHandle texture) = 0;
        [[nodiscard]] virtual Result<ResidencyState> queryResidency(TextureHandle) const
        {
            return std::unexpected(unsupportedOperation("queryResidency"));
        }
        [[nodiscard]] virtual Result<TextureTransferLayout> queryTextureTransferLayout(
            TextureHandle texture, const BufferTextureCopyRegion& region) const = 0;
        [[nodiscard]] virtual Result<TimelinePoint> enqueueMakeResident(QueueType,
                                                                        const ResidencySet&,
                                                                        ResidencyBudgetPolicy)
        {
            return std::unexpected(unsupportedOperation("enqueueMakeResident"));
        }
        [[nodiscard]] virtual Status evict(const ResidencySet&)
        {
            return std::unexpected(unsupportedOperation("evict"));
        }
        [[nodiscard]] virtual Status copyBufferToTextureRegions(BackendCommandBuffer& commandBuffer,
                                                                BufferHandle source,
                                                                TextureHandle destination,
                                                                std::span<const BufferTextureCopyRegion> regions) = 0;
        [[nodiscard]] virtual Status copyTextureToBufferRegions(BackendCommandBuffer& commandBuffer,
                                                                TextureHandle source,
                                                                BufferHandle destination,
                                                                std::span<const BufferTextureCopyRegion> regions) = 0;
        [[nodiscard]] virtual Status copyTextureRegions(BackendCommandBuffer& commandBuffer,
                                                        TextureHandle source,
                                                        TextureHandle destination,
                                                        std::span<const TextureCopyRegion> regions) = 0;
        [[nodiscard]] virtual Status resolveTexture(BackendCommandBuffer& commandBuffer,
                                                    const TextureResolve& resolve) = 0;
        [[nodiscard]] virtual Status generateMipmaps(BackendCommandBuffer& commandBuffer,
                                                     const TextureMipmapGeneration& generation) = 0;
        [[nodiscard]] virtual Status textureBarrier(BackendCommandBuffer& commandBuffer,
                                                    const TextureBarrier& barrier) = 0;
        [[nodiscard]] virtual Status activateTextureAlias(BackendCommandBuffer&, const TextureAliasActivation&)
        {
            return std::unexpected(unsupportedOperation("activateTextureAlias"));
        }
        [[nodiscard]] virtual Result<TextureViewHandle> createTextureView(const TextureViewDesc& desc) = 0;
        [[nodiscard]] virtual Status destroyTextureView(TextureViewHandle view) = 0;
        [[nodiscard]] virtual Result<SamplerHandle> createSampler(const SamplerDesc& desc) = 0;
        [[nodiscard]] virtual Status destroySampler(SamplerHandle sampler) = 0;
        [[nodiscard]] virtual Result<BindingLayoutHandle> createBindingLayout(const BindingLayoutDesc& desc) = 0;
        [[nodiscard]] virtual Status destroyBindingLayout(BindingLayoutHandle layout) = 0;
        [[nodiscard]] virtual Result<BindingPoolHandle> createBindingPool(const BindingPoolDesc& desc) = 0;
        [[nodiscard]] virtual Status destroyBindingPool(BindingPoolHandle pool) = 0;
        [[nodiscard]] virtual Result<BindingSetHandle> createBindingSet(const BindingSetDesc& desc) = 0;
        [[nodiscard]] virtual Status updateBindingSet(BindingSetHandle set, std::span<const BindingWrite> writes) = 0;
        [[nodiscard]] virtual Status destroyBindingSet(BindingSetHandle set) = 0;
        [[nodiscard]] virtual Result<ShaderLibraryHandle> createShaderLibrary(const ShaderLibraryDesc& desc) = 0;
        [[nodiscard]] virtual Status destroyShaderLibrary(ShaderLibraryHandle shaderLibrary) = 0;
        [[nodiscard]] virtual Result<ShaderModuleHandle> createShaderModule(const ShaderModuleDesc& desc) = 0;
        [[nodiscard]] virtual Result<ShaderInterfaceSnapshot> shaderModuleInterface(
            ShaderModuleHandle shaderModule) const = 0;
        [[nodiscard]] virtual Status destroyShaderModule(ShaderModuleHandle shaderModule) = 0;
        [[nodiscard]] virtual Result<PipelineLayoutHandle> createPipelineLayout(const PipelineLayoutDesc& desc) = 0;
        [[nodiscard]] virtual Status destroyPipelineLayout(PipelineLayoutHandle pipelineLayout) = 0;
        [[nodiscard]] virtual Result<PipelineCacheHandle> createPipelineCache(const PipelineCacheDesc& desc) = 0;
        [[nodiscard]] virtual Status destroyPipelineCache(PipelineCacheHandle pipelineCache) = 0;
        [[nodiscard]] virtual Result<std::vector<std::byte>> pipelineCacheData(PipelineCacheHandle pipelineCache) = 0;
        [[nodiscard]] virtual Result<PipelineHandle> createComputePipeline(const ComputePipelineDesc& desc) = 0;
        [[nodiscard]] virtual Result<PipelineHandle> createGraphicsPipeline(const GraphicsPipelineDesc& desc) = 0;
        [[nodiscard]] virtual Result<PipelineCompilationHandle> beginComputePipelineCompilation(
            const ComputePipelineDesc& desc) = 0;
        [[nodiscard]] virtual Result<PipelineCompilationHandle> beginGraphicsPipelineCompilation(
            const GraphicsPipelineDesc& desc) = 0;
        [[nodiscard]] virtual Result<PipelineCompilationState> pipelineCompilationState(
            PipelineCompilationHandle compilation) = 0;
        [[nodiscard]] virtual Result<PipelineHandle> waitPipelineCompilation(PipelineCompilationHandle compilation) = 0;
        [[nodiscard]] virtual Status destroyPipelineCompilation(PipelineCompilationHandle compilation) = 0;
        [[nodiscard]] virtual Result<RayTracingPipelineLibraryHandle> createRayTracingPipelineLibrary(
            const RayTracingPipelineLibraryDesc&)
        {
            return std::unexpected(unsupportedOperation("createRayTracingPipelineLibrary"));
        }
        [[nodiscard]] virtual Status destroyRayTracingPipelineLibrary(RayTracingPipelineLibraryHandle)
        {
            return std::unexpected(unsupportedOperation("destroyRayTracingPipelineLibrary"));
        }
        [[nodiscard]] virtual Result<PipelineHandle> createRayTracingPipeline(const RayTracingPipelineDesc&)
        {
            return std::unexpected(unsupportedOperation("createRayTracingPipeline"));
        }
        [[nodiscard]] virtual Status getRayTracingShaderGroupHandles(PipelineHandle,
                                                                     std::uint32_t,
                                                                     std::uint32_t,
                                                                     std::span<std::byte>)
        {
            return std::unexpected(unsupportedOperation("getRayTracingShaderGroupHandles"));
        }
        [[nodiscard]] virtual Result<std::uint32_t> rayTracingShaderGroupRecordSize(PipelineHandle, std::uint32_t)
        {
            return std::unexpected(unsupportedOperation("rayTracingShaderGroupRecordSize"));
        }
        [[nodiscard]] virtual Status writeRayTracingShaderGroupRecord(PipelineHandle,
                                                                      std::uint32_t,
                                                                      std::span<const std::byte>,
                                                                      std::span<std::byte>)
        {
            return std::unexpected(unsupportedOperation("writeRayTracingShaderGroupRecord"));
        }
        [[nodiscard]] virtual Status destroyPipeline(PipelineHandle pipeline) = 0;
        [[nodiscard]] virtual Status bindComputePipeline(BackendCommandBuffer& commandBuffer,
                                                         PipelineHandle pipeline) = 0;
        [[nodiscard]] virtual Status bindComputeBindingSet(BackendCommandBuffer& commandBuffer,
                                                           std::uint32_t setIndex,
                                                           BindingSetHandle bindingSet,
                                                           std::span<const std::uint32_t> dynamicOffsets) = 0;
        [[nodiscard]] virtual Status bindComputeBindingSets(BackendCommandBuffer& commandBuffer,
                                                            std::uint32_t firstSet,
                                                            std::span<const BindingSetHandle> bindingSets,
                                                            std::span<const std::uint32_t> dynamicOffsets) = 0;
        [[nodiscard]] virtual Status setComputePushConstants(BackendCommandBuffer& commandBuffer,
                                                             std::uint32_t offset,
                                                             std::span<const std::byte> data) = 0;
        [[nodiscard]] virtual Status dispatch(BackendCommandBuffer& commandBuffer,
                                              std::uint32_t groupCountX,
                                              std::uint32_t groupCountY,
                                              std::uint32_t groupCountZ) = 0;
        [[nodiscard]] virtual Status dispatchIndirect(BackendCommandBuffer& commandBuffer,
                                                      const IndirectDispatchDesc& desc) = 0;
        [[nodiscard]] virtual Status bindRayTracingPipeline(BackendCommandBuffer&, PipelineHandle)
        {
            return std::unexpected(unsupportedOperation("bindRayTracingPipeline"));
        }
        [[nodiscard]] virtual Status bindRayTracingBindingSet(BackendCommandBuffer&,
                                                              std::uint32_t,
                                                              BindingSetHandle,
                                                              std::span<const std::uint32_t>)
        {
            return std::unexpected(unsupportedOperation("bindRayTracingBindingSet"));
        }
        [[nodiscard]] virtual Status bindRayTracingBindingSets(BackendCommandBuffer&,
                                                               std::uint32_t,
                                                               std::span<const BindingSetHandle>,
                                                               std::span<const std::uint32_t>)
        {
            return std::unexpected(unsupportedOperation("bindRayTracingBindingSets"));
        }
        [[nodiscard]] virtual Status setRayTracingPushConstants(BackendCommandBuffer&,
                                                                std::uint32_t,
                                                                std::span<const std::byte>)
        {
            return std::unexpected(unsupportedOperation("setRayTracingPushConstants"));
        }
        [[nodiscard]] virtual Status dispatchRays(BackendCommandBuffer&, const RayTracingDispatchDesc&)
        {
            return std::unexpected(unsupportedOperation("dispatchRays"));
        }
        [[nodiscard]] virtual Status dispatchRaysIndirect(BackendCommandBuffer&, const RayTracingIndirectDispatchDesc&)
        {
            return std::unexpected(unsupportedOperation("dispatchRaysIndirect"));
        }
        [[nodiscard]] virtual Status beginRendering(BackendCommandBuffer& commandBuffer, const RenderingDesc& desc) = 0;
        [[nodiscard]] virtual Status endRendering(BackendCommandBuffer& commandBuffer) = 0;
        [[nodiscard]] virtual Status clearColorAttachment(BackendCommandBuffer& commandBuffer,
                                                          std::uint32_t attachmentIndex,
                                                          const ClearColor& clearValue,
                                                          std::span<const ScissorRect> rectangles) = 0;
        [[nodiscard]] virtual Status clearDepthStencilAttachment(BackendCommandBuffer& commandBuffer,
                                                                 DepthStencilClearAspect aspects,
                                                                 float depth,
                                                                 std::uint32_t stencil,
                                                                 std::span<const ScissorRect> rectangles) = 0;
        [[nodiscard]] virtual Status bindGraphicsPipeline(BackendCommandBuffer& commandBuffer,
                                                          PipelineHandle pipeline) = 0;
        [[nodiscard]] virtual Status bindGraphicsBindingSet(BackendCommandBuffer& commandBuffer,
                                                            std::uint32_t setIndex,
                                                            BindingSetHandle bindingSet,
                                                            std::span<const std::uint32_t> dynamicOffsets) = 0;
        [[nodiscard]] virtual Status bindGraphicsBindingSets(BackendCommandBuffer& commandBuffer,
                                                             std::uint32_t firstSet,
                                                             std::span<const BindingSetHandle> bindingSets,
                                                             std::span<const std::uint32_t> dynamicOffsets) = 0;
        [[nodiscard]] virtual Status setGraphicsPushConstants(BackendCommandBuffer& commandBuffer,
                                                              ShaderStage visibility,
                                                              std::uint32_t offset,
                                                              std::span<const std::byte> data) = 0;
        [[nodiscard]] virtual Status setViewports(BackendCommandBuffer& commandBuffer,
                                                  std::span<const Viewport> viewports) = 0;
        [[nodiscard]] virtual Status setScissors(BackendCommandBuffer& commandBuffer,
                                                 std::span<const ScissorRect> scissors) = 0;
        [[nodiscard]] virtual Status setBlendConstants(BackendCommandBuffer& commandBuffer,
                                                       const BlendConstants& constants) = 0;
        [[nodiscard]] virtual Status setDepthBias(BackendCommandBuffer&, const DepthBias&)
        {
            return std::unexpected(unsupportedOperation("setDepthBias"));
        }
        [[nodiscard]] virtual Status setDepthBounds(BackendCommandBuffer&, const DepthBounds&)
        {
            return std::unexpected(unsupportedOperation("setDepthBounds"));
        }
        [[nodiscard]] virtual Status setStencilReference(BackendCommandBuffer& commandBuffer,
                                                         StencilFace face,
                                                         std::uint32_t reference) = 0;
        [[nodiscard]] virtual Status setShadingRate(BackendCommandBuffer&, ShadingRate, const ShadingRateCombiners&)
        {
            return std::unexpected(unsupportedOperation("setShadingRate"));
        }
        [[nodiscard]] virtual Status setShadingRateImage(BackendCommandBuffer&, TextureHandle)
        {
            return std::unexpected(unsupportedOperation("setShadingRateImage"));
        }
        [[nodiscard]] virtual Status bindVertexBuffers(BackendCommandBuffer& commandBuffer,
                                                       std::uint32_t firstSlot,
                                                       std::span<const VertexBufferBinding> bindings) = 0;
        [[nodiscard]] virtual Status bindStreamOutputBuffers(BackendCommandBuffer&,
                                                             std::uint32_t,
                                                             std::span<const StreamOutputBufferBinding>)
        {
            return std::unexpected(unsupportedOperation("bindStreamOutputBuffers"));
        }
        [[nodiscard]] virtual Status bindIndexBuffer(BackendCommandBuffer& commandBuffer,
                                                     BufferHandle buffer,
                                                     IndexFormat format,
                                                     std::uint64_t offset) = 0;
        [[nodiscard]] virtual Status draw(BackendCommandBuffer& commandBuffer,
                                          std::uint32_t vertexCount,
                                          std::uint32_t instanceCount,
                                          std::uint32_t firstVertex,
                                          std::uint32_t firstInstance) = 0;
        [[nodiscard]] virtual Status drawIndexed(BackendCommandBuffer& commandBuffer,
                                                 std::uint32_t indexCount,
                                                 std::uint32_t instanceCount,
                                                 std::uint32_t firstIndex,
                                                 std::int32_t baseVertex,
                                                 std::uint32_t firstInstance) = 0;
        [[nodiscard]] virtual Status dispatchMesh(BackendCommandBuffer&, std::uint32_t, std::uint32_t, std::uint32_t)
        {
            return std::unexpected(unsupportedOperation("dispatchMesh"));
        }
        [[nodiscard]] virtual Status dispatchMeshIndirect(BackendCommandBuffer&, const IndirectMeshDispatchDesc&)
        {
            return std::unexpected(unsupportedOperation("dispatchMeshIndirect"));
        }
        [[nodiscard]] virtual Status drawIndirect(BackendCommandBuffer& commandBuffer,
                                                  const IndirectDrawDesc& desc) = 0;
        [[nodiscard]] virtual Status drawIndexedIndirect(BackendCommandBuffer& commandBuffer,
                                                         const IndirectDrawDesc& desc) = 0;
        [[nodiscard]] virtual Status beginConditionalExecution(BackendCommandBuffer&, const ConditionalExecutionDesc&)
        {
            return std::unexpected(unsupportedOperation("beginConditionalExecution"));
        }
        [[nodiscard]] virtual Status endConditionalExecution(BackendCommandBuffer&)
        {
            return std::unexpected(unsupportedOperation("endConditionalExecution"));
        }
        [[nodiscard]] virtual Result<QueryPoolHandle> createTimestampQueryPool(const TimestampQueryPoolDesc&)
        {
            return std::unexpected(unsupportedOperation("createTimestampQueryPool"));
        }
        [[nodiscard]] virtual Status destroyQueryPool(QueryPoolHandle)
        {
            return std::unexpected(unsupportedOperation("destroyQueryPool"));
        }
        [[nodiscard]] virtual Status resetQueryPool(BackendCommandBuffer&,
                                                    QueryPoolHandle,
                                                    std::uint32_t,
                                                    std::uint32_t)
        {
            return std::unexpected(unsupportedOperation("resetQueryPool"));
        }
        [[nodiscard]] virtual Status writeTimestamp(BackendCommandBuffer&, QueryPoolHandle, std::uint32_t)
        {
            return std::unexpected(unsupportedOperation("writeTimestamp"));
        }
        [[nodiscard]] virtual Status beginGpuProfileScope(BackendCommandBuffer&, const GpuProfileScopeDesc&)
        {
            return std::unexpected(unsupportedOperation("beginGpuProfileScope"));
        }
        [[nodiscard]] virtual Status endGpuProfileScope(BackendCommandBuffer&)
        {
            return std::unexpected(unsupportedOperation("endGpuProfileScope"));
        }
        [[nodiscard]] virtual Status resolveTimestamps(BackendCommandBuffer&, const TimestampQueryResolveDesc&)
        {
            return std::unexpected(unsupportedOperation("resolveTimestamps"));
        }
        [[nodiscard]] virtual Result<QueryPoolHandle> createOcclusionQueryPool(const OcclusionQueryPoolDesc&)
        {
            return std::unexpected(unsupportedOperation("createOcclusionQueryPool"));
        }
        [[nodiscard]] virtual Status beginOcclusionQuery(BackendCommandBuffer&, QueryPoolHandle, std::uint32_t)
        {
            return std::unexpected(unsupportedOperation("beginOcclusionQuery"));
        }
        [[nodiscard]] virtual Status endOcclusionQuery(BackendCommandBuffer&)
        {
            return std::unexpected(unsupportedOperation("endOcclusionQuery"));
        }
        [[nodiscard]] virtual Status resolveOcclusionQueries(BackendCommandBuffer&, const OcclusionQueryResolveDesc&)
        {
            return std::unexpected(unsupportedOperation("resolveOcclusionQueries"));
        }
        [[nodiscard]] virtual Result<QueryPoolHandle> createPipelineStatisticsQueryPool(
            const PipelineStatisticsQueryPoolDesc&)
        {
            return std::unexpected(unsupportedOperation("createPipelineStatisticsQueryPool"));
        }
        [[nodiscard]] virtual Status beginPipelineStatisticsQuery(BackendCommandBuffer&, QueryPoolHandle, std::uint32_t)
        {
            return std::unexpected(unsupportedOperation("beginPipelineStatisticsQuery"));
        }
        [[nodiscard]] virtual Status endPipelineStatisticsQuery(BackendCommandBuffer&)
        {
            return std::unexpected(unsupportedOperation("endPipelineStatisticsQuery"));
        }
        [[nodiscard]] virtual Status resolvePipelineStatisticsQueries(BackendCommandBuffer&,
                                                                      const PipelineStatisticsQueryResolveDesc&)
        {
            return std::unexpected(unsupportedOperation("resolvePipelineStatisticsQueries"));
        }
        [[nodiscard]] virtual Result<QueryPoolHandle> createStreamOutputStatisticsQueryPool(
            const StreamOutputStatisticsQueryPoolDesc&)
        {
            return std::unexpected(unsupportedOperation("createStreamOutputStatisticsQueryPool"));
        }
        [[nodiscard]] virtual Status beginStreamOutputStatisticsQuery(BackendCommandBuffer&,
                                                                      QueryPoolHandle,
                                                                      std::uint32_t)
        {
            return std::unexpected(unsupportedOperation("beginStreamOutputStatisticsQuery"));
        }
        [[nodiscard]] virtual Status endStreamOutputStatisticsQuery(BackendCommandBuffer&, std::uint32_t)
        {
            return std::unexpected(unsupportedOperation("endStreamOutputStatisticsQuery"));
        }
        [[nodiscard]] virtual Status resolveStreamOutputStatisticsQueries(BackendCommandBuffer&,
                                                                          const StreamOutputStatisticsQueryResolveDesc&)
        {
            return std::unexpected(unsupportedOperation("resolveStreamOutputStatisticsQueries"));
        }
        [[nodiscard]] virtual Result<QueryPoolHandle> createAccelerationStructureCompactedSizeQueryPool(
            const AccelerationStructureCompactedSizeQueryPoolDesc&)
        {
            return std::unexpected(unsupportedOperation("createAccelerationStructureCompactedSizeQueryPool"));
        }
        [[nodiscard]] virtual Status writeAccelerationStructureCompactedSize(BackendCommandBuffer&,
                                                                             QueryPoolHandle,
                                                                             std::uint32_t,
                                                                             AccelerationStructureHandle)
        {
            return std::unexpected(unsupportedOperation("writeAccelerationStructureCompactedSize"));
        }
        [[nodiscard]] virtual Status resolveAccelerationStructureCompactedSizes(
            BackendCommandBuffer&, const AccelerationStructureCompactedSizeQueryResolveDesc&)
        {
            return std::unexpected(unsupportedOperation("resolveAccelerationStructureCompactedSizes"));
        }
        [[nodiscard]] virtual Result<QueryPoolHandle> createAccelerationStructureSerializationInfoQueryPool(
            const AccelerationStructureSerializationInfoQueryPoolDesc&)
        {
            return std::unexpected(unsupportedOperation("createAccelerationStructureSerializationInfoQueryPool"));
        }
        [[nodiscard]] virtual Status writeAccelerationStructureSerializationInfo(BackendCommandBuffer&,
                                                                                 QueryPoolHandle,
                                                                                 std::uint32_t,
                                                                                 AccelerationStructureHandle)
        {
            return std::unexpected(unsupportedOperation("writeAccelerationStructureSerializationInfo"));
        }
        [[nodiscard]] virtual Status resolveAccelerationStructureSerializationInfo(
            BackendCommandBuffer&, const AccelerationStructureSerializationInfoQueryResolveDesc&)
        {
            return std::unexpected(unsupportedOperation("resolveAccelerationStructureSerializationInfo"));
        }
        [[nodiscard]] virtual Result<SurfaceHandle> createSurface(const SurfaceDesc& desc) = 0;
        [[nodiscard]] virtual Result<SurfaceCapabilities> querySurfaceCapabilities(SurfaceHandle surface) const = 0;
        [[nodiscard]] virtual Status destroySurface(SurfaceHandle surface) = 0;
        [[nodiscard]] virtual Result<SwapchainHandle> createSwapchain(const SwapchainDesc& desc) = 0;
        [[nodiscard]] virtual Status destroySwapchain(SwapchainHandle swapchain) = 0;
        [[nodiscard]] virtual Status resizeSwapchain(SwapchainHandle swapchain,
                                                     std::uint32_t width,
                                                     std::uint32_t height) = 0;
        [[nodiscard]] virtual Result<AcquiredSwapchainImage> acquireNextImage(SwapchainHandle swapchain) = 0;
        [[nodiscard]] virtual bool hasAcquiredSwapchainImage(SwapchainHandle swapchain) const noexcept = 0;
        [[nodiscard]] virtual Status cancelSwapchainImage(SwapchainHandle swapchain) = 0;
        [[nodiscard]] virtual Result<PresentStatus> present(SwapchainHandle swapchain,
                                                            TimelinePoint renderComplete) = 0;
        [[nodiscard]] virtual Result<SwapchainPresentationInfo> querySwapchainPresentation(
            SwapchainHandle swapchain) const = 0;
        [[nodiscard]] virtual Result<SwapchainFrameTiming> querySwapchainFrameTiming(
            SwapchainHandle swapchain) const = 0;

    protected:
        BackendDevice() = default;

        [[nodiscard]] Error unsupportedOperation(std::string_view operation) const
        {
            if (operationalState() == DeviceOperationalState::lost)
            {
                return deviceLossError().value_or(Error{
                    ErrorCode::deviceLost,
                    0,
                    "The device is lost.",
                });
            }
            return unsupportedBackendOperation(backendName(), operation);
        }
    };

} // namespace rachis::detail
