#pragma once

#include "Common/TimedMutex.h"
#include "Internal/BackendDevice.h"
#include "Internal/MetalDiagnostics.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

namespace rachis::metal
{
#ifdef RACHIS_TESTING
    class MetalCommitFeedbackGate;
#endif

    struct MetalDeviceNative;

    class MetalDevice final : public detail::BackendDevice
    {
    public:
        MetalDevice(AdapterInfo adapter,
                    DeviceCapabilities capabilities,
                    std::vector<FormatProperties> formats,
                    QueueProperties graphicsQueue,
                    std::shared_ptr<MetalDeviceNative> native);
        ~MetalDevice() override;

        MetalDevice(const MetalDevice&) = delete;
        MetalDevice& operator=(const MetalDevice&) = delete;
        MetalDevice(MetalDevice&&) = delete;
        MetalDevice& operator=(MetalDevice&&) = delete;

        [[nodiscard]] std::string_view backendName() const noexcept override
        {
            return "Metal";
        }
        [[nodiscard]] std::uint64_t deviceToken() const noexcept override
        {
            return deviceToken_;
        }
        [[nodiscard]] const AdapterInfo& adapterInfo() const noexcept override;
        [[nodiscard]] const DeviceCapabilities& capabilities() const noexcept override;
        [[nodiscard]] DeviceOperationalState operationalState() const noexcept override;
        [[nodiscard]] std::optional<Error> deviceLossError() const override;
        [[nodiscard]] Result<std::vector<MemoryBudget>> queryMemoryBudgets(std::uint32_t nodeIndex) const override;
        [[nodiscard]] Result<ResourceMemorySnapshot> captureResourceMemorySnapshot() const override;
        [[nodiscard]] Result<DeviceFaultReport> captureDeviceFaultReport() const override;
        [[nodiscard]] BackendAllocationSnapshot captureBackendAllocationSnapshot() const noexcept override;
        [[nodiscard]] FormatProperties queryFormatProperties(Format format) const noexcept override;
        [[nodiscard]] QueueProperties queueProperties(QueueType queueType) const noexcept override;
        [[nodiscard]] DiagnosticCounts diagnosticCounts() const noexcept override;
        [[nodiscard]] BindingUpdateStatistics bindingUpdateStatistics() const noexcept override;
        [[nodiscard]] MetalValidationConfiguration validationConfiguration() const;
        [[nodiscard]] Result<MetalCaptureRecord> captureLabeledHeadlessCommand(const MetalCaptureRequest& request);
        [[nodiscard]] Status emitIntentionalValidationFailure();
#ifdef RACHIS_TESTING
        void failNextCommitFeedback() noexcept;
        void holdNextCommitFeedback(std::shared_ptr<MetalCommitFeedbackGate> gate) noexcept;
        [[nodiscard]] std::weak_ptr<void> bufferOwnerObserverForTesting(BufferHandle buffer) const noexcept;
        [[nodiscard]] std::size_t allocationTelemetryCountForTesting() const noexcept;
        [[nodiscard]] Status injectDeviceLoss(DeviceFaultStatus status = DeviceFaultStatus::removed);
        [[nodiscard]] Status notifySurfaceHostCloseForTesting(SurfaceHandle surface);
        [[nodiscard]] void* nativeMetalDeviceHandle() const noexcept;
        [[nodiscard]] std::uint64_t samplerGpuResourceIdForTesting(SamplerHandle sampler) const noexcept;
#endif
        [[nodiscard]] Status copyBuffersImmediate(BufferHandle source,
                                                  BufferHandle destination,
                                                  std::uint64_t sourceOffset,
                                                  std::uint64_t destinationOffset,
                                                  std::uint64_t size);
#ifdef RACHIS_TESTING
        [[nodiscard]] Status copyCapturedBindingBuffer(detail::BackendCommandBuffer& commandBuffer,
                                                       std::uint32_t setIndex,
                                                       BufferHandle destination,
                                                       std::uint64_t size);
#endif
        [[nodiscard]] Status waitIdle(QueueType queueType) override;
        [[nodiscard]] Status waitIdle() override;
        [[nodiscard]] std::uint64_t completedValue(QueueType queueType) const noexcept override;
        void collectGarbage() noexcept override;

        [[nodiscard]] Result<QueueClockCalibration> calibrateQueueClock(QueueType queueType) const override;
        [[nodiscard]] Result<CommandPoolHandle> createCommandPool(const CommandPoolDesc& desc) override;
        [[nodiscard]] Status destroyCommandPool(CommandPoolHandle commandPool) override;
        [[nodiscard]] Status resetCommandPool(CommandPoolHandle commandPool) override;
        [[nodiscard]] Result<std::unique_ptr<detail::BackendCommandBuffer>> acquireCommandBuffer(
            CommandPoolHandle commandPool) override;
        [[nodiscard]] Status resetCommandBuffer(detail::BackendCommandBuffer& commandBuffer) override;
        [[nodiscard]] Result<TimelinePoint> submit(
            std::span<std::unique_ptr<detail::BackendCommandBuffer>> commandBuffers,
            std::span<const TimelinePoint> waits) override;
        [[nodiscard]] Result<TimelinePointStatus> queryTimelinePoint(TimelinePoint point) const override;
        [[nodiscard]] Result<TimelinePointStatus> wait(TimelinePoint point, std::chrono::nanoseconds timeout) override;
        [[nodiscard]] Status wait(TimelinePoint point) override;
        [[nodiscard]] Status beginDebugGroup(detail::BackendCommandBuffer& commandBuffer,
                                             const DebugMarkerDesc& desc) override;
        [[nodiscard]] Status endDebugGroup(detail::BackendCommandBuffer& commandBuffer) override;
        [[nodiscard]] Status insertDebugMarker(detail::BackendCommandBuffer& commandBuffer,
                                               const DebugMarkerDesc& desc) override;
        [[nodiscard]] Result<ResourceMemoryRequirements> queryBufferMemoryRequirements(
            const BufferDesc& desc) const override;
        [[nodiscard]] Result<ResourceHeapHandle> createResourceHeap(const ResourceHeapDesc& desc) override;
        [[nodiscard]] Status destroyResourceHeap(ResourceHeapHandle heap) override;
        [[nodiscard]] Result<ResidencyState> queryResidency(ResourceHeapHandle heap) const override;
        [[nodiscard]] Result<BufferHandle> createBuffer(const BufferDesc& desc) override;
        [[nodiscard]] Result<BufferHandle> createPlacedBuffer(const ResourcePlacement& placement,
                                                              const BufferDesc& desc) override;
        [[nodiscard]] Result<ResidencyState> queryResidency(BufferHandle buffer) const override;
        [[nodiscard]] Status destroyBuffer(BufferHandle buffer) override;
        [[nodiscard]] Result<MappedMemory> mapBuffer(BufferHandle buffer,
                                                     std::uint64_t offset,
                                                     std::uint64_t size) override;
        [[nodiscard]] Status unmapBuffer(BufferHandle buffer, std::uint64_t offset, std::uint64_t size) override;
        [[nodiscard]] Status flushMappedMemoryRanges(BufferHandle buffer,
                                                     std::span<const MappedMemoryRange> ranges) override;
        [[nodiscard]] Status invalidateMappedMemoryRanges(BufferHandle buffer,
                                                          std::span<const MappedMemoryRange> ranges) override;
        [[nodiscard]] Status writeBuffer(BufferHandle buffer,
                                         std::uint64_t offset,
                                         std::span<const std::byte> data) override;
        [[nodiscard]] Status readBuffer(BufferHandle buffer,
                                        std::uint64_t offset,
                                        std::span<std::byte> destination) override;
        [[nodiscard]] Status copyBufferRegions(detail::BackendCommandBuffer& commandBuffer,
                                               BufferHandle source,
                                               BufferHandle destination,
                                               std::span<const BufferCopyRegion> regions) override;
        [[nodiscard]] Status bufferBarrier(detail::BackendCommandBuffer& commandBuffer,
                                           const BufferBarrier& barrier) override;
        [[nodiscard]] Status resourceBarriers(detail::BackendCommandBuffer& commandBuffer,
                                              const ResourceBarrierBatch& barriers) override;
        [[nodiscard]] Status activateBufferAlias(detail::BackendCommandBuffer& commandBuffer,
                                                 const BufferAliasActivation& activation) override;
        [[nodiscard]] Result<ResourceMemoryRequirements> queryTextureMemoryRequirements(
            const TextureDesc& desc) const override;
        [[nodiscard]] Result<TextureHandle> createTexture(const TextureDesc& desc) override;
        [[nodiscard]] Result<TextureHandle> createPlacedTexture(const ResourcePlacement& placement,
                                                                const TextureDesc& desc) override;
        [[nodiscard]] Result<ResidencyState> queryResidency(TextureHandle texture) const override;
        [[nodiscard]] Status destroyTexture(TextureHandle texture) override;
        [[nodiscard]] Result<TextureTransferLayout> queryTextureTransferLayout(
            TextureHandle texture, const BufferTextureCopyRegion& region) const override;
        [[nodiscard]] Status copyBufferToTextureRegions(detail::BackendCommandBuffer& commandBuffer,
                                                        BufferHandle source,
                                                        TextureHandle destination,
                                                        std::span<const BufferTextureCopyRegion> regions) override;
        [[nodiscard]] Status copyTextureToBufferRegions(detail::BackendCommandBuffer& commandBuffer,
                                                        TextureHandle source,
                                                        BufferHandle destination,
                                                        std::span<const BufferTextureCopyRegion> regions) override;
        [[nodiscard]] Status copyTextureRegions(detail::BackendCommandBuffer& commandBuffer,
                                                TextureHandle source,
                                                TextureHandle destination,
                                                std::span<const TextureCopyRegion> regions) override;
        [[nodiscard]] Status resolveTexture(detail::BackendCommandBuffer& commandBuffer,
                                            const TextureResolve& resolve) override;
        [[nodiscard]] Status generateMipmaps(detail::BackendCommandBuffer& commandBuffer,
                                             const TextureMipmapGeneration& generation) override;
        [[nodiscard]] Status textureBarrier(detail::BackendCommandBuffer& commandBuffer,
                                            const TextureBarrier& barrier) override;
        [[nodiscard]] Status activateTextureAlias(detail::BackendCommandBuffer& commandBuffer,
                                                  const TextureAliasActivation& activation) override;
        [[nodiscard]] Result<TextureViewHandle> createTextureView(const TextureViewDesc& desc) override;
        [[nodiscard]] Status destroyTextureView(TextureViewHandle view) override;
        [[nodiscard]] Result<SamplerHandle> createSampler(const SamplerDesc& desc) override;
        [[nodiscard]] Status destroySampler(SamplerHandle sampler) override;
        [[nodiscard]] Result<BindingLayoutHandle> createBindingLayout(const BindingLayoutDesc& desc) override;
        [[nodiscard]] Status destroyBindingLayout(BindingLayoutHandle layout) override;
        [[nodiscard]] Result<BindingPoolHandle> createBindingPool(const BindingPoolDesc& desc) override;
        [[nodiscard]] Status destroyBindingPool(BindingPoolHandle pool) override;
        [[nodiscard]] Result<BindingSetHandle> createBindingSet(const BindingSetDesc& desc) override;
        [[nodiscard]] Status updateBindingSet(BindingSetHandle set, std::span<const BindingWrite> writes) override;
        [[nodiscard]] Status destroyBindingSet(BindingSetHandle set) override;
        [[nodiscard]] Result<ShaderLibraryHandle> createShaderLibrary(const ShaderLibraryDesc& desc) override;
        [[nodiscard]] Status destroyShaderLibrary(ShaderLibraryHandle shaderLibrary) override;
        [[nodiscard]] Result<ShaderModuleHandle> createShaderModule(const ShaderModuleDesc& desc) override;
        [[nodiscard]] Result<ShaderInterfaceSnapshot> shaderModuleInterface(
            ShaderModuleHandle shaderModule) const override;
        [[nodiscard]] Status destroyShaderModule(ShaderModuleHandle shaderModule) override;
        [[nodiscard]] Result<PipelineLayoutHandle> createPipelineLayout(const PipelineLayoutDesc& desc) override;
        [[nodiscard]] Status destroyPipelineLayout(PipelineLayoutHandle pipelineLayout) override;
        [[nodiscard]] Result<PipelineCacheHandle> createPipelineCache(const PipelineCacheDesc& desc) override;
        [[nodiscard]] Status destroyPipelineCache(PipelineCacheHandle pipelineCache) override;
        [[nodiscard]] Result<std::vector<std::byte>> pipelineCacheData(PipelineCacheHandle pipelineCache) override;
        [[nodiscard]] Result<PipelineHandle> createComputePipeline(const ComputePipelineDesc& desc) override;
        [[nodiscard]] Result<PipelineHandle> createGraphicsPipeline(const GraphicsPipelineDesc& desc) override;
        [[nodiscard]] Result<PipelineCompilationHandle> beginComputePipelineCompilation(
            const ComputePipelineDesc& desc) override;
        [[nodiscard]] Result<PipelineCompilationHandle> beginGraphicsPipelineCompilation(
            const GraphicsPipelineDesc& desc) override;
        [[nodiscard]] Result<PipelineCompilationState> pipelineCompilationState(
            PipelineCompilationHandle compilation) override;
        [[nodiscard]] Result<PipelineHandle> waitPipelineCompilation(PipelineCompilationHandle compilation) override;
        [[nodiscard]] Status destroyPipelineCompilation(PipelineCompilationHandle compilation) override;
        [[nodiscard]] Status destroyPipeline(PipelineHandle pipeline) override;
        [[nodiscard]] Status bindComputePipeline(detail::BackendCommandBuffer& commandBuffer,
                                                 PipelineHandle pipeline) override;
        [[nodiscard]] Status bindComputeBindingSet(detail::BackendCommandBuffer& commandBuffer,
                                                   std::uint32_t setIndex,
                                                   BindingSetHandle bindingSet,
                                                   std::span<const std::uint32_t> dynamicOffsets) override;
        [[nodiscard]] Status bindComputeBindingSets(detail::BackendCommandBuffer& commandBuffer,
                                                    std::uint32_t firstSet,
                                                    std::span<const BindingSetHandle> bindingSets,
                                                    std::span<const std::uint32_t> dynamicOffsets) override;
        [[nodiscard]] Status setComputePushConstants(detail::BackendCommandBuffer& commandBuffer,
                                                     std::uint32_t offset,
                                                     std::span<const std::byte> data) override;
        [[nodiscard]] Status dispatch(detail::BackendCommandBuffer& commandBuffer,
                                      std::uint32_t groupCountX,
                                      std::uint32_t groupCountY,
                                      std::uint32_t groupCountZ) override;
        [[nodiscard]] Status dispatchIndirect(detail::BackendCommandBuffer& commandBuffer,
                                              const IndirectDispatchDesc& desc) override;
        [[nodiscard]] Status beginRendering(detail::BackendCommandBuffer& commandBuffer,
                                            const RenderingDesc& desc) override;
        [[nodiscard]] Status endRendering(detail::BackendCommandBuffer& commandBuffer) override;
        [[nodiscard]] Status clearColorAttachment(detail::BackendCommandBuffer& commandBuffer,
                                                  std::uint32_t attachmentIndex,
                                                  const ClearColor& clearValue,
                                                  std::span<const ScissorRect> rectangles) override;
        [[nodiscard]] Status clearDepthStencilAttachment(detail::BackendCommandBuffer& commandBuffer,
                                                         DepthStencilClearAspect aspects,
                                                         float depth,
                                                         std::uint32_t stencil,
                                                         std::span<const ScissorRect> rectangles) override;
        [[nodiscard]] Status bindGraphicsPipeline(detail::BackendCommandBuffer& commandBuffer,
                                                  PipelineHandle pipeline) override;
        [[nodiscard]] Status bindGraphicsBindingSet(detail::BackendCommandBuffer& commandBuffer,
                                                    std::uint32_t setIndex,
                                                    BindingSetHandle bindingSet,
                                                    std::span<const std::uint32_t> dynamicOffsets) override;
        [[nodiscard]] Status bindGraphicsBindingSets(detail::BackendCommandBuffer& commandBuffer,
                                                     std::uint32_t firstSet,
                                                     std::span<const BindingSetHandle> bindingSets,
                                                     std::span<const std::uint32_t> dynamicOffsets) override;
        [[nodiscard]] Status setGraphicsPushConstants(detail::BackendCommandBuffer& commandBuffer,
                                                      ShaderStage visibility,
                                                      std::uint32_t offset,
                                                      std::span<const std::byte> data) override;
        [[nodiscard]] Status setViewports(detail::BackendCommandBuffer& commandBuffer,
                                          std::span<const Viewport> viewports) override;
        [[nodiscard]] Status setScissors(detail::BackendCommandBuffer& commandBuffer,
                                         std::span<const ScissorRect> scissors) override;
        [[nodiscard]] Status setBlendConstants(detail::BackendCommandBuffer& commandBuffer,
                                               const BlendConstants& constants) override;
        [[nodiscard]] Status setStencilReference(detail::BackendCommandBuffer& commandBuffer,
                                                 StencilFace face,
                                                 std::uint32_t reference) override;
        [[nodiscard]] Status bindVertexBuffers(detail::BackendCommandBuffer& commandBuffer,
                                               std::uint32_t firstSlot,
                                               std::span<const VertexBufferBinding> bindings) override;
        [[nodiscard]] Status bindIndexBuffer(detail::BackendCommandBuffer& commandBuffer,
                                             BufferHandle buffer,
                                             IndexFormat format,
                                             std::uint64_t offset) override;
        [[nodiscard]] Status draw(detail::BackendCommandBuffer& commandBuffer,
                                  std::uint32_t vertexCount,
                                  std::uint32_t instanceCount,
                                  std::uint32_t firstVertex,
                                  std::uint32_t firstInstance) override;
        [[nodiscard]] Status drawIndexed(detail::BackendCommandBuffer& commandBuffer,
                                         std::uint32_t indexCount,
                                         std::uint32_t instanceCount,
                                         std::uint32_t firstIndex,
                                         std::int32_t baseVertex,
                                         std::uint32_t firstInstance) override;
        [[nodiscard]] Status drawIndirect(detail::BackendCommandBuffer& commandBuffer,
                                          const IndirectDrawDesc& desc) override;
        [[nodiscard]] Status drawIndexedIndirect(detail::BackendCommandBuffer& commandBuffer,
                                                 const IndirectDrawDesc& desc) override;
        [[nodiscard]] Result<QueryPoolHandle> createTimestampQueryPool(const TimestampQueryPoolDesc& desc) override;
        [[nodiscard]] Status destroyQueryPool(QueryPoolHandle queryPool) override;
        [[nodiscard]] Status resetQueryPool(detail::BackendCommandBuffer& commandBuffer,
                                            QueryPoolHandle queryPool,
                                            std::uint32_t firstQuery,
                                            std::uint32_t queryCount) override;
        [[nodiscard]] Status writeTimestamp(detail::BackendCommandBuffer& commandBuffer,
                                            QueryPoolHandle queryPool,
                                            std::uint32_t queryIndex) override;
        [[nodiscard]] Status beginGpuProfileScope(detail::BackendCommandBuffer& commandBuffer,
                                                  const GpuProfileScopeDesc& desc) override;
        [[nodiscard]] Status endGpuProfileScope(detail::BackendCommandBuffer& commandBuffer) override;
        [[nodiscard]] Status resolveTimestamps(detail::BackendCommandBuffer& commandBuffer,
                                               const TimestampQueryResolveDesc& desc) override;
        [[nodiscard]] Result<SurfaceHandle> createSurface(const SurfaceDesc& desc) override;
        [[nodiscard]] Result<SurfaceCapabilities> querySurfaceCapabilities(SurfaceHandle surface) const override;
        [[nodiscard]] Status destroySurface(SurfaceHandle surface) override;
        [[nodiscard]] Result<SwapchainHandle> createSwapchain(const SwapchainDesc& desc) override;
        [[nodiscard]] Status destroySwapchain(SwapchainHandle swapchain) override;
        [[nodiscard]] Status resizeSwapchain(SwapchainHandle swapchain,
                                             std::uint32_t width,
                                             std::uint32_t height) override;
        [[nodiscard]] Result<AcquiredSwapchainImage> acquireNextImage(
            SwapchainHandle swapchain, std::optional<std::chrono::nanoseconds> timeout) override;
        [[nodiscard]] bool hasAcquiredSwapchainImage(SwapchainHandle swapchain) const noexcept override;
        [[nodiscard]] Status cancelSwapchainImage(SwapchainHandle swapchain) override;
        [[nodiscard]] Result<PresentStatus> present(SwapchainHandle swapchain, TimelinePoint renderComplete) override;
        [[nodiscard]] Result<SwapchainPresentationInfo> querySwapchainPresentation(
            SwapchainHandle swapchain) const override;
        [[nodiscard]] Result<SwapchainFrameTiming> querySwapchainFrameTiming(SwapchainHandle swapchain) const override;
        [[nodiscard]] Result<SwapchainPresentTimings> querySwapchainPresentTimings(
            SwapchainHandle swapchain,
            std::uint64_t firstSequence,
            const detail::PresentTimingSink& records) const override;

    private:
        [[nodiscard]] Status requireActive() const;
        void publishLoss(const Error& error, DeviceFaultStatus status, bool injected) const;
        [[nodiscard]] Status ensureMipmapPipelines();
        void destroyMipmapPipelines() noexcept;

        AdapterInfo adapter_{};
        DeviceCapabilities capabilities_{};
        std::vector<FormatProperties> formats_{};
        QueueProperties graphicsQueue_{};
        std::shared_ptr<MetalDeviceNative> native_{};
        std::uint64_t deviceToken_ = 0;
        mutable rachis::detail::TimedMutex mutex_{};
        bool mipmapsReady_ = false;
        SamplerHandle mipmapSampler_{};
        std::array<BindingLayoutHandle, 3> mipmapBindingLayouts_{};
        PipelineLayoutHandle mipmapPipelineLayout_{};
        std::array<ShaderModuleHandle, 2> mipmapShaders_{};
        std::array<PipelineHandle, 2> mipmapPipelines_{};
    };

} // namespace rachis::metal
