#pragma once

#include "D3D12/D3D12AccelerationStructureLease.h"
#include "D3D12/D3D12Descriptor.h"
#include "D3D12/D3D12Queue.h"
#include "Rachis/AccelerationStructure.h"
#include "Rachis/Interop.h"
#include "Rachis/Resource.h"

#include <d3d12.h>
#include <dxgi.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace rachis::d3d12
{

    class BindingManager;
    class PipelineManager;
    struct NativeTextureView;

    inline constexpr std::size_t memorySegmentGroupCount = 2;

    struct MemoryAllocationStatistics final
    {
        std::uint64_t blockBytes = 0;
        std::uint64_t allocationBytes = 0;
        std::uint64_t blockCount = 0;
        std::uint64_t allocationCount = 0;
    };

    struct TextureViewSource final
    {
        std::shared_ptr<void> owner;
        std::shared_ptr<ResidencyRecord> residency;
        std::shared_ptr<SwapchainImageLeaseState> swapchainLease;
        ID3D12Resource* resource = nullptr;
        DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
        TextureDesc desc;
        bool sparse = false;
    };

    enum class SamplerFeedbackMapType : std::uint8_t
    {
        minMip,
        mipRegionUsed,
    };

    struct SamplerFeedbackBindingSource final
    {
        std::shared_ptr<void> owner;
        std::shared_ptr<ResidencyRecord> targetedResidency;
        ID3D12Resource* targetedResource = nullptr;
        ID3D12Resource* feedbackResource = nullptr;
    };

    struct BufferBindingSource final
    {
        std::shared_ptr<void> owner;
        std::shared_ptr<ResidencyRecord> residency;
        ID3D12Resource* resource = nullptr;
        BufferDesc desc;
        std::uint64_t nativeWidth = 0;
        bool sparse = false;
    };

    [[nodiscard]] constexpr std::uint64_t constantBufferViewSize(std::uint64_t size) noexcept
    {
        constexpr std::uint64_t mask = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT - 1;
        return size > std::numeric_limits<std::uint64_t>::max() - mask ? size : (size + mask) & ~mask;
    }

    struct SparseBindingSnapshot final
    {
        std::shared_ptr<void> owner;
        TimelinePoint publication;
    };

    struct RayTracingIndirectDispatchSource final
    {
        ID3D12Resource* argumentResource = nullptr;
        std::uint64_t argumentOffset = 0;
    };

    struct AccelerationStructureBindingSource final
    {
        std::shared_ptr<void> record;
        D3D12_GPU_VIRTUAL_ADDRESS address = 0;
    };

    struct AccelerationStructureBindingSnapshot final
    {
        std::shared_ptr<void> owner;
        std::vector<std::shared_ptr<ResidencyRecord>> residencies;
        AccelerationStructureExecutionLease lease;
    };

    struct AccelerationStructureCompactionSource final
    {
        std::shared_ptr<void> owner;
        std::vector<std::shared_ptr<ResidencyRecord>> residencies;
        D3D12_GPU_VIRTUAL_ADDRESS address = 0;
        AccelerationStructureExecutionLease lease;
    };

    struct AccelerationStructureSerializationSource final
    {
        std::shared_ptr<void> owner;
        std::vector<std::shared_ptr<ResidencyRecord>> residencies;
        D3D12_GPU_VIRTUAL_ADDRESS address = 0;
        AccelerationStructureExecutionLease lease;
    };

    class NativeTextureBarrierSequence final
    {
    public:
        NativeTextureBarrierSequence();
        ~NativeTextureBarrierSequence();

        NativeTextureBarrierSequence(const NativeTextureBarrierSequence&) = delete;
        NativeTextureBarrierSequence& operator=(const NativeTextureBarrierSequence&) = delete;
        NativeTextureBarrierSequence(NativeTextureBarrierSequence&&) noexcept;
        NativeTextureBarrierSequence& operator=(NativeTextureBarrierSequence&&) noexcept;

        [[nodiscard]] std::size_t size() const noexcept;
        void record(std::size_t index, ID3D12GraphicsCommandList& commandList) const noexcept;

    private:
        friend class ResourceManager;
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    class ResourceManager final
    {
    public:
        ResourceManager();
        ~ResourceManager();

        ResourceManager(const ResourceManager&) = delete;
        ResourceManager& operator=(const ResourceManager&) = delete;
        ResourceManager(ResourceManager&&) = delete;
        ResourceManager& operator=(ResourceManager&&) = delete;

        [[nodiscard]] Status initialize(ID3D12Device& device,
                                        IDXGIAdapter& adapter,
                                        bool enhancedBarriers,
                                        bool variableRateShadingTier2,
                                        bool rayTracing,
                                        bool sparseResources,
                                        bool samplerFeedback);
        [[nodiscard]] std::array<MemoryAllocationStatistics, memorySegmentGroupCount> memoryAllocationStatistics()
            const noexcept;
        [[nodiscard]] ResourceMemorySnapshot captureMemorySnapshot() const;
        [[nodiscard]] DescriptorAllocationStatistics descriptorAllocationStatistics() const noexcept;
        [[nodiscard]] Result<ResourceHeapHandle> createResourceHeap(const ResourceHeapDesc& desc);
        [[nodiscard]] Status destroyResourceHeap(ResourceHeapHandle heap);
        [[nodiscard]] Result<ResidencyState> queryResidency(const QueueManager& queues, ResourceHeapHandle heap) const;
        [[nodiscard]] Result<ResourceMemoryRequirements> queryBufferMemoryRequirements(const BufferDesc& desc) const;
        [[nodiscard]] Result<BufferHandle> createBuffer(const BufferDesc& desc);
        [[nodiscard]] Result<BufferHandle> createPlacedBuffer(const ResourcePlacement& placement,
                                                              const BufferDesc& desc);
        [[nodiscard]] Result<BufferHandle> createSparseBuffer(const BufferDesc& desc);
        [[nodiscard]] Result<SparseBufferProperties> querySparseBufferProperties(BufferHandle buffer) const;
        [[nodiscard]] Result<TimelinePoint> enqueueSparseBufferMappings(
            QueueManager& queues, QueueType queueType, std::span<const SparseBufferTileMapping> mappings);
        [[nodiscard]] Result<BufferHandle> createSharedBuffer(const BufferDesc& desc);
        [[nodiscard]] Result<SharedHandle> exportSharedBuffer(BufferHandle buffer);
        [[nodiscard]] Result<BufferHandle> importSharedBuffer(SharedHandleView handle, const BufferDesc& desc);
        [[nodiscard]] Status destroyBuffer(BufferHandle buffer);
        [[nodiscard]] Result<ResidencyState> queryResidency(const QueueManager& queues, BufferHandle buffer) const;
        [[nodiscard]] Result<MappedMemory> mapBuffer(BufferHandle buffer, std::uint64_t offset, std::uint64_t size);
        [[nodiscard]] Status unmapBuffer(BufferHandle buffer, std::uint64_t offset, std::uint64_t size);
        [[nodiscard]] Status flushMappedMemoryRanges(BufferHandle buffer, std::span<const MappedMemoryRange> ranges);
        [[nodiscard]] Status invalidateMappedMemoryRanges(BufferHandle buffer,
                                                          std::span<const MappedMemoryRange> ranges);
        [[nodiscard]] Status writeBuffer(BufferHandle buffer, std::uint64_t offset, std::span<const std::byte> data);
        [[nodiscard]] Status readBuffer(BufferHandle buffer, std::uint64_t offset, std::span<std::byte> destination);
        [[nodiscard]] Status copyBufferRegions(QueueManager& queues,
                                               detail::BackendCommandBuffer& commandBuffer,
                                               BufferHandle source,
                                               BufferHandle destination,
                                               std::span<const BufferCopyRegion> regions);
        [[nodiscard]] Status fillBuffer(QueueManager& queues,
                                        BindingManager& bindings,
                                        detail::BackendCommandBuffer& commandBuffer,
                                        BufferHandle buffer,
                                        std::uint64_t offset,
                                        std::uint64_t size,
                                        std::uint32_t value);
        [[nodiscard]] Status clearStorageTexture(QueueManager& queues,
                                                 BindingManager& bindings,
                                                 detail::BackendCommandBuffer& commandBuffer,
                                                 const NativeTextureView& view,
                                                 const StorageTextureClearValue& value);
        [[nodiscard]] Status bufferBarrier(QueueManager& queues,
                                           detail::BackendCommandBuffer& commandBuffer,
                                           const BufferBarrier& barrier);
        [[nodiscard]] Status samplerFeedbackBarrier(QueueManager& queues,
                                                    detail::BackendCommandBuffer& commandBuffer,
                                                    const SamplerFeedbackBarrier& barrier);
        [[nodiscard]] Status activateBufferAlias(QueueManager& queues,
                                                 detail::BackendCommandBuffer& commandBuffer,
                                                 const BufferAliasActivation& activation);
        [[nodiscard]] Status resourceBarriers(QueueManager& queues,
                                              detail::BackendCommandBuffer& commandBuffer,
                                              const ResourceBarrierBatch& barriers);
        [[nodiscard]] Result<AccelerationStructureBuildSizes> queryBottomLevelAccelerationStructureBuildSizes(
            const BottomLevelAccelerationStructureBuildInput& input) const;
        [[nodiscard]] Result<AccelerationStructureBuildSizes> queryTopLevelAccelerationStructureBuildSizes(
            const TopLevelAccelerationStructureBuildInput& input) const;
        [[nodiscard]] Result<AccelerationStructureHandle> createAccelerationStructure(
            const AccelerationStructureDesc& desc);
        [[nodiscard]] Status destroyAccelerationStructure(AccelerationStructureHandle accelerationStructure);
        [[nodiscard]] Status buildBottomLevelAccelerationStructure(
            QueueManager& queues,
            detail::BackendCommandBuffer& commandBuffer,
            const BottomLevelAccelerationStructureBuildDesc& desc);
        [[nodiscard]] Status buildTopLevelAccelerationStructure(QueueManager& queues,
                                                                detail::BackendCommandBuffer& commandBuffer,
                                                                const TopLevelAccelerationStructureBuildDesc& desc);
        [[nodiscard]] Status updateBottomLevelAccelerationStructure(
            QueueManager& queues,
            detail::BackendCommandBuffer& commandBuffer,
            const BottomLevelAccelerationStructureUpdateDesc& desc);
        [[nodiscard]] Status updateTopLevelAccelerationStructure(QueueManager& queues,
                                                                 detail::BackendCommandBuffer& commandBuffer,
                                                                 const TopLevelAccelerationStructureUpdateDesc& desc);
        [[nodiscard]] Status updateBottomLevelAccelerationStructureInPlace(
            QueueManager& queues,
            detail::BackendCommandBuffer& commandBuffer,
            const BottomLevelAccelerationStructureInPlaceUpdateDesc& desc);
        [[nodiscard]] Status updateTopLevelAccelerationStructureInPlace(
            QueueManager& queues,
            detail::BackendCommandBuffer& commandBuffer,
            const TopLevelAccelerationStructureInPlaceUpdateDesc& desc);
        [[nodiscard]] Status compactAccelerationStructure(QueueManager& queues,
                                                          detail::BackendCommandBuffer& commandBuffer,
                                                          const AccelerationStructureCompactionDesc& desc);
        [[nodiscard]] Status serializeAccelerationStructure(QueueManager& queues,
                                                            detail::BackendCommandBuffer& commandBuffer,
                                                            const AccelerationStructureSerializationDesc& desc);
        [[nodiscard]] Result<AccelerationStructureArchiveInfo> inspectAccelerationStructureArchive(
            std::span<const std::byte> archive) const;
        [[nodiscard]] Status deserializeAccelerationStructure(QueueManager& queues,
                                                              detail::BackendCommandBuffer& commandBuffer,
                                                              const AccelerationStructureDeserializationDesc& desc);
        [[nodiscard]] Result<ResourceMemoryRequirements> queryTextureMemoryRequirements(const TextureDesc& desc) const;
        [[nodiscard]] Result<TextureHandle> createTexture(const TextureDesc& desc);
        [[nodiscard]] Result<TextureHandle> createPlacedTexture(const ResourcePlacement& placement,
                                                                const TextureDesc& desc);
        [[nodiscard]] Result<TextureHandle> createSparseTexture(const TextureDesc& desc);
        [[nodiscard]] Result<SparseTextureProperties> querySparseTextureProperties(TextureHandle texture) const;
        [[nodiscard]] Result<TimelinePoint> enqueueSparseTextureMappings(QueueManager& queues,
                                                                         QueueType queueType,
                                                                         const SparseTextureMappingBatch& mappings);
        [[nodiscard]] Result<SamplerFeedbackMapHandle> createMinMipFeedbackMap(const MinMipFeedbackMapDesc& desc);
        [[nodiscard]] Result<MinMipFeedbackMapProperties> queryMinMipFeedbackMapProperties(
            SamplerFeedbackMapHandle feedbackMap) const;
        [[nodiscard]] Result<SamplerFeedbackMapHandle> createMipRegionUsedFeedbackMap(
            const MipRegionUsedFeedbackMapDesc& desc);
        [[nodiscard]] Result<MipRegionUsedFeedbackMapProperties> queryMipRegionUsedFeedbackMapProperties(
            SamplerFeedbackMapHandle feedbackMap) const;
        [[nodiscard]] Status destroySamplerFeedbackMap(SamplerFeedbackMapHandle feedbackMap);
        [[nodiscard]] Status resetSamplerFeedbackMap(QueueManager& queues,
                                                     BindingManager& bindings,
                                                     detail::BackendCommandBuffer& commandBuffer,
                                                     SamplerFeedbackMapHandle feedbackMap);
        [[nodiscard]] Status decodeMinMipFeedback(QueueManager& queues,
                                                  detail::BackendCommandBuffer& commandBuffer,
                                                  const MinMipFeedbackDecode& decode);
        [[nodiscard]] Status decodeMipRegionUsedFeedback(QueueManager& queues,
                                                         detail::BackendCommandBuffer& commandBuffer,
                                                         const MipRegionUsedFeedbackDecode& decode);
        [[nodiscard]] Status encodeMinMipFeedback(QueueManager& queues,
                                                  detail::BackendCommandBuffer& commandBuffer,
                                                  const MinMipFeedbackEncode& encode);
        [[nodiscard]] Status encodeMipRegionUsedFeedback(QueueManager& queues,
                                                         detail::BackendCommandBuffer& commandBuffer,
                                                         const MipRegionUsedFeedbackEncode& encode);
        [[nodiscard]] Result<TextureHandle> createSharedTexture(const TextureDesc& desc);
        [[nodiscard]] Result<SharedHandle> exportSharedTexture(TextureHandle texture);
        [[nodiscard]] Result<TextureHandle> importSharedTexture(SharedHandleView handle, const TextureDesc& desc);
        [[nodiscard]] Result<TextureHandle> importTexture(ID3D12Resource& resource, const TextureDesc& desc);
        [[nodiscard]] Result<TextureHandle> importSwapchainTexture(ID3D12Resource& resource,
                                                                   const TextureDesc& desc,
                                                                   std::shared_ptr<SwapchainImageLeaseState> lease);
        [[nodiscard]] Status destroyTexture(TextureHandle texture, bool externallyOwned = false);
        [[nodiscard]] Result<ResidencyState> queryResidency(const QueueManager& queues, TextureHandle texture) const;
        [[nodiscard]] Result<TextureTransferLayout> queryTextureTransferLayout(
            TextureHandle texture, const BufferTextureCopyRegion& region) const;
        [[nodiscard]] Result<TimelinePoint> enqueueMakeResident(QueueManager& queues,
                                                                QueueType queueType,
                                                                const ResidencySet& resources,
                                                                ResidencyBudgetPolicy budgetPolicy);
        [[nodiscard]] Status evict(const QueueManager& queues, const ResidencySet& resources);
        [[nodiscard]] Status copyBufferToTextureRegions(QueueManager& queues,
                                                        detail::BackendCommandBuffer& commandBuffer,
                                                        BufferHandle source,
                                                        TextureHandle destination,
                                                        std::span<const BufferTextureCopyRegion> regions);
        [[nodiscard]] Status copyTextureToBufferRegions(QueueManager& queues,
                                                        detail::BackendCommandBuffer& commandBuffer,
                                                        TextureHandle source,
                                                        BufferHandle destination,
                                                        std::span<const BufferTextureCopyRegion> regions);
        [[nodiscard]] Status copyTextureRegions(QueueManager& queues,
                                                detail::BackendCommandBuffer& commandBuffer,
                                                TextureHandle source,
                                                TextureHandle destination,
                                                std::span<const TextureCopyRegion> regions);
        [[nodiscard]] Status resolveTexture(QueueManager& queues,
                                            BindingManager& bindings,
                                            PipelineManager& pipelines,
                                            detail::BackendCommandBuffer& commandBuffer,
                                            const TextureResolve& resolve);
        [[nodiscard]] Status textureBarrier(QueueManager& queues,
                                            detail::BackendCommandBuffer& commandBuffer,
                                            const TextureBarrier& barrier);
        [[nodiscard]] Result<NativeTextureBarrierSequence> prepareNativeTextureBarrierSequence(
            QueueManager& queues,
            detail::BackendCommandBuffer& commandBuffer,
            std::span<const TextureBarrier> barriers);
        [[nodiscard]] Status activateTextureAlias(QueueManager& queues,
                                                  detail::BackendCommandBuffer& commandBuffer,
                                                  const TextureAliasActivation& activation);
        [[nodiscard]] Result<TextureViewSource> textureViewSource(TextureHandle texture);
        [[nodiscard]] Result<SamplerFeedbackBindingSource> samplerFeedbackBindingSource(
            SamplerFeedbackMapHandle feedbackMap, SamplerFeedbackMapType expectedType);
        [[nodiscard]] Result<BufferBindingSource> bufferBindingSource(BufferHandle buffer);
        [[nodiscard]] Result<std::optional<SparseBindingSnapshot>> captureSparseBufferBinding(
            const BufferBindingSource& source);
        [[nodiscard]] Result<std::optional<SparseBindingSnapshot>> captureSparseTextureBinding(
            const TextureViewSource& source);
        [[nodiscard]] Result<RayTracingIndirectDispatchSource> recordRayTracingIndirectDispatchArguments(
            QueueManager& queues,
            detail::BackendCommandBuffer& commandBuffer,
            const D3D12_DISPATCH_RAYS_DESC& nativeTemplate,
            BufferHandle argumentBuffer,
            std::uint64_t argumentOffset,
            std::span<const BufferBindingSource* const> shaderTables);
        [[nodiscard]] Result<AccelerationStructureBindingSource> accelerationStructureBindingSource(
            AccelerationStructureHandle accelerationStructure);
        [[nodiscard]] Result<AccelerationStructureBindingSnapshot> captureAccelerationStructureBinding(
            const AccelerationStructureBindingSource& source, std::uint64_t recordingId);
        [[nodiscard]] Result<AccelerationStructureCompactionSource> accelerationStructureCompactionSource(
            AccelerationStructureHandle accelerationStructure, std::uint64_t recordingId);
        [[nodiscard]] Result<AccelerationStructureSerializationSource> accelerationStructureSerializationSource(
            AccelerationStructureHandle accelerationStructure, std::uint64_t recordingId);

    private:
        [[nodiscard]] Result<SamplerFeedbackMapHandle> createSamplerFeedbackMap(TextureHandle texture,
                                                                                Extent3D mipRegion,
                                                                                std::string_view label,
                                                                                SamplerFeedbackMapType type);
        [[nodiscard]] Status decodeSamplerFeedback(QueueManager& queues,
                                                   detail::BackendCommandBuffer& commandBuffer,
                                                   SamplerFeedbackMapHandle source,
                                                   TextureHandle destination,
                                                   SamplerFeedbackMapType type);
        [[nodiscard]] Status encodeSamplerFeedback(QueueManager& queues,
                                                   detail::BackendCommandBuffer& commandBuffer,
                                                   TextureHandle source,
                                                   SamplerFeedbackMapHandle destination,
                                                   SamplerFeedbackMapType type);
        [[nodiscard]] Status updateBottomLevelAccelerationStructure(
            QueueManager& queues,
            detail::BackendCommandBuffer& commandBuffer,
            const BottomLevelAccelerationStructureUpdateDesc& desc,
            bool inPlace);
        [[nodiscard]] Status updateTopLevelAccelerationStructure(QueueManager& queues,
                                                                 detail::BackendCommandBuffer& commandBuffer,
                                                                 const TopLevelAccelerationStructureUpdateDesc& desc,
                                                                 bool inPlace);
        [[nodiscard]] Result<BufferHandle> createBuffer(const BufferDesc& desc, bool shared);
        [[nodiscard]] Result<BufferHandle> importBuffer(ID3D12Resource& resource, const BufferDesc& desc, bool shared);
        [[nodiscard]] Result<TextureHandle> createTexture(const TextureDesc& desc, bool shared);
        [[nodiscard]] Result<TextureHandle> importTexture(
            ID3D12Resource& resource,
            const TextureDesc& desc,
            bool externallyOwned,
            bool shared,
            std::shared_ptr<SwapchainImageLeaseState> swapchainLease = nullptr);

        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace rachis::d3d12
