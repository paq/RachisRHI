#pragma once

#include "Rachis/AccelerationStructure.h"
#include "Rachis/Binding.h"
#include "Rachis/Command.h"
#include "Rachis/Diagnostics.h"
#include "Rachis/Error.h"
#include "Rachis/Interop.h"
#include "Rachis/Pipeline.h"
#include "Rachis/Presentation.h"
#include "Rachis/Query.h"
#include "Rachis/Rendering.h"
#include "Rachis/Resource.h"
#include "Rachis/Sampler.h"

#include <array>
#include <chrono>
#include <compare>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace rachis
{

    struct AdapterId final
    {
        std::uint64_t value = 0;

        friend constexpr auto operator<=>(const AdapterId&, const AdapterId&) noexcept = default;
    };

    enum class AdapterType : std::uint8_t
    {
        unknown,
        integrated,
        discrete,
        software,
    };

    enum class AdapterPreference : std::uint8_t
    {
        highPerformance,
        minimumPower,
        warp,
    };

    enum class Backend : std::uint8_t
    {
        d3d12,
        vulkan,
        metal,
    };

    [[nodiscard]] inline constexpr Backend defaultHostBackend() noexcept
    {
#if defined(_WIN32)
        return Backend::d3d12;
#elif defined(__APPLE__)
        return Backend::metal;
#else
        return Backend::vulkan;
#endif
    }

    struct AdapterInfo final
    {
        AdapterId id = {};
        std::string name = {};
        AdapterType type = AdapterType::unknown;
        std::uint32_t vendorId = 0;
        std::uint32_t deviceId = 0;
        std::uint64_t dedicatedVideoMemory = 0;
        std::uint64_t sharedSystemMemory = 0;
        Backend backend = Backend::d3d12;
    };

    enum class MemoryHeapProperty : std::uint32_t
    {
        none = 0,
        deviceLocal = 1U << 0U,
    };

    [[nodiscard]] constexpr bool hasAny(MemoryHeapProperty value, MemoryHeapProperty flags) noexcept
    {
        return (static_cast<std::uint32_t>(value) & static_cast<std::uint32_t>(flags)) != 0;
    }

    enum class MemoryBudgetFeature : std::uint32_t
    {
        none = 0,
        reservation = 1U << 0U,
        allocationStatistics = 1U << 1U,
    };

    [[nodiscard]] constexpr MemoryBudgetFeature operator|(MemoryBudgetFeature left, MemoryBudgetFeature right) noexcept
    {
        return static_cast<MemoryBudgetFeature>(static_cast<std::uint32_t>(left) | static_cast<std::uint32_t>(right));
    }

    [[nodiscard]] constexpr bool hasAny(MemoryBudgetFeature value, MemoryBudgetFeature flags) noexcept
    {
        return (static_cast<std::uint32_t>(value) & static_cast<std::uint32_t>(flags)) != 0;
    }

    struct MemoryBudget final
    {
        std::uint32_t heapIndex = 0;
        MemoryHeapProperty properties = MemoryHeapProperty::none;
        MemoryBudgetFeature features = MemoryBudgetFeature::none;
        std::uint64_t blockBytes = 0;
        std::uint64_t allocationBytes = 0;
        std::uint64_t blockCount = 0;
        std::uint64_t allocationCount = 0;
        std::uint64_t budgetBytes = 0;
        std::uint64_t usageBytes = 0;
        std::uint64_t availableForReservationBytes = 0;
        std::uint64_t reservationBytes = 0;
    };

    enum class ValidationLevel : std::uint8_t
    {
        disabled,
        standard,
        gpu,
    };

    struct DeviceDesc final
    {
        Backend backend = defaultHostBackend();
        AdapterPreference adapterPreference = AdapterPreference::highPerformance;
        std::optional<AdapterId> adapterId = {};
        ValidationLevel validation = ValidationLevel::disabled;
        bool requireValidation = false;
        bool enableDred = false;
        bool enableDeviceFault = false;
        DiagnosticCallback diagnosticCallback = nullptr;
        void* diagnosticUserData = nullptr;

        std::string applicationName = {};
        std::string engineName = {};
    };

    struct ApiVersion final
    {
        std::uint16_t major = 0;
        std::uint16_t minor = 0;

        friend constexpr auto operator<=>(const ApiVersion&, const ApiVersion&) noexcept = default;
    };

    enum class NdcYAxisDirection : std::uint8_t
    {
        unknown,
        up,
        down,
    };

    enum class NdcDepthRange : std::uint8_t
    {
        unknown,
        zeroToOne,
        minusOneToOne,
    };

    enum class TextureOrigin : std::uint8_t
    {
        unknown,
        upperLeft,
        lowerLeft,
    };

    struct CoordinateConventions final
    {
        NdcYAxisDirection ndcYAxis = NdcYAxisDirection::unknown;
        NdcDepthRange ndcDepthRange = NdcDepthRange::unknown;
        TextureOrigin textureOrigin = TextureOrigin::unknown;
    };

    enum class LayeredRenderingMode : std::uint8_t
    {
        none = 0,

        vertexStage = 1U << 0U,
    };

    [[nodiscard]] constexpr LayeredRenderingMode operator|(LayeredRenderingMode left,
                                                           LayeredRenderingMode right) noexcept
    {
        return static_cast<LayeredRenderingMode>(static_cast<std::uint8_t>(left) | static_cast<std::uint8_t>(right));
    }

    [[nodiscard]] constexpr bool hasAny(LayeredRenderingMode value, LayeredRenderingMode flags) noexcept
    {
        return (static_cast<std::uint8_t>(value) & static_cast<std::uint8_t>(flags)) != 0;
    }

    struct DeviceLimits final
    {
        std::uint32_t maxTextureDimension1D = 0;
        std::uint32_t maxTextureDimension2D = 0;
        std::uint32_t maxTextureDimension3D = 0;
        std::uint32_t maxTextureArrayLayers = 0;

        std::uint32_t maxFramebufferLayers = 1;
        std::uint32_t maxTextureMipLevels = 0;
        std::uint32_t maxComputeWorkgroupInvocations = 0;
        std::array<std::uint32_t, 3> maxComputeWorkgroupSize{};
        std::array<std::uint32_t, 3> maxComputeWorkgroupCount{};
        std::uint64_t maxBufferSize = 0;
        std::uint64_t maxUniformBufferSize = 0;
        std::uint64_t minUniformBufferOffsetAlignment = 1;
        std::uint32_t maxSamplerAnisotropy = 1;
        float maxSamplerLodBias = 0.0F;
        std::uint32_t maxColorAttachments = 0;

        std::uint32_t maxViewports = 1;
        std::uint32_t maxVertexBuffers = 0;
        std::uint32_t maxVertexAttributes = 0;
        std::uint32_t maxBindingSets = 0;
        std::uint32_t maxPushConstantBytes = 0;
        std::uint32_t maxVariableResourceDescriptors = 0;
        std::uint32_t maxVariableBufferDescriptors = 0;
        std::uint32_t maxVariableSamplerDescriptors = 0;
        std::uint32_t maxBindingPoolResourceDescriptors = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t maxBindingSamplerDescriptors = std::numeric_limits<std::uint32_t>::max();

        std::uint32_t maxClipDistances = 0;
        std::uint32_t maxCullDistances = 0;
        std::uint32_t maxCombinedClipAndCullDistances = 0;
    };

    struct SubgroupCapabilities final
    {
        std::uint32_t minSize = 0;
        std::uint32_t maxSize = 0;
    };

    struct TessellationCapabilities final
    {
        std::uint32_t maxPatchControlPoints = 0;
    };

    struct RayTracingCapabilities final
    {
        std::uint32_t tier = 0;
        std::uint32_t maxGeometriesPerBottomLevel = 0;
        std::uint32_t maxInstancesPerTopLevel = 0;
        std::uint64_t maxPrimitivesPerBottomLevel = 0;

        std::uint32_t shaderGroupHandleSize = 0;
        std::uint32_t shaderGroupHandleAlignment = 0;
        std::uint32_t shaderGroupBaseAlignment = 0;
        std::uint32_t maxShaderGroupStride = 0;
        std::uint32_t maxRecursionDepth = 0;
        std::uint32_t maxAttributeSize = 0;
        std::uint64_t maxDispatchInvocations = 0;
    };

    struct MappedMemoryCapabilities final
    {
        bool hostCoherent = false;

        std::uint64_t nonCoherentAtomSize = 1;
    };

    struct DeviceCapabilities final
    {
        ApiVersion featureLevel = {};
        ApiVersion shaderModel = {};
        CoordinateConventions coordinateConventions = {};
        std::uint32_t resourceBindingTier = 0;
        std::uint32_t resourceHeapTier = 0;
        std::uint32_t nodeCount = 0;
        bool enhancedBarriers = false;

        bool bufferBarrierRanges = false;
        bool unifiedMemory = false;
        bool cacheCoherentUnifiedMemory = false;

        bool persistentMapping = false;
        MappedMemoryCapabilities mappedMemory = {};

        bool shaderFloat16 = false;
        bool shaderInt16 = false;

        bool shaderInt64 = false;

        bool shaderInt64AtomicGroupShared = false;
        bool shaderInt64AtomicTypedResource = false;

        bool subgroupOperations = false;
        SubgroupCapabilities subgroup = {};

        bool dualSourceBlending = false;

        bool logicOperations = false;

        bool fragmentPrimitiveId = false;

        bool fragmentBarycentricCoordinates = false;

        bool rasterizerOrderedStorageAccess = false;

        bool geometryShaders = false;

        bool tessellationShaders = false;
        TessellationCapabilities tessellation = {};

        bool meshShaders = false;
        bool taskShaders = false;
        MeshShaderCapabilities meshShader = {};

        bool rayTracing = false;

        bool indirectRayDispatch = false;

        bool rayQuery = false;
        RayTracingCapabilities rayTracingLimits = {};

        LayeredRenderingMode layeredRenderingModes = LayeredRenderingMode::none;

        bool multiviewRendering = false;

        std::uint32_t maxMultiviewViewCount = 0;
        bool indirectDispatch = false;

        bool multiDispatchIndirect = false;

        bool dispatchIndirectCount = false;
        bool indirectDraws = false;
        bool indirectDrawBaseInstance = false;
        bool multiDrawIndirect = false;
        bool drawIndirectCount = false;

        bool multiViewport = false;
        bool debugMarkers = false;
        bool pipelineCaches = false;
        bool occlusionQueries = false;
        bool pipelineStatisticsQueries = false;

        bool meshPipelineStatisticsQueries = false;
        bool streamOutput = false;

        bool streamOutputRasterizationDisable = false;
        bool streamOutputStatisticsQueries = false;

        bool conditionalExecution = false;

        bool depthBoundsTest = false;

        bool independentFrontAndBackStencilReference = false;

        bool variableRateShading = false;
        bool additionalShadingRates = false;

        bool variableRateShadingTier2 = false;
        bool perPrimitiveShadingRateWithViewportIndexing = false;

        std::uint32_t shadingRateImageTileSize = 0;
        bool partiallyBoundBindingArrays = false;
        bool variableDescriptorCountBindingArrays = false;
        bool wireframeRasterization = false;

        bool dynamicDepthBias = false;

        bool conservativeRasterization = false;
        bool unclippedDepth = false;
        bool primitiveRestart = false;

        bool triangleFans = false;

        bool sparseBuffers = false;

        bool sparseTextures = false;

        bool samplerFeedback = false;
        bool mipmapGeneration = false;

        bool samplerMinMaxReduction = false;

        bool textureViewComponentSwizzle = false;

        bool sharedTextureHandles = false;

        bool sharedBufferHandles = false;

        bool sharedTimelineHandles = false;
        bool memoryBudgets = false;
        bool residencyPriorities = false;
        bool manualResidency = false;
        VertexInputCapabilities vertexInput = {};
        DeviceLimits limits = {};
    };

    namespace detail
    {
        class BackendDevice;
        struct DeviceAccess;
    } // namespace detail

    class Device final
    {
    public:
        Device() noexcept;
        ~Device();

        Device(Device&& other) noexcept;
        Device& operator=(Device&& other) noexcept;

        Device(const Device&) = delete;
        Device& operator=(const Device&) = delete;

        [[nodiscard]] bool isValid() const noexcept;
        [[nodiscard]] explicit operator bool() const noexcept;

        [[nodiscard]] DeviceOperationalState operationalState() const noexcept;
        [[nodiscard]] std::optional<Error> deviceLossError() const;
        [[nodiscard]] std::optional<AdapterInfo> adapterInfo() const;
        [[nodiscard]] std::optional<DeviceCapabilities> capabilities() const noexcept;
        [[nodiscard]] Result<std::vector<MemoryBudget>> queryMemoryBudgets(std::uint32_t nodeIndex = 0) const;
        [[nodiscard]] Result<ResourceMemorySnapshot> captureResourceMemorySnapshot() const;
        [[nodiscard]] Result<DeviceFaultReport> captureDeviceFaultReport() const;

        [[nodiscard]] BackendAllocationSnapshot captureBackendAllocationSnapshot() const noexcept;
        [[nodiscard]] FormatProperties queryFormatProperties(Format format) const noexcept;
        [[nodiscard]] QueueProperties queueProperties(QueueType queueType) const noexcept;
        [[nodiscard]] Result<QueueClockCalibration> calibrateQueueClock(QueueType queueType) const;
        [[nodiscard]] DiagnosticCounts diagnosticCounts() const noexcept;
        [[nodiscard]] BindingUpdateStatistics bindingUpdateStatistics() const noexcept;
        [[nodiscard]] Result<CommandPoolHandle> createCommandPool(const CommandPoolDesc& desc);

        [[nodiscard]] Status destroyCommandPool(CommandPoolHandle commandPool);

        [[nodiscard]] Status resetCommandPool(CommandPoolHandle commandPool);
        [[nodiscard]] Result<CommandBuffer> acquireCommandBuffer(CommandPoolHandle commandPool);

        [[nodiscard]] Status resetCommandBuffer(CommandBuffer& commandBuffer);

        [[nodiscard]] Result<TimelinePoint> submit(CommandBuffer&& commandBuffer,
                                                   std::span<const TimelinePoint> waits = {});

        [[nodiscard]] Result<TimelinePoint> submit(std::span<CommandBuffer> commandBuffers,
                                                   std::span<const TimelinePoint> waits = {});

        [[nodiscard]] Result<TimelinePointStatus> queryTimelinePoint(TimelinePoint point) const;

        [[nodiscard]] Result<TimelinePointStatus> wait(TimelinePoint point, std::chrono::nanoseconds timeout);
        [[nodiscard]] Status wait(TimelinePoint point);

        [[nodiscard]] Result<SharedTimelinePoint> exportTimelinePoint(TimelinePoint point);

        [[nodiscard]] Result<TimelinePoint> enqueueSharedTimelineWait(QueueType queueType,
                                                                      SharedTimelinePointView point);

        [[nodiscard]] Status waitIdle(QueueType queueType);

        [[nodiscard]] Status waitIdle();
        [[nodiscard]] std::uint64_t completedValue(QueueType queueType) const noexcept;
        void collectGarbage() noexcept;
        [[nodiscard]] Status beginDebugGroup(CommandBuffer& commandBuffer, const DebugMarkerDesc& desc);
        [[nodiscard]] Status endDebugGroup(CommandBuffer& commandBuffer);
        [[nodiscard]] Status insertDebugMarker(CommandBuffer& commandBuffer, const DebugMarkerDesc& desc);
        [[nodiscard]] Result<ResourceHeapHandle> createResourceHeap(const ResourceHeapDesc& desc);
        [[nodiscard]] Status destroyResourceHeap(ResourceHeapHandle heap);
        [[nodiscard]] Result<ResidencyState> queryResidency(ResourceHeapHandle heap) const;
        [[nodiscard]] Result<ResourceMemoryRequirements> queryBufferMemoryRequirements(const BufferDesc& desc) const;
        [[nodiscard]] Result<BufferHandle> createBuffer(const BufferDesc& desc);
        [[nodiscard]] Result<BufferHandle> createPlacedBuffer(const ResourcePlacement& placement,
                                                              const BufferDesc& desc);

        [[nodiscard]] Result<BufferHandle> createSparseBuffer(const BufferDesc& desc);
        [[nodiscard]] Result<SparseBufferProperties> querySparseBufferProperties(BufferHandle buffer) const;

        [[nodiscard]] Result<TimelinePoint> enqueueSparseBufferMappings(
            QueueType queueType, std::span<const SparseBufferTileMapping> mappings);

        [[nodiscard]] Result<BufferHandle> createSharedBuffer(const BufferDesc& desc);
        [[nodiscard]] Result<SharedHandle> exportSharedBuffer(BufferHandle buffer);

        [[nodiscard]] Result<BufferHandle> importSharedBuffer(SharedHandleView handle, const BufferDesc& desc);
        [[nodiscard]] Status destroyBuffer(BufferHandle buffer);
        [[nodiscard]] Result<ResidencyState> queryResidency(BufferHandle buffer) const;
        [[nodiscard]] Result<MappedMemory> mapBuffer(BufferHandle buffer,
                                                     std::uint64_t offset = 0,
                                                     std::uint64_t size = wholeResource);
        [[nodiscard]] Status unmapBuffer(BufferHandle buffer,
                                         std::uint64_t offset = 0,
                                         std::uint64_t size = wholeResource);

        [[nodiscard]] Status flushMappedMemoryRanges(BufferHandle buffer, std::span<const MappedMemoryRange> ranges);
        [[nodiscard]] Status invalidateMappedMemoryRanges(BufferHandle buffer,
                                                          std::span<const MappedMemoryRange> ranges);
        [[nodiscard]] Status writeBuffer(BufferHandle buffer, std::uint64_t offset, std::span<const std::byte> data);
        [[nodiscard]] Status readBuffer(BufferHandle buffer, std::uint64_t offset, std::span<std::byte> destination);
        [[nodiscard]] Status copyBuffer(CommandBuffer& commandBuffer, const BufferCopy& copy);
        [[nodiscard]] Status copyBuffer(CommandBuffer& commandBuffer,
                                        BufferHandle source,
                                        BufferHandle destination,
                                        std::span<const BufferCopyRegion> regions);
        [[nodiscard]] Status fillBuffer(CommandBuffer& commandBuffer,
                                        BufferHandle buffer,
                                        std::uint64_t offset,
                                        std::uint64_t size,
                                        std::uint32_t value);
        [[nodiscard]] Status globalBarrier(CommandBuffer& commandBuffer, const GlobalBarrier& barrier);
        [[nodiscard]] Status bufferBarrier(CommandBuffer& commandBuffer, const BufferBarrier& barrier);
        [[nodiscard]] Status samplerFeedbackBarrier(CommandBuffer& commandBuffer,
                                                    const SamplerFeedbackBarrier& barrier);
        [[nodiscard]] Status activateBufferAlias(CommandBuffer& commandBuffer, const BufferAliasActivation& activation);
        [[nodiscard]] Status resourceBarriers(CommandBuffer& commandBuffer, const ResourceBarrierBatch& barriers);
        [[nodiscard]] Result<AccelerationStructureBuildSizes> queryBottomLevelAccelerationStructureBuildSizes(
            const BottomLevelAccelerationStructureBuildInput& input) const;
        [[nodiscard]] Result<AccelerationStructureBuildSizes> queryTopLevelAccelerationStructureBuildSizes(
            const TopLevelAccelerationStructureBuildInput& input) const;
        [[nodiscard]] Result<AccelerationStructureHandle> createAccelerationStructure(
            const AccelerationStructureDesc& desc);
        [[nodiscard]] Status destroyAccelerationStructure(AccelerationStructureHandle accelerationStructure);
        [[nodiscard]] Status buildBottomLevelAccelerationStructure(
            CommandBuffer& commandBuffer, const BottomLevelAccelerationStructureBuildDesc& desc);
        [[nodiscard]] Status buildTopLevelAccelerationStructure(CommandBuffer& commandBuffer,
                                                                const TopLevelAccelerationStructureBuildDesc& desc);
        [[nodiscard]] Status updateBottomLevelAccelerationStructure(
            CommandBuffer& commandBuffer, const BottomLevelAccelerationStructureUpdateDesc& desc);
        [[nodiscard]] Status updateTopLevelAccelerationStructure(CommandBuffer& commandBuffer,
                                                                 const TopLevelAccelerationStructureUpdateDesc& desc);
        [[nodiscard]] Status updateBottomLevelAccelerationStructureInPlace(
            CommandBuffer& commandBuffer, const BottomLevelAccelerationStructureInPlaceUpdateDesc& desc);
        [[nodiscard]] Status updateTopLevelAccelerationStructureInPlace(
            CommandBuffer& commandBuffer, const TopLevelAccelerationStructureInPlaceUpdateDesc& desc);
        [[nodiscard]] Status compactAccelerationStructure(CommandBuffer& commandBuffer,
                                                          const AccelerationStructureCompactionDesc& desc);
        [[nodiscard]] Status serializeAccelerationStructure(CommandBuffer& commandBuffer,
                                                            const AccelerationStructureSerializationDesc& desc);
        [[nodiscard]] Result<AccelerationStructureArchiveInfo> inspectAccelerationStructureArchive(
            std::span<const std::byte> archive) const;
        [[nodiscard]] Status deserializeAccelerationStructure(CommandBuffer& commandBuffer,
                                                              const AccelerationStructureDeserializationDesc& desc);
        [[nodiscard]] Result<ResourceMemoryRequirements> queryTextureMemoryRequirements(const TextureDesc& desc) const;
        [[nodiscard]] Result<TextureHandle> createTexture(const TextureDesc& desc);
        [[nodiscard]] Result<TextureHandle> createPlacedTexture(const ResourcePlacement& placement,
                                                                const TextureDesc& desc);

        [[nodiscard]] Result<TextureHandle> createSparseTexture(const TextureDesc& desc);
        [[nodiscard]] Result<SparseTextureProperties> querySparseTextureProperties(TextureHandle texture) const;

        [[nodiscard]] Result<TimelinePoint> enqueueSparseTextureMappings(QueueType queueType,
                                                                         const SparseTextureMappingBatch& mappings);
        [[nodiscard]] Result<SamplerFeedbackMapHandle> createMinMipFeedbackMap(const MinMipFeedbackMapDesc& desc);
        [[nodiscard]] Result<MinMipFeedbackMapProperties> queryMinMipFeedbackMapProperties(
            SamplerFeedbackMapHandle feedbackMap) const;
        [[nodiscard]] Result<SamplerFeedbackMapHandle> createMipRegionUsedFeedbackMap(
            const MipRegionUsedFeedbackMapDesc& desc);
        [[nodiscard]] Result<MipRegionUsedFeedbackMapProperties> queryMipRegionUsedFeedbackMapProperties(
            SamplerFeedbackMapHandle feedbackMap) const;
        [[nodiscard]] Status destroySamplerFeedbackMap(SamplerFeedbackMapHandle feedbackMap);

        [[nodiscard]] Status resetSamplerFeedbackMap(CommandBuffer& commandBuffer,
                                                     SamplerFeedbackMapHandle feedbackMap);
        [[nodiscard]] Status decodeMinMipFeedback(CommandBuffer& commandBuffer, const MinMipFeedbackDecode& decode);
        [[nodiscard]] Status decodeMipRegionUsedFeedback(CommandBuffer& commandBuffer,
                                                         const MipRegionUsedFeedbackDecode& decode);
        [[nodiscard]] Status encodeMinMipFeedback(CommandBuffer& commandBuffer, const MinMipFeedbackEncode& encode);
        [[nodiscard]] Status encodeMipRegionUsedFeedback(CommandBuffer& commandBuffer,
                                                         const MipRegionUsedFeedbackEncode& encode);

        [[nodiscard]] Result<TextureHandle> createSharedTexture(const TextureDesc& desc);
        [[nodiscard]] Result<SharedHandle> exportSharedTexture(TextureHandle texture);

        [[nodiscard]] Result<TextureHandle> importSharedTexture(SharedHandleView handle, const TextureDesc& desc);
        [[nodiscard]] Status destroyTexture(TextureHandle texture);
        [[nodiscard]] Result<ResidencyState> queryResidency(TextureHandle texture) const;
        [[nodiscard]] Result<TextureTransferLayout> queryTextureTransferLayout(
            TextureHandle texture, const BufferTextureCopyRegion& region) const;
        [[nodiscard]] Result<TimelinePoint> enqueueMakeResident(
            QueueType queueType,
            const ResidencySet& resources,
            ResidencyBudgetPolicy budgetPolicy = ResidencyBudgetPolicy::denyOverBudget);
        [[nodiscard]] Status evict(const ResidencySet& resources);
        [[nodiscard]] Status copyBufferToTexture(CommandBuffer& commandBuffer, const BufferTextureCopy& copy);
        [[nodiscard]] Status copyBufferToTexture(CommandBuffer& commandBuffer,
                                                 BufferHandle source,
                                                 TextureHandle destination,
                                                 std::span<const BufferTextureCopyRegion> regions);
        [[nodiscard]] Status copyTextureToBuffer(CommandBuffer& commandBuffer, const BufferTextureCopy& copy);
        [[nodiscard]] Status copyTextureToBuffer(CommandBuffer& commandBuffer,
                                                 TextureHandle source,
                                                 BufferHandle destination,
                                                 std::span<const BufferTextureCopyRegion> regions);
        [[nodiscard]] Status copyTexture(CommandBuffer& commandBuffer, const TextureCopy& copy);
        [[nodiscard]] Status copyTexture(CommandBuffer& commandBuffer,
                                         TextureHandle source,
                                         TextureHandle destination,
                                         std::span<const TextureCopyRegion> regions);
        [[nodiscard]] Status clearStorageTexture(CommandBuffer& commandBuffer,
                                                 TextureViewHandle view,
                                                 const StorageTextureClearValue& value);
        [[nodiscard]] Status resolveTexture(CommandBuffer& commandBuffer, const TextureResolve& resolve);
        [[nodiscard]] Status generateMipmaps(CommandBuffer& commandBuffer, const TextureMipmapGeneration& generation);
        [[nodiscard]] Status textureBarrier(CommandBuffer& commandBuffer, const TextureBarrier& barrier);
        [[nodiscard]] Status activateTextureAlias(CommandBuffer& commandBuffer,
                                                  const TextureAliasActivation& activation);
        [[nodiscard]] Result<TextureViewHandle> createTextureView(const TextureViewDesc& desc);
        [[nodiscard]] Status destroyTextureView(TextureViewHandle view);
        [[nodiscard]] Result<SamplerHandle> createSampler(const SamplerDesc& desc);
        [[nodiscard]] Status destroySampler(SamplerHandle sampler);
        [[nodiscard]] Result<BindingLayoutHandle> createBindingLayout(const BindingLayoutDesc& desc);
        [[nodiscard]] Status destroyBindingLayout(BindingLayoutHandle layout);
        [[nodiscard]] Result<BindingPoolHandle> createBindingPool(const BindingPoolDesc& desc);
        [[nodiscard]] Status destroyBindingPool(BindingPoolHandle pool);
        [[nodiscard]] Result<BindingSetHandle> createBindingSet(const BindingSetDesc& desc);
        [[nodiscard]] Status updateBindingSet(BindingSetHandle set, std::span<const BindingWrite> writes);
        [[nodiscard]] Status destroyBindingSet(BindingSetHandle set);
        [[nodiscard]] Result<ShaderLibraryHandle> createShaderLibrary(const ShaderLibraryDesc& desc);
        [[nodiscard]] Status destroyShaderLibrary(ShaderLibraryHandle shaderLibrary);
        [[nodiscard]] Result<ShaderModuleHandle> createShaderModule(const ShaderModuleDesc& desc);

        [[nodiscard]] Result<ShaderInterfaceSnapshot> shaderModuleInterface(ShaderModuleHandle shaderModule) const;
        [[nodiscard]] Status destroyShaderModule(ShaderModuleHandle shaderModule);
        [[nodiscard]] Result<PipelineLayoutHandle> createPipelineLayout(const PipelineLayoutDesc& desc);
        [[nodiscard]] Status destroyPipelineLayout(PipelineLayoutHandle pipelineLayout);
        [[nodiscard]] Result<PipelineCacheHandle> createPipelineCache(const PipelineCacheDesc& desc = {});
        [[nodiscard]] Status destroyPipelineCache(PipelineCacheHandle pipelineCache);
        [[nodiscard]] Result<std::vector<std::byte>> pipelineCacheData(PipelineCacheHandle pipelineCache);
        [[nodiscard]] Result<PipelineHandle> createComputePipeline(const ComputePipelineDesc& desc);
        [[nodiscard]] Result<PipelineHandle> createGraphicsPipeline(const GraphicsPipelineDesc& desc);

        [[nodiscard]] Result<PipelineCompilationHandle> beginComputePipelineCompilation(
            const ComputePipelineDesc& desc);
        [[nodiscard]] Result<PipelineCompilationHandle> beginGraphicsPipelineCompilation(
            const GraphicsPipelineDesc& desc);
        [[nodiscard]] Result<PipelineCompilationState> pipelineCompilationState(PipelineCompilationHandle compilation);
        [[nodiscard]] Result<PipelineHandle> waitPipelineCompilation(PipelineCompilationHandle compilation);
        [[nodiscard]] Status destroyPipelineCompilation(PipelineCompilationHandle compilation);
        [[nodiscard]] Result<RayTracingPipelineLibraryHandle> createRayTracingPipelineLibrary(
            const RayTracingPipelineLibraryDesc& desc);
        [[nodiscard]] Status destroyRayTracingPipelineLibrary(RayTracingPipelineLibraryHandle library);
        [[nodiscard]] Result<PipelineHandle> createRayTracingPipeline(const RayTracingPipelineDesc& desc);
        [[nodiscard]] Status getRayTracingShaderGroupHandles(PipelineHandle pipeline,
                                                             std::uint32_t firstGroup,
                                                             std::uint32_t groupCount,
                                                             std::span<std::byte> destination);
        [[nodiscard]] Result<std::uint32_t> rayTracingShaderGroupRecordSize(PipelineHandle pipeline,
                                                                            std::uint32_t groupIndex);
        [[nodiscard]] Status writeRayTracingShaderGroupRecord(PipelineHandle pipeline,
                                                              std::uint32_t groupIndex,
                                                              std::span<const std::byte> localConstants,
                                                              std::span<std::byte> destination);
        [[nodiscard]] Status destroyPipeline(PipelineHandle pipeline);
        [[nodiscard]] Status bindComputePipeline(CommandBuffer& commandBuffer, PipelineHandle pipeline);
        [[nodiscard]] Status bindComputeBindingSet(CommandBuffer& commandBuffer,
                                                   std::uint32_t setIndex,
                                                   BindingSetHandle bindingSet,
                                                   std::span<const std::uint32_t> dynamicOffsets = {});

        [[nodiscard]] Status bindComputeBindingSets(CommandBuffer& commandBuffer,
                                                    std::uint32_t firstSet,
                                                    std::span<const BindingSetHandle> bindingSets,
                                                    std::span<const std::uint32_t> dynamicOffsets = {});
        [[nodiscard]] Status setComputePushConstants(CommandBuffer& commandBuffer,
                                                     std::uint32_t offset,
                                                     std::span<const std::byte> data);
        [[nodiscard]] Status dispatch(CommandBuffer& commandBuffer,
                                      std::uint32_t groupCountX,
                                      std::uint32_t groupCountY = 1,
                                      std::uint32_t groupCountZ = 1);
        [[nodiscard]] Status dispatchIndirect(CommandBuffer& commandBuffer,
                                              BufferHandle argumentBuffer,
                                              std::uint64_t argumentOffset = 0);
        [[nodiscard]] Status dispatchIndirect(CommandBuffer& commandBuffer, const IndirectDispatchDesc& desc);
        [[nodiscard]] Status bindRayTracingPipeline(CommandBuffer& commandBuffer, PipelineHandle pipeline);
        [[nodiscard]] Status bindRayTracingBindingSet(CommandBuffer& commandBuffer,
                                                      std::uint32_t setIndex,
                                                      BindingSetHandle bindingSet,
                                                      std::span<const std::uint32_t> dynamicOffsets = {});
        [[nodiscard]] Status bindRayTracingBindingSets(CommandBuffer& commandBuffer,
                                                       std::uint32_t firstSet,
                                                       std::span<const BindingSetHandle> bindingSets,
                                                       std::span<const std::uint32_t> dynamicOffsets = {});
        [[nodiscard]] Status setRayTracingPushConstants(CommandBuffer& commandBuffer,
                                                        std::uint32_t offset,
                                                        std::span<const std::byte> data);
        [[nodiscard]] Status dispatchRays(CommandBuffer& commandBuffer, const RayTracingDispatchDesc& desc);
        [[nodiscard]] Status dispatchRaysIndirect(CommandBuffer& commandBuffer,
                                                  const RayTracingIndirectDispatchDesc& desc);
        [[nodiscard]] Status beginRendering(CommandBuffer& commandBuffer, const RenderingDesc& desc);
        [[nodiscard]] Status endRendering(CommandBuffer& commandBuffer);
        [[nodiscard]] Status clearColorAttachment(CommandBuffer& commandBuffer,
                                                  std::uint32_t attachmentIndex,
                                                  const ClearColor& clearValue,
                                                  std::span<const ScissorRect> rectangles = {});
        [[nodiscard]] Status clearDepthStencilAttachment(CommandBuffer& commandBuffer,
                                                         DepthStencilClearAspect aspects,
                                                         float depth,
                                                         std::uint32_t stencil,
                                                         std::span<const ScissorRect> rectangles = {});
        [[nodiscard]] Status bindGraphicsPipeline(CommandBuffer& commandBuffer, PipelineHandle pipeline);
        [[nodiscard]] Status bindGraphicsBindingSet(CommandBuffer& commandBuffer,
                                                    std::uint32_t setIndex,
                                                    BindingSetHandle bindingSet,
                                                    std::span<const std::uint32_t> dynamicOffsets = {});

        [[nodiscard]] Status bindGraphicsBindingSets(CommandBuffer& commandBuffer,
                                                     std::uint32_t firstSet,
                                                     std::span<const BindingSetHandle> bindingSets,
                                                     std::span<const std::uint32_t> dynamicOffsets = {});
        [[nodiscard]] Status setGraphicsPushConstants(CommandBuffer& commandBuffer,
                                                      ShaderStage visibility,
                                                      std::uint32_t offset,
                                                      std::span<const std::byte> data);
        [[nodiscard]] Status setViewport(CommandBuffer& commandBuffer, const Viewport& viewport);
        [[nodiscard]] Status setViewports(CommandBuffer& commandBuffer, std::span<const Viewport> viewports);
        [[nodiscard]] Status setScissor(CommandBuffer& commandBuffer, const ScissorRect& scissor);
        [[nodiscard]] Status setScissors(CommandBuffer& commandBuffer, std::span<const ScissorRect> scissors);
        [[nodiscard]] Status setBlendConstants(CommandBuffer& commandBuffer, const BlendConstants& constants);
        [[nodiscard]] Status setDepthBias(CommandBuffer& commandBuffer, const DepthBias& bias);
        [[nodiscard]] Status setDepthBounds(CommandBuffer& commandBuffer, const DepthBounds& bounds);
        [[nodiscard]] Status setStencilReference(CommandBuffer& commandBuffer, std::uint32_t reference);
        [[nodiscard]] Status setStencilReference(CommandBuffer& commandBuffer,
                                                 StencilFace face,
                                                 std::uint32_t reference);
        [[nodiscard]] Status setShadingRate(CommandBuffer& commandBuffer, ShadingRate rate);
        [[nodiscard]] Status setShadingRate(CommandBuffer& commandBuffer,
                                            ShadingRate rate,
                                            const ShadingRateCombiners& combiners);

        [[nodiscard]] Status setShadingRateImage(CommandBuffer& commandBuffer, TextureHandle image = {});
        [[nodiscard]] Status bindVertexBuffer(CommandBuffer& commandBuffer,
                                              std::uint32_t slot,
                                              BufferHandle buffer,
                                              std::uint64_t offset = 0);

        [[nodiscard]] Status bindVertexBuffers(CommandBuffer& commandBuffer,
                                               std::uint32_t firstSlot,
                                               std::span<const VertexBufferBinding> bindings);
        [[nodiscard]] Status bindStreamOutputBuffers(CommandBuffer& commandBuffer,
                                                     std::uint32_t firstSlot,
                                                     std::span<const StreamOutputBufferBinding> bindings);
        [[nodiscard]] Status bindIndexBuffer(CommandBuffer& commandBuffer,
                                             BufferHandle buffer,
                                             IndexFormat format,
                                             std::uint64_t offset = 0);
        [[nodiscard]] Status draw(CommandBuffer& commandBuffer,
                                  std::uint32_t vertexCount,
                                  std::uint32_t instanceCount = 1,
                                  std::uint32_t firstVertex = 0,
                                  std::uint32_t firstInstance = 0);
        [[nodiscard]] Status drawIndexed(CommandBuffer& commandBuffer,
                                         std::uint32_t indexCount,
                                         std::uint32_t instanceCount = 1,
                                         std::uint32_t firstIndex = 0,
                                         std::int32_t baseVertex = 0,
                                         std::uint32_t firstInstance = 0);
        [[nodiscard]] Status dispatchMesh(CommandBuffer& commandBuffer,
                                          std::uint32_t groupCountX,
                                          std::uint32_t groupCountY = 1,
                                          std::uint32_t groupCountZ = 1);
        [[nodiscard]] Status dispatchMeshIndirect(CommandBuffer& commandBuffer, const IndirectMeshDispatchDesc& desc);
        [[nodiscard]] Status drawIndirect(CommandBuffer& commandBuffer, const IndirectDrawDesc& desc);
        [[nodiscard]] Status drawIndexedIndirect(CommandBuffer& commandBuffer, const IndirectDrawDesc& desc);
        [[nodiscard]] Status beginConditionalExecution(CommandBuffer& commandBuffer,
                                                       const ConditionalExecutionDesc& desc);
        [[nodiscard]] Status endConditionalExecution(CommandBuffer& commandBuffer);
        [[nodiscard]] Result<QueryPoolHandle> createTimestampQueryPool(const TimestampQueryPoolDesc& desc);
        [[nodiscard]] Status destroyQueryPool(QueryPoolHandle queryPool);

        [[nodiscard]] Status resetQueryPool(CommandBuffer& commandBuffer,
                                            QueryPoolHandle queryPool,
                                            std::uint32_t firstQuery,
                                            std::uint32_t queryCount);
        [[nodiscard]] Status writeTimestamp(CommandBuffer& commandBuffer,
                                            QueryPoolHandle queryPool,
                                            std::uint32_t queryIndex);
        [[nodiscard]] Status beginGpuProfileScope(CommandBuffer& commandBuffer, const GpuProfileScopeDesc& desc);
        [[nodiscard]] Status endGpuProfileScope(CommandBuffer& commandBuffer);
        [[nodiscard]] Status resolveTimestamps(CommandBuffer& commandBuffer, const TimestampQueryResolveDesc& desc);
        [[nodiscard]] Result<QueryPoolHandle> createOcclusionQueryPool(const OcclusionQueryPoolDesc& desc);
        [[nodiscard]] Status beginOcclusionQuery(CommandBuffer& commandBuffer,
                                                 QueryPoolHandle queryPool,
                                                 std::uint32_t queryIndex);
        [[nodiscard]] Status endOcclusionQuery(CommandBuffer& commandBuffer);
        [[nodiscard]] Status resolveOcclusionQueries(CommandBuffer& commandBuffer,
                                                     const OcclusionQueryResolveDesc& desc);
        [[nodiscard]] Result<QueryPoolHandle> createPipelineStatisticsQueryPool(
            const PipelineStatisticsQueryPoolDesc& desc);
        [[nodiscard]] Status beginPipelineStatisticsQuery(CommandBuffer& commandBuffer,
                                                          QueryPoolHandle queryPool,
                                                          std::uint32_t queryIndex);
        [[nodiscard]] Status endPipelineStatisticsQuery(CommandBuffer& commandBuffer);
        [[nodiscard]] Status resolvePipelineStatisticsQueries(CommandBuffer& commandBuffer,
                                                              const PipelineStatisticsQueryResolveDesc& desc);
        [[nodiscard]] Result<QueryPoolHandle> createStreamOutputStatisticsQueryPool(
            const StreamOutputStatisticsQueryPoolDesc& desc);
        [[nodiscard]] Status beginStreamOutputStatisticsQuery(CommandBuffer& commandBuffer,
                                                              QueryPoolHandle queryPool,
                                                              std::uint32_t queryIndex);
        [[nodiscard]] Status endStreamOutputStatisticsQuery(CommandBuffer& commandBuffer, std::uint32_t stream);
        [[nodiscard]] Status resolveStreamOutputStatisticsQueries(CommandBuffer& commandBuffer,
                                                                  const StreamOutputStatisticsQueryResolveDesc& desc);
        [[nodiscard]] Result<QueryPoolHandle> createAccelerationStructureCompactedSizeQueryPool(
            const AccelerationStructureCompactedSizeQueryPoolDesc& desc);
        [[nodiscard]] Status writeAccelerationStructureCompactedSize(CommandBuffer& commandBuffer,
                                                                     QueryPoolHandle queryPool,
                                                                     std::uint32_t queryIndex,
                                                                     AccelerationStructureHandle source);
        [[nodiscard]] Status resolveAccelerationStructureCompactedSizes(
            CommandBuffer& commandBuffer, const AccelerationStructureCompactedSizeQueryResolveDesc& desc);
        [[nodiscard]] Result<QueryPoolHandle> createAccelerationStructureSerializationInfoQueryPool(
            const AccelerationStructureSerializationInfoQueryPoolDesc& desc);
        [[nodiscard]] Status writeAccelerationStructureSerializationInfo(CommandBuffer& commandBuffer,
                                                                         QueryPoolHandle queryPool,
                                                                         std::uint32_t queryIndex,
                                                                         AccelerationStructureHandle source);
        [[nodiscard]] Status resolveAccelerationStructureSerializationInfo(
            CommandBuffer& commandBuffer, const AccelerationStructureSerializationInfoQueryResolveDesc& desc);
        [[nodiscard]] Result<SurfaceHandle> createSurface(const SurfaceDesc& desc);
        [[nodiscard]] Result<SurfaceCapabilities> querySurfaceCapabilities(SurfaceHandle surface) const;
        [[nodiscard]] Status destroySurface(SurfaceHandle surface);
        [[nodiscard]] Result<SwapchainHandle> createSwapchain(const SwapchainDesc& desc);
        [[nodiscard]] Status destroySwapchain(SwapchainHandle swapchain);
        [[nodiscard]] Status resizeSwapchain(SwapchainHandle swapchain, std::uint32_t width, std::uint32_t height);
        [[nodiscard]] Result<AcquiredSwapchainImage> acquireNextImage(SwapchainHandle swapchain);

        [[nodiscard]] bool hasAcquiredSwapchainImage(SwapchainHandle swapchain) const noexcept;

        [[nodiscard]] Status cancelSwapchainImage(SwapchainHandle swapchain);

        [[nodiscard]] Result<PresentStatus> present(SwapchainHandle swapchain, TimelinePoint renderComplete);
        [[nodiscard]] Result<SwapchainPresentationInfo> querySwapchainPresentation(SwapchainHandle swapchain) const;
        [[nodiscard]] Result<SwapchainFrameTiming> querySwapchainFrameTiming(SwapchainHandle swapchain) const;

    private:
        explicit Device(std::unique_ptr<detail::BackendDevice> backend) noexcept;

        std::unique_ptr<detail::BackendDevice> backend_;

        friend Result<Device> createDevice(const DeviceDesc& desc);
        friend struct detail::DeviceAccess;
    };

    [[nodiscard]] Result<std::vector<AdapterInfo>> enumerateAdapters();
    [[nodiscard]] Result<std::vector<AdapterInfo>> enumerateAdapters(Backend backend);
    [[nodiscard]] Result<Device> createDevice(const DeviceDesc& desc = {});

} // namespace rachis
