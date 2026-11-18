#pragma once

#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <dispatch/dispatch.h>

#include "Common/HandleTable.h"
#include "Common/MonotonicIdentitySource.h"
#include "Internal/BackendCommandBuffer.h"
#include "Internal/CommandBufferAccess.h"
#include "Internal/MetalDevice.h"
#include "Internal/MetalDiagnostics.h"
#include "Internal/MetalFactory.h"
#include "Internal/MetalFormatSupport.h"
#include "Internal/MetalMacosProfile.h"
#include "Internal/MetalShaderBinding.h"
#include "Internal/MetalShaderManifest.h"
#include "Internal/TimelinePointAccess.h"
#include "Internal/VertexInputValidation.h"
#include "Metal/MetalPresentationFeedback.h"
#include "Metal/MetalSubmissionFeedback.h"
#include "Metal/MetalTestHooks.h"

#include "Rachis/Command.h"
#include "Rachis/Pipeline.h"
#include "Rachis/Query.h"
#include "Rachis/Rendering.h"
#include "Rachis/Resource.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <ranges>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void objc_autoreleasePoolPop(void* context);

namespace rachis::metal
{

    class MetalAutoreleasePool final
    {
    public:
        MetalAutoreleasePool() noexcept : context_(objc_autoreleasePoolPush())
        {
        }

        ~MetalAutoreleasePool()
        {
            objc_autoreleasePoolPop(context_);
        }

        MetalAutoreleasePool(const MetalAutoreleasePool&) = delete;
        MetalAutoreleasePool& operator=(const MetalAutoreleasePool&) = delete;
        MetalAutoreleasePool(MetalAutoreleasePool&&) = delete;
        MetalAutoreleasePool& operator=(MetalAutoreleasePool&&) = delete;

    private:
        void* context_ = nullptr;
    };

    inline constexpr std::uint32_t metalMaxBufferArguments = 31;
    inline constexpr std::uint32_t metalMaxTextureArguments = 128;
    inline constexpr std::uint32_t metalMaxSamplerArguments = 16;
    inline constexpr std::uint32_t metalMaxMeshPayloadBytes = 16384 - 32;
    inline constexpr std::uint32_t metalMaxMeshOutputVertices = 256;
    inline constexpr std::uint32_t metalMaxMeshOutputPrimitives = 512;
    inline constexpr std::uint32_t metalMaxMeshOutputMemoryBytes = 16384;
    inline constexpr std::uint32_t metalMaxMeshGridThreadgroups = 1048575;

    inline constexpr std::string_view kMetalDeviceLabel = "RHI.Metal";
    inline constexpr std::string_view kMetalQueueLabel = "RHI.Metal.Graphics";

    inline constexpr MTLStages kMetalComputeEncoderStages =
        MTLStageDispatch | MTLStageBlit | MTLStageAccelerationStructure;
    inline constexpr MTLStages kMetalRenderEncoderStages =
        MTLStageVertex | MTLStageFragment | MTLStageObject | MTLStageMesh | MTLStageTile;

    inline constexpr std::uint32_t kMetalPushConstantSliceAlignment = 256;
    inline constexpr std::uint32_t kMetalPushConstantPageBytes = 16384;

    class MetalPointerSet final
    {
    public:
        MetalPointerSet() = default;

        [[nodiscard]] bool insertIfAbsent(const void* ptr) noexcept
        {
            if (ptr == nullptr)
            {
                return false;
            }
            if (!slots_.empty())
            {
                const std::size_t mask = slots_.size() - 1U;
                std::size_t index = hashPointer(ptr) & mask;
                for (;;)
                {
                    const void* const existing = slots_[index];
                    if (existing == ptr)
                    {
                        return false;
                    }
                    if (existing == nullptr)
                    {
                        if ((size_ + 1U) * 10U <= slots_.size() * 7U)
                        {
                            slots_[index] = ptr;
                            ++size_;
                            return true;
                        }
                        break;
                    }
                    index = (index + 1U) & mask;
                }
            }
            const std::size_t nextCapacity = slots_.empty() ? std::size_t{64} : slots_.size() * 2U;
            if (!rehash(nextCapacity))
            {
                return true;
            }
            const std::size_t mask = slots_.size() - 1U;
            std::size_t index = hashPointer(ptr) & mask;
            while (slots_[index] != nullptr)
            {
                index = (index + 1U) & mask;
            }
            slots_[index] = ptr;
            ++size_;
            return true;
        }

        void clear() noexcept
        {
            if (size_ != 0)
            {
                std::ranges::fill(slots_, nullptr);
                size_ = 0;
            }
        }

    private:
        [[nodiscard]] static constexpr std::size_t hashPointer(const void* ptr) noexcept
        {
            std::uint64_t x = reinterpret_cast<std::uintptr_t>(ptr);
            x ^= x >> 33U;
            x *= 0xff51afd7ed558ccdULL;
            x ^= x >> 33U;
            return static_cast<std::size_t>(x);
        }

        [[nodiscard]] bool rehash(std::size_t nextCapacity) noexcept
        {
            try
            {
                std::vector<const void*> grown(nextCapacity, nullptr);
                const std::size_t mask = nextCapacity - 1U;
                for (const void* const existing : slots_)
                {
                    if (existing == nullptr)
                    {
                        continue;
                    }
                    std::size_t index = hashPointer(existing) & mask;
                    while (grown[index] != nullptr)
                    {
                        index = (index + 1U) & mask;
                    }
                    grown[index] = existing;
                }
                slots_ = std::move(grown);
                return true;
            }
            catch (const std::bad_alloc&)
            {
                return false;
            }
        }

        std::vector<const void*> slots_;
        std::size_t size_ = 0;
    };

    enum class MetalArgumentTableStage : std::uint8_t
    {
        compute,
        vertex,
        fragment,
        object,
        mesh,
    };

    inline constexpr std::size_t kMetalArgumentTableStageCount = 5;

    struct MetalStageArgumentTable final
    {
        __strong id<MTL4ArgumentTable> table = nil;
        std::uint32_t maxBuffers = 0;
        std::uint32_t maxTextures = 0;
        std::uint32_t maxSamplers = 0;
    };

    struct MetalPushConstantSlice final
    {
        __strong id<MTLBuffer> buffer = nil;
        std::uint64_t gpuAddress = 0;
        std::uint32_t byteLength = 0;
    };

    struct MetalAllocationTelemetry final
    {
        std::uint64_t allocationId = 0;
        ResourceMemoryKind kind = ResourceMemoryKind::buffer;
        MemoryType memoryType = MemoryType::deviceLocal;
        std::uint64_t allocationBytes = 0;
        std::string label;
        bool logicalAlive = true;
    };

    struct MetalHeapRecord;

    struct MetalPlacedLease final
    {
        std::shared_ptr<MetalHeapRecord> heap;
        std::uint64_t offset = 0;
        std::uint64_t size = 0;
    };

    struct MetalHeapRecord final
    {
        ResourceHeapDesc desc{};
        std::string label;
        __strong id<MTLHeap> heap;
        std::shared_ptr<MetalAllocationTelemetry> telemetry;
        std::mutex mutex;
        std::vector<std::weak_ptr<MetalPlacedLease>> placements;
        bool alive = true;
    };

    struct MetalBufferOwner final
    {
        __strong id<MTLBuffer> buffer;
        std::shared_ptr<MetalAllocationTelemetry> telemetry;
        std::shared_ptr<MetalPlacedLease> placement;
    };

    struct MetalTextureOwner final
    {
        __strong id<MTLTexture> texture;
        std::shared_ptr<MetalAllocationTelemetry> telemetry;
        std::shared_ptr<MetalPlacedLease> placement;
    };

    struct MetalSamplerOwner final
    {
        __strong id<MTLSamplerState> sampler;
    };

    struct MetalBufferRecord
    {
        std::shared_ptr<MetalBufferOwner> owner;
        __strong id<MTLBuffer> buffer;
        BufferDesc desc{};
        std::string label;
        std::uint64_t allocationId = 0;
        std::uint64_t allocationBytes = 0;
        bool mapped = false;
        std::uint64_t mapOffset = 0;
        std::uint64_t mapSize = 0;
    };

    struct MetalTextureLayoutTable final
    {
        std::vector<TextureLayout> layouts;
        std::uint32_t mipLevels = 1;
        std::uint32_t arrayLayers = 1;
    };

    struct MetalTextureRecord
    {
        std::shared_ptr<MetalTextureOwner> owner;
        std::shared_ptr<MetalTextureLayoutTable> layouts;
        __strong id<MTLTexture> texture;
        TextureDesc desc{};
        std::string label;
        std::vector<Format> viewFormats;
        std::uint64_t allocationId = 0;
        std::uint64_t allocationBytes = 0;
        std::uint32_t liveViews = 0;
        bool swapchainImage = false;
        __strong id<CAMetalDrawable> drawable = nil;
        SwapchainHandle swapchain{};
        std::uint64_t swapchainAcquireId = 0;
    };

    struct MetalTextureViewRecord
    {
        std::shared_ptr<MetalTextureOwner> owner;
        std::shared_ptr<MetalTextureOwner> parent;
        __strong id<MTLTexture> view;
        TextureHandle texture{};
        TextureViewDesc desc{};
        std::string label;
        bool swapchainView = false;
    };

    struct MetalSamplerRecord
    {
        std::shared_ptr<MetalSamplerOwner> owner;
        __strong id<MTLSamplerState> sampler;
        SamplerDesc desc{};
        std::string label;
    };

    enum class MetalQueryPoolKind : std::uint8_t
    {
        timestamp,
    };

    struct MetalQueryPoolRecord final
    {
        MetalQueryPoolKind kind = MetalQueryPoolKind::timestamp;
        QueueType queueType = QueueType::graphics;
        std::uint32_t queryCount = 0;
        std::uint64_t entrySize = timestampQueryResultSize;
        std::string label;
        __strong id<MTL4CounterHeap> heap;
        mutable std::mutex stateMutex;
        std::vector<std::uint8_t> pendingSlots;
    };

    static_assert(sizeof(MTL4TimestampHeapEntry) == timestampQueryResultSize);

    struct MetalShaderLibraryEntryRecord final
    {
        ShaderStage stage = ShaderStage::none;
        std::vector<std::byte> bytecode;
        std::string entryPoint;
        std::vector<SpecializationConstant> specializationConstants;
    };

    struct MetalShaderLibraryRecord final
    {
        std::vector<MetalShaderLibraryEntryRecord> entries;
        std::string label;
    };

    inline constexpr std::size_t kMetalShaderCodeDigestBytes = 32;

    struct MetalShaderModuleRecord
    {
        __strong id<MTLLibrary> library;
        __strong id<MTLFunction> function;
        std::array<std::uint8_t, kMetalShaderCodeDigestBytes> codeDigest{};
        ShaderInterfaceSnapshot snapshot{};
        MetalResolvedStageBindings stageBindings{};
        std::optional<MetalReflectedStageInterface> stageInterface{};
        std::shared_ptr<MetalShaderLibraryRecord> sourceLibrary;
        std::string entryPoint;
        std::string label;
    };

    struct MetalBindingLayoutRecord final
    {
        std::vector<BindingLayoutEntry> entries;
        std::uint32_t resourceDescriptorCount = 0;
        std::uint32_t samplerDescriptorCount = 0;
        std::uint32_t dynamicBufferCount = 0;
        std::optional<std::uint32_t> variableBinding;
        std::string label;
    };

    struct MetalBindingPoolState;

    struct MetalBindingPoolSetLease final
    {
        std::shared_ptr<MetalBindingPoolState> state;
        std::uint64_t activeCommandRetentions = 0;
        bool armed = false;
        MetalBindingPoolSetLease() = default;
        MetalBindingPoolSetLease(const MetalBindingPoolSetLease&) = delete;
        MetalBindingPoolSetLease& operator=(const MetalBindingPoolSetLease&) = delete;
        MetalBindingPoolSetLease(MetalBindingPoolSetLease&&) = delete;
        MetalBindingPoolSetLease& operator=(MetalBindingPoolSetLease&&) = delete;
        ~MetalBindingPoolSetLease();
    };

    struct MetalBindingPoolVersionLease final
    {
        std::shared_ptr<MetalBindingPoolState> state;
        std::array<std::uint32_t, bindingTypeCount> counts{};
        bool armed = false;
        MetalBindingPoolVersionLease() = default;
        MetalBindingPoolVersionLease(const MetalBindingPoolVersionLease&) = delete;
        MetalBindingPoolVersionLease& operator=(const MetalBindingPoolVersionLease&) = delete;
        MetalBindingPoolVersionLease(MetalBindingPoolVersionLease&&) = delete;
        MetalBindingPoolVersionLease& operator=(MetalBindingPoolVersionLease&&) = delete;
        ~MetalBindingPoolVersionLease();
    };

    struct MetalBindingPoolState final
    {
        std::mutex mutex;
        std::uint32_t maxBindingSets = 0;
        std::uint32_t liveBindingSets = 0;
        std::array<std::uint32_t, bindingTypeCount> descriptorCapacity{};
        std::array<std::uint32_t, bindingTypeCount> liveDescriptors{};
        bool logicalAlive = true;
        std::string label;
    };

    struct MetalBindingSlot final
    {
        BindingType type = BindingType::uniformBuffer;
        BindingResource resource{};
        std::shared_ptr<void> keepAlive{};
        __strong id<MTLBuffer> buffer = nil;
        __strong id<MTLTexture> texture = nil;
        __strong id<MTLSamplerState> sampler = nil;
        std::uint64_t gpuAddress = 0;
        std::uint64_t offset = 0;
        std::uint64_t range = 0;
        std::uint32_t binding = 0;
        std::uint32_t arrayElement = 0;
        std::uint32_t metalIndex = 0;
    };

    struct MetalBindingSetVersion final
    {
        std::shared_ptr<MetalBindingLayoutRecord> layout;
        std::shared_ptr<MetalBindingPoolState> pool;
        std::shared_ptr<MetalBindingPoolVersionLease> versionLease;
        std::vector<MetalBindingSlot> slots;
        __strong id<MTL4ArgumentTable> table;
        std::uint32_t variableDescriptorCount = 0;
    };

    struct MetalBindingSetRecord final
    {
        std::shared_ptr<MetalBindingLayoutRecord> layout;
        std::shared_ptr<MetalBindingPoolSetLease> poolSet;
        std::shared_ptr<MetalBindingSetVersion> version;
        std::string label;
    };

    struct MetalBindingSetCommandRetention final
    {
        std::shared_ptr<MetalBindingPoolSetLease> poolSet;
        std::shared_ptr<MetalBindingPoolVersionLease> versionLease;
        MetalBindingSetCommandRetention() = default;
        MetalBindingSetCommandRetention(const MetalBindingSetCommandRetention&) = delete;
        MetalBindingSetCommandRetention& operator=(const MetalBindingSetCommandRetention&) = delete;
        MetalBindingSetCommandRetention(MetalBindingSetCommandRetention&&) = delete;
        MetalBindingSetCommandRetention& operator=(MetalBindingSetCommandRetention&&) = delete;
        ~MetalBindingSetCommandRetention();
    };

    struct MetalLayoutBindingSlotMap final
    {
        std::uint32_t set = 0;
        std::uint32_t binding = 0;
        std::uint32_t count = 0;
        std::uint32_t metalBaseIndex = 0;
    };

    struct MetalPipelineLayoutRecord final
    {
        std::vector<std::shared_ptr<MetalBindingLayoutRecord>> sets;
        std::vector<PushConstantRange> pushConstantRanges;
        std::vector<MetalLayoutBindingSlotMap> remappedBindings;
        std::uint32_t pushConstantBytes = 0;
        std::uint32_t pushConstantMetalIndex = 0;
        std::uint32_t vertexStageBufferSlots = 0;
        bool reservePushConstantBuffer = false;
        std::string label;
    };

    enum class MetalPipelineKind : std::uint8_t
    {
        compute,
        graphics,
    };

    struct MetalGraphicsPipelineState final
    {
        PrimitiveTopology topology = PrimitiveTopology::triangleList;
        FrontFace frontFace = FrontFace::counterClockwise;
        CullMode cullMode = CullMode::back;
        FillMode fillMode = FillMode::solid;
        std::int32_t depthBias = 0;
        float depthBiasClamp = 0.0F;
        float depthBiasSlopeScale = 0.0F;
        bool depthClipEnabled = true;
        DepthStencilState depthStencil{};
        std::uint32_t sampleCount = 1;
        std::uint32_t sampleMask = std::numeric_limits<std::uint32_t>::max();
        bool alphaToCoverageEnabled = false;
        StripIndexFormat stripIndexFormat{};
        std::vector<Format> colorFormats;
        Format depthStencilFormat = Format::unknown;
        std::uint32_t viewMask = 0;
        std::uint32_t vertexBufferCount = 0;
        std::uint32_t vertexBufferBaseIndex = 0;
        std::vector<std::uint32_t> vertexBufferAlignments;
        bool usesBlendConstants = false;
        bool stencilEnabled = false;
        bool rasterizationEnabled = true;
    };

    struct MetalPipelineRecord final
    {
        MetalPipelineKind kind = MetalPipelineKind::compute;
        std::shared_ptr<MetalPipelineLayoutRecord> layout;
        std::vector<std::uint8_t> setRequired;
        std::vector<std::shared_ptr<void>> retained{};
        __strong id<MTLComputePipelineState> compute;
        __strong id<MTLRenderPipelineState> graphics;
        __strong id<MTLDepthStencilState> depthStencil;
        ShaderWorkgroupSize workgroup{};
        MetalResolvedStageBindings computeStageBindings{};
        MetalResolvedStageBindings vertexStageBindings{};
        MetalResolvedStageBindings pixelStageBindings{};
        MetalResolvedStageBindings objectStageBindings{};
        MetalResolvedStageBindings meshStageBindings{};
        bool hasPixelShader = false;
        bool mesh = false;
        bool hasObjectShader = false;
        MTLSize objectThreadgroupSize{1, 1, 1};
        MTLSize meshThreadgroupSize{1, 1, 1};
        std::uint32_t pushConstantBytes = 0;
        bool reservePushConstantBuffer = false;
        std::optional<MetalGraphicsPipelineState> graphicsState;
        std::string label;
    };

    struct MetalBoundBuffer final
    {
        __strong id<MTLBuffer> buffer;
        std::uint64_t offset = 0;
        std::uint64_t size = 0;
    };

    struct MetalRenderingAttachment final
    {
        __strong id<MTLTexture> texture;
        Format format = Format::unknown;
        LoadOperation loadOperation = LoadOperation::load;
        StoreOperation storeOperation = StoreOperation::store;
        ClearColor clearColor{};
        float depthClear = 1.0F;
        std::uint32_t stencilClear = 0;
        LoadOperation stencilLoadOperation = LoadOperation::load;
        StoreOperation stencilStoreOperation = StoreOperation::store;
    };

    struct MetalRenderingScope final
    {
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::uint32_t sampleCount = 1;
        std::uint32_t layerCount = 1;
        std::vector<MetalRenderingAttachment> colors;
        MetalRenderingAttachment depth{};
        bool hasDepth = false;
        bool hasStencil = false;
    };

    struct MetalPipelineCompilationRecord final
    {
        PipelineCompilationState state = PipelineCompilationState::failed;
        PipelineHandle pipeline{};
        Error error{ErrorCode::invalidState, 0, "Metal pipeline compilation has no result."};
    };

    enum class MetalPipelineCacheKind : std::uint32_t
    {
        compute = 0,
        graphics = 1,
    };

    struct MetalPipelineCacheIdentity final
    {
        PipelineCacheKey key{};
        MetalPipelineCacheKind kind = MetalPipelineCacheKind::compute;

        friend constexpr auto operator<=>(const MetalPipelineCacheIdentity&,
                                          const MetalPipelineCacheIdentity&) noexcept = default;
    };

    struct MetalPipelineCacheSegment final
    {
        std::vector<std::byte> bytes;
        __strong id<MTL4Archive> archive;
        __strong NSURL* archiveFileURL;

        MetalPipelineCacheSegment() = default;
        MetalPipelineCacheSegment(const MetalPipelineCacheSegment&) = delete;
        MetalPipelineCacheSegment& operator=(const MetalPipelineCacheSegment&) = delete;
        MetalPipelineCacheSegment(MetalPipelineCacheSegment&&) = delete;
        MetalPipelineCacheSegment& operator=(MetalPipelineCacheSegment&&) = delete;
        ~MetalPipelineCacheSegment()
        {
            if (archiveFileURL != nil)
            {
                [[NSFileManager defaultManager] removeItemAtURL:archiveFileURL error:nil];
            }
        }
    };

    struct MetalPipelineCacheLive final
    {
        __strong id<MTLComputePipelineState> compute;
        __strong id<MTLRenderPipelineState> graphics;
        std::vector<std::byte> fingerprint;
        std::shared_ptr<MetalPipelineCacheSegment> segment;
        bool inSerializer = false;
    };

    struct MetalPipelineCacheRecord final
    {
        std::mutex mutex;
        std::shared_mutex serializeMutex;
        std::string label;
        std::map<MetalPipelineCacheIdentity, MetalPipelineCacheLive> live;
        std::vector<std::shared_ptr<MetalPipelineCacheSegment>> segments;
        __strong id<MTL4PipelineDataSetSerializer> serializer;
        __strong id<MTL4Compiler> compiler;
        bool serializerDirty = false;

        MetalPipelineCacheRecord() = default;
        MetalPipelineCacheRecord(const MetalPipelineCacheRecord&) = delete;
        MetalPipelineCacheRecord& operator=(const MetalPipelineCacheRecord&) = delete;
        MetalPipelineCacheRecord(MetalPipelineCacheRecord&&) = delete;
        MetalPipelineCacheRecord& operator=(MetalPipelineCacheRecord&&) = delete;
        ~MetalPipelineCacheRecord() = default;
    };

    enum class MetalEncoderKind : std::uint8_t
    {
        none,
        copy,
        compute,
        render,
    };

    class MetalCommandBuffer;
    struct MetalSwapchainRecord;

    struct MetalWaitedDrawable final
    {
        __strong id<CAMetalDrawable> drawable = nil;
        std::shared_ptr<MetalSwapchainRecord> swapchain;
        std::uint64_t acquireId = 0;
    };

    enum class MetalQuerySlotPrecondition : std::uint8_t
    {
        none = 0,
        mustNotBePending = 1,
        mustBePending = 2,
    };

    struct MetalStagedQuerySlot final
    {
        QueryPoolHandle queryPool{};
        std::shared_ptr<MetalQueryPoolRecord> pool;
        std::uint32_t queryIndex = 0;
        MetalQuerySlotPrecondition precondition = MetalQuerySlotPrecondition::none;
        bool pendingAfter = false;
    };

    struct MetalDeferredStoreActions final
    {
        std::array<MTLStoreAction, maxColorAttachments> colors{};
        std::uint32_t colorCount = 0;
        MTLStoreAction depth = MTLStoreActionDontCare;
        MTLStoreAction stencil = MTLStoreActionDontCare;
        bool hasDepth = false;
        bool hasStencil = false;
        bool pending = false;
    };

    struct MetalCommandEntry final
    {
        __strong id<MTL4CommandAllocator> allocator;
        __strong id<MTL4CommandBuffer> command;
        __strong id<MTL4ComputeCommandEncoder> computeEncoder;
        __strong id<MTL4RenderCommandEncoder> renderEncoder;
        MetalDeferredStoreActions renderStoreActions{};
        __strong id<MTLResidencySet> residency;
        MetalPointerSet residentAllocations;
        MetalCommandBuffer* lease = nullptr;
        bool inFlight = false;
        bool begun = false;
        bool residencyBound = false;
        bool residencyDirty = false;
        MetalEncoderKind encoder = MetalEncoderKind::none;
        MTLStages pendingBarrierAfterStages = static_cast<MTLStages>(0);
        MTLStages pendingBarrierBeforeStages = static_cast<MTLStages>(0);
        std::vector<std::shared_ptr<MetalBindingSetVersion>> boundComputeVersions;
        std::vector<std::shared_ptr<MetalBindingSetVersion>> boundGraphicsVersions;

        std::array<MetalStageArgumentTable, kMetalArgumentTableStageCount> stageArgumentTables{};
        std::vector<id<MTL4ArgumentTable>> supersededArgumentTables;
        bool computeTableDirty = true;
        bool vertexTableDirty = true;
        bool fragmentTableDirty = true;
        __strong id<MTL4ArgumentTable> activeComputeTable = nil;
        __strong id<MTL4ArgumentTable> activeVertexTable = nil;
        __strong id<MTL4ArgumentTable> activeFragmentTable = nil;
        const MetalPipelineRecord* activeComputePipeline = nullptr;
        const MetalPipelineRecord* activeGraphicsPipeline = nullptr;
        std::array<std::byte, maxPushConstantBytes> computePushConstants{};
        std::uint32_t computePushConstantBytes = 0;
        bool computePushConstantsDirty = true;
        MetalPushConstantSlice activeComputePushSlice{};
        std::array<std::byte, maxPushConstantBytes> graphicsPushConstants{};
        std::uint32_t graphicsPushConstantBytes = 0;
        bool graphicsPushConstantsDirty = true;
        MetalPushConstantSlice activeGraphicsPushSlice{};

        std::vector<id<MTLBuffer>> pushConstantPages;
        std::vector<id<MTLBuffer>> indirectScratchBuffers;
        std::size_t pushConstantPageIndex = 0;
        std::uint32_t pushConstantPageOffset = 0;
        std::shared_ptr<MetalPipelineRecord> boundComputePipeline;
        std::shared_ptr<MetalPipelineRecord> boundGraphicsPipeline;
        std::array<MetalBoundBuffer, maxVertexBuffers> vertexBuffers{};
        MetalBoundBuffer indexBuffer{};
        IndexFormat indexFormat = IndexFormat::uint16;
        std::array<Viewport, maxViewportCount> viewports{};
        std::array<ScissorRect, maxViewportCount> scissors{};
        bool scissorsEmpty = false;
        BlendConstants blendConstants{};
        std::uint32_t stencilReference = 0;
        std::vector<MetalWaitedDrawable> waitedDrawables;
    };

    struct MetalCommandPoolState final
    {
        mutable std::mutex mutex;
        __strong id<MTLDevice> device;
        __strong id<MTLLogState> logState;
        QueueType queueType = QueueType::graphics;
        CommandPoolFlag flags = CommandPoolFlag::none;
        std::string label;
        std::vector<std::unique_ptr<MetalCommandEntry>> entries;
        std::vector<MetalCommandEntry*> available;
        std::vector<MetalCommandEntry*> quarantined;
    };

    struct MetalLayoutCell final
    {
        const MetalTextureLayoutTable* table = nullptr;
        std::size_t index = 0;

        [[nodiscard]] bool operator==(const MetalLayoutCell&) const noexcept = default;
    };

    struct MetalLayoutCellHash final
    {
        [[nodiscard]] std::size_t operator()(const MetalLayoutCell& cell) const noexcept
        {
            const auto pointer = std::hash<const void*>{}(cell.table);
            return pointer ^
                   (std::hash<std::size_t>{}(cell.index) + 0x9e3779b97f4a7c15ULL + (pointer << 6U) + (pointer >> 2U));
        }
    };

    struct MetalLayoutWrite final
    {
        std::shared_ptr<MetalTextureLayoutTable> table;
        std::size_t index = 0;
        TextureLayout layout = TextureLayout::undefined;
    };

    struct MetalLayoutEntryRequirement final
    {
        std::shared_ptr<MetalTextureLayoutTable> table;
        std::size_t index = 0;
        TextureLayout layout = TextureLayout::undefined;
        TextureSubresource subresource{};
        std::string_view operation;
    };

    struct MetalCommandLayoutState final
    {
        std::vector<MetalLayoutWrite> writes;
        std::unordered_map<MetalLayoutCell, std::size_t, MetalLayoutCellHash> writeIndex;
        std::vector<MetalLayoutEntryRequirement> entries;

        void clear() noexcept
        {
            writes.clear();
            writeIndex.clear();
            entries.clear();
        }
    };

    struct MetalLayoutTransition final
    {
        std::shared_ptr<MetalTextureLayoutTable> table;
        std::uint32_t firstMip = 0;
        std::uint32_t mipCount = 1;
        std::uint32_t firstLayer = 0;
        std::uint32_t layerCount = 1;
        std::uint32_t aspects = 1;
        TextureLayout before = TextureLayout::undefined;
        TextureLayout after = TextureLayout::undefined;
        std::string_view operation;
    };

    class MetalCommandBuffer final : public detail::BackendCommandBuffer
    {
    public:
        MetalCommandBuffer(std::shared_ptr<MetalCommandPoolState> owner,
                           MetalCommandEntry& entry,
                           std::uint64_t deviceToken) noexcept
            : detail::BackendCommandBuffer(detail::BackendCommandBufferKind::metal), owner_(std::move(owner)),
              entry_(&entry), deviceToken_(deviceToken),
              queueType_(owner_ != nullptr ? owner_->queueType : QueueType::graphics)
        {
        }

        ~MetalCommandBuffer() override
        {
            abandon();
        }

        MetalCommandBuffer(const MetalCommandBuffer&) = delete;
        MetalCommandBuffer& operator=(const MetalCommandBuffer&) = delete;
        MetalCommandBuffer(MetalCommandBuffer&&) = delete;
        MetalCommandBuffer& operator=(MetalCommandBuffer&&) = delete;

        [[nodiscard]] QueueType queueType() const noexcept override
        {
            return queueType_;
        }

        [[nodiscard]] std::uint64_t deviceToken() const noexcept override
        {
            return deviceToken_;
        }

        [[nodiscard]] detail::RecordingStateLock lockRecordingState() const override
        {
            return detail::RecordingStateLock::lockPool(owner_);
        }

        [[nodiscard]] MetalCommandPoolState* owner() const noexcept
        {
            return owner_.get();
        }
        [[nodiscard]] const std::shared_ptr<MetalCommandPoolState>& ownerShared() const noexcept
        {
            return owner_;
        }
        [[nodiscard]] MetalCommandEntry* entry() const noexcept
        {
            return entry_;
        }

        [[nodiscard]] Status validateReset(bool requireIndividualOptIn) const;
        [[nodiscard]] Status resetRecording();
        void clearRecordingState() noexcept;
#ifdef RACHIS_TESTING
        void setInFlightForTesting(bool inFlight) noexcept
        {
            if (entry_ != nullptr)
            {
                entry_->inFlight = inFlight;
            }
        }
#endif
        [[nodiscard]] Status ensureCopyEncoder();
        [[nodiscard]] Status ensureComputeEncoder();
        [[nodiscard]] Status encodeStageBarrier(MTLStages afterStages, MTLStages beforeStages);
        void endEncoders() noexcept;
        void detachForShutdown() noexcept;
        [[nodiscard]] MetalCommandEntry* release() noexcept;
        [[nodiscard]] std::vector<std::shared_ptr<void>>& retained() noexcept
        {
            return retained_;
        }
        [[nodiscard]] std::vector<std::shared_ptr<void>> takeRetained() noexcept
        {
            return std::move(retained_);
        }
        [[nodiscard]] std::vector<MetalStagedQuerySlot>& stagedQuerySlots() noexcept
        {
            return stagedQuerySlots_;
        }
        [[nodiscard]] const std::vector<MetalStagedQuerySlot>& stagedQuerySlots() const noexcept
        {
            return stagedQuerySlots_;
        }
        [[nodiscard]] MetalCommandLayoutState& layoutState() noexcept
        {
            return layoutState_;
        }
        [[nodiscard]] const MetalCommandLayoutState& layoutState() const noexcept
        {
            return layoutState_;
        }

    private:
        void abandon() noexcept;

        std::shared_ptr<MetalCommandPoolState> owner_;
        MetalCommandEntry* entry_ = nullptr;
        std::uint64_t deviceToken_ = 0;
        QueueType queueType_ = QueueType::graphics;
        std::vector<std::shared_ptr<void>> retained_;
        std::vector<MetalStagedQuerySlot> stagedQuerySlots_;
        MetalCommandLayoutState layoutState_;
    };

    struct MetalPendingEntry final
    {
        std::uint64_t timelineValue = std::numeric_limits<std::uint64_t>::max();
        std::shared_ptr<MetalCommandPoolState> pool;
        MetalCommandEntry* entry = nullptr;
        std::vector<std::shared_ptr<void>> retained{};
    };

    struct MetalNativeQueueGate final
    {
        std::mutex mutex;
    };

    struct MetalQueueTimeline final
    {
        QueueType type = QueueType::graphics;
        __strong id<MTLSharedEvent> event;
        std::uint64_t nextValue = 1;
        std::atomic<std::uint64_t> lastSubmitted{0};
        MetalSubmissionFeedback feedback;
        mutable std::mutex retirementMutex;
        std::deque<MetalPendingEntry> pending;
    };

    inline constexpr std::size_t metalQueueCount = 3;

    struct MetalViewVisibility final
    {
        __strong id view = nil;
        std::atomic<bool> hidden{false};
        std::atomic<bool> refreshQueued{false};
    };

    struct MetalSurfaceRecord final
    {
        __strong CAMetalLayer* layer = nil;
        std::shared_ptr<MetalViewVisibility> viewVisibility;
        std::string label;
        bool swapchainClaimed = false;
    };

    struct MetalInFlightPresent final
    {
        std::uint64_t timelineValue = 0;
        __strong id<CAMetalDrawable> drawable = nil;
    };

    struct MetalSwapchainRecord final
    {
        SurfaceHandle surface{};
        SwapchainDesc desc{};
        std::string label;
        __strong id<CAMetalDrawable> drawable = nil;
        TextureHandle texture{};
        TextureViewHandle view{};
        std::mutex operationMutex;
        bool acquired = false;
        bool leaseDroppedByHostClose = false;
        std::uint64_t lastAcquireId = 0;
        std::atomic<std::uint64_t> currentAcquireId{0};
        std::atomic<const void*> waitedDrawableToken{nullptr};
        std::uint64_t generation = 1;
        std::uint32_t inFlightPresents = 0;
        std::uint64_t lastAcquireNs = 0;
        std::uint64_t lastPresentSubmitNs = 0;
        std::uint64_t lastPresentIntervalNs = 0;
        std::uint32_t presentsCompleted = 0;
        std::uint64_t submittedPresentationSequence = 0;
        std::shared_ptr<MetalPresentationFeedback> presentationFeedback = std::make_shared<MetalPresentationFeedback>();
        std::vector<MetalInFlightPresent> inFlight;
    };

    struct MetalDeviceNative
    {
        __strong id<MTLDevice> device;
        __strong id<MTL4CommandQueue> commandQueue;
        __strong id<MTL4Compiler> compiler;
        __strong id<MTLLogState> logState;
        std::shared_ptr<MetalDiagnosticSink> sink = std::make_shared<MetalDiagnosticSink>();
        MetalValidationConfiguration validation{};
        std::unique_ptr<MetalNativeQueueGate> queueGate = std::make_unique<MetalNativeQueueGate>();
        std::array<MetalQueueTimeline, metalQueueCount> timelines{};
        detail::HandleTable<std::shared_ptr<MetalHeapRecord>, ResourceHeapHandle> heaps;
        detail::HandleTable<MetalBufferRecord, BufferHandle> buffers;
        detail::HandleTable<MetalTextureRecord, TextureHandle> textures;
        detail::HandleTable<MetalTextureViewRecord, TextureViewHandle> views;
        detail::HandleTable<MetalSamplerRecord, SamplerHandle> samplers;
        detail::HandleTable<MetalShaderModuleRecord, ShaderModuleHandle> shaderModules;
        detail::HandleTable<std::shared_ptr<MetalShaderLibraryRecord>, ShaderLibraryHandle> shaderLibraries;
        detail::HandleTable<std::shared_ptr<MetalBindingLayoutRecord>, BindingLayoutHandle> bindingLayouts;
        detail::HandleTable<std::shared_ptr<MetalBindingPoolState>, BindingPoolHandle> bindingPools;
        detail::HandleTable<std::shared_ptr<MetalBindingSetRecord>, BindingSetHandle> bindingSets;
        detail::HandleTable<std::shared_ptr<MetalPipelineLayoutRecord>, PipelineLayoutHandle> pipelineLayouts;
        detail::HandleTable<std::shared_ptr<MetalPipelineRecord>, PipelineHandle> pipelines;
        detail::HandleTable<std::shared_ptr<MetalPipelineCompilationRecord>, PipelineCompilationHandle>
            pipelineCompilations;
        detail::HandleTable<std::shared_ptr<MetalPipelineCacheRecord>, PipelineCacheHandle> pipelineCaches;
        BindingUpdateStatistics bindingStatistics{};
        detail::HandleTable<std::shared_ptr<MetalCommandPoolState>, CommandPoolHandle> commandPools;
        std::vector<std::weak_ptr<MetalCommandPoolState>> allCommandPools;
        std::size_t nextCommandPoolPruneThreshold = 64;
        detail::HandleTable<std::shared_ptr<MetalQueryPoolRecord>, QueryPoolHandle> queryPools;
        detail::HandleTable<std::shared_ptr<MetalSurfaceRecord>, SurfaceHandle> surfaces;
        detail::HandleTable<std::shared_ptr<MetalSwapchainRecord>, SwapchainHandle> swapchains;
        __strong id<MTLBuffer> mipmapEncodeOff;
        __strong id<MTLBuffer> mipmapEncodeOn;
        __strong id<MTLComputePipelineState> indirectClampCompute;
        __strong id<MTLRenderPipelineState> indirectClampRender;
        std::uint64_t nextAllocationId = 1;
        mutable std::vector<std::weak_ptr<MetalAllocationTelemetry>> allocations;
        std::size_t nextAllocationPruneThreshold = 64;
        std::uint64_t deviceToken = 1;
        mutable std::mutex lossMutex;
        std::shared_ptr<const Error> lossError{};
        std::shared_ptr<const DeviceFaultReport> lossReport{};
        std::atomic<DeviceOperationalState> operationalState{DeviceOperationalState::active};
        std::atomic<bool> lossInjected{false};
        std::atomic<bool> nativeLost{false};
#ifdef RACHIS_TESTING
        std::atomic<bool> failNextCommitFeedback{false};
        std::mutex feedbackGateMutex;
        std::shared_ptr<MetalCommitFeedbackGate> nextCommitFeedbackGate;
#endif
    };

    struct MetalFormatFootprint final
    {
        std::uint32_t blockWidth = 1;
        std::uint32_t blockHeight = 1;
        std::uint32_t bytesPerBlock = 0;
        bool compressed = false;
    };

    struct MetalTextureBox final
    {
        std::uint32_t mipLevel = 0;
        std::uint32_t arrayLayer = 0;
        TextureAspect aspect = TextureAspect::color;
        Origin3D origin{};
        Extent3D extent{};
    };

    struct MetalResolvedTransfer final
    {
        TextureTransferLayout layout{};
        MTLSize size{};
        MTLOrigin origin{};
        NSUInteger slice = 0;
        NSUInteger level = 0;
        MTLBlitOption options = MTLBlitOptionNone;
    };

    [[nodiscard]] constexpr bool metalCheckedAdd(std::uint64_t left,
                                                 std::uint64_t right,
                                                 std::uint64_t& result) noexcept
    {
        if (right > std::numeric_limits<std::uint64_t>::max() - left)
        {
            return false;
        }
        result = left + right;
        return true;
    }

    [[nodiscard]] constexpr bool metalCheckedMultiply(std::uint64_t left,
                                                      std::uint64_t right,
                                                      std::uint64_t& result) noexcept
    {
        if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)
        {
            return false;
        }
        result = left * right;
        return true;
    }

    [[nodiscard]] constexpr std::uint32_t metalMipExtent(std::uint32_t value, std::uint32_t mipLevel) noexcept
    {
        return std::max(1U, value >> mipLevel);
    }

    [[nodiscard]] constexpr std::uint32_t metalAlignUp(std::uint32_t value, std::uint32_t alignment) noexcept
    {
        if (alignment <= 1)
        {
            return value;
        }
        return (value + alignment - 1U) / alignment * alignment;
    }

    [[nodiscard]] constexpr MetalFormatFootprint metalFormatFootprint(Format format, TextureAspect aspect) noexcept
    {
        MetalFormatFootprint footprint{};
        switch (format)
        {
        case Format::r8Uint:
        case Format::r8Sint:
        case Format::r8Unorm:
            footprint.bytesPerBlock = 1;
            break;
        case Format::rg8Uint:
        case Format::rg8Sint:
        case Format::r16Uint:
        case Format::r16Sint:
        case Format::rg8Unorm:
        case Format::r16Float:
            footprint.bytesPerBlock = 2;
            break;
        case Format::rgba8Uint:
        case Format::rgba8Sint:
        case Format::rg16Uint:
        case Format::rg16Sint:
        case Format::rgba8Unorm:
        case Format::rgba8UnormSrgb:
        case Format::bgra8Unorm:
        case Format::bgra8UnormSrgb:
        case Format::r32Float:
        case Format::r32Uint:
        case Format::r32Sint:
        case Format::rg16Float:
            footprint.bytesPerBlock = 4;
            break;
        case Format::rgba16Uint:
        case Format::rgba16Sint:
        case Format::rg32Uint:
        case Format::rg32Sint:
        case Format::rgba16Float:
        case Format::rg32Float:
            footprint.bytesPerBlock = 8;
            break;
        case Format::rgba32Uint:
        case Format::rgba32Sint:
        case Format::rgba32Float:
            footprint.bytesPerBlock = 16;
            break;
        case Format::d32Float:
            footprint.bytesPerBlock = 4;
            break;
        case Format::d24UnormS8Uint:
            footprint.bytesPerBlock = aspect == TextureAspect::stencil ? 1 : 4;
            break;
        case Format::bc1RgbaUnorm:
        case Format::bc1RgbaUnormSrgb:
        case Format::bc4RUnorm:
        case Format::bc4RSnorm:
            footprint.blockWidth = 4;
            footprint.blockHeight = 4;
            footprint.bytesPerBlock = 8;
            footprint.compressed = true;
            break;
        case Format::bc2RgbaUnorm:
        case Format::bc2RgbaUnormSrgb:
        case Format::bc3RgbaUnorm:
        case Format::bc3RgbaUnormSrgb:
        case Format::bc5RgUnorm:
        case Format::bc5RgSnorm:
        case Format::bc6hRgbUfloat:
        case Format::bc6hRgbSfloat:
        case Format::bc7RgbaUnorm:
        case Format::bc7RgbaUnormSrgb:
            footprint.blockWidth = 4;
            footprint.blockHeight = 4;
            footprint.bytesPerBlock = 16;
            footprint.compressed = true;
            break;
        default:
            break;
        }
        return footprint;
    }

    [[nodiscard]] constexpr bool metalAspectMatchesFormat(Format format, TextureAspect aspect) noexcept
    {
        switch (aspect)
        {
        case TextureAspect::depth:
            return format == Format::d32Float || format == Format::d24UnormS8Uint;
        case TextureAspect::stencil:
            return format == Format::d24UnormS8Uint;
        case TextureAspect::color:
            return format != Format::d32Float && format != Format::d24UnormS8Uint;
        }
        return false;
    }

    [[nodiscard]] constexpr bool metalTextureBoxesOverlap(const MetalTextureBox& left,
                                                          const MetalTextureBox& right) noexcept
    {
        if (left.mipLevel != right.mipLevel || left.arrayLayer != right.arrayLayer || left.aspect != right.aspect)
        {
            return false;
        }
        return left.origin.x < right.origin.x + right.extent.width &&
               right.origin.x < left.origin.x + left.extent.width &&
               left.origin.y < right.origin.y + right.extent.height &&
               right.origin.y < left.origin.y + left.extent.height &&
               left.origin.z < right.origin.z + right.extent.depth &&
               right.origin.z < left.origin.z + left.extent.depth;
    }

    [[nodiscard]] constexpr bool metalBlockAligned(std::uint32_t origin,
                                                   std::uint32_t extent,
                                                   std::uint32_t mipSize,
                                                   std::uint32_t block) noexcept
    {
        if (block <= 1)
        {
            return true;
        }
        if (origin % block != 0)
        {
            return false;
        }
        return extent % block == 0 || origin + extent == mipSize;
    }

    [[nodiscard]] constexpr std::uint64_t metalAlignUp(std::uint64_t value, std::uint64_t alignment) noexcept
    {
        if (alignment <= 1)
        {
            return value;
        }
        return (value + alignment - 1U) / alignment * alignment;
    }

    [[nodiscard]] constexpr Extent3D metalSubresourceExtent(const TextureDesc& desc, std::uint32_t mipLevel) noexcept
    {
        return Extent3D{
            .width = metalMipExtent(desc.width, mipLevel),
            .height = desc.dimension == TextureDimension::texture1D ? 1U : metalMipExtent(desc.height, mipLevel),
            .depth =
                desc.dimension == TextureDimension::texture3D ? metalMipExtent(desc.depthOrArrayLayers, mipLevel) : 1U,
        };
    }

    [[nodiscard]] constexpr bool metalLayoutCompatible(TextureUsage usage, TextureLayout layout) noexcept
    {
        switch (layout)
        {
        case TextureLayout::common:
            return true;
        case TextureLayout::shaderResource:
            return hasAny(usage, TextureUsage::sampled);
        case TextureLayout::unorderedAccess:
            return hasAny(usage, TextureUsage::storage);
        case TextureLayout::renderTarget:
            return hasAny(usage, TextureUsage::renderTarget);
        case TextureLayout::copySource:
            return hasAny(usage, TextureUsage::copySource);
        case TextureLayout::copyDestination:
            return hasAny(usage, TextureUsage::copyDestination);
        case TextureLayout::resolveSource:
        case TextureLayout::resolveDestination:
            return hasAny(usage, TextureUsage::renderTarget | TextureUsage::depthStencil);
        default:
            return false;
        }
    }

    [[nodiscard]] constexpr bool isValidMetalQueueType(QueueType type) noexcept
    {
        switch (type)
        {
        case QueueType::graphics:
        case QueueType::compute:
        case QueueType::copy:
            return true;
        }
        return false;
    }

    [[nodiscard]] constexpr std::size_t metalQueueIndex(QueueType type) noexcept
    {
        switch (type)
        {
        case QueueType::graphics:
            return 0;
        case QueueType::compute:
            return 1;
        case QueueType::copy:
            return 2;
        }
        return 0;
    }

    [[nodiscard]] constexpr const char* metalQueueTimelineLabel(QueueType type) noexcept
    {
        switch (type)
        {
        case QueueType::graphics:
            return "RHI.Metal.Timeline.Graphics";
        case QueueType::compute:
            return "RHI.Metal.Timeline.Compute";
        case QueueType::copy:
            return "RHI.Metal.Timeline.Copy";
        }
        return "RHI.Metal.Timeline";
    }

    [[nodiscard]] constexpr bool hasUnreservedMetalTimelineSlot(std::uint64_t nextValue) noexcept
    {
        return nextValue != 0 && nextValue != std::numeric_limits<std::uint64_t>::max();
    }

    [[nodiscard]] constexpr bool isValidMetalCommandPoolFlags(CommandPoolFlag flags) noexcept
    {
        constexpr auto allowed = static_cast<std::uint32_t>(CommandPoolFlag::allowIndividualReset) |
                                 static_cast<std::uint32_t>(CommandPoolFlag::unretainedResources);
        return (static_cast<std::uint32_t>(flags) & ~allowed) == 0;
    }

    [[nodiscard]] constexpr bool metalFormatHasStencil(Format format) noexcept
    {
        return format == Format::d24UnormS8Uint;
    }

    [[nodiscard]] constexpr bool metalIsDynamicBuffer(BindingType type) noexcept
    {
        return type == BindingType::dynamicUniformBuffer || type == BindingType::dynamicReadOnlyStorageBuffer ||
               type == BindingType::dynamicStorageBuffer;
    }

    [[nodiscard]] constexpr bool metalIsBufferType(BindingType type) noexcept
    {
        return type == BindingType::uniformBuffer || type == BindingType::readOnlyStorageBuffer ||
               type == BindingType::storageBuffer || metalIsDynamicBuffer(type);
    }

    [[nodiscard]] constexpr std::uint32_t metalArgumentLimit(BindingType type) noexcept
    {
        if (metalIsBufferType(type))
        {
            return metalMaxBufferArguments;
        }
        return type == BindingType::sampler ? metalMaxSamplerArguments : metalMaxTextureArguments;
    }

    [[nodiscard]] constexpr BindingType metalStaticBufferType(BindingType type) noexcept
    {
        switch (type)
        {
        case BindingType::dynamicUniformBuffer:
            return BindingType::uniformBuffer;
        case BindingType::dynamicReadOnlyStorageBuffer:
            return BindingType::readOnlyStorageBuffer;
        case BindingType::dynamicStorageBuffer:
            return BindingType::storageBuffer;
        default:
            return type;
        }
    }

    [[nodiscard]] constexpr bool metalIsSamplerFeedback(BindingType type) noexcept
    {
        return type == BindingType::minMipFeedback || type == BindingType::mipRegionUsedFeedback;
    }

    [[nodiscard]] constexpr std::size_t metalRangeClassIndex(BindingType type) noexcept
    {
        if (metalIsBufferType(type))
        {
            return 0;
        }
        return type == BindingType::sampler ? 2 : 1;
    }

    [[nodiscard]] constexpr bool metalIsValidVisibility(ShaderStage visibility) noexcept
    {
        const auto bits = static_cast<std::uint16_t>(visibility);
        const auto allowed = static_cast<std::uint16_t>(ShaderStage::all);
        return bits != 0 && (bits & ~allowed) == 0;
    }

    [[nodiscard]] constexpr bool metalPushRangesOverlap(const PushConstantRange& left,
                                                        const PushConstantRange& right) noexcept
    {
        return left.offset < right.offset + right.size && right.offset < left.offset + left.size;
    }

    [[nodiscard]] constexpr bool metalPushConstantWriteCovered(std::span<const PushConstantRange> ranges,
                                                               ShaderStage visibility,
                                                               std::uint32_t offset,
                                                               std::size_t size) noexcept
    {
        const auto requiredMask = static_cast<std::uint16_t>(visibility);
        if (requiredMask == 0 || size == 0 || size > maxPushConstantBytes || offset > maxPushConstantBytes - size)
        {
            return false;
        }
        std::array<std::uint16_t, maxPushConstantBytes> coveredStages{};
        for (const auto& range : ranges)
        {
            const auto matchingStages =
                static_cast<std::uint16_t>(static_cast<std::uint16_t>(range.visibility) & requiredMask);
            if (matchingStages == 0)
            {
                continue;
            }
            const auto maxBytes = static_cast<std::size_t>(maxPushConstantBytes);
            const auto rangeBegin = (std::min)(static_cast<std::size_t>(range.offset), maxBytes);
            const auto rangeEnd = (std::min)(static_cast<std::size_t>(range.offset) + range.size, maxBytes);
            for (std::size_t byte = rangeBegin; byte < rangeEnd; ++byte)
            {
                coveredStages[byte] = static_cast<std::uint16_t>(coveredStages[byte] | matchingStages);
            }
        }
        for (std::size_t byte = offset; byte < offset + size; ++byte)
        {
            if ((coveredStages[byte] & requiredMask) != requiredMask)
            {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] constexpr std::uint32_t metalLogicalDescriptorCount(const BindingLayoutEntry& entry,
                                                                      std::uint32_t variableDescriptorCount) noexcept
    {
        return hasAny(entry.flags, BindingFlag::variableDescriptorCount) ? variableDescriptorCount : entry.count;
    }

    [[nodiscard]] Error metalSelectionError(ErrorCode code, std::string message);
    [[nodiscard]] Error metalProfileError(const MetalProfileDecision& decision);
    [[nodiscard]] AdapterInfo adapterFromDevice(id<MTLDevice> device);
    [[nodiscard]] bool formatIsBc(Format format) noexcept;
    [[nodiscard]] bool formatIsDepth(Format format) noexcept;
    [[nodiscard]] bool formatIsInteger(Format format) noexcept;
    [[nodiscard]] bool formatAllowsViewReinterpretation(Format format) noexcept;
    [[nodiscard]] MTLPixelFormat metalPixelFormat(Format format) noexcept;
    [[nodiscard]] NSString* metalNsString(std::string_view text) noexcept;
    [[nodiscard]] Error metalNsError(ErrorCode code,
                                     NSError* error,
                                     std::string_view operation,
                                     std::string_view encoderContext = {});
    [[nodiscard]] DeviceFaultStatus metalFaultStatusFromError(NSError* error) noexcept;
    void publishMetalDeviceLoss(MetalDeviceNative& native,
                                const Error& error,
                                DeviceFaultStatus status,
                                bool injected,
                                std::string_view queueLabel = {},
                                std::string_view commandLabel = {});
    [[nodiscard]] Error metalExceptionError(NSException* exception, std::string_view operation);
    [[nodiscard]] DiagnosticSeverity metalLogSeverity(MTLLogLevel level) noexcept;
    [[nodiscard]] std::string sanitizeCaptureToken(std::string_view text);

#ifdef RACHIS_TESTING
    struct MetalTestingState final
    {
        std::mutex mutex;
        std::optional<MetalHostReport> hostReportOverride;
        MetalDevice* boundDevice = nullptr;
        bool failNextCommandApiCreate = false;
        bool failNextLogStateCreate = false;
        bool failNextDepthStencilStateCreate = false;
        bool failNextDrawable = false;
        bool failNextResidencySetCreate = false;
        bool failNextPipelineCacheSerialize = false;
        std::string pipelineCacheTemporaryDirectory;
        std::atomic<std::uint64_t> residencyAddCount{0};
        std::atomic<std::uint64_t> residencyDeduplicatedSkipCount{0};
        std::atomic<std::uint64_t> residencyCommitCount{0};
        std::atomic<std::uint64_t> argumentTableCreateCount{0};
        std::atomic<std::uint64_t> argumentTablePoolReuseCount{0};
        std::atomic<std::uint64_t> argumentTableEncoderReuseCount{0};
        std::atomic<std::uint64_t> pushConstantPageAllocateCount{0};
        std::atomic<std::uint64_t> pushConstantSliceSuballocateCount{0};
        std::atomic<std::uint64_t> teardownQueueDrainCount{0};
        std::atomic<std::uint64_t> renderTimestampVertexStageCount{0};
        std::atomic<std::uint64_t> renderTimestampFragmentStageCount{0};
        std::shared_ptr<MetalCommitFeedbackGate> nextCommandPoolResetGate;
        std::shared_ptr<MetalCommitFeedbackGate> nextDrawableAcquireGate;
    };

    [[nodiscard]] MetalTestingState& metalTestingState() noexcept;
    void bindMetalDiagnosticTestDevice(MetalDevice* device) noexcept;
    void unbindMetalDiagnosticTestDevice(MetalDevice* device) noexcept;
    [[nodiscard]] Status consumeInjectedMetalPresentLoss(MetalDevice& device);
    void recordMetalResidencyAdd() noexcept;
    void recordMetalResidencySkip() noexcept;
    void recordMetalResidencyCommit() noexcept;
    void recordMetalArgumentTableCreate() noexcept;
    void recordMetalArgumentTablePoolReuse() noexcept;
    void recordMetalArgumentTableEncoderReuse() noexcept;
    void recordMetalPushConstantPageAllocate() noexcept;
    void recordMetalPushConstantSliceSuballocate() noexcept;
    void recordMetalTeardownQueueDrain() noexcept;
    void recordMetalRenderTimestampStage(MTLRenderStages stage) noexcept;
    void holdMetalCommandPoolResetForTesting() noexcept;
#endif

    [[nodiscard]] FormatProperties probeFormat(id<MTLDevice> device, Format format) noexcept;

    [[nodiscard]] Status validateMetalMarkerDesc(const DebugMarkerDesc& desc);
    void flushMetalComputePendingBarriers(MetalCommandEntry& entry) noexcept;
    void flushMetalRenderPendingBarriers(MetalCommandEntry& entry) noexcept;
    [[nodiscard]] Status flushMetalDeferredBarriers(MetalCommandEntry& entry);
    void endMetalEncoders(MetalCommandEntry& entry) noexcept;
    void endMetalRecording(MetalCommandEntry& entry) noexcept;
    [[nodiscard]] Status resetMetalRecording(MetalCommandEntry& entry);
    [[nodiscard]] Status beginMetalRecording(MetalCommandPoolState& pool, MetalCommandEntry& entry);
    [[nodiscard]] MetalCommandBuffer* asMetalCommand(detail::BackendCommandBuffer& commandBuffer,
                                                     std::uint64_t deviceToken) noexcept;
    [[nodiscard]] Status retainMetalApplicationResource(MetalCommandBuffer& command, std::shared_ptr<void> resource);
    [[nodiscard]] Status retainMetalUtilityResource(MetalCommandBuffer& command, std::shared_ptr<void> resource);
    [[nodiscard]] std::shared_ptr<MetalBindingSetCommandRetention> metalRetainBindingSetForCommand(
        const MetalBindingSetRecord& record);
    [[nodiscard]] Status retainMetalBindingSetRetentions(
        MetalCommandBuffer& command, std::span<std::shared_ptr<MetalBindingSetCommandRetention>> retentions);
    [[nodiscard]] Status metalRecordingReady(const MetalCommandBuffer& native);
    [[nodiscard]] Result<std::shared_ptr<MetalCommandPoolState>> metalRecordingPool(const MetalCommandBuffer& native);

    [[nodiscard]] std::shared_ptr<MetalTextureLayoutTable> metalTextureLayoutTable(MetalTextureRecord& record);
    [[nodiscard]] Status metalCheckLayoutTransitions(const MetalCommandLayoutState& state,
                                                     std::span<const MetalLayoutTransition> transitions);
    void metalStageLayoutTransitions(MetalCommandLayoutState& state,
                                     std::span<const MetalLayoutTransition> transitions);

    struct MetalBatchLayouts final
    {
        std::vector<MetalLayoutWrite> writes;
        std::unordered_map<MetalLayoutCell, std::size_t, MetalLayoutCellHash> writeIndex;
    };
    [[nodiscard]] Result<MetalBatchLayouts> metalPlanBatchLayouts(std::span<MetalCommandBuffer* const> commands);
    void metalPublishBatchLayouts(MetalBatchLayouts& layouts) noexcept;
    void metalRetainResidency(MetalCommandEntry& entry, id<MTLAllocation> allocation) noexcept;
    [[nodiscard]] MetalWaitedDrawable metalSwapchainImageReference(const MetalDeviceNative& native,
                                                                   const MetalTextureRecord& texture);
    [[nodiscard]] Status metalReserveWaitedDrawableTracking(MetalCommandEntry* entry,
                                                            std::span<const MetalWaitedDrawable> references);
    void metalMergeWaitedDrawables(MetalCommandEntry* entry, std::span<const MetalWaitedDrawable> references) noexcept;
    void commitMetalResidency(MetalCommandEntry& entry) noexcept;
    [[nodiscard]] Result<id<MTL4ArgumentTable>> acquireMetalArgumentTable(id<MTLDevice> device,
                                                                          MetalCommandEntry& entry,
                                                                          MetalArgumentTableStage stage,
                                                                          std::uint32_t maxBuffers,
                                                                          std::uint32_t maxTextures,
                                                                          std::uint32_t maxSamplers);
    [[nodiscard]] Result<MetalPushConstantSlice> metalCapturePushConstantSlice(id<MTLDevice> device,
                                                                               MetalCommandEntry& entry,
                                                                               std::span<const std::byte> sourceBytes,
                                                                               std::uint32_t copyBytes);
    void collectMetalQueue(MetalQueueTimeline& state) noexcept;
    void retireMetalPresents(MetalDeviceNative& native) noexcept;
    [[nodiscard]] Status checkMetalSubmission(const MetalDeviceNative& native);
    void publishMetalSubmissionFailure(MetalDeviceNative& native,
                                       NSError* error,
                                       std::string_view label,
                                       std::string_view queueLabel = {});
    [[nodiscard]] Result<TimelinePointStatus> queryMetalValue(const MetalDeviceNative& native,
                                                              MetalQueueTimeline& state,
                                                              std::uint64_t value);
    [[nodiscard]] Result<TimelinePointStatus> waitForMetalValue(const MetalDeviceNative& native,
                                                                MetalQueueTimeline& state,
                                                                std::uint64_t value,
                                                                std::uint64_t timeoutMs);
    [[nodiscard]] Status drainMetalQueue(const MetalDeviceNative& native, MetalQueueTimeline& state);

    [[nodiscard]] bool metalPlacementsAlias(const std::shared_ptr<MetalPlacedLease>& before,
                                            const std::shared_ptr<MetalPlacedLease>& after) noexcept;
    void pruneMetalAllocations(MetalDeviceNative& native) noexcept;
    [[nodiscard]] ResourceMemorySnapshot captureMetalAllocationSnapshot(MetalDeviceNative& native);
    [[nodiscard]] Status rejectMetalBufferWorkingSetOverflow(id<MTLDevice> device, std::uint64_t bytes);
    [[nodiscard]] Status rejectMetalTextureWorkingSetOverflow(id<MTLDevice> device, std::uint64_t bytes);
    [[nodiscard]] Status reserveMetalAllocationTelemetry(MetalDeviceNative& native);
    void publishMetalAllocation(MetalDeviceNative& native, const std::shared_ptr<MetalAllocationTelemetry>& allocation);

    [[nodiscard]] Result<MetalResolvedTransfer> resolveMetalTransfer(const TextureDesc& desc,
                                                                     const BufferTextureCopyRegion& region,
                                                                     std::uint32_t bufferOffsetAlignment,
                                                                     std::uint32_t rowPitchAlignment);
    [[nodiscard]] std::pair<std::uint32_t, std::uint32_t> metalTransferAlignments(id<MTLDevice> device,
                                                                                  Format format,
                                                                                  TextureAspect aspect) noexcept;

    [[nodiscard]] bool metalIsTimestampSlotPending(const MetalCommandBuffer& command,
                                                   QueryPoolHandle queryPool,
                                                   const MetalQueryPoolRecord& pool,
                                                   std::uint32_t queryIndex) noexcept;
    [[nodiscard]] Status metalReserveStagedQuerySlots(std::vector<MetalStagedQuerySlot>& staged,
                                                      QueryPoolHandle queryPool,
                                                      std::span<const std::uint32_t> queryIndices);
    void metalStageTimestampWrite(std::vector<MetalStagedQuerySlot>& staged,
                                  QueryPoolHandle queryPool,
                                  const std::shared_ptr<MetalQueryPoolRecord>& pool,
                                  std::uint32_t queryIndex) noexcept;
    [[nodiscard]] Status metalEncodeTimestamp(MetalCommandEntry& entry,
                                              const std::shared_ptr<MetalQueryPoolRecord>& pool,
                                              std::uint32_t queryIndex,
                                              MTLRenderStages renderStage);

    [[nodiscard]] Result<std::uint32_t> metalValidatePushConstantRanges(std::span<const PushConstantRange> ranges);
    [[nodiscard]] Status metalApplyDynamicOffsets(std::vector<std::shared_ptr<MetalBindingSetVersion>>& versions,
                                                  std::span<const std::uint32_t> offsets);

    struct MetalIndirectClampPipelines final
    {
        __strong id<MTLComputePipelineState> compute = nil;
        __strong id<MTLRenderPipelineState> render = nil;
    };

    struct MetalIndirectClampRequest final
    {
        std::uint64_t sourceAddress = 0;
        std::uint64_t countAddress = 0;
        std::uint32_t maxRecords = 0;
        std::uint32_t strideBytes = 0;
        std::uint32_t recordWords = 0;
        std::uint32_t zeroFirstWord = 0;
        std::uint32_t zeroWordCount = 0;
    };

    struct MetalClampedRecords final
    {
        std::uint64_t headerAddress = 0;
        std::uint64_t recordsAddress = 0;
        std::uint32_t recordBytes = 0;
    };

    [[nodiscard]] Result<MetalIndirectClampPipelines> metalIndirectClampPipelines(
        std::mutex& deviceMutex, const std::shared_ptr<MetalDeviceNative>& native);
    [[nodiscard]] Result<MetalClampedRecords> metalPrepareClampedRecords(id<MTLDevice> device,
                                                                         MetalCommandEntry& entry,
                                                                         const MetalIndirectClampRequest& request);
    [[nodiscard]] Status metalEncodeComputeRecordClamp(id<MTLDevice> device,
                                                       MetalCommandEntry& entry,
                                                       id<MTLComputePipelineState> pipeline,
                                                       const MetalClampedRecords& records,
                                                       std::uint32_t maxRecords);
    [[nodiscard]] Status metalEncodeRenderRecordClamp(id<MTLDevice> device,
                                                      MetalCommandEntry& entry,
                                                      id<MTLRenderPipelineState> pipeline,
                                                      const MetalClampedRecords& records,
                                                      std::uint32_t maxRecords);

} // namespace rachis::metal
