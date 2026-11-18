#pragma once

#include "D3D12/D3D12Resource.h"

#include "Common/HandleTable.h"
#include "Common/MonotonicIdentitySource.h"
#include "D3D12/D3D12Descriptor.h"
#include "D3D12/D3D12Format.h"
#include "D3D12/D3D12LegacyTextureStates.h"
#include "D3D12/D3D12NativeHelpers.h"
#include "D3D12/D3D12RuntimePolicies.h"
#include "D3D12/D3D12TestHooks.h"
#include "D3D12/D3D12UploadArena.h"

#include <wrl/client.h>

#include <D3D12MemAlloc.h>
#include <Windows.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <ranges>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rachis::d3d12
{

    using Microsoft::WRL::ComPtr;

    inline constexpr std::uint64_t sparseBufferTileSize = D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;

    enum class ResidencyPoolClass : std::uint8_t
    {
        deviceBuffer,
        uploadBuffer,
        readbackBuffer,
        nonAttachmentTexture,
        attachmentTexture,
        count,
    };

    inline constexpr std::size_t explicitResidencyPriorityCount = 4;

    inline constexpr std::size_t residencyPoolClassCount = static_cast<std::size_t>(ResidencyPoolClass::count);

    struct TrackedResourceMemory;

    struct ResourceMemoryTrackingState final
    {
        ResourceMemoryTrackingState() noexcept : allocationIdentities(test::consumeNextResourceMemoryIdentitySeed())
        {
        }

        std::mutex mutex;
        detail::MonotonicIdentitySource allocationIdentities;
        std::unordered_map<std::uint64_t, std::weak_ptr<TrackedResourceMemory>> entries;
    };

    struct TrackedResourceMemory final
    {
        TrackedResourceMemory() = default;
        TrackedResourceMemory(const TrackedResourceMemory&) = delete;
        TrackedResourceMemory& operator=(const TrackedResourceMemory&) = delete;
        TrackedResourceMemory(TrackedResourceMemory&&) = delete;
        TrackedResourceMemory& operator=(TrackedResourceMemory&&) = delete;

        ~TrackedResourceMemory()
        {
            const auto tracking = state.lock();
            if (tracking == nullptr)
            {
                return;
            }
            const std::scoped_lock lock(tracking->mutex);
            tracking->entries.erase(entry.allocationId);
        }

        std::weak_ptr<ResourceMemoryTrackingState> state;
        ResourceMemoryEntry entry;
    };

    struct PlacedResourceLease;

    struct SparseHeapTileLease;

    struct ResourceHeapRecord final
    {
        std::shared_ptr<TrackedResourceMemory> trackedMemory;
        ComPtr<ID3D12Heap> heap;
        std::shared_ptr<ResidencyRecord> residency;
        ResourceHeapDesc desc;
        std::string label;
        std::mutex mutex;
        std::vector<std::weak_ptr<PlacedResourceLease>> placements;
        std::vector<std::weak_ptr<SparseHeapTileLease>> sparseTiles;
        bool alive = true;
    };

    struct PlacedResourceLease final
    {
        std::shared_ptr<ResourceHeapRecord> heap;
        std::uint64_t offset = 0;
        std::uint64_t size = 0;
    };

    struct SparseHeapTileLease final
    {
        std::shared_ptr<ResourceHeapRecord> heap;
        SmallVector<std::weak_ptr<SparseHeapTileLease>, 2> ancestors;
        TimelinePoint reusableAfter;
        std::uint32_t firstTile = 0;
        std::uint32_t tileCount = 0;
    };

    struct SparseMappingRegion final
    {
        std::uint32_t firstTile = 0;
        std::uint32_t tileCount = 0;
        std::uint32_t heapFirstTile = 0;
        std::shared_ptr<SparseHeapTileLease> lease;
    };

    struct SparseMappingState final
    {
        std::vector<SparseMappingRegion> regions;
        TimelinePoint publication;
        bool publicationUnavailable = false;
        mutable std::atomic_uint32_t commandRetentions = 0;
    };

    struct SparseMappingController final
    {
        std::shared_ptr<const SparseMappingState> current;
    };

    struct SparseMappingCommandLease final
    {
        explicit SparseMappingCommandLease(std::shared_ptr<const SparseMappingState> mapping)
            : mapping_(std::move(mapping))
        {
            mapping_->commandRetentions.fetch_add(1, std::memory_order_acq_rel);
        }

        ~SparseMappingCommandLease()
        {
            mapping_->commandRetentions.fetch_sub(1, std::memory_order_acq_rel);
        }

        SparseMappingCommandLease(const SparseMappingCommandLease&) = delete;
        SparseMappingCommandLease& operator=(const SparseMappingCommandLease&) = delete;
        SparseMappingCommandLease(SparseMappingCommandLease&&) = delete;
        SparseMappingCommandLease& operator=(SparseMappingCommandLease&&) = delete;

        [[nodiscard]] const std::shared_ptr<const SparseMappingState>& mapping() const noexcept
        {
            return mapping_;
        }

    private:
        std::shared_ptr<const SparseMappingState> mapping_;
    };

    struct BufferRecord final
    {
        BufferRecord() = default;
        BufferRecord(const BufferRecord&) = delete;
        BufferRecord& operator=(const BufferRecord&) = delete;
        BufferRecord(BufferRecord&&) = delete;
        BufferRecord& operator=(BufferRecord&&) = delete;

        ComPtr<ID3D12Device> device;
        ComPtr<D3D12MA::Allocator> allocator;
        ComPtr<D3D12MA::Pool> residencyPool;
        std::shared_ptr<TrackedResourceMemory> trackedMemory;
        ComPtr<D3D12MA::Allocation> allocation;
        std::shared_ptr<PlacedResourceLease> placement;
        ComPtr<ID3D12Resource> resource;
        std::shared_ptr<ResidencyRecord> residency;
        BufferDesc desc;
        std::uint64_t nativeWidth = 0;
        std::string label;
        SparseBufferProperties sparseProperties;
        std::shared_ptr<const SparseMappingState> sparseMappings;
        std::shared_ptr<SparseMappingController> sparseState;
        std::mutex mapMutex;
        bool alive = true;
        bool shared = false;
        bool sparse = false;
        bool mapped = false;
        std::byte* mappedData = nullptr;
        std::uint64_t mappedOffset = 0;
        std::uint64_t mappedSize = 0;

        ~BufferRecord()
        {
            if (mapped && resource != nullptr)
            {
                resource->Unmap(0, nullptr);
            }
        }
    };

    struct TextureRecord;

    struct TextureRecord final
    {
        ComPtr<ID3D12Device> device;
        ComPtr<D3D12MA::Allocator> allocator;
        ComPtr<D3D12MA::Pool> residencyPool;
        std::shared_ptr<TrackedResourceMemory> trackedMemory;
        ComPtr<D3D12MA::Allocation> allocation;
        std::shared_ptr<PlacedResourceLease> placement;
        ComPtr<ID3D12Resource> resource;
        std::shared_ptr<ResidencyRecord> residency;
        std::shared_ptr<SwapchainImageLeaseState> swapchainLease;
        std::shared_ptr<LegacyTextureStateTracker> legacyStates;
        TextureDesc desc;
        std::string label;
        std::vector<Format> viewFormats;
        SparseTextureProperties sparseProperties;
        std::vector<D3D12_SUBRESOURCE_TILING> sparseSubresources;
        std::uint32_t sparseMipTailFirstTile = 0;
        std::shared_ptr<const SparseMappingState> sparseMappings;
        std::shared_ptr<SparseMappingController> sparseState;
        bool externallyOwned = false;
        bool shared = false;
        bool sparse = false;
    };

    [[nodiscard]] inline Status trackLegacyTextureStates(TextureRecord& texture, bool enhancedBarriers)
    {
        static_cast<void>(enhancedBarriers);
        const auto info = format::info(texture.desc.format);
        const std::uint32_t planeCount = info.has_value() ? info->planeCount : 1U;
        const std::uint32_t arraySize =
            texture.desc.dimension == TextureDimension::texture3D ? 1U : texture.desc.depthOrArrayLayers;
        try
        {
            texture.legacyStates = std::make_shared<LegacyTextureStateTracker>(
                texture.resource.Get(), texture.desc.mipLevels * arraySize * planeCount);
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to allocate legacy texture state tracking.",
            });
        }
        texture.legacyStates->setInitialLayout(texture.swapchainLease != nullptr ? TextureLayout::present
                                                                                 : TextureLayout::common);
        if (texture.shared || (texture.externallyOwned && texture.swapchainLease == nullptr))
        {
            texture.legacyStates->markLayoutUnknown();
        }
        return {};
    }

    struct SamplerFeedbackMapRecord final
    {
        ComPtr<ID3D12Device> device;
        ComPtr<D3D12MA::Allocator> allocator;
        std::shared_ptr<TrackedResourceMemory> trackedMemory;
        ComPtr<D3D12MA::Allocation> allocation;
        ComPtr<ID3D12Resource> resource;
        std::shared_ptr<TextureRecord> targetedTexture;
        SamplerFeedbackMapType type = SamplerFeedbackMapType::minMip;
        Extent3D mipRegion;
        Extent3D decodedExtent;
        Extent3D pairedExtent;
        std::uint32_t mipLevels = 1;
        std::string label;
    };

    struct AccelerationStructureStorageLease final
    {
        std::shared_ptr<BufferRecord> storage;
        std::uint64_t offset = 0;
        std::uint64_t size = 0;
    };

    struct AccelerationStructureVersion final
    {
        std::shared_ptr<BufferRecord> storage;
        std::shared_ptr<AccelerationStructureStorageLease> storageLease;
        std::vector<std::shared_ptr<void>> dependencies;
        std::vector<std::shared_ptr<AccelerationStructureExecutionTracker>> dependencyTrackers;
        std::vector<std::shared_ptr<ResidencyRecord>> residencies;
        std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geometryLayout;
        AccelerationStructureBuildFlag buildFlags = AccelerationStructureBuildFlag::none;
        std::uint32_t instanceCount = 0;
        std::uint64_t resultSize = 0;
        D3D12_GPU_VIRTUAL_ADDRESS address = 0;
    };

    struct AccelerationStructureRecord final
    {
        AccelerationStructureType type = AccelerationStructureType::bottomLevel;
        std::shared_ptr<BufferRecord> storage;
        std::uint64_t offset = 0;
        std::uint64_t size = 0;
        std::string label;
        std::shared_ptr<AccelerationStructureStorageLease> storageLease;
        std::shared_ptr<AccelerationStructureExecutionTracker> executionTracker =
            std::make_shared<AccelerationStructureExecutionTracker>();
        std::mutex mutex;
        std::shared_ptr<AccelerationStructureVersion> currentVersion;
    };

    struct PreparedBottomLevelInput final
    {
        std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geometries;
        std::vector<std::shared_ptr<BufferRecord>> owners;
        std::vector<std::shared_ptr<ResidencyRecord>> residencies;
        std::uint64_t primitiveCount = 0;
    };

    struct PreparedResourceBarrierBatch;

    struct NativeUploadSlab final
    {
        NativeUploadSlab() noexcept = default;
        NativeUploadSlab(const NativeUploadSlab&) = delete;
        NativeUploadSlab& operator=(const NativeUploadSlab&) = delete;

        NativeUploadSlab(NativeUploadSlab&& other) noexcept
            : device(std::move(other.device)), allocator(std::move(other.allocator)),
              allocation(std::move(other.allocation)), resource(std::move(other.resource)),
              mappedData(std::exchange(other.mappedData, nullptr)),
              gpuVirtualAddress(std::exchange(other.gpuVirtualAddress, 0))
        {
        }

        NativeUploadSlab& operator=(NativeUploadSlab&& other) noexcept
        {
            if (this != &other)
            {
                if (mappedData != nullptr && resource != nullptr)
                {
                    resource->Unmap(0, nullptr);
                }
                resource.Reset();
                allocation.Reset();
                allocator.Reset();
                device.Reset();

                device = std::move(other.device);
                allocator = std::move(other.allocator);
                allocation = std::move(other.allocation);
                resource = std::move(other.resource);
                mappedData = std::exchange(other.mappedData, nullptr);
                gpuVirtualAddress = std::exchange(other.gpuVirtualAddress, 0);
            }
            return *this;
        }

        ~NativeUploadSlab()
        {
            if (mappedData != nullptr && resource != nullptr)
            {
                resource->Unmap(0, nullptr);
            }
        }

        ComPtr<ID3D12Device> device;
        ComPtr<D3D12MA::Allocator> allocator;
        ComPtr<D3D12MA::Allocation> allocation;
        ComPtr<ID3D12Resource> resource;
        std::byte* mappedData = nullptr;
        D3D12_GPU_VIRTUAL_ADDRESS gpuVirtualAddress = 0;
    };

    struct UploadSliceLeaseOwner final
    {
        explicit UploadSliceLeaseOwner(UploadArena<NativeUploadSlab>::Lease arenaLease) noexcept
            : lease(std::move(arenaLease))
        {
        }

        UploadArena<NativeUploadSlab>::Lease lease;
    };

    struct UploadSlice final
    {
        std::shared_ptr<void> owner;
        ID3D12Resource* resource = nullptr;
        std::byte* mappedData = nullptr;
        D3D12_GPU_VIRTUAL_ADDRESS gpuVirtualAddress = 0;
        std::uint64_t offset = 0;
        std::uint64_t size = 0;
    };

    struct NativeIndirectArgumentSlab final
    {
        ComPtr<ID3D12Device> device;
        ComPtr<D3D12MA::Allocator> allocator;
        ComPtr<D3D12MA::Allocation> allocation;
        ComPtr<ID3D12Resource> resource;
        std::uint64_t size = 0;
    };

    struct IndirectArgumentSliceLeaseOwner final
    {
        explicit IndirectArgumentSliceLeaseOwner(UploadArena<NativeIndirectArgumentSlab>::Lease arenaLease) noexcept
            : lease(std::move(arenaLease))
        {
        }

        UploadArena<NativeIndirectArgumentSlab>::Lease lease;
    };

    struct IndirectArgumentSlice final
    {
        std::shared_ptr<void> owner;
        ID3D12Resource* resource = nullptr;
        std::uint64_t resourceSize = 0;
        std::uint64_t offset = 0;
    };

    inline constexpr std::size_t indirectArgumentArenaQueueCount = 3;
    inline constexpr std::size_t indirectArgumentArenaMaxFreeSlabs = 2;

    namespace resource_detail
    {

        [[nodiscard]] inline bool validRange(std::uint64_t resourceSize,
                                             std::uint64_t offset,
                                             std::uint64_t size) noexcept
        {
            return offset < resourceSize && size != 0 && (size == wholeResource || size <= resourceSize - offset);
        }

        [[nodiscard]] inline std::uint64_t resolveSize(std::uint64_t resourceSize,
                                                       std::uint64_t offset,
                                                       std::uint64_t size) noexcept
        {
            return size == wholeResource ? resourceSize - offset : size;
        }

        [[nodiscard]] constexpr bool validResidencyPriority(ResidencyPriority priority) noexcept
        {
            switch (priority)
            {
            case ResidencyPriority::automatic:
            case ResidencyPriority::minimum:
            case ResidencyPriority::low:
            case ResidencyPriority::normal:
            case ResidencyPriority::high:
                return true;
            }
            return false;
        }

        [[nodiscard]] constexpr bool validAllocationFlags(ResourceAllocationFlag flags) noexcept
        {
            constexpr auto allowed = static_cast<std::uint32_t>(ResourceAllocationFlag::manualResidency);
            return (static_cast<std::uint32_t>(flags) & ~allowed) == 0;
        }

        [[nodiscard]] constexpr bool validTextureDimension(TextureDimension dimension) noexcept
        {
            switch (dimension)
            {
            case TextureDimension::texture1D:
            case TextureDimension::texture2D:
            case TextureDimension::texture3D:
                return true;
            }
            return false;
        }

        [[nodiscard]] constexpr bool validTextureUsage(TextureUsage usage) noexcept
        {
            constexpr auto allowed = static_cast<std::uint32_t>(TextureUsage::copySource) |
                                     static_cast<std::uint32_t>(TextureUsage::copyDestination) |
                                     static_cast<std::uint32_t>(TextureUsage::sampled) |
                                     static_cast<std::uint32_t>(TextureUsage::storage) |
                                     static_cast<std::uint32_t>(TextureUsage::renderTarget) |
                                     static_cast<std::uint32_t>(TextureUsage::depthStencil) |
                                     static_cast<std::uint32_t>(TextureUsage::shadingRateSource);
            const auto raw = static_cast<std::uint32_t>(usage);
            return raw != 0 && (raw & ~allowed) == 0;
        }

        [[nodiscard]] inline D3D12_RESOURCE_FLAGS bufferFlags(BufferUsage usage) noexcept
        {
            return hasAny(usage,
                          BufferUsage::storage | BufferUsage::accelerationStructureStorage |
                              BufferUsage::accelerationStructureScratch)
                       ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                       : D3D12_RESOURCE_FLAG_NONE;
        }

        [[nodiscard]] constexpr std::uint64_t nativeBufferWidth(const BufferDesc& desc) noexcept
        {
            return hasAny(desc.usage, BufferUsage::constant) ? constantBufferViewSize(desc.size) : desc.size;
        }

        [[nodiscard]] inline D3D12_RESOURCE_DESC nativeBufferDesc(const BufferDesc& desc) noexcept
        {
            return {
                .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
                .Alignment = 0,
                .Width = nativeBufferWidth(desc),
                .Height = 1,
                .DepthOrArraySize = 1,
                .MipLevels = 1,
                .Format = DXGI_FORMAT_UNKNOWN,
                .SampleDesc = {.Count = 1, .Quality = 0},
                .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                .Flags = bufferFlags(desc.usage),
            };
        }

        [[nodiscard]] constexpr DXGI_FORMAT nativeResourceFormat(const TextureDesc& desc,
                                                                 const format::Info& info) noexcept
        {
            return desc.viewFormats.empty() ? info.resource : format::mutableResource(desc.format);
        }

        [[nodiscard]] constexpr D3D12_RESOURCE_DIMENSION nativeDimension(TextureDimension dimension) noexcept
        {
            switch (dimension)
            {
            case TextureDimension::texture1D:
                return D3D12_RESOURCE_DIMENSION_TEXTURE1D;
            case TextureDimension::texture3D:
                return D3D12_RESOURCE_DIMENSION_TEXTURE3D;
            default:
                return D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            }
        }

        [[nodiscard]] constexpr D3D12_RESOURCE_FLAGS textureFlags(TextureUsage usage) noexcept
        {
            D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE;
            if (hasAny(usage, TextureUsage::storage))
            {
                flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            }
            if (hasAny(usage, TextureUsage::renderTarget))
            {
                flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            }
            if (hasAny(usage, TextureUsage::depthStencil))
            {
                flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
            }
            return flags;
        }

        [[nodiscard]] inline D3D12_RESOURCE_DESC nativeTextureDesc(const TextureDesc& desc,
                                                                   const format::Info& info) noexcept
        {
            return {
                .Dimension = nativeDimension(desc.dimension),
                .Alignment = 0,
                .Width = desc.width,
                .Height = desc.height,
                .DepthOrArraySize = static_cast<std::uint16_t>(desc.depthOrArrayLayers),
                .MipLevels = static_cast<std::uint16_t>(desc.mipLevels),
                .Format = nativeResourceFormat(desc, info),
                .SampleDesc = {.Count = desc.sampleCount, .Quality = 0},
                .Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
                .Flags = textureFlags(desc.usage),
            };
        }

        [[nodiscard]] constexpr std::uint32_t fullMipCount(const TextureDesc& desc) noexcept
        {
            std::uint32_t largest = std::max(desc.width, desc.height);
            if (desc.dimension == TextureDimension::texture3D)
            {
                largest = std::max(largest, desc.depthOrArrayLayers);
            }
            std::uint32_t count = 0;
            while (largest != 0)
            {
                ++count;
                largest >>= 1U;
            }
            return count;
        }

        [[nodiscard]] constexpr bool isDepthFormat(const format::Info& info) noexcept
        {
            return info.depth;
        }

        [[nodiscard]] constexpr Format linearStorageViewFormat(Format format) noexcept
        {
            switch (format)
            {
            case Format::rgba8UnormSrgb:
                return Format::rgba8Unorm;
            case Format::bgra8UnormSrgb:
                return Format::bgra8Unorm;
            default:
                return Format::unknown;
            }
        }

        [[nodiscard]] constexpr std::optional<std::uint32_t> aspectPlane(const format::Info& info,
                                                                         TextureAspect aspect) noexcept
        {
            switch (aspect)
            {
            case TextureAspect::color:
                return info.depth ? std::nullopt : std::optional<std::uint32_t>{0};
            case TextureAspect::depth:
                return info.depth ? std::optional<std::uint32_t>{0} : std::nullopt;
            case TextureAspect::stencil:
                return info.stencil ? std::optional<std::uint32_t>{1} : std::nullopt;
            }
            return std::nullopt;
        }

        [[nodiscard]] Result<std::uint64_t> reserveResourceMemoryIdentity(
            const std::shared_ptr<ResourceMemoryTrackingState>& state) noexcept;

        [[nodiscard]] Result<std::shared_ptr<TrackedResourceMemory>> trackResourceMemory(
            const std::shared_ptr<ResourceMemoryTrackingState>& state,
            std::uint64_t identity,
            ResourceMemoryKind kind,
            MemoryType memoryType,
            std::uint64_t allocationBytes,
            bool dedicatedAllocation,
            bool shared,
            std::string_view label);

        [[nodiscard]] Result<std::shared_ptr<TrackedResourceMemory>> trackResourceMemory(
            const std::shared_ptr<ResourceMemoryTrackingState>& state,
            std::uint64_t identity,
            ResourceMemoryKind kind,
            MemoryType memoryType,
            D3D12MA::Allocation& allocation,
            bool shared,
            std::string_view label);

        void markResourceMemoryDestroyed(const std::shared_ptr<TrackedResourceMemory>& tracked);

        using SparseMappingCommandLeases = SmallVector<std::shared_ptr<SparseMappingCommandLease>, 2>;
        using SparseMappingDependencies = SmallVector<TimelinePoint, 2>;

        [[nodiscard]] Status captureSparseBufferMapping(const std::shared_ptr<BufferRecord>& record,
                                                        SparseMappingCommandLeases& leases,
                                                        SparseMappingDependencies& dependencies);

        [[nodiscard]] Status retainResourceCommandDependencies(
            QueueManager& queues,
            detail::BackendCommandBuffer& commandBuffer,
            std::shared_ptr<void> applicationOwner,
            std::span<const std::shared_ptr<ResidencyRecord>> residencies,
            std::span<const std::shared_ptr<SparseMappingCommandLease>> sparseMappings,
            std::span<const TimelinePoint> sparseMappingDependencies,
            std::span<const std::shared_ptr<SwapchainImageLeaseState>> swapchainLeases,
            std::shared_ptr<void> backendOwner = nullptr);

        [[nodiscard]] Status captureSparseTextureMapping(const std::shared_ptr<TextureRecord>& record,
                                                         SparseMappingCommandLeases& leases,
                                                         SparseMappingDependencies& dependencies);

        [[nodiscard]] bool sparseTextureRegionIsMapped(const TextureRecord& record,
                                                       std::uint32_t subresource,
                                                       Origin3D origin,
                                                       Extent3D extent) noexcept;

        struct TextureLayoutUse final
        {
            LegacyTextureStateRecording* recording = nullptr;
            std::size_t use = 0;
            std::vector<std::uint32_t> subresources;
            TextureLayout layout = TextureLayout::undefined;
        };

        [[nodiscard]] Result<TextureLayoutUse> prepareTextureLayoutUse(QueueManager& queues,
                                                                       detail::BackendCommandBuffer& commandBuffer,
                                                                       const TextureRecord& texture,
                                                                       const TextureSubresourceRange& subresources,
                                                                       TextureLayout layout,
                                                                       std::string_view operation);

        void commitTextureLayoutUse(const TextureLayoutUse& use) noexcept;

        [[nodiscard]] Status requireFormatSupport(ID3D12Device& device,
                                                  DXGI_FORMAT format,
                                                  D3D12_FORMAT_SUPPORT1 requiredSupport1,
                                                  D3D12_FORMAT_SUPPORT2 requiredSupport2,
                                                  std::string_view purpose);

        [[nodiscard]] DXGI_FORMAT storageValidationFormat(const TextureDesc& desc, const format::Info& info) noexcept;

        [[nodiscard]] Status validateTextureDesc(ID3D12Device& device,
                                                 const TextureDesc& desc,
                                                 const format::Info& info,
                                                 bool variableRateShadingTier2);

        [[nodiscard]] Status validateTextureAllocationDesc(ID3D12Device& device,
                                                           const TextureDesc& desc,
                                                           const format::Info& info,
                                                           bool variableRateShadingTier2,
                                                           bool residencyPrioritySupported,
                                                           bool manualResidencySupported,
                                                           bool shared);

    } // namespace resource_detail

    struct ResourceManager::Impl final
    {
        ID3D12Device* device = nullptr;
        ComPtr<ID3D12Device1> residencyDevice;
        ComPtr<ID3D12Device3> residencyControlDevice;
        ComPtr<ID3D12Device5> rayTracingDevice;
        ComPtr<ID3D12Device8> samplerFeedbackDevice;
        ComPtr<D3D12MA::Allocator> allocator;
        std::mutex residencyPoolsMutex;
        std::array<std::array<ComPtr<D3D12MA::Pool>, explicitResidencyPriorityCount>, residencyPoolClassCount>
            residencyPools;
        bool enhancedBarriers = false;
        bool variableRateShadingTier2 = false;
        bool rayTracing = false;
        bool sparseResources = false;
        bool samplerFeedback = false;
        ComPtr<ID3D12RootSignature> depthResolveRootSignature;
        ComPtr<ID3D12PipelineState> depthResolvePipeline;
        CpuDescriptorPool depthResolveDescriptors{D3D12_DESCRIPTOR_HEAP_TYPE_DSV};
        mutable std::mutex resourceHeapsMutex;
        detail::HandleTable<std::shared_ptr<ResourceHeapRecord>, ResourceHeapHandle> resourceHeaps;
        mutable std::mutex buffersMutex;
        detail::HandleTable<std::shared_ptr<BufferRecord>, BufferHandle> buffers;
        std::mutex sparseMappingsMutex;
        std::atomic<std::size_t> activeSparseResourceCount{0};
        mutable std::mutex texturesMutex;
        detail::HandleTable<std::shared_ptr<TextureRecord>, TextureHandle> textures;
        mutable std::mutex samplerFeedbackMapsMutex;
        detail::HandleTable<std::shared_ptr<SamplerFeedbackMapRecord>, SamplerFeedbackMapHandle> samplerFeedbackMaps;
        std::shared_ptr<ResourceMemoryTrackingState> memoryTracking = std::make_shared<ResourceMemoryTrackingState>();
        mutable std::mutex accelerationStructuresMutex;
        detail::HandleTable<std::shared_ptr<AccelerationStructureRecord>, AccelerationStructureHandle>
            accelerationStructures;
        std::mutex accelerationStructureStorageLeasesMutex;
        std::vector<std::weak_ptr<AccelerationStructureStorageLease>> accelerationStructureStorageLeases;
        std::shared_ptr<UploadArena<NativeUploadSlab>> uploadArena = UploadArena<NativeUploadSlab>::create();
        std::array<std::shared_ptr<UploadArena<NativeIndirectArgumentSlab>>, indirectArgumentArenaQueueCount>
            indirectArgumentArenas{
                UploadArena<NativeIndirectArgumentSlab>::create(defaultUploadArenaSlabBytes,
                                                                indirectArgumentArenaMaxFreeSlabs),
                UploadArena<NativeIndirectArgumentSlab>::create(defaultUploadArenaSlabBytes,
                                                                indirectArgumentArenaMaxFreeSlabs),
                UploadArena<NativeIndirectArgumentSlab>::create(defaultUploadArenaSlabBytes,
                                                                indirectArgumentArenaMaxFreeSlabs),
            };

        [[nodiscard]] Result<std::shared_ptr<ResourceHeapRecord>> resourceHeap(ResourceHeapHandle handle) const;
        [[nodiscard]] Result<D3D12MA::Pool*> residencyPool(ResidencyPoolClass poolClass, ResidencyPriority priority);
        [[nodiscard]] Result<std::shared_ptr<BufferRecord>> buffer(BufferHandle handle) const;
        [[nodiscard]] Result<PreparedBottomLevelInput> prepareBottomLevel(
            const BottomLevelAccelerationStructureBuildInput& input) const;
        [[nodiscard]] Result<std::shared_ptr<TextureRecord>> texture(TextureHandle handle) const;
        [[nodiscard]] Result<std::shared_ptr<SamplerFeedbackMapRecord>> samplerFeedbackMap(
            SamplerFeedbackMapHandle handle) const;
        [[nodiscard]] Result<std::shared_ptr<AccelerationStructureRecord>> accelerationStructure(
            AccelerationStructureHandle handle) const;
        [[nodiscard]] Status prepareBufferBarriers(PreparedResourceBarrierBatch& prepared,
                                                   std::span<const BufferBarrier> barriers,
                                                   QueueType queueType,
                                                   std::unique_lock<std::mutex>& sparseMappingLock) const;
        [[nodiscard]] Status prepareTextureBarriers(PreparedResourceBarrierBatch& prepared,
                                                    std::span<const TextureBarrier> barriers,
                                                    QueueType queueType,
                                                    std::unique_lock<std::mutex>& sparseMappingLock) const;
        [[nodiscard]] Status prepareSamplerFeedbackBarriers(PreparedResourceBarrierBatch& prepared,
                                                            std::span<const SamplerFeedbackBarrier> barriers,
                                                            QueueType queueType) const;
        [[nodiscard]] Result<std::shared_ptr<PreparedResourceBarrierBatch>> prepareResourceBarrierBatch(
            const ResourceBarrierBatch& barriers,
            QueueType queueType,
            std::unique_lock<std::mutex>& sparseMappingLock) const;
        [[nodiscard]] Result<UploadSlice> allocateUploadSlice(std::uint64_t size,
                                                              std::uint64_t alignment,
                                                              std::string_view operation);
        [[nodiscard]] Result<IndirectArgumentSlice> allocateIndirectArgumentSlice(QueueType queueType,
                                                                                  std::uint64_t size,
                                                                                  std::uint64_t alignment,
                                                                                  std::string_view operation);
        [[nodiscard]] Status allocateBuffer(BufferRecord& record, D3D12MA::Pool* residencyPool);
        [[nodiscard]] Status allocateTexture(TextureRecord& record,
                                             const format::Info& info,
                                             D3D12MA::Pool* residencyPool);
    };

} // namespace rachis::d3d12
