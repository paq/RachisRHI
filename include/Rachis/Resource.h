#pragma once

#include "Rachis/Handle.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace rachis
{

    inline constexpr std::uint64_t wholeResource = std::numeric_limits<std::uint64_t>::max();
    inline constexpr std::uint32_t remainingSubresources = std::numeric_limits<std::uint32_t>::max();

    enum class MemoryType : std::uint8_t
    {
        deviceLocal,

        upload,

        readback,
    };

    enum class ResidencyPriority : std::uint8_t
    {
        automatic,
        minimum,
        low,
        normal,
        high,
    };

    enum class ResourceAllocationFlag : std::uint32_t
    {
        none = 0,

        manualResidency = 1U << 0U,
    };

    [[nodiscard]] constexpr ResourceAllocationFlag operator|(ResourceAllocationFlag left,
                                                             ResourceAllocationFlag right) noexcept
    {
        return static_cast<ResourceAllocationFlag>(static_cast<std::uint32_t>(left) |
                                                   static_cast<std::uint32_t>(right));
    }

    [[nodiscard]] constexpr bool hasAny(ResourceAllocationFlag value, ResourceAllocationFlag flags) noexcept
    {
        return (static_cast<std::uint32_t>(value) & static_cast<std::uint32_t>(flags)) != 0;
    }

    enum class ResourceMemoryKind : std::uint8_t
    {
        buffer,
        texture,
        resourceHeap,
    };

    enum class ResourceHeapClass : std::uint8_t
    {
        buffer,
        nonAttachmentTexture,
        attachmentTexture,
    };

    enum class ResourceHeapFlag : std::uint32_t
    {
        none = 0,

        allowAliasing = 1U << 0U,

        sparseBacking = 1U << 1U,
    };

    [[nodiscard]] constexpr ResourceHeapFlag operator|(ResourceHeapFlag left, ResourceHeapFlag right) noexcept
    {
        return static_cast<ResourceHeapFlag>(static_cast<std::uint32_t>(left) | static_cast<std::uint32_t>(right));
    }

    [[nodiscard]] constexpr bool hasAny(ResourceHeapFlag value, ResourceHeapFlag flags) noexcept
    {
        return (static_cast<std::uint32_t>(value) & static_cast<std::uint32_t>(flags)) != 0;
    }

    struct ResourceMemoryRequirements final
    {
        std::uint64_t size = 0;
        std::uint64_t alignment = 0;
        bool dedicatedAllocation = false;
    };

    struct ResourceHeapDesc final
    {
        std::uint64_t size = 0;
        std::uint64_t alignment = 0;
        MemoryType memoryType = MemoryType::deviceLocal;
        ResourceHeapClass resourceClass = ResourceHeapClass::buffer;
        ResourceHeapFlag flags = ResourceHeapFlag::none;
        ResidencyPriority residencyPriority = ResidencyPriority::automatic;
        ResourceAllocationFlag allocationFlags = ResourceAllocationFlag::none;
        std::string_view label = {};
    };

    struct ResourcePlacement final
    {
        ResourceHeapHandle heap = {};
        std::uint64_t offset = 0;
    };

    struct ResourceMemoryEntry final
    {
        std::uint64_t allocationId = 0;
        ResourceMemoryKind kind = ResourceMemoryKind::buffer;
        MemoryType memoryType = MemoryType::deviceLocal;
        std::uint64_t allocationBytes = 0;
        bool logicalAlive = false;
        bool dedicatedAllocation = false;
        bool shared = false;
        std::string label = {};
    };

    struct ResourceMemorySnapshot final
    {
        std::vector<ResourceMemoryEntry> entries = {};
        std::uint64_t liveAllocationBytes = 0;
        std::uint64_t retainedAllocationBytes = 0;
        std::uint64_t liveAllocationCount = 0;
        std::uint64_t retainedAllocationCount = 0;
    };

    enum class ResidencyState : std::uint8_t
    {
        automatic,
        resident,
        evicted,
        pending,
    };

    enum class ResidencyBudgetPolicy : std::uint8_t
    {
        allowOverBudget,
        denyOverBudget,
    };

    struct ResidencySet final
    {
        std::span<const ResourceHeapHandle> heaps = {};
        std::span<const BufferHandle> buffers = {};
        std::span<const TextureHandle> textures = {};
    };

    enum class BufferUsage : std::uint32_t
    {
        none = 0,
        copySource = 1U << 0U,
        copyDestination = 1U << 1U,
        vertex = 1U << 2U,
        index = 1U << 3U,
        constant = 1U << 4U,
        storage = 1U << 5U,
        indirect = 1U << 6U,
        predicate = 1U << 7U,

        accelerationStructureStorage = 1U << 8U,

        accelerationStructureBuildInput = 1U << 9U,

        accelerationStructureScratch = 1U << 10U,

        shaderBindingTable = 1U << 11U,

        streamOutput = 1U << 12U,
    };

    enum class IndexFormat : std::uint8_t
    {
        uint16,
        uint32,
    };

    enum class Format : std::uint8_t
    {
        unknown,
        r8Unorm,
        rg8Unorm,
        rgba8Unorm,
        rgba8UnormSrgb,
        bgra8Unorm,
        bgra8UnormSrgb,
        r16Float,
        rg16Float,
        rgba16Float,
        r32Float,
        rg32Float,
        rgba32Float,
        d32Float,
        d24UnormS8Uint,
        bc1RgbaUnorm,
        bc1RgbaUnormSrgb,
        bc3RgbaUnorm,
        bc3RgbaUnormSrgb,
        bc5RgUnorm,
        bc7RgbaUnorm,
        bc7RgbaUnormSrgb,
        r32Uint,
        r8Uint,
        rg8Uint,
        rgba8Uint,
        r8Sint,
        rg8Sint,
        rgba8Sint,
        r16Uint,
        rg16Uint,
        rgba16Uint,
        r16Sint,
        rg16Sint,
        rgba16Sint,
        rg32Uint,
        rgba32Uint,
        r32Sint,
        rg32Sint,
        rgba32Sint,
        r64Uint,
        bc2RgbaUnorm,
        bc2RgbaUnormSrgb,
        bc4RUnorm,
        bc4RSnorm,
        bc5RgSnorm,
        bc6hRgbUfloat,
        bc6hRgbSfloat,
    };

    [[nodiscard]] constexpr bool isTextureViewFormatCompatible(Format textureFormat, Format viewFormat) noexcept
    {
        if (textureFormat == Format::unknown || viewFormat == Format::unknown)
        {
            return false;
        }
        if (textureFormat == viewFormat)
        {
            return true;
        }
        switch (textureFormat)
        {
        case Format::rgba8Unorm:
            return viewFormat == Format::rgba8UnormSrgb;
        case Format::rgba8UnormSrgb:
            return viewFormat == Format::rgba8Unorm;
        case Format::bgra8Unorm:
            return viewFormat == Format::bgra8UnormSrgb;
        case Format::bgra8UnormSrgb:
            return viewFormat == Format::bgra8Unorm;
        case Format::bc1RgbaUnorm:
            return viewFormat == Format::bc1RgbaUnormSrgb;
        case Format::bc1RgbaUnormSrgb:
            return viewFormat == Format::bc1RgbaUnorm;
        case Format::bc2RgbaUnorm:
            return viewFormat == Format::bc2RgbaUnormSrgb;
        case Format::bc2RgbaUnormSrgb:
            return viewFormat == Format::bc2RgbaUnorm;
        case Format::bc3RgbaUnorm:
            return viewFormat == Format::bc3RgbaUnormSrgb;
        case Format::bc3RgbaUnormSrgb:
            return viewFormat == Format::bc3RgbaUnorm;
        case Format::bc7RgbaUnorm:
            return viewFormat == Format::bc7RgbaUnormSrgb;
        case Format::bc7RgbaUnormSrgb:
            return viewFormat == Format::bc7RgbaUnorm;
        default:
            return false;
        }
    }

    enum class FormatFeature : std::uint32_t
    {
        none = 0,
        texture1D = 1U << 0U,
        texture2D = 1U << 1U,
        texture3D = 1U << 2U,
        cube = 1U << 3U,
        sampled = 1U << 4U,
        filterable = 1U << 5U,
        storageRead = 1U << 6U,
        storageWrite = 1U << 7U,
        colorAttachment = 1U << 8U,
        depthStencilAttachment = 1U << 9U,
        blendable = 1U << 10U,
        copySource = 1U << 11U,
        copyDestination = 1U << 12U,
        multisampleResolve = 1U << 13U,
        viewFormatReinterpretation = 1U << 14U,
        logicOperation = 1U << 15U,
    };

    [[nodiscard]] constexpr FormatFeature operator|(FormatFeature left, FormatFeature right) noexcept
    {
        return static_cast<FormatFeature>(static_cast<std::underlying_type_t<FormatFeature>>(left) |
                                          static_cast<std::underlying_type_t<FormatFeature>>(right));
    }

    [[nodiscard]] constexpr FormatFeature operator&(FormatFeature left, FormatFeature right) noexcept
    {
        return static_cast<FormatFeature>(static_cast<std::underlying_type_t<FormatFeature>>(left) &
                                          static_cast<std::underlying_type_t<FormatFeature>>(right));
    }

    constexpr FormatFeature& operator|=(FormatFeature& left, FormatFeature right) noexcept
    {
        left = left | right;
        return left;
    }

    [[nodiscard]] constexpr bool hasAny(FormatFeature value, FormatFeature flags) noexcept
    {
        return (value & flags) != FormatFeature::none;
    }

    [[nodiscard]] constexpr bool hasAll(FormatFeature value, FormatFeature flags) noexcept
    {
        return (value & flags) == flags;
    }

    enum class SampleCountMask : std::uint8_t
    {
        none = 0,
        x1 = 1U << 0U,
        x2 = 1U << 1U,
        x4 = 1U << 2U,
        x8 = 1U << 3U,
    };

    enum class ResolveMode : std::uint8_t
    {
        average,
        minimum,
        maximum,
    };

    enum class ResolveModeMask : std::uint8_t
    {
        none = 0,
        average = 1U << 0U,
        minimum = 1U << 1U,
        maximum = 1U << 2U,
    };

    [[nodiscard]] constexpr ResolveModeMask operator|(ResolveModeMask left, ResolveModeMask right) noexcept
    {
        return static_cast<ResolveModeMask>(static_cast<std::underlying_type_t<ResolveModeMask>>(left) |
                                            static_cast<std::underlying_type_t<ResolveModeMask>>(right));
    }

    constexpr ResolveModeMask& operator|=(ResolveModeMask& left, ResolveModeMask right) noexcept
    {
        left = left | right;
        return left;
    }

    [[nodiscard]] constexpr bool supportsResolveMode(ResolveModeMask mask, ResolveMode mode) noexcept
    {
        const auto required = [&]() {
            switch (mode)
            {
            case ResolveMode::average:
                return ResolveModeMask::average;
            case ResolveMode::minimum:
                return ResolveModeMask::minimum;
            case ResolveMode::maximum:
                return ResolveModeMask::maximum;
            }
            return ResolveModeMask::none;
        }();
        return (static_cast<std::uint8_t>(mask) & static_cast<std::uint8_t>(required)) != 0;
    }

    [[nodiscard]] constexpr SampleCountMask operator|(SampleCountMask left, SampleCountMask right) noexcept
    {
        return static_cast<SampleCountMask>(static_cast<std::underlying_type_t<SampleCountMask>>(left) |
                                            static_cast<std::underlying_type_t<SampleCountMask>>(right));
    }

    constexpr SampleCountMask& operator|=(SampleCountMask& left, SampleCountMask right) noexcept
    {
        left = left | right;
        return left;
    }

    [[nodiscard]] constexpr bool supportsSampleCount(SampleCountMask mask, std::uint32_t sampleCount) noexcept
    {
        switch (sampleCount)
        {
        case 1:
            return (static_cast<std::uint8_t>(mask) & static_cast<std::uint8_t>(SampleCountMask::x1)) != 0;
        case 2:
            return (static_cast<std::uint8_t>(mask) & static_cast<std::uint8_t>(SampleCountMask::x2)) != 0;
        case 4:
            return (static_cast<std::uint8_t>(mask) & static_cast<std::uint8_t>(SampleCountMask::x4)) != 0;
        case 8:
            return (static_cast<std::uint8_t>(mask) & static_cast<std::uint8_t>(SampleCountMask::x8)) != 0;
        default:
            return false;
        }
    }

    struct FormatProperties final
    {
        FormatFeature features = FormatFeature::none;
        SampleCountMask sampleCounts = SampleCountMask::none;

        ResolveModeMask colorResolveModes = ResolveModeMask::none;
        ResolveModeMask depthResolveModes = ResolveModeMask::none;
        ResolveModeMask stencilResolveModes = ResolveModeMask::none;
    };

    enum class TextureDimension : std::uint8_t
    {
        texture1D,
        texture2D,
        texture3D,
    };

    enum class TextureUsage : std::uint32_t
    {
        none = 0,
        copySource = 1U << 0U,
        copyDestination = 1U << 1U,
        sampled = 1U << 2U,
        storage = 1U << 3U,
        renderTarget = 1U << 4U,
        depthStencil = 1U << 5U,
        shadingRateSource = 1U << 6U,
    };

    [[nodiscard]] constexpr TextureUsage operator|(TextureUsage left, TextureUsage right) noexcept
    {
        return static_cast<TextureUsage>(static_cast<std::underlying_type_t<TextureUsage>>(left) |
                                         static_cast<std::underlying_type_t<TextureUsage>>(right));
    }

    [[nodiscard]] constexpr TextureUsage operator&(TextureUsage left, TextureUsage right) noexcept
    {
        return static_cast<TextureUsage>(static_cast<std::underlying_type_t<TextureUsage>>(left) &
                                         static_cast<std::underlying_type_t<TextureUsage>>(right));
    }

    [[nodiscard]] constexpr bool hasAny(TextureUsage value, TextureUsage flags) noexcept
    {
        return (value & flags) != TextureUsage::none;
    }

    enum class OptimizedClearValueKind : std::uint8_t
    {
        automatic,
        color,
        depthStencil,
    };

    struct TextureOptimizedClearValue final
    {
        OptimizedClearValueKind kind = OptimizedClearValueKind::automatic;
        std::array<float, 4> color{};
        float depth = 1.0F;
        std::uint32_t stencil = 0;
    };

    struct TextureDesc final
    {
        TextureDimension dimension = TextureDimension::texture2D;
        std::uint32_t width = 1;
        std::uint32_t height = 1;
        std::uint32_t depthOrArrayLayers = 1;
        std::uint32_t mipLevels = 1;
        std::uint32_t sampleCount = 1;
        Format format = Format::unknown;
        TextureUsage usage = TextureUsage::none;

        std::span<const Format> viewFormats = {};
        ResidencyPriority residencyPriority = ResidencyPriority::automatic;
        ResourceAllocationFlag allocationFlags = ResourceAllocationFlag::none;
        TextureOptimizedClearValue optimizedClearValue = {};
        std::string_view label = {};
    };

    struct Origin3D final
    {
        std::uint32_t x = 0;
        std::uint32_t y = 0;
        std::uint32_t z = 0;
    };

    struct Extent3D final
    {
        std::uint32_t width = 1;
        std::uint32_t height = 1;
        std::uint32_t depth = 1;
    };

    enum class TextureAspect : std::uint8_t
    {
        color,
        depth,
        stencil,
    };

    struct TextureSubresource final
    {
        std::uint32_t mipLevel = 0;
        std::uint32_t arrayLayer = 0;
        TextureAspect aspect = TextureAspect::color;
    };

    struct BufferTextureCopyRegion final
    {
        std::uint64_t bufferOffset = 0;

        std::uint32_t bytesPerRow = 0;

        std::uint32_t rowsPerImage = 0;
        TextureSubresource textureSubresource = {};
        Origin3D textureOrigin = {};
        Extent3D extent = {};
    };

    struct TextureTransferLayout final
    {
        std::uint32_t bytesPerRow = 0;
        std::uint32_t rowsPerImage = 0;
        std::uint32_t rowCount = 0;
        std::uint64_t rowDataSize = 0;
        std::uint64_t bytesPerImage = 0;
        std::uint64_t requiredBufferSize = 0;
        std::uint32_t bufferOffsetAlignment = 1;
        std::uint32_t rowPitchAlignment = 1;
    };

    struct TextureCopyRegion final
    {
        TextureSubresource sourceSubresource = {};
        Origin3D sourceOrigin = {};
        TextureSubresource destinationSubresource = {};
        Origin3D destinationOrigin = {};
        Extent3D extent = {};
    };

    struct BufferTextureCopy final
    {
        BufferHandle buffer = {};
        std::uint64_t bufferOffset = 0;

        std::uint32_t bytesPerRow = 0;

        std::uint32_t rowsPerImage = 0;
        TextureHandle texture = {};
        TextureSubresource textureSubresource = {};
        Origin3D textureOrigin = {};
        Extent3D extent = {};
    };

    struct TextureCopy final
    {
        TextureHandle source = {};
        TextureSubresource sourceSubresource = {};
        Origin3D sourceOrigin = {};
        TextureHandle destination = {};
        TextureSubresource destinationSubresource = {};
        Origin3D destinationOrigin = {};
        Extent3D extent = {};
    };

    struct TextureResolve final
    {
        TextureHandle source = {};
        TextureSubresource sourceSubresource = {};
        TextureHandle destination = {};
        TextureSubresource destinationSubresource = {};
        ResolveMode mode = ResolveMode::average;
    };

    [[nodiscard]] constexpr BufferUsage operator|(BufferUsage left, BufferUsage right) noexcept
    {
        return static_cast<BufferUsage>(static_cast<std::underlying_type_t<BufferUsage>>(left) |
                                        static_cast<std::underlying_type_t<BufferUsage>>(right));
    }

    [[nodiscard]] constexpr BufferUsage operator&(BufferUsage left, BufferUsage right) noexcept
    {
        return static_cast<BufferUsage>(static_cast<std::underlying_type_t<BufferUsage>>(left) &
                                        static_cast<std::underlying_type_t<BufferUsage>>(right));
    }

    [[nodiscard]] constexpr bool hasAny(BufferUsage value, BufferUsage flags) noexcept
    {
        return (value & flags) != BufferUsage::none;
    }

    struct BufferDesc final
    {
        std::uint64_t size = 0;
        BufferUsage usage = BufferUsage::none;
        MemoryType memoryType = MemoryType::deviceLocal;
        ResidencyPriority residencyPriority = ResidencyPriority::automatic;
        ResourceAllocationFlag allocationFlags = ResourceAllocationFlag::none;
        std::string_view label = {};
    };

    struct SparseBufferProperties final
    {
        std::uint64_t tileSize = 0;
        std::uint32_t tileCount = 0;
    };

    struct SparseBufferTileMapping final
    {
        BufferHandle buffer = {};
        std::uint32_t bufferTileOffset = 0;
        std::uint32_t tileCount = 0;
        ResourceHeapHandle heap = {};
        std::uint32_t heapTileOffset = 0;
    };

    struct SparseTextureProperties final
    {
        std::uint64_t tileSize = 0;
        std::uint32_t totalTileCount = 0;
        std::uint32_t standardMipCount = 0;
        std::uint32_t packedMipCount = 0;
        std::uint32_t mipTailTileCount = 0;
        Extent3D standardTileExtent = {};
        std::vector<Extent3D> standardMipTileGrids = {};
    };

    struct SparseTextureTileMapping final
    {
        TextureHandle texture = {};
        TextureSubresource subresource = {};
        Origin3D textureTileOrigin = {};
        Extent3D textureTileExtent = {};
        ResourceHeapHandle heap = {};
        std::uint32_t heapTileOffset = 0;
    };

    struct SparseTextureMipTailMapping final
    {
        TextureHandle texture = {};
        std::uint32_t arrayLayer = 0;
        ResourceHeapHandle heap = {};
        std::uint32_t heapTileOffset = 0;
    };

    struct SparseTextureMappingBatch final
    {
        std::span<const SparseTextureTileMapping> standardTiles = {};
        std::span<const SparseTextureMipTailMapping> mipTails = {};
    };

    struct MinMipFeedbackMapDesc final
    {
        TextureHandle texture = {};
        Extent3D mipRegion{4, 4, 1};
        std::string_view label = {};
    };

    struct MinMipFeedbackMapProperties final
    {
        Extent3D mipRegion = {};
        Extent3D decodedExtent = {};
    };

    struct MinMipFeedbackDecode final
    {
        SamplerFeedbackMapHandle source = {};
        TextureHandle destination = {};
    };

    struct MinMipFeedbackEncode final
    {
        TextureHandle source = {};
        SamplerFeedbackMapHandle destination = {};
    };

    struct MipRegionUsedFeedbackMapDesc final
    {
        TextureHandle texture = {};
        Extent3D mipRegion{4, 4, 1};
        std::string_view label = {};
    };

    struct MipRegionUsedFeedbackMapProperties final
    {
        Extent3D mipRegion = {};

        Extent3D decodedExtent = {};

        Extent3D pairedExtent = {};
        std::uint32_t mipLevels = 1;
    };

    struct MipRegionUsedFeedbackDecode final
    {
        SamplerFeedbackMapHandle source = {};
        TextureHandle destination = {};
    };

    struct MipRegionUsedFeedbackEncode final
    {
        TextureHandle source = {};
        SamplerFeedbackMapHandle destination = {};
    };

    struct MappedMemory final
    {
        std::byte* data = nullptr;
        std::uint64_t size = 0;
    };

    struct MappedMemoryRange final
    {
        std::uint64_t offset = 0;
        std::uint64_t size = wholeResource;
    };

    struct BufferCopyRegion final
    {
        std::uint64_t sourceOffset = 0;
        std::uint64_t destinationOffset = 0;
        std::uint64_t size = 0;
    };

    struct BufferCopy final
    {
        BufferHandle source = {};
        std::uint64_t sourceOffset = 0;
        BufferHandle destination = {};
        std::uint64_t destinationOffset = 0;
        std::uint64_t size = 0;
    };

    enum class PipelineStage : std::uint32_t
    {
        none = 0,
        all = 1U << 0U,
        draw = 1U << 1U,
        vertexInput = 1U << 2U,
        vertexShader = 1U << 3U,
        pixelShader = 1U << 4U,
        computeShader = 1U << 5U,
        copy = 1U << 6U,
        resolve = 1U << 7U,
        renderTarget = 1U << 8U,
        depthStencil = 1U << 9U,
        indirect = 1U << 10U,
        geometryShader = 1U << 11U,
        hullShader = 1U << 12U,
        domainShader = 1U << 13U,
        taskShader = 1U << 14U,
        meshShader = 1U << 15U,
        conditionalExecution = 1U << 16U,
        accelerationStructureBuild = 1U << 17U,
        rayTracingShader = 1U << 18U,
        accelerationStructureCopy = 1U << 19U,
        streamOutput = 1U << 20U,
    };

    [[nodiscard]] constexpr PipelineStage operator|(PipelineStage left, PipelineStage right) noexcept
    {
        return static_cast<PipelineStage>(static_cast<std::underlying_type_t<PipelineStage>>(left) |
                                          static_cast<std::underlying_type_t<PipelineStage>>(right));
    }

    [[nodiscard]] constexpr PipelineStage operator&(PipelineStage left, PipelineStage right) noexcept
    {
        return static_cast<PipelineStage>(static_cast<std::underlying_type_t<PipelineStage>>(left) &
                                          static_cast<std::underlying_type_t<PipelineStage>>(right));
    }

    [[nodiscard]] constexpr bool hasAny(PipelineStage value, PipelineStage flags) noexcept
    {
        return (value & flags) != PipelineStage::none;
    }

    enum class Access : std::uint32_t
    {
        none = 0,
        common = 1U << 0U,
        vertexBuffer = 1U << 1U,
        constantBuffer = 1U << 2U,
        indexBuffer = 1U << 3U,
        shaderResource = 1U << 4U,
        unorderedAccess = 1U << 5U,
        copySource = 1U << 6U,
        copyDestination = 1U << 7U,
        indirectArgument = 1U << 8U,
        renderTarget = 1U << 9U,
        depthStencilRead = 1U << 10U,
        depthStencilWrite = 1U << 11U,
        resolveSource = 1U << 12U,
        resolveDestination = 1U << 13U,
        predicate = 1U << 14U,
        shadingRateSource = 1U << 15U,
        accelerationStructureBuildInput = 1U << 16U,
        accelerationStructureScratch = 1U << 17U,
        streamOutputWrite = 1U << 18U,
    };

    [[nodiscard]] constexpr Access operator|(Access left, Access right) noexcept
    {
        return static_cast<Access>(static_cast<std::underlying_type_t<Access>>(left) |
                                   static_cast<std::underlying_type_t<Access>>(right));
    }

    [[nodiscard]] constexpr Access operator&(Access left, Access right) noexcept
    {
        return static_cast<Access>(static_cast<std::underlying_type_t<Access>>(left) &
                                   static_cast<std::underlying_type_t<Access>>(right));
    }

    [[nodiscard]] constexpr bool hasAny(Access value, Access flags) noexcept
    {
        return (value & flags) != Access::none;
    }

    struct GlobalBarrier final
    {
        PipelineStage stagesBefore = PipelineStage::none;
        PipelineStage stagesAfter = PipelineStage::none;
        Access accessBefore = Access::none;
        Access accessAfter = Access::none;
    };

    enum class OwnershipTransfer : std::uint8_t
    {
        none,
        releaseExternal,
        acquireExternal,
    };

    struct BufferBarrier final
    {
        BufferHandle buffer = {};
        PipelineStage stagesBefore = PipelineStage::none;
        PipelineStage stagesAfter = PipelineStage::none;
        Access accessBefore = Access::none;
        Access accessAfter = Access::none;

        std::uint64_t offset = 0;
        std::uint64_t size = wholeResource;
        OwnershipTransfer ownership = OwnershipTransfer::none;
    };

    struct BufferAliasActivation final
    {
        BufferHandle before = {};
        BufferHandle after = {};
        PipelineStage stagesBefore = PipelineStage::none;
        PipelineStage stagesAfter = PipelineStage::none;
        Access accessBefore = Access::none;
        Access accessAfter = Access::none;
    };

    enum class TextureLayout : std::uint8_t
    {
        undefined,
        common,
        present,
        shaderResource,
        unorderedAccess,
        renderTarget,
        depthStencilWrite,
        depthStencilRead,
        copySource,
        copyDestination,
        resolveSource,
        resolveDestination,
        shadingRateSource,
    };

    struct TextureSubresourceRange final
    {
        std::uint32_t firstMipLevel = 0;
        std::uint32_t mipLevelCount = remainingSubresources;
        std::uint32_t firstArrayLayer = 0;
        std::uint32_t arrayLayerCount = remainingSubresources;
        TextureAspect aspect = TextureAspect::color;
    };

    struct TextureBarrier final
    {
        TextureHandle texture = {};
        PipelineStage stagesBefore = PipelineStage::none;
        PipelineStage stagesAfter = PipelineStage::none;
        Access accessBefore = Access::none;
        Access accessAfter = Access::none;
        TextureLayout layoutBefore = TextureLayout::undefined;
        TextureLayout layoutAfter = TextureLayout::undefined;
        TextureSubresourceRange subresources = {};
        OwnershipTransfer ownership = OwnershipTransfer::none;
    };

    struct SamplerFeedbackBarrier final
    {
        SamplerFeedbackMapHandle feedbackMap = {};
        PipelineStage stagesBefore = PipelineStage::none;
        PipelineStage stagesAfter = PipelineStage::none;
        Access accessBefore = Access::none;
        Access accessAfter = Access::none;
        TextureLayout layoutBefore = TextureLayout::undefined;
        TextureLayout layoutAfter = TextureLayout::undefined;
    };

    struct TextureAliasActivation final
    {
        TextureHandle before = {};
        TextureHandle after = {};
        PipelineStage stagesBefore = PipelineStage::none;
        PipelineStage stagesAfter = PipelineStage::none;
        Access accessBefore = Access::none;
        Access accessAfter = Access::none;
        TextureLayout layoutBefore = TextureLayout::undefined;
        TextureLayout layoutAfter = TextureLayout::undefined;
    };

    struct ResourceBarrierBatch final
    {
        std::span<const GlobalBarrier> globalBarriers = {};
        std::span<const BufferBarrier> bufferBarriers = {};
        std::span<const TextureBarrier> textureBarriers = {};
        std::span<const SamplerFeedbackBarrier> samplerFeedbackBarriers = {};
    };

    struct TextureMipmapGeneration final
    {
        TextureHandle texture = {};
        TextureSubresourceRange subresources = {};
        TextureLayout baseLayout = TextureLayout::undefined;
        TextureLayout finalLayout = TextureLayout::shaderResource;
    };

    enum class TextureViewUsage : std::uint8_t
    {
        sampled,
        storage,
        renderTarget,
        depthStencil,
    };

    enum class TextureViewDimension : std::uint8_t
    {
        texture1D,
        texture1DArray,
        texture2D,
        texture2DArray,
        texture3D,
        cube,
        cubeArray,
    };

    enum class ComponentSwizzle : std::uint8_t
    {
        identity,
        zero,
        one,
        red,
        green,
        blue,
        alpha,
    };

    struct ComponentMapping final
    {
        ComponentSwizzle red = ComponentSwizzle::identity;
        ComponentSwizzle green = ComponentSwizzle::identity;
        ComponentSwizzle blue = ComponentSwizzle::identity;
        ComponentSwizzle alpha = ComponentSwizzle::identity;

        friend constexpr bool operator==(const ComponentMapping&, const ComponentMapping&) noexcept = default;
    };

    inline constexpr ComponentMapping identityComponentMapping{};

    struct TextureViewDesc final
    {
        TextureHandle texture = {};
        TextureViewUsage usage = TextureViewUsage::sampled;
        TextureViewDimension dimension = TextureViewDimension::texture2D;

        Format format = Format::unknown;
        TextureAspect aspect = TextureAspect::color;
        std::uint32_t firstMipLevel = 0;
        std::uint32_t mipLevelCount = remainingSubresources;
        std::uint32_t firstArrayLayer = 0;
        std::uint32_t arrayLayerCount = remainingSubresources;

        ComponentMapping components = {};
        std::string_view label = {};
    };

} // namespace rachis
