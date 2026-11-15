#include "Metal/MetalInternal.h"

namespace rachis::metal
{
    namespace
    {

        [[nodiscard]] bool validMetalMemoryType(rachis::MemoryType type) noexcept
        {
            return type == rachis::MemoryType::deviceLocal || type == rachis::MemoryType::upload ||
                   type == rachis::MemoryType::readback;
        }

        [[nodiscard]] rachis::Status validateMetalBufferDesc(const rachis::BufferDesc& desc)
        {
            using namespace rachis;
            if (desc.size == 0 || desc.usage == BufferUsage::none || !validMetalMemoryType(desc.memoryType) ||
                desc.residencyPriority != ResidencyPriority::automatic)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "A buffer requires a non-zero size, at least one usage flag, and valid memory/residency values."));
            }
            if (hasAny(desc.allocationFlags, ResourceAllocationFlag::manualResidency))
            {
                return std::unexpected(metalSelectionError(ErrorCode::unsupported,
                                                           "DeviceDesc manual residency is not supported on Metal."));
            }
            if (desc.label.size() > maxDebugLabelBytes || desc.label.contains('\0'))
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, "A buffer label must be a portable UTF-8 string."));
            }
            constexpr auto followUp = BufferUsage::accelerationStructureStorage |
                                      BufferUsage::accelerationStructureBuildInput |
                                      BufferUsage::accelerationStructureScratch | BufferUsage::shaderBindingTable |
                                      BufferUsage::streamOutput | BufferUsage::predicate;
            if (hasAny(desc.usage, followUp))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::unsupported, "The requested buffer usage is not in the macOS Metal core set."));
            }
            return {};
        }

        [[nodiscard]] MTLResourceOptions metalBufferOptions(rachis::MemoryType memoryType) noexcept
        {
            const auto storage = memoryType == rachis::MemoryType::deviceLocal ? MTLResourceStorageModePrivate
                                                                               : MTLResourceStorageModeShared;

            return storage | MTLResourceCPUCacheModeDefaultCache | MTLResourceHazardTrackingModeTracked;
        }

        [[nodiscard]] bool metalBufferHostVisible(rachis::MemoryType memoryType) noexcept
        {
            return memoryType != rachis::MemoryType::deviceLocal;
        }

        [[nodiscard]] constexpr bool validMetalHostRange(std::uint64_t resourceSize,
                                                         std::uint64_t offset,
                                                         std::uint64_t size) noexcept
        {
            return offset < resourceSize && size != 0 && size <= resourceSize - offset;
        }

        [[nodiscard]] constexpr bool validMetalTextureUsage(rachis::TextureUsage usage) noexcept
        {
            constexpr auto allowed = static_cast<std::uint32_t>(rachis::TextureUsage::copySource) |
                                     static_cast<std::uint32_t>(rachis::TextureUsage::copyDestination) |
                                     static_cast<std::uint32_t>(rachis::TextureUsage::sampled) |
                                     static_cast<std::uint32_t>(rachis::TextureUsage::storage) |
                                     static_cast<std::uint32_t>(rachis::TextureUsage::renderTarget) |
                                     static_cast<std::uint32_t>(rachis::TextureUsage::depthStencil) |
                                     static_cast<std::uint32_t>(rachis::TextureUsage::shadingRateSource);
            const auto raw = static_cast<std::uint32_t>(usage);
            return raw != 0 && (raw & ~allowed) == 0;
        }

        [[nodiscard]] std::uint32_t metalFullMipCount(const rachis::TextureDesc& desc) noexcept
        {
            auto extent = std::max(desc.width, desc.height);
            if (desc.dimension == rachis::TextureDimension::texture3D)
            {
                extent = std::max(extent, desc.depthOrArrayLayers);
            }
            std::uint32_t levels = 1;
            while (extent > 1)
            {
                extent /= 2;
                ++levels;
            }
            return levels;
        }

        [[nodiscard]] constexpr bool validMetalComponentSwizzle(rachis::ComponentSwizzle swizzle) noexcept
        {
            switch (swizzle)
            {
            case rachis::ComponentSwizzle::identity:
            case rachis::ComponentSwizzle::zero:
            case rachis::ComponentSwizzle::one:
            case rachis::ComponentSwizzle::red:
            case rachis::ComponentSwizzle::green:
            case rachis::ComponentSwizzle::blue:
            case rachis::ComponentSwizzle::alpha:
                return true;
            }
            return false;
        }

        [[nodiscard]] constexpr bool validMetalComponentMapping(const rachis::ComponentMapping& mapping) noexcept
        {
            return validMetalComponentSwizzle(mapping.red) && validMetalComponentSwizzle(mapping.green) &&
                   validMetalComponentSwizzle(mapping.blue) && validMetalComponentSwizzle(mapping.alpha);
        }

        [[nodiscard]] constexpr bool metalViewDimensionAllowed(rachis::TextureDimension textureDimension,
                                                               rachis::TextureViewDimension viewDimension) noexcept
        {
            switch (textureDimension)
            {
            case rachis::TextureDimension::texture1D:
                return viewDimension == rachis::TextureViewDimension::texture1D ||
                       viewDimension == rachis::TextureViewDimension::texture1DArray;
            case rachis::TextureDimension::texture2D:
                return viewDimension == rachis::TextureViewDimension::texture2D ||
                       viewDimension == rachis::TextureViewDimension::texture2DArray ||
                       viewDimension == rachis::TextureViewDimension::cube ||
                       viewDimension == rachis::TextureViewDimension::cubeArray;
            case rachis::TextureDimension::texture3D:
                return viewDimension == rachis::TextureViewDimension::texture3D;
            }
            return false;
        }

        [[nodiscard]] rachis::FormatProperties metalFormatAt(std::span<const rachis::FormatProperties> formats,
                                                             rachis::Format format) noexcept
        {
            const auto index = static_cast<std::size_t>(format);
            if (index >= formats.size())
            {
                return {};
            }
            return formats[index];
        }

        [[nodiscard]] bool metalCubeCompatible(const rachis::TextureDesc& desc) noexcept
        {
            return desc.dimension == rachis::TextureDimension::texture2D && desc.width == desc.height &&
                   desc.sampleCount == 1 && desc.depthOrArrayLayers >= 6 && desc.depthOrArrayLayers % 6 == 0 &&
                   !formatIsDepth(desc.format);
        }

        [[nodiscard]] MTLTextureType metalTextureType(const rachis::TextureDesc& desc) noexcept
        {
            switch (desc.dimension)
            {
            case rachis::TextureDimension::texture1D:
                return desc.depthOrArrayLayers > 1 ? MTLTextureType1DArray : MTLTextureType1D;
            case rachis::TextureDimension::texture3D:
                return MTLTextureType3D;
            case rachis::TextureDimension::texture2D:
                if (desc.sampleCount > 1)
                {
                    return desc.depthOrArrayLayers > 1 ? MTLTextureType2DMultisampleArray : MTLTextureType2DMultisample;
                }
                if (metalCubeCompatible(desc))
                {
                    return desc.depthOrArrayLayers == 6 ? MTLTextureTypeCube : MTLTextureTypeCubeArray;
                }
                return desc.depthOrArrayLayers > 1 ? MTLTextureType2DArray : MTLTextureType2D;
            }
            return MTLTextureType2D;
        }

        [[nodiscard]] MTLTextureUsage metalTextureUsage(rachis::TextureUsage usage, bool pixelFormatView) noexcept
        {
            MTLTextureUsage native = MTLTextureUsageUnknown;
            if (rachis::hasAny(usage,
                               rachis::TextureUsage::sampled | rachis::TextureUsage::copySource |
                                   rachis::TextureUsage::copyDestination | rachis::TextureUsage::storage))
            {
                native |= MTLTextureUsageShaderRead;
            }
            if (rachis::hasAny(usage, rachis::TextureUsage::storage))
            {
                native |= MTLTextureUsageShaderWrite;
            }
            if (rachis::hasAny(usage, rachis::TextureUsage::renderTarget | rachis::TextureUsage::depthStencil))
            {
                native |= MTLTextureUsageRenderTarget;
            }
            if (pixelFormatView)
            {
                native |= MTLTextureUsagePixelFormatView;
            }
            return native;
        }

        [[nodiscard]] MTLTextureDescriptor* metalTextureDescriptor(const rachis::TextureDesc& desc) noexcept
        {
            MTLTextureDescriptor* const descriptor = [MTLTextureDescriptor new];
            const auto type = metalTextureType(desc);
            descriptor.textureType = type;
            descriptor.pixelFormat = metalPixelFormat(desc.format);
            descriptor.width = desc.width;
            descriptor.height = desc.dimension == rachis::TextureDimension::texture1D ? 1 : desc.height;
            descriptor.depth = desc.dimension == rachis::TextureDimension::texture3D ? desc.depthOrArrayLayers : 1;
            descriptor.mipmapLevelCount = desc.mipLevels;
            descriptor.sampleCount = desc.sampleCount;

            descriptor.storageMode = MTLStorageModePrivate;
            descriptor.cpuCacheMode = MTLCPUCacheModeDefaultCache;
            descriptor.hazardTrackingMode = MTLHazardTrackingModeTracked;
            descriptor.usage = metalTextureUsage(desc.usage, !desc.viewFormats.empty());
            if (type == MTLTextureTypeCubeArray)
            {
                descriptor.arrayLength = desc.depthOrArrayLayers / 6;
            }
            else if (type == MTLTextureType1DArray || type == MTLTextureType2DArray ||
                     type == MTLTextureType2DMultisampleArray)
            {
                descriptor.arrayLength = desc.depthOrArrayLayers;
            }
            else
            {
                descriptor.arrayLength = 1;
            }
            return descriptor;
        }

        [[nodiscard]] rachis::Status validateMetalTextureDescFields(const rachis::TextureDesc& desc)
        {
            using namespace rachis;
            if (desc.width == 0 || desc.height == 0 || desc.depthOrArrayLayers == 0 || desc.mipLevels == 0 ||
                !validMetalTextureUsage(desc.usage) || metalPixelFormat(desc.format) == MTLPixelFormatInvalid)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "A texture requires a valid dimension, non-zero extents and mip count, plus valid usage flags."));
            }
            if (desc.sampleCount != 1 && desc.sampleCount != 2 && desc.sampleCount != 4 && desc.sampleCount != 8)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, "A texture sample count must be 1, 2, 4, or 8."));
            }
            if (desc.residencyPriority != ResidencyPriority::automatic)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "Metal textures require automatic residency priority."));
            }
            if (hasAny(desc.allocationFlags, ResourceAllocationFlag::manualResidency))
            {
                return std::unexpected(metalSelectionError(ErrorCode::unsupported,
                                                           "DeviceDesc manual residency is not supported on Metal."));
            }
            if (desc.label.size() > maxDebugLabelBytes || desc.label.contains('\0'))
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "A texture label must be a portable UTF-8 string."));
            }
            return {};
        }

        [[nodiscard]] rachis::Status validateMetalTextureUsageAndShape(const rachis::TextureDesc& desc)
        {
            using namespace rachis;
            if (hasAny(desc.usage, TextureUsage::shadingRateSource))
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::unsupported, "Shading-rate images are not supported on Metal."));
            }
            if (desc.sampleCount != 1 &&
                (desc.dimension != TextureDimension::texture2D || desc.mipLevels != 1 ||
                 hasAny(desc.usage, TextureUsage::sampled | TextureUsage::storage) ||
                 (hasAny(desc.usage, TextureUsage::renderTarget) == hasAny(desc.usage, TextureUsage::depthStencil))))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "A multisampled texture must be a single-mip 2D attachment with optional copy usage."));
            }
            if (desc.mipLevels > metalFullMipCount(desc))
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, "The texture mip count exceeds its dimensions."));
            }
            if (formatIsDepth(desc.format) != hasAny(desc.usage, TextureUsage::depthStencil))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "Depth formats and depth-stencil usage must be specified together."));
            }
            if (formatIsDepth(desc.format) && (hasAny(desc.usage, TextureUsage::renderTarget | TextureUsage::storage) ||
                                               desc.dimension == TextureDimension::texture3D))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::unsupported, "Depth textures cannot be render-target, storage, or 3D resources."));
            }
            if (formatIsBc(desc.format) &&
                hasAny(desc.usage, TextureUsage::renderTarget | TextureUsage::storage | TextureUsage::depthStencil))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::unsupported, "Block-compressed textures support copy and sampled usages only."));
            }
            if (formatIsBc(desc.format) && ((desc.width % 4U) != 0 || (desc.height % 4U) != 0))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "Block-compressed textures require extents that are multiples of the block size."));
            }
            return {};
        }

        [[nodiscard]] rachis::Status validateMetalTextureExtents(const rachis::TextureDesc& desc,
                                                                 const rachis::DeviceCapabilities& capabilities)
        {
            using namespace rachis;
            if (desc.dimension == TextureDimension::texture1D && desc.height != 1)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, "A 1D texture requires height 1."));
            }
            auto maxExtent = capabilities.limits.maxTextureDimension2D;
            if (desc.dimension == TextureDimension::texture1D)
            {
                maxExtent = capabilities.limits.maxTextureDimension1D;
            }
            else if (desc.dimension == TextureDimension::texture3D)
            {
                maxExtent = capabilities.limits.maxTextureDimension3D;
            }
            if (desc.width > maxExtent || desc.height > maxExtent ||
                (desc.dimension == TextureDimension::texture3D && desc.depthOrArrayLayers > maxExtent) ||
                (desc.dimension != TextureDimension::texture3D &&
                 desc.depthOrArrayLayers > capabilities.limits.maxTextureArrayLayers))
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "The texture dimensions exceed the Metal device limits."));
            }
            return {};
        }

        [[nodiscard]] rachis::Status validateMetalTextureViewFormats(const rachis::TextureDesc& desc)
        {
            using namespace rachis;
            for (std::size_t index = 0; index < desc.viewFormats.size(); ++index)
            {
                const auto viewFormat = desc.viewFormats[index];
                bool duplicate = false;
                for (std::size_t previous = 0; previous < index; ++previous)
                {
                    duplicate = duplicate || desc.viewFormats[previous] == viewFormat;
                }
                if (viewFormat == desc.format || duplicate || !isTextureViewFormatCompatible(desc.format, viewFormat) ||
                    metalPixelFormat(viewFormat) == MTLPixelFormatInvalid)
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument,
                        "Alternate texture view formats must be unique, non-base members of one compatible family."));
                }
            }
            return {};
        }

        [[nodiscard]] rachis::FormatFeature metalTextureRequiredFormatFeatures(const rachis::TextureDesc& desc)
        {
            using namespace rachis;
            FormatFeature required = FormatFeature::none;
            switch (desc.dimension)
            {
            case TextureDimension::texture1D:
                required |= FormatFeature::texture1D;
                break;
            case TextureDimension::texture3D:
                required |= FormatFeature::texture3D;
                break;
            case TextureDimension::texture2D:
                required |= FormatFeature::texture2D;
                break;
            }
            if (hasAny(desc.usage, TextureUsage::sampled))
            {
                required |= FormatFeature::sampled;
            }
            if (hasAny(desc.usage, TextureUsage::renderTarget))
            {
                required |= FormatFeature::colorAttachment;
            }
            if (hasAny(desc.usage, TextureUsage::depthStencil))
            {
                required |= FormatFeature::depthStencilAttachment;
            }
            if (hasAny(desc.usage, TextureUsage::copySource))
            {
                required |= FormatFeature::copySource;
            }
            if (hasAny(desc.usage, TextureUsage::copyDestination))
            {
                required |= FormatFeature::copyDestination;
            }
            return required;
        }

        [[nodiscard]] rachis::Format metalStorageValidationFormat(const rachis::TextureDesc& desc) noexcept
        {
            const auto linearFormat = rachis::metalLinearViewFormat(desc.format);
            if (linearFormat == rachis::Format::unknown ||
                std::ranges::find(desc.viewFormats, linearFormat) == desc.viewFormats.end())
            {
                return desc.format;
            }
            return linearFormat;
        }

        [[nodiscard]] bool metalTextureStorageSupported(const rachis::TextureDesc& desc,
                                                        std::span<const rachis::FormatProperties> formats) noexcept
        {
            using namespace rachis;
            return !hasAny(desc.usage, TextureUsage::storage) ||
                   hasAny(metalFormatAt(formats, metalStorageValidationFormat(desc)).features,
                          FormatFeature::storageWrite);
        }

        [[nodiscard]] rachis::Status validateMetalTextureDesc(const rachis::TextureDesc& desc,
                                                              const rachis::DeviceCapabilities& capabilities,
                                                              std::span<const rachis::FormatProperties> formats)
        {
            using namespace rachis;
            if (auto valid = validateMetalTextureDescFields(desc); !valid)
            {
                return valid;
            }
            if (auto valid = validateMetalTextureUsageAndShape(desc); !valid)
            {
                return valid;
            }
            if (auto valid = validateMetalTextureExtents(desc, capabilities); !valid)
            {
                return valid;
            }
            if (auto valid = validateMetalTextureViewFormats(desc); !valid)
            {
                return valid;
            }
            if (desc.dimension == TextureDimension::texture1D && desc.mipLevels != 1)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::unsupported, "Metal 1D textures and 1D texture arrays support only one mip level."));
            }
            if (desc.dimension == TextureDimension::texture1D &&
                hasAny(desc.usage, TextureUsage::renderTarget | TextureUsage::depthStencil))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::unsupported,
                    "Metal 1D textures and 1D texture arrays cannot be render-target or depth-stencil attachments."));
            }

            const auto properties = metalFormatAt(formats, desc.format);
            const auto required = metalTextureRequiredFormatFeatures(desc);
            if (!hasAll(properties.features, required) || !metalTextureStorageSupported(desc, formats) ||
                !supportsSampleCount(properties.sampleCounts, desc.sampleCount))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::unsupported,
                    "The Metal format does not support the requested texture usage, dimension, or sample count."));
            }
            return {};
        }

        [[nodiscard]] MTLTextureType metalViewTextureType(rachis::TextureViewDimension dimension,
                                                          std::uint32_t sampleCount) noexcept
        {
            switch (dimension)
            {
            case rachis::TextureViewDimension::texture1D:
                return MTLTextureType1D;
            case rachis::TextureViewDimension::texture1DArray:
                return MTLTextureType1DArray;
            case rachis::TextureViewDimension::texture2DArray:
                return sampleCount > 1 ? MTLTextureType2DMultisampleArray : MTLTextureType2DArray;
            case rachis::TextureViewDimension::texture3D:
                return MTLTextureType3D;
            case rachis::TextureViewDimension::cube:
                return MTLTextureTypeCube;
            case rachis::TextureViewDimension::cubeArray:
                return MTLTextureTypeCubeArray;
            case rachis::TextureViewDimension::texture2D:
                return sampleCount > 1 ? MTLTextureType2DMultisample : MTLTextureType2D;
            }
            return sampleCount > 1 ? MTLTextureType2DMultisample : MTLTextureType2D;
        }

        [[nodiscard]] MTLPixelFormat metalViewPixelFormat(rachis::Format textureFormat,
                                                          rachis::Format viewFormat,
                                                          rachis::TextureAspect aspect) noexcept
        {
            if (aspect == rachis::TextureAspect::stencil)
            {
                return textureFormat == rachis::Format::d24UnormS8Uint ? MTLPixelFormatX24_Stencil8
                                                                       : MTLPixelFormatStencil8;
            }
            if (aspect == rachis::TextureAspect::depth && textureFormat == rachis::Format::d24UnormS8Uint)
            {
                return MTLPixelFormatDepth24Unorm_Stencil8;
            }
            return metalPixelFormat(viewFormat);
        }

        [[nodiscard]] MTLTextureSwizzle metalSwizzle(rachis::ComponentSwizzle swizzle,
                                                     MTLTextureSwizzle identity) noexcept
        {
            switch (swizzle)
            {
            case rachis::ComponentSwizzle::zero:
                return MTLTextureSwizzleZero;
            case rachis::ComponentSwizzle::one:
                return MTLTextureSwizzleOne;
            case rachis::ComponentSwizzle::red:
                return MTLTextureSwizzleRed;
            case rachis::ComponentSwizzle::green:
                return MTLTextureSwizzleGreen;
            case rachis::ComponentSwizzle::blue:
                return MTLTextureSwizzleBlue;
            case rachis::ComponentSwizzle::alpha:
                return MTLTextureSwizzleAlpha;
            case rachis::ComponentSwizzle::identity:
                break;
            }
            return identity;
        }

        struct ResolvedMetalTextureView final
        {
            rachis::Format format = rachis::Format::unknown;
            std::uint32_t mipCount = 1;
            std::uint32_t layerCount = 1;
        };

        [[nodiscard]] rachis::Result<rachis::Format> resolveMetalTextureViewFormat(
            const rachis::TextureDesc& textureDesc,
            std::span<const rachis::Format> viewFormats,
            const rachis::TextureViewDesc& desc)
        {
            using namespace rachis;
            const auto format = desc.format == Format::unknown ? textureDesc.format : desc.format;
            if (format != textureDesc.format)
            {
                const auto allowed = std::ranges::find(viewFormats, format);
                if (allowed == viewFormats.end() || !isTextureViewFormatCompatible(textureDesc.format, format))
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument,
                        "A texture view format must be the base format or a declared compatible alternate."));
                }
            }
            return format;
        }

        [[nodiscard]] rachis::Status validateMetalTextureViewUsage(const rachis::TextureDesc& textureDesc,
                                                                   const rachis::TextureViewDesc& desc)
        {
            using namespace rachis;
            if (desc.usage != TextureViewUsage::sampled && desc.components != identityComponentMapping)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "Component swizzle is valid only on sampled texture views."));
            }
            switch (desc.usage)
            {
            case TextureViewUsage::sampled:
                if (!hasAny(textureDesc.usage, TextureUsage::sampled))
                {
                    return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                               "A sampled view requires sampled texture usage."));
                }
                break;
            case TextureViewUsage::storage:
                if (!hasAny(textureDesc.usage, TextureUsage::storage))
                {
                    return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                               "A storage view requires storage texture usage."));
                }
                break;
            case TextureViewUsage::renderTarget:
                if (!hasAny(textureDesc.usage, TextureUsage::renderTarget))
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument, "A render-target view requires render-target texture usage."));
                }
                break;
            case TextureViewUsage::depthStencil:
                if (!hasAny(textureDesc.usage, TextureUsage::depthStencil))
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument, "A depth-stencil view requires depth-stencil texture usage."));
                }
                break;
            }
            return {};
        }

        [[nodiscard]] rachis::Status validateMetalTextureViewDimension(const rachis::TextureDesc& textureDesc,
                                                                       const rachis::TextureViewDesc& desc)
        {
            using namespace rachis;
            if (!metalViewDimensionAllowed(textureDesc.dimension, desc.dimension))
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "The texture view dimension is incompatible with the texture dimension."));
            }
            if ((desc.dimension == TextureViewDimension::texture1D ||
                 desc.dimension == TextureViewDimension::texture1DArray) &&
                (desc.usage == TextureViewUsage::renderTarget || desc.usage == TextureViewUsage::depthStencil))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::unsupported,
                    "Metal 1D texture views cannot be render-target or depth-stencil attachments."));
            }
            if ((desc.dimension == TextureViewDimension::cube || desc.dimension == TextureViewDimension::cubeArray) &&
                (desc.usage != TextureViewUsage::sampled || textureDesc.width != textureDesc.height))
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, "Cube views require a sampled square 2D texture."));
            }
            if ((desc.dimension == TextureViewDimension::cube || desc.dimension == TextureViewDimension::cubeArray) &&
                !metalCubeCompatible(textureDesc))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "Cube views require a cube-compatible Metal texture backing."));
            }
            return {};
        }

        [[nodiscard]] rachis::FormatFeature metalTextureViewRequiredFeature(rachis::TextureViewUsage usage)
        {
            using namespace rachis;
            switch (usage)
            {
            case TextureViewUsage::storage:
                return FormatFeature::storageWrite;
            case TextureViewUsage::renderTarget:
                return FormatFeature::colorAttachment;
            case TextureViewUsage::depthStencil:
                return FormatFeature::depthStencilAttachment;
            case TextureViewUsage::sampled:
                break;
            }
            return FormatFeature::sampled;
        }

        [[nodiscard]] rachis::Status validateMetalTextureViewFormatSupport(
            const rachis::TextureDesc& textureDesc,
            const rachis::TextureViewDesc& desc,
            rachis::Format format,
            std::span<const rachis::FormatProperties> formats)
        {
            using namespace rachis;
            const auto properties = metalFormatAt(formats, format);
            const auto required = metalTextureViewRequiredFeature(desc.usage);
            if (!hasAny(properties.features, required))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::unsupported, "The texture view format does not support the requested usage."));
            }
            if ((desc.usage == TextureViewUsage::storage || desc.usage == TextureViewUsage::renderTarget) &&
                (formatIsDepth(format) || formatIsBc(format)))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::unsupported, "Storage and render-target views require an uncompressed color format."));
            }
            if (desc.usage == TextureViewUsage::depthStencil && !formatIsDepth(format))
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, "A depth-stencil view requires a depth format."));
            }
            if ((desc.aspect == TextureAspect::stencil && textureDesc.format != Format::d24UnormS8Uint) ||
                (desc.aspect == TextureAspect::depth && !formatIsDepth(textureDesc.format)) ||
                (desc.aspect == TextureAspect::color && formatIsDepth(textureDesc.format)))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "The texture view aspect does not match the texture format."));
            }
            return {};
        }

        [[nodiscard]] rachis::Result<std::uint32_t> resolveMetalTextureViewMipCount(
            const rachis::TextureDesc& textureDesc, const rachis::TextureViewDesc& desc)
        {
            using namespace rachis;
            const bool singleMip = desc.usage != TextureViewUsage::sampled;
            std::uint32_t mipCount = desc.mipLevelCount;
            if (mipCount == remainingSubresources)
            {
                mipCount = singleMip ? 1U : textureDesc.mipLevels - std::min(desc.firstMipLevel, textureDesc.mipLevels);
            }
            if (desc.firstMipLevel >= textureDesc.mipLevels || mipCount == 0 ||
                mipCount > textureDesc.mipLevels - desc.firstMipLevel || (singleMip && mipCount != 1))
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "The texture view mip range is invalid for its usage."));
            }
            return mipCount;
        }

        [[nodiscard]] rachis::Result<std::uint32_t> resolveMetalTextureViewLayerCount(
            const rachis::TextureDesc& textureDesc, const rachis::TextureViewDesc& desc)
        {
            using namespace rachis;
            const auto textureArraySize =
                textureDesc.dimension == TextureDimension::texture3D ? 1U : textureDesc.depthOrArrayLayers;
            const bool singleLayer = desc.dimension == TextureViewDimension::texture1D ||
                                     desc.dimension == TextureViewDimension::texture2D ||
                                     desc.dimension == TextureViewDimension::texture3D;
            std::uint32_t defaultArrayCount = textureArraySize - std::min(desc.firstArrayLayer, textureArraySize);
            if (singleLayer)
            {
                defaultArrayCount = 1;
            }
            else if (desc.dimension == TextureViewDimension::cube)
            {
                defaultArrayCount = 6;
            }
            const auto layerCount =
                desc.arrayLayerCount == remainingSubresources ? defaultArrayCount : desc.arrayLayerCount;
            if (desc.firstArrayLayer >= textureArraySize || layerCount == 0 ||
                layerCount > textureArraySize - desc.firstArrayLayer ||
                (singleLayer && (textureArraySize != 1 || layerCount != 1)) ||
                (desc.dimension == TextureViewDimension::cube && (desc.firstArrayLayer != 0 || layerCount != 6)) ||
                (desc.dimension == TextureViewDimension::cubeArray &&
                 (desc.firstArrayLayer % 6 != 0 || layerCount % 6 != 0)) ||
                (textureDesc.dimension == TextureDimension::texture3D && desc.firstArrayLayer != 0))
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "The texture view array or cube-face range is invalid."));
            }
            return layerCount;
        }

        [[nodiscard]] rachis::Result<ResolvedMetalTextureView> resolveMetalTextureView(
            const rachis::TextureDesc& textureDesc,
            std::span<const rachis::Format> viewFormats,
            const rachis::TextureViewDesc& desc,
            std::span<const rachis::FormatProperties> formats)
        {
            using namespace rachis;
            if (!validMetalComponentMapping(desc.components))
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "The texture view contains an invalid component swizzle."));
            }
            auto format = resolveMetalTextureViewFormat(textureDesc, viewFormats, desc);
            if (!format)
            {
                return std::unexpected(std::move(format.error()));
            }
            if (auto valid = validateMetalTextureViewUsage(textureDesc, desc); !valid)
            {
                return std::unexpected(std::move(valid.error()));
            }
            if (auto valid = validateMetalTextureViewDimension(textureDesc, desc); !valid)
            {
                return std::unexpected(std::move(valid.error()));
            }
            if (auto valid = validateMetalTextureViewFormatSupport(textureDesc, desc, *format, formats); !valid)
            {
                return std::unexpected(std::move(valid.error()));
            }
            auto mipCount = resolveMetalTextureViewMipCount(textureDesc, desc);
            if (!mipCount)
            {
                return std::unexpected(std::move(mipCount.error()));
            }
            auto layerCount = resolveMetalTextureViewLayerCount(textureDesc, desc);
            if (!layerCount)
            {
                return std::unexpected(std::move(layerCount.error()));
            }
            if (desc.label.size() > maxDebugLabelBytes || desc.label.contains('\0'))
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "A texture view label must be a portable UTF-8 string."));
            }
            return ResolvedMetalTextureView{
                .format = *format,
                .mipCount = *mipCount,
                .layerCount = *layerCount,
            };
        }

        [[nodiscard]] rachis::Status validateMetalSamplerDesc(const rachis::SamplerDesc& desc,
                                                              const rachis::DeviceCapabilities& capabilities)
        {
            using namespace rachis;
            if (desc.maxAnisotropy == 0 || desc.maxAnisotropy > 16 ||
                desc.maxAnisotropy > capabilities.limits.maxSamplerAnisotropy)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "Sampler maxAnisotropy must be in [1, min(16, device limit)]."));
            }
            if (!std::isfinite(desc.lodBias) || !std::isfinite(desc.minLod) ||
                (!std::isfinite(desc.maxLod) && desc.maxLod != std::numeric_limits<float>::max()) ||
                desc.minLod > desc.maxLod)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "Sampler LOD values must be finite and minLod must not exceed maxLod."));
            }
            if (std::fabs(desc.lodBias) > capabilities.limits.maxSamplerLodBias)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "Sampler lodBias must be in [-maxSamplerLodBias, maxSamplerLodBias] of the device limits."));
            }
            if (desc.mipFilter == MipmapFilterMode::none && desc.minLod != 0.0F)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "A sampler without mip filtering cannot request a non-zero minLod."));
            }
            if (desc.reductionMode != SamplerReductionMode::standard && desc.compareEnabled)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "Sampler min/max reduction and comparison sampling are mutually exclusive."));
            }
            if (desc.reductionMode != SamplerReductionMode::standard)
            {
                return std::unexpected(metalSelectionError(ErrorCode::unsupported,
                                                           "Sampler min/max reduction is not supported on Metal."));
            }
            if (desc.label.size() > maxDebugLabelBytes || desc.label.contains('\0'))
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "A sampler label must be a portable UTF-8 string."));
            }
            return {};
        }

        [[nodiscard]] MTLSamplerMinMagFilter metalFilter(rachis::FilterMode mode) noexcept
        {
            return mode == rachis::FilterMode::linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
        }

        [[nodiscard]] MTLSamplerMinMagFilter metalSamplerFilter(const rachis::SamplerDesc& desc,
                                                                rachis::FilterMode mode) noexcept
        {
            return desc.maxAnisotropy > 1 ? MTLSamplerMinMagFilterLinear : metalFilter(mode);
        }

        [[nodiscard]] MTLSamplerMipFilter metalMipFilter(rachis::MipmapFilterMode mode) noexcept
        {
            switch (mode)
            {
            case rachis::MipmapFilterMode::linear:
                return MTLSamplerMipFilterLinear;
            case rachis::MipmapFilterMode::nearest:
                return MTLSamplerMipFilterNearest;
            case rachis::MipmapFilterMode::none:
                break;
            }
            return MTLSamplerMipFilterNotMipmapped;
        }

        [[nodiscard]] MTLSamplerAddressMode metalAddressMode(rachis::AddressMode mode) noexcept
        {
            switch (mode)
            {
            case rachis::AddressMode::repeat:
                return MTLSamplerAddressModeRepeat;
            case rachis::AddressMode::mirrorRepeat:
                return MTLSamplerAddressModeMirrorRepeat;
            case rachis::AddressMode::clampToBorder:
                return MTLSamplerAddressModeClampToBorderColor;
            case rachis::AddressMode::clampToEdge:
                break;
            }
            return MTLSamplerAddressModeClampToEdge;
        }

        [[nodiscard]] MTLSamplerBorderColor metalBorderColor(rachis::BorderColor color) noexcept
        {
            switch (color)
            {
            case rachis::BorderColor::opaqueBlack:
                return MTLSamplerBorderColorOpaqueBlack;
            case rachis::BorderColor::opaqueWhite:
                return MTLSamplerBorderColorOpaqueWhite;
            case rachis::BorderColor::transparentBlack:
                break;
            }
            return MTLSamplerBorderColorTransparentBlack;
        }

        [[nodiscard]] MTLCompareFunction metalCompare(rachis::CompareOp op) noexcept
        {
            switch (op)
            {
            case rachis::CompareOp::never:
                return MTLCompareFunctionNever;
            case rachis::CompareOp::less:
                return MTLCompareFunctionLess;
            case rachis::CompareOp::equal:
                return MTLCompareFunctionEqual;
            case rachis::CompareOp::lessEqual:
                return MTLCompareFunctionLessEqual;
            case rachis::CompareOp::greater:
                return MTLCompareFunctionGreater;
            case rachis::CompareOp::notEqual:
                return MTLCompareFunctionNotEqual;
            case rachis::CompareOp::greaterEqual:
                return MTLCompareFunctionGreaterEqual;
            case rachis::CompareOp::always:
                return MTLCompareFunctionAlways;
            }
            return MTLCompareFunctionAlways;
        }

        [[nodiscard]] constexpr bool validMetalResourceHeapClass(ResourceHeapClass resourceClass) noexcept
        {
            switch (resourceClass)
            {
            case ResourceHeapClass::buffer:
            case ResourceHeapClass::nonAttachmentTexture:
            case ResourceHeapClass::attachmentTexture:
                return true;
            }
            return false;
        }

        [[nodiscard]] constexpr bool validMetalResourceHeapFlags(ResourceHeapFlag flags) noexcept
        {
            constexpr auto known = static_cast<std::uint32_t>(ResourceHeapFlag::allowAliasing) |
                                   static_cast<std::uint32_t>(ResourceHeapFlag::sparseBacking);
            return (static_cast<std::uint32_t>(flags) & ~known) == 0;
        }

        [[nodiscard]] constexpr TextureUsage metalDummyTextureUsage(ResourceHeapClass resourceClass) noexcept
        {
            return resourceClass == ResourceHeapClass::attachmentTexture
                       ? TextureUsage::renderTarget | TextureUsage::copySource | TextureUsage::copyDestination
                       : TextureUsage::sampled | TextureUsage::copySource | TextureUsage::copyDestination;
        }

        [[nodiscard]] std::uint64_t metalHeapClassAlignment(id<MTLDevice> device,
                                                            ResourceHeapClass resourceClass,
                                                            MemoryType memoryType) noexcept
        {
            if (device == nil)
            {
                return 16;
            }
            if (resourceClass == ResourceHeapClass::buffer)
            {
                const MTLSizeAndAlign sizeAlign =
                    [device heapBufferSizeAndAlignWithLength:1 options:metalBufferOptions(memoryType)];
                return sizeAlign.align == 0 ? std::uint64_t{16} : static_cast<std::uint64_t>(sizeAlign.align);
            }
            const TextureDesc dummy{
                .width = 1,
                .height = 1,
                .format = Format::rgba8Unorm,
                .usage = metalDummyTextureUsage(resourceClass),
            };
            MTLTextureDescriptor* const descriptor = metalTextureDescriptor(dummy);
            const MTLSizeAndAlign sizeAlign = [device heapTextureSizeAndAlignWithDescriptor:descriptor];
            return sizeAlign.align == 0 ? std::uint64_t{16} : static_cast<std::uint64_t>(sizeAlign.align);
        }

        [[nodiscard]] Result<std::shared_ptr<MetalPlacedLease>> metalReservePlacement(
            const std::shared_ptr<MetalHeapRecord>& heap,
            ResourceHeapClass requiredClass,
            MemoryType memoryType,
            std::uint64_t offset,
            const ResourceMemoryRequirements& requirements,
            std::string_view resourceKind)
        {
            if (heap == nullptr)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           std::string("A placed ") + std::string(resourceKind) +
                                                               " requires a live resource heap."));
            }
            if (heap->desc.resourceClass != requiredClass || heap->desc.memoryType != memoryType)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           std::string("A placed ") + std::string(resourceKind) +
                                                               " requires a compatible resource-heap class."));
            }
            if (hasAny(heap->desc.flags, ResourceHeapFlag::sparseBacking))
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           std::string("A placed ") + std::string(resourceKind) +
                                                               " cannot use a sparse-backing heap."));
            }
            if (requirements.alignment > heap->desc.alignment || offset % requirements.alignment != 0 ||
                requirements.size > heap->desc.size || offset > heap->desc.size - requirements.size)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    std::string("A placed ") + std::string(resourceKind) +
                        " requires a heap with sufficient alignment and an aligned, in-bounds range."));
            }

            auto lease = std::make_shared<MetalPlacedLease>();
            lease->heap = heap;
            lease->offset = offset;
            lease->size = requirements.size;

            const std::scoped_lock lock(heap->mutex);
            if (!heap->alive)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           std::string("A placed ") + std::string(resourceKind) +
                                                               " requires a live resource heap."));
            }
            auto iterator = heap->placements.begin();
            while (iterator != heap->placements.end())
            {
                const auto existing = iterator->lock();
                if (existing == nullptr)
                {
                    iterator = heap->placements.erase(iterator);
                    continue;
                }
                const bool overlaps =
                    offset < existing->offset + existing->size && existing->offset < offset + requirements.size;
                if (overlaps && !hasAny(heap->desc.flags, ResourceHeapFlag::allowAliasing))
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidState,
                        "Placed-resource ranges cannot overlap while either resource remains physically retained."));
                }
                ++iterator;
            }
            heap->placements.emplace_back(lease);
            return lease;
        }

        [[nodiscard]] constexpr ResourceHeapClass metalTextureResourceHeapClass(TextureUsage usage) noexcept
        {
            return hasAny(usage, TextureUsage::renderTarget | TextureUsage::depthStencil)
                       ? ResourceHeapClass::attachmentTexture
                       : ResourceHeapClass::nonAttachmentTexture;
        }

        [[nodiscard]] ResourceMemoryRequirements metalBufferHeapRequirements(id<MTLDevice> device,
                                                                             const BufferDesc& desc) noexcept
        {
            const MTLSizeAndAlign sizeAlign =
                [device heapBufferSizeAndAlignWithLength:static_cast<NSUInteger>(desc.size)
                                                 options:metalBufferOptions(desc.memoryType)];
            const auto alignment =
                sizeAlign.align == 0 ? std::uint64_t{16} : static_cast<std::uint64_t>(sizeAlign.align);
            const auto size = metalAlignUp(std::max<std::uint64_t>(sizeAlign.size, 1), alignment);
            return ResourceMemoryRequirements{
                .size = size,
                .alignment = alignment,
                .dedicatedAllocation = false,
            };
        }

        [[nodiscard]] ResourceMemoryRequirements metalTextureHeapRequirements(id<MTLDevice> device,
                                                                              const TextureDesc& desc) noexcept
        {
            MTLTextureDescriptor* const descriptor = metalTextureDescriptor(desc);
            const MTLSizeAndAlign sizeAlign = [device heapTextureSizeAndAlignWithDescriptor:descriptor];
            const auto alignment =
                sizeAlign.align == 0 ? std::uint64_t{16} : static_cast<std::uint64_t>(sizeAlign.align);
            const auto size = metalAlignUp(std::max<std::uint64_t>(sizeAlign.size, 1), alignment);
            return ResourceMemoryRequirements{
                .size = size,
                .alignment = alignment,
                .dedicatedAllocation = false,
            };
        }

    } // namespace

    void pruneMetalAllocations(MetalDeviceNative& native) noexcept
    {
        std::erase_if(native.allocations,
                      [](const std::weak_ptr<MetalAllocationTelemetry>& allocation) { return allocation.expired(); });
    }

    [[nodiscard]] ResourceMemorySnapshot captureMetalAllocationSnapshot(MetalDeviceNative& native)
    {
        pruneMetalAllocations(native);
        ResourceMemorySnapshot snapshot{};
        for (const auto& weak : native.allocations)
        {
            const auto allocation = weak.lock();
            if (allocation == nullptr)
            {
                continue;
            }
            snapshot.entries.push_back(ResourceMemoryEntry{
                .allocationId = allocation->allocationId,
                .kind = allocation->kind,
                .memoryType = allocation->memoryType,
                .allocationBytes = allocation->allocationBytes,
                .logicalAlive = allocation->logicalAlive,
                .dedicatedAllocation = true,
                .label = allocation->label,
            });
            if (allocation->logicalAlive)
            {
                snapshot.liveAllocationBytes += allocation->allocationBytes;
                ++snapshot.liveAllocationCount;
            }
            else
            {
                snapshot.retainedAllocationBytes += allocation->allocationBytes;
                ++snapshot.retainedAllocationCount;
            }
        }
        std::ranges::sort(snapshot.entries, {}, &ResourceMemoryEntry::allocationId);
        return snapshot;
    }

    [[nodiscard]] Status rejectMetalBufferWorkingSetOverflow(id<MTLDevice> device, std::uint64_t bytes)
    {
        if (device == nil)
        {
            return {};
        }
        if (bytes > device.maxBufferLength || bytes > device.recommendedMaxWorkingSetSize)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::outOfMemory,
                "The Metal allocation exceeds the device working-set budget. Partial Metal objects were not created."));
        }
        return {};
    }

    [[nodiscard]] Status rejectMetalTextureWorkingSetOverflow(id<MTLDevice> device, std::uint64_t bytes)
    {
        if (device == nil)
        {
            return {};
        }
        if (bytes > device.recommendedMaxWorkingSetSize)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::outOfMemory,
                "The Metal allocation exceeds the device working-set budget. Partial Metal objects were not created."));
        }
        return {};
    }

    Status reserveMetalAllocationTelemetry(MetalDeviceNative& native)
    {
        if (native.allocations.size() >= native.nextAllocationPruneThreshold)
        {
            pruneMetalAllocations(native);
            native.nextAllocationPruneThreshold = std::max<std::size_t>(64, native.allocations.size() * 2);
        }
        if (native.allocations.size() == native.allocations.capacity())
        {
            try
            {
                const auto currentCapacity = native.allocations.capacity();
                const auto grownCapacity =
                    currentCapacity == 0
                        ? std::size_t{16}
                        : (currentCapacity > std::numeric_limits<std::size_t>::max() / 2 ? native.allocations.size() + 1
                                                                                         : currentCapacity * 2);
                native.allocations.reserve(grownCapacity);
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(metalSelectionError(ErrorCode::outOfMemory,
                                                           "Failed to grow the Metal allocation telemetry registry."));
            }
        }
        return {};
    }

    void publishMetalAllocation(MetalDeviceNative& native, const std::shared_ptr<MetalAllocationTelemetry>& allocation)
    {
        native.allocations.push_back(allocation);
    }

    [[nodiscard]] bool metalPlacementsAlias(const std::shared_ptr<MetalPlacedLease>& before,
                                            const std::shared_ptr<MetalPlacedLease>& after) noexcept
    {
        return before != nullptr && after != nullptr && before->heap != nullptr && before->heap == after->heap &&
               hasAny(before->heap->desc.flags, ResourceHeapFlag::allowAliasing) &&
               before->offset < after->offset + after->size && after->offset < before->offset + before->size;
    }

    Result<ResourceMemoryRequirements> MetalDevice::queryBufferMemoryRequirements(const BufferDesc& desc) const
    {
        auto valid = validateMetalBufferDesc(desc);
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || native_->device == nil)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState, "Metal queryBufferMemoryRequirements requires an active device."));
        }
        return metalBufferHeapRequirements(native_->device, desc);
    }

    Result<ResourceHeapHandle> MetalDevice::createResourceHeap(const ResourceHeapDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || native_->device == nil)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal createResourceHeap requires an active device."));
        }
        const auto classAlignment = metalHeapClassAlignment(native_->device, desc.resourceClass, desc.memoryType);
        const auto alignment = desc.alignment == 0 ? classAlignment : desc.alignment;

        if (desc.size == 0 || !std::has_single_bit(alignment) || desc.size % alignment != 0 ||
            !validMetalMemoryType(desc.memoryType) || !validMetalResourceHeapClass(desc.resourceClass) ||
            !validMetalResourceHeapFlags(desc.flags) ||
            (desc.resourceClass != ResourceHeapClass::buffer && desc.memoryType != MemoryType::deviceLocal) ||
            (hasAny(desc.flags, ResourceHeapFlag::allowAliasing) && desc.memoryType != MemoryType::deviceLocal) ||
            desc.residencyPriority != ResidencyPriority::automatic || desc.label.size() > maxDebugLabelBytes ||
            desc.label.contains('\0'))
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument,
                "A resource heap requires compatible memory, placement, residency, and allocation properties."));
        }
        if (hasAny(desc.allocationFlags, ResourceAllocationFlag::manualResidency))
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::unsupported, "DeviceDesc manual residency is not supported on Metal."));
        }
        if (hasAny(desc.flags, ResourceHeapFlag::sparseBacking))
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::unsupported, "Sparse-backing heaps are unavailable on this device."));
        }
        auto budget = rejectMetalBufferWorkingSetOverflow(native_->device, desc.size);
        if (!budget)
        {
            return std::unexpected(std::move(budget.error()));
        }
        auto reserved = reserveMetalAllocationTelemetry(*native_);
        if (!reserved)
        {
            return std::unexpected(std::move(reserved.error()));
        }

        @autoreleasepool
        {
            MTLHeapDescriptor* const descriptor = [MTLHeapDescriptor new];
            descriptor.type = MTLHeapTypePlacement;
            descriptor.storageMode =
                desc.memoryType == MemoryType::deviceLocal ? MTLStorageModePrivate : MTLStorageModeShared;
            descriptor.cpuCacheMode = MTLCPUCacheModeDefaultCache;
            descriptor.hazardTrackingMode = MTLHazardTrackingModeTracked;
            descriptor.size = static_cast<NSUInteger>(desc.size);
            id<MTLHeap> const heap = [native_->device newHeapWithDescriptor:descriptor];
            if (heap == nil)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::outOfMemory, "newHeapWithDescriptor failed. Partial Metal objects were released."));
            }

            auto record = std::make_shared<MetalHeapRecord>();
            record->desc = desc;
            record->desc.alignment = alignment;
            record->label = std::string(desc.label);
            record->desc.label = record->label;
            record->heap = heap;
            record->telemetry = std::make_shared<MetalAllocationTelemetry>();
            record->telemetry->allocationId = native_->nextAllocationId;
            record->telemetry->kind = ResourceMemoryKind::resourceHeap;
            record->telemetry->memoryType = desc.memoryType;
            record->telemetry->allocationBytes = desc.size;
            record->telemetry->label = record->label;
            if (!record->label.empty())
            {
                heap.label = metalNsString(record->label);
            }
            auto handle = native_->heaps.emplace(record);
            if (!handle)
            {
                return std::unexpected(std::move(handle.error()));
            }
            ++native_->nextAllocationId;
            publishMetalAllocation(*native_, record->telemetry);
            return handle;
        }
    }

    Status MetalDevice::destroyResourceHeap(ResourceHeapHandle heap)
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal destroyResourceHeap requires an active device."));
        }
        auto* stored = native_->heaps.find(heap);
        if (stored == nullptr || *stored == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The resource-heap handle is invalid or stale."));
        }
        const auto record = *stored;
        {
            const std::scoped_lock recordLock(record->mutex);
            record->alive = false;
            if (record->telemetry != nullptr)
            {
                record->telemetry->logicalAlive = false;
            }
        }
        if (!native_->heaps.erase(heap))
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The resource-heap handle is invalid or stale."));
        }
        return {};
    }

    Result<ResidencyState> MetalDevice::queryResidency(ResourceHeapHandle heap) const
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal queryResidency requires an active device."));
        }
        auto* stored = native_->heaps.find(heap);
        if (stored == nullptr || *stored == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The resource-heap handle is invalid or stale."));
        }
        return ResidencyState::automatic;
    }

    Result<BufferHandle> MetalDevice::createBuffer(const BufferDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || native_->device == nil)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal createBuffer requires an active device."));
        }
        auto valid = validateMetalBufferDesc(desc);
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        auto budget = rejectMetalBufferWorkingSetOverflow(native_->device, desc.size);
        if (!budget)
        {
            return std::unexpected(std::move(budget.error()));
        }
        auto reserved = reserveMetalAllocationTelemetry(*native_);
        if (!reserved)
        {
            return std::unexpected(std::move(reserved.error()));
        }

        @autoreleasepool
        {
            id<MTLBuffer> const buffer = [native_->device newBufferWithLength:static_cast<NSUInteger>(desc.size)
                                                                      options:metalBufferOptions(desc.memoryType)];
            if (buffer == nil)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::outOfMemory, "newBufferWithLength failed. Partial Metal objects were released."));
            }
            auto allocation = std::make_shared<MetalAllocationTelemetry>();
            allocation->allocationId = native_->nextAllocationId;
            allocation->kind = ResourceMemoryKind::buffer;
            allocation->memoryType = desc.memoryType;
            allocation->allocationBytes = buffer.length;
            allocation->label = std::string(desc.label);
            MetalBufferRecord record{};
            record.owner = std::make_shared<MetalBufferOwner>();
            record.owner->buffer = buffer;
            record.owner->telemetry = allocation;
            record.buffer = buffer;
            record.label = allocation->label;
            record.desc = desc;
            record.desc.label = record.label;
            record.allocationId = allocation->allocationId;
            record.allocationBytes = allocation->allocationBytes;
            if (!record.label.empty())
            {
                buffer.label = metalNsString(record.label);
            }
            auto handle = native_->buffers.emplace(std::move(record));
            if (!handle)
            {
                return std::unexpected(std::move(handle.error()));
            }
            ++native_->nextAllocationId;
            publishMetalAllocation(*native_, allocation);
            return handle;
        }
    }

    Result<BufferHandle> MetalDevice::createPlacedBuffer(const ResourcePlacement& placement, const BufferDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (desc.residencyPriority != ResidencyPriority::automatic ||
            desc.allocationFlags != ResourceAllocationFlag::none)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument,
                "Placed buffers inherit heap residency and cannot request resource-level residency policy."));
        }
        auto valid = validateMetalBufferDesc(desc);
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || native_->device == nil)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal createPlacedBuffer requires an active device."));
        }
        const auto requirements = metalBufferHeapRequirements(native_->device, desc);
        if (requirements.dedicatedAllocation)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument, "A dedicated buffer cannot be placed on a shared resource heap."));
        }
        std::shared_ptr<MetalHeapRecord> heap;
        auto* stored = native_->heaps.find(placement.heap);
        if (stored != nullptr)
        {
            heap = *stored;
        }
        auto lease = metalReservePlacement(
            heap, ResourceHeapClass::buffer, desc.memoryType, placement.offset, requirements, "buffer");
        if (!lease)
        {
            return std::unexpected(std::move(lease.error()));
        }

        @autoreleasepool
        {
            id<MTLBuffer> const buffer = [heap->heap newBufferWithLength:static_cast<NSUInteger>(desc.size)
                                                                 options:metalBufferOptions(desc.memoryType)
                                                                  offset:static_cast<NSUInteger>(placement.offset)];
            if (buffer == nil)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::outOfMemory, "newBufferWithLength offset failed. Partial Metal objects were released."));
            }
            MetalBufferRecord record{};
            record.owner = std::make_shared<MetalBufferOwner>();
            record.owner->buffer = buffer;
            record.owner->placement = *lease;
            record.buffer = buffer;
            record.label = std::string(desc.label);
            record.desc = desc;
            record.desc.label = record.label;
            if (!record.label.empty())
            {
                buffer.label = metalNsString(record.label);
            }
            auto handle = native_->buffers.emplace(std::move(record));
            if (!handle)
            {
                return std::unexpected(std::move(handle.error()));
            }
            return handle;
        }
    }

    Result<ResidencyState> MetalDevice::queryResidency(BufferHandle buffer) const
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal queryResidency requires an active device."));
        }
        auto* record = native_->buffers.find(buffer);
        if (record == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The buffer handle is not a live Metal allocation."));
        }
        return ResidencyState::automatic;
    }

    Status MetalDevice::destroyBuffer(BufferHandle buffer)
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal destroyBuffer requires an active device."));
        }
        auto* record = native_->buffers.find(buffer);
        if (record == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The buffer handle is not a live Metal allocation."));
        }
        if (record->owner != nullptr && record->owner->placement == nullptr && record->owner->telemetry != nullptr)
        {
            record->owner->telemetry->logicalAlive = false;
        }
        if (!native_->buffers.erase(buffer))
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The buffer handle is not a live Metal allocation."));
        }
        return {};
    }

    Result<MappedMemory> MetalDevice::mapBuffer(BufferHandle buffer, std::uint64_t offset, std::uint64_t size)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal mapBuffer requires an active device."));
        }
        auto* record = native_->buffers.find(buffer);
        if (record == nullptr || record->buffer == nil)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The buffer handle is not a live Metal allocation."));
        }
        if (!metalBufferHostVisible(record->desc.memoryType) || record->buffer.contents == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::unsupported, "Device-local Metal buffers are not host-mappable."));
        }
        if (record->mapped)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidState, "The Metal buffer is already mapped."));
        }
        if (offset > record->desc.size)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The Metal map range is outside the buffer."));
        }
        const auto remaining = record->desc.size - offset;
        const auto mappedSize = size == wholeResource ? remaining : size;
        if (mappedSize == 0 || mappedSize > remaining)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The Metal map range is outside the buffer."));
        }
        record->mapped = true;
        record->mapOffset = offset;
        record->mapSize = mappedSize;
        return MappedMemory{
            .data = static_cast<std::byte*>(record->buffer.contents) + offset,
            .size = mappedSize,
        };
    }

    Status MetalDevice::unmapBuffer(BufferHandle buffer, std::uint64_t offset, std::uint64_t size)
    {
        std::scoped_lock const lock(mutex_);
        static_cast<void>(offset);
        static_cast<void>(size);
        if (native_ == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal unmapBuffer requires an active device."));
        }
        auto* record = native_->buffers.find(buffer);
        if (record == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The buffer handle is not a live Metal allocation."));
        }
        if (!record->mapped)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidState, "The Metal buffer is not mapped."));
        }
        record->mapped = false;
        record->mapOffset = 0;
        record->mapSize = 0;
        return {};
    }

    Status MetalDevice::flushMappedMemoryRanges(BufferHandle buffer, std::span<const MappedMemoryRange> ranges)
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                       "Metal flushMappedMemoryRanges requires an active device."));
        }
        auto* record = native_->buffers.find(buffer);
        if (record == nullptr || !record->mapped)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Flush requires a mapped Metal buffer."));
        }
        static_cast<void>(ranges);
        return {};
    }

    Status MetalDevice::invalidateMappedMemoryRanges(BufferHandle buffer, std::span<const MappedMemoryRange> ranges)
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState, "Metal invalidateMappedMemoryRanges requires an active device."));
        }
        auto* record = native_->buffers.find(buffer);
        if (record == nullptr || !record->mapped)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Invalidate requires a mapped Metal buffer."));
        }
        static_cast<void>(ranges);
        return {};
    }

    Status MetalDevice::writeBuffer(BufferHandle buffer, std::uint64_t offset, std::span<const std::byte> data)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal writeBuffer requires an active device."));
        }
        auto* record = native_->buffers.find(buffer);
        if (record == nullptr || record->buffer == nil)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The buffer handle is not a live Metal allocation."));
        }

        if (!validMetalHostRange(record->desc.size, offset, data.size()))
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The host buffer write range is empty or outside the buffer."));
        }
        if (record->desc.memoryType != MemoryType::upload || record->buffer.contents == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::unsupported, "Host writes are upload-only on Metal."));
        }
        if (record->mapped)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidState, "writeBuffer is mapping-exclusive."));
        }
        std::memcpy(static_cast<std::byte*>(record->buffer.contents) + offset, data.data(), data.size());
        return {};
    }

    Status MetalDevice::readBuffer(BufferHandle buffer, std::uint64_t offset, std::span<std::byte> destination)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal readBuffer requires an active device."));
        }
        auto* record = native_->buffers.find(buffer);
        if (record == nullptr || record->buffer == nil)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The buffer handle is not a live Metal allocation."));
        }

        if (!validMetalHostRange(record->desc.size, offset, destination.size()))
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The host buffer read range is empty or outside the buffer."));
        }
        if (record->desc.memoryType != MemoryType::readback || record->buffer.contents == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::unsupported, "Host reads are readback-only on Metal."));
        }
        if (record->mapped)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidState, "readBuffer is mapping-exclusive."));
        }
        std::memcpy(
            destination.data(), static_cast<const std::byte*>(record->buffer.contents) + offset, destination.size());
        return {};
    }

    Result<ResourceMemoryRequirements> MetalDevice::queryTextureMemoryRequirements(const TextureDesc& desc) const
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || native_->device == nil)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState, "Metal queryTextureMemoryRequirements requires an active device."));
        }
        auto valid = validateMetalTextureDesc(desc, capabilities_, formats_);
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        @autoreleasepool
        {
            MTLTextureDescriptor* const descriptor = metalTextureDescriptor(desc);
            const MTLSizeAndAlign sizeAlign = [native_->device heapTextureSizeAndAlignWithDescriptor:descriptor];
            const auto alignment =
                sizeAlign.align == 0 ? std::uint64_t{16} : static_cast<std::uint64_t>(sizeAlign.align);
            const auto size = metalAlignUp(std::max<std::uint64_t>(sizeAlign.size, 1), alignment);
            return ResourceMemoryRequirements{
                .size = size,
                .alignment = alignment,
                .dedicatedAllocation = false,
            };
        }
    }

    Result<TextureHandle> MetalDevice::createTexture(const TextureDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || native_->device == nil)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal createTexture requires an active device."));
        }
        auto valid = validateMetalTextureDesc(desc, capabilities_, formats_);
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        @autoreleasepool
        {
            MTLTextureDescriptor* const descriptor = metalTextureDescriptor(desc);
            const MTLSizeAndAlign sizeAlign = [native_->device heapTextureSizeAndAlignWithDescriptor:descriptor];
            auto budget =
                rejectMetalTextureWorkingSetOverflow(native_->device, std::max<std::uint64_t>(sizeAlign.size, 1));
            if (!budget)
            {
                return std::unexpected(std::move(budget.error()));
            }
            auto reserved = reserveMetalAllocationTelemetry(*native_);
            if (!reserved)
            {
                return std::unexpected(std::move(reserved.error()));
            }
            id<MTLTexture> const texture = [native_->device newTextureWithDescriptor:descriptor];
            if (texture == nil)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::outOfMemory, "newTextureWithDescriptor failed. Partial Metal objects were released."));
            }
            auto allocation = std::make_shared<MetalAllocationTelemetry>();
            allocation->allocationId = native_->nextAllocationId;
            allocation->kind = ResourceMemoryKind::texture;
            allocation->memoryType = MemoryType::deviceLocal;
            allocation->allocationBytes = texture.allocatedSize;
            allocation->label = std::string(desc.label);
            MetalTextureRecord record{};
            record.owner = std::make_shared<MetalTextureOwner>();
            record.owner->texture = texture;
            record.owner->telemetry = allocation;
            record.texture = texture;
            record.label = allocation->label;
            record.desc = desc;
            record.viewFormats.assign(desc.viewFormats.begin(), desc.viewFormats.end());
            record.desc.viewFormats = {};
            record.desc.label = record.label;
            record.allocationId = allocation->allocationId;
            record.allocationBytes = allocation->allocationBytes;
            if (!record.label.empty())
            {
                texture.label = metalNsString(record.label);
            }
            auto handle = native_->textures.emplace(std::move(record));
            if (!handle)
            {
                return std::unexpected(std::move(handle.error()));
            }
            ++native_->nextAllocationId;
            publishMetalAllocation(*native_, allocation);
            return handle;
        }
    }

    Result<TextureHandle> MetalDevice::createPlacedTexture(const ResourcePlacement& placement, const TextureDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (desc.residencyPriority != ResidencyPriority::automatic ||
            desc.allocationFlags != ResourceAllocationFlag::none)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument,
                "Placed textures inherit heap residency and cannot request resource-level residency policy."));
        }
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || native_->device == nil)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal createPlacedTexture requires an active device."));
        }
        auto valid = validateMetalTextureDesc(desc, capabilities_, formats_);
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        const auto requirements = metalTextureHeapRequirements(native_->device, desc);
        if (requirements.dedicatedAllocation)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument, "A dedicated texture cannot be placed on a shared resource heap."));
        }
        std::shared_ptr<MetalHeapRecord> heap;
        auto* stored = native_->heaps.find(placement.heap);
        if (stored != nullptr)
        {
            heap = *stored;
        }
        auto lease = metalReservePlacement(heap,
                                           metalTextureResourceHeapClass(desc.usage),
                                           MemoryType::deviceLocal,
                                           placement.offset,
                                           requirements,
                                           "texture");
        if (!lease)
        {
            return std::unexpected(std::move(lease.error()));
        }

        @autoreleasepool
        {
            MTLTextureDescriptor* const descriptor = metalTextureDescriptor(desc);
            id<MTLTexture> const texture =
                [heap->heap newTextureWithDescriptor:descriptor offset:static_cast<NSUInteger>(placement.offset)];
            if (texture == nil)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::outOfMemory,
                    "newTextureWithDescriptor offset failed. Partial Metal objects were released."));
            }
            MetalTextureRecord record{};
            record.owner = std::make_shared<MetalTextureOwner>();
            record.owner->texture = texture;
            record.owner->placement = *lease;
            record.texture = texture;
            record.label = std::string(desc.label);
            record.desc = desc;
            record.viewFormats.assign(desc.viewFormats.begin(), desc.viewFormats.end());
            record.desc.viewFormats = {};
            record.desc.label = record.label;
            if (!record.label.empty())
            {
                texture.label = metalNsString(record.label);
            }
            auto handle = native_->textures.emplace(std::move(record));
            if (!handle)
            {
                return std::unexpected(std::move(handle.error()));
            }
            return handle;
        }
    }

    Result<ResidencyState> MetalDevice::queryResidency(TextureHandle texture) const
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal queryResidency requires an active device."));
        }
        auto* record = native_->textures.find(texture);
        if (record == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The texture handle is not a live Metal allocation."));
        }
        return ResidencyState::automatic;
    }

    Status MetalDevice::destroyTexture(TextureHandle texture)
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal destroyTexture requires an active device."));
        }
        auto* record = native_->textures.find(texture);
        if (record == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The texture handle is not a live Metal allocation."));
        }
        if (record->swapchainImage)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState, "Swapchain images are borrowed and cannot be destroyed directly."));
        }
        if (record->liveViews != 0)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "A texture with live views cannot be destroyed."));
        }
        if (record->owner != nullptr && record->owner->placement == nullptr && record->owner->telemetry != nullptr)
        {
            record->owner->telemetry->logicalAlive = false;
        }
        if (!native_->textures.erase(texture))
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The texture handle is not a live Metal allocation."));
        }
        return {};
    }

    Result<TextureTransferLayout> MetalDevice::queryTextureTransferLayout(TextureHandle texture,
                                                                          const BufferTextureCopyRegion& region) const
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || native_->device == nil)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                       "Metal queryTextureTransferLayout requires an active device."));
        }
        auto* record = native_->textures.find(texture);
        if (record == nullptr || record->texture == nil)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument, "A texture transfer layout query references an invalid or stale texture."));
        }
        const auto alignments =
            metalTransferAlignments(native_->device, record->desc.format, region.textureSubresource.aspect);
        auto resolved = resolveMetalTransfer(record->desc, region, alignments.first, alignments.second);
        if (!resolved)
        {
            return std::unexpected(std::move(resolved.error()));
        }
        return resolved->layout;
    }

    Result<TextureViewHandle> MetalDevice::createTextureView(const TextureViewDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || native_->device == nil)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal createTextureView requires an active device."));
        }
        auto* texture = native_->textures.find(desc.texture);
        if (texture == nullptr || texture->texture == nil)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument, "createTextureView received a stale or foreign texture handle."));
        }
        auto resolved = resolveMetalTextureView(texture->desc, texture->viewFormats, desc, formats_);
        if (!resolved)
        {
            return std::unexpected(std::move(resolved.error()));
        }
        @autoreleasepool
        {
            const auto pixel = metalViewPixelFormat(texture->desc.format, resolved->format, desc.aspect);
            const auto type = metalViewTextureType(desc.dimension, texture->desc.sampleCount);
            const NSRange levels = NSMakeRange(desc.firstMipLevel, resolved->mipCount);
            const NSRange slices =
                NSMakeRange(desc.dimension == TextureViewDimension::texture3D ? 0 : desc.firstArrayLayer,
                            desc.dimension == TextureViewDimension::texture3D ? 1 : resolved->layerCount);
            const MTLTextureSwizzleChannels swizzle =
                MTLTextureSwizzleChannelsMake(metalSwizzle(desc.components.red, MTLTextureSwizzleRed),
                                              metalSwizzle(desc.components.green, MTLTextureSwizzleGreen),
                                              metalSwizzle(desc.components.blue, MTLTextureSwizzleBlue),
                                              metalSwizzle(desc.components.alpha, MTLTextureSwizzleAlpha));
            id<MTLTexture> const view = [texture->texture newTextureViewWithPixelFormat:pixel
                                                                            textureType:type
                                                                                 levels:levels
                                                                                 slices:slices
                                                                                swizzle:swizzle];
            if (view == nil)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::outOfMemory,
                                        "newTextureViewWithPixelFormat failed. Partial Metal objects were released."));
            }
            MetalTextureViewRecord record{};
            record.owner = std::make_shared<MetalTextureOwner>();
            record.owner->texture = view;
            record.parent = texture->owner;
            record.view = view;
            record.texture = desc.texture;
            record.label = std::string(desc.label);
            record.desc = desc;
            record.desc.label = record.label;
            if (!record.label.empty())
            {
                view.label = metalNsString(record.label);
            }
            auto handle = native_->views.emplace(std::move(record));
            if (!handle)
            {
                return std::unexpected(std::move(handle.error()));
            }
            texture->liveViews += 1;
            return handle;
        }
    }

    Status MetalDevice::destroyTextureView(TextureViewHandle view)
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal destroyTextureView requires an active device."));
        }
        auto* record = native_->views.find(view);
        if (record == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The texture view handle is not a live Metal object."));
        }
        if (record->swapchainView)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState, "Swapchain image views are borrowed and cannot be destroyed directly."));
        }
        if (auto* texture = native_->textures.find(record->texture); texture != nullptr && texture->liveViews > 0)
        {
            texture->liveViews -= 1;
        }
        if (!native_->views.erase(view))
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The texture view handle is not a live Metal object."));
        }
        return {};
    }

    Result<SamplerHandle> MetalDevice::createSampler(const SamplerDesc& desc)
    {
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr || native_->device == nil)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal createSampler requires an active device."));
        }
        auto valid = validateMetalSamplerDesc(desc, capabilities_);
        if (!valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        @autoreleasepool
        {
            MTLSamplerDescriptor* const descriptor = [MTLSamplerDescriptor new];
            descriptor.minFilter = metalSamplerFilter(desc, desc.minFilter);
            descriptor.magFilter = metalSamplerFilter(desc, desc.magFilter);
            descriptor.mipFilter = metalMipFilter(desc.mipFilter);
            descriptor.sAddressMode = metalAddressMode(desc.addressU);
            descriptor.tAddressMode = metalAddressMode(desc.addressV);
            descriptor.rAddressMode = metalAddressMode(desc.addressW);
            descriptor.lodMinClamp = desc.minLod;
            descriptor.lodMaxClamp = desc.maxLod;
            descriptor.lodBias = desc.lodBias;
            descriptor.maxAnisotropy = desc.maxAnisotropy;
            descriptor.compareFunction = desc.compareEnabled ? metalCompare(desc.compare) : MTLCompareFunctionNever;
            descriptor.borderColor = metalBorderColor(desc.borderColor);
            descriptor.normalizedCoordinates = YES;
            descriptor.supportArgumentBuffers = YES;
            id<MTLSamplerState> const sampler = [native_->device newSamplerStateWithDescriptor:descriptor];
            if (sampler == nil)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::outOfMemory,
                                        "newSamplerStateWithDescriptor failed. Partial Metal objects were released."));
            }
            MetalSamplerRecord record{};
            record.owner = std::make_shared<MetalSamplerOwner>();
            record.owner->sampler = sampler;
            record.sampler = sampler;
            record.label = std::string(desc.label);
            record.desc = desc;
            record.desc.label = record.label;
            auto handle = native_->samplers.emplace(std::move(record));
            if (!handle)
            {
                return std::unexpected(std::move(handle.error()));
            }
            return handle;
        }
    }

    Status MetalDevice::destroySampler(SamplerHandle sampler)
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Metal destroySampler requires an active device."));
        }
        if (native_->samplers.find(sampler) == nullptr || !native_->samplers.erase(sampler))
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "The sampler handle is not a live Metal object."));
        }
        return {};
    }

#ifdef RACHIS_TESTING
    std::uint64_t MetalDevice::samplerGpuResourceIdForTesting(SamplerHandle sampler) const noexcept
    {
        std::scoped_lock const lock(mutex_);
        if (native_ == nullptr)
        {
            return 0;
        }
        const auto* record = native_->samplers.find(sampler);
        return record != nullptr && record->sampler != nil ? record->sampler.gpuResourceID._impl : 0;
    }
#endif

} // namespace rachis::metal
