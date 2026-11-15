#include "D3D12/D3D12View.h"

#include "Common/HandleTable.h"
#include "D3D12/D3D12Descriptor.h"
#include "D3D12/D3D12Format.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace rachis::d3d12
{
    namespace
    {

        [[nodiscard]] constexpr bool isDepthFormat(Format value) noexcept
        {
            const auto metadata = format::info(value);
            return metadata && metadata->depth;
        }

        [[nodiscard]] constexpr bool isCompressed(Format value) noexcept
        {
            const auto metadata = format::info(value);
            return metadata && metadata->compressed;
        }

        [[nodiscard]] constexpr DXGI_FORMAT colorFormat(Format value) noexcept
        {
            return format::view(value);
        }

        [[nodiscard]] constexpr DXGI_FORMAT nativeViewFormat(Format format,
                                                             TextureViewUsage usage,
                                                             TextureAspect aspect) noexcept
        {
            if (!isDepthFormat(format))
            {
                return aspect == TextureAspect::color ? colorFormat(format) : DXGI_FORMAT_UNKNOWN;
            }
            if (usage == TextureViewUsage::depthStencil)
            {
                if (aspect != TextureAspect::depth)
                {
                    return DXGI_FORMAT_UNKNOWN;
                }
                return format == Format::d32Float ? DXGI_FORMAT_D32_FLOAT : DXGI_FORMAT_D24_UNORM_S8_UINT;
            }
            if (usage != TextureViewUsage::sampled)
            {
                return DXGI_FORMAT_UNKNOWN;
            }
            if (format == Format::d32Float)
            {
                return aspect == TextureAspect::depth ? DXGI_FORMAT_R32_FLOAT : DXGI_FORMAT_UNKNOWN;
            }
            if (aspect == TextureAspect::depth)
            {
                return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
            }
            return aspect == TextureAspect::stencil ? DXGI_FORMAT_X24_TYPELESS_G8_UINT : DXGI_FORMAT_UNKNOWN;
        }

        [[nodiscard]] constexpr std::uint32_t planeSlice(Format format, TextureAspect aspect) noexcept
        {
            return format == Format::d24UnormS8Uint && aspect == TextureAspect::stencil ? 1U : 0U;
        }

        [[nodiscard]] constexpr bool validComponentSwizzle(ComponentSwizzle swizzle) noexcept
        {
            switch (swizzle)
            {
            case ComponentSwizzle::identity:
            case ComponentSwizzle::zero:
            case ComponentSwizzle::one:
            case ComponentSwizzle::red:
            case ComponentSwizzle::green:
            case ComponentSwizzle::blue:
            case ComponentSwizzle::alpha:
                return true;
            }
            return false;
        }

        [[nodiscard]] constexpr bool validComponentMapping(const ComponentMapping& mapping) noexcept
        {
            return validComponentSwizzle(mapping.red) && validComponentSwizzle(mapping.green) &&
                   validComponentSwizzle(mapping.blue) && validComponentSwizzle(mapping.alpha);
        }

        [[nodiscard]] constexpr D3D12_SHADER_COMPONENT_MAPPING nativeComponentSwizzle(
            ComponentSwizzle swizzle, std::uint32_t destination) noexcept
        {
            switch (swizzle)
            {
            case ComponentSwizzle::identity:
                return static_cast<D3D12_SHADER_COMPONENT_MAPPING>(destination);
            case ComponentSwizzle::zero:
                return D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0;
            case ComponentSwizzle::one:
                return D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_1;
            case ComponentSwizzle::red:
                return D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_0;
            case ComponentSwizzle::green:
                return D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_1;
            case ComponentSwizzle::blue:
                return D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_2;
            case ComponentSwizzle::alpha:
                return D3D12_SHADER_COMPONENT_MAPPING_FROM_MEMORY_COMPONENT_3;
            }
            return D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0;
        }

        [[nodiscard]] constexpr std::uint32_t nativeComponentMapping(const ComponentMapping& mapping) noexcept
        {
            return D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(nativeComponentSwizzle(mapping.red, 0),
                                                           nativeComponentSwizzle(mapping.green, 1),
                                                           nativeComponentSwizzle(mapping.blue, 2),
                                                           nativeComponentSwizzle(mapping.alpha, 3));
        }

        [[nodiscard]] constexpr bool usageAllowed(TextureUsage textureUsage, TextureViewUsage viewUsage) noexcept
        {
            switch (viewUsage)
            {
            case TextureViewUsage::sampled:
                return hasAny(textureUsage, TextureUsage::sampled);
            case TextureViewUsage::storage:
                return hasAny(textureUsage, TextureUsage::storage);
            case TextureViewUsage::renderTarget:
                return hasAny(textureUsage, TextureUsage::renderTarget);
            case TextureViewUsage::depthStencil:
                return hasAny(textureUsage, TextureUsage::depthStencil);
            }
            return false;
        }

        [[nodiscard]] constexpr bool dimensionAllowed(TextureDimension textureDimension,
                                                      TextureViewDimension viewDimension) noexcept
        {
            switch (textureDimension)
            {
            case TextureDimension::texture1D:
                return viewDimension == TextureViewDimension::texture1D ||
                       viewDimension == TextureViewDimension::texture1DArray;
            case TextureDimension::texture2D:
                return viewDimension == TextureViewDimension::texture2D ||
                       viewDimension == TextureViewDimension::texture2DArray ||
                       viewDimension == TextureViewDimension::cube || viewDimension == TextureViewDimension::cubeArray;
            case TextureDimension::texture3D:
                return viewDimension == TextureViewDimension::texture3D;
            }
            return false;
        }

        struct ResolvedViewDesc final
        {
            TextureViewDesc desc;
            Format format = Format::unknown;
            std::uint32_t mipCount = 0;
            std::uint32_t arrayCount = 0;
        };

        [[nodiscard]] Status requireViewFormatSupport(ID3D12Device& device, DXGI_FORMAT format, TextureViewUsage usage)
        {
            D3D12_FORMAT_SUPPORT1 requiredSupport1 = D3D12_FORMAT_SUPPORT1_NONE;
            D3D12_FORMAT_SUPPORT2 requiredSupport2 = D3D12_FORMAT_SUPPORT2_NONE;
            switch (usage)
            {
            case TextureViewUsage::sampled:
                break;
            case TextureViewUsage::storage:
                requiredSupport1 = D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW;
                requiredSupport2 = D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE;
                break;
            case TextureViewUsage::renderTarget:
                requiredSupport1 = D3D12_FORMAT_SUPPORT1_RENDER_TARGET;
                break;
            case TextureViewUsage::depthStencil:
                requiredSupport1 = D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL;
                break;
            }

            D3D12_FEATURE_DATA_FORMAT_SUPPORT support{
                .Format = format,
                .Support1 = {},
                .Support2 = {},
            };
            const HRESULT result = device.CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support));
            if (FAILED(result))
            {
                return std::unexpected(Error{
                    ErrorCode::nativeFailure,
                    static_cast<std::int64_t>(result),
                    "ID3D12Device::CheckFeatureSupport failed while validating a texture view format.",
                });
            }
            const bool supportsSampledView =
                usage != TextureViewUsage::sampled || format::supportsSampledUsage(support.Support1);
            if (!supportsSampledView || (support.Support1 & requiredSupport1) != requiredSupport1 ||
                (support.Support2 & requiredSupport2) != requiredSupport2)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The selected texture view format does not support the requested view usage.",
                });
            }
            return {};
        }

        [[nodiscard]] Status requireValidComponentMapping(const TextureViewDesc& desc)
        {
            if (!validComponentMapping(desc.components))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The texture view contains an invalid component swizzle.",
                });
            }
            if (desc.usage != TextureViewUsage::sampled && desc.components != identityComponentMapping)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Non-identity component mappings require a sampled texture view.",
                });
            }
            return {};
        }

        [[nodiscard]] Status requireViewAllowedByTexture(const TextureViewSource& source, const TextureViewDesc& desc)
        {
            if (!usageAllowed(source.desc.usage, desc.usage))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The texture view usage was not enabled on the texture.",
                });
            }
            if (!dimensionAllowed(source.desc.dimension, desc.dimension))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The texture view dimension is incompatible with the texture dimension.",
                });
            }
            return {};
        }

        [[nodiscard]] Result<Format> resolveSupportedViewFormat(ID3D12Device& device,
                                                                const TextureViewSource& source,
                                                                const TextureViewDesc& desc)
        {
            const Format resolvedFormat = desc.format == Format::unknown ? source.desc.format : desc.format;
            if (resolvedFormat != source.desc.format)
            {
                const auto declared = std::ranges::find(source.desc.viewFormats, resolvedFormat);
                if (declared == source.desc.viewFormats.end() ||
                    !isTextureViewFormatCompatible(source.desc.format, resolvedFormat))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "The texture view format was not declared as a compatible alternate when the texture was "
                        "created.",
                    });
                }
            }
            const auto nativeFormat = nativeViewFormat(resolvedFormat, desc.usage, desc.aspect);
            if (nativeFormat == DXGI_FORMAT_UNKNOWN)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The texture view format, usage, and aspect combination is invalid.",
                });
            }
            auto formatSupported = requireViewFormatSupport(device, nativeFormat, desc.usage);
            if (!formatSupported)
            {
                return std::unexpected(std::move(formatSupported.error()));
            }
            return resolvedFormat;
        }

        [[nodiscard]] Status requireFormatAllowedForUsage(const TextureViewDesc& desc, Format resolvedFormat)
        {
            if ((desc.usage == TextureViewUsage::storage || desc.usage == TextureViewUsage::renderTarget) &&
                (isDepthFormat(resolvedFormat) || isCompressed(resolvedFormat)))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Storage and render-target views require an uncompressed color format.",
                });
            }
            if (desc.usage == TextureViewUsage::depthStencil && !isDepthFormat(resolvedFormat))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A depth-stencil view requires a depth format.",
                });
            }
            return {};
        }

        [[nodiscard]] Status requireDimensionAllowedForUsage(const TextureViewSource& source,
                                                             const TextureViewDesc& desc)
        {
            const bool cubeView =
                desc.dimension == TextureViewDimension::cube || desc.dimension == TextureViewDimension::cubeArray;
            if (cubeView && (desc.usage != TextureViewUsage::sampled || source.desc.width != source.desc.height))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Cube views require a sampled square 2D texture.",
                });
            }
            if (desc.usage == TextureViewUsage::depthStencil && desc.dimension == TextureViewDimension::texture3D)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "D3D12 does not support 3D depth-stencil views.",
                });
            }
            const bool multisampledTexture = source.desc.sampleCount != 1;
            const bool attachmentView =
                desc.usage != TextureViewUsage::sampled && desc.usage != TextureViewUsage::storage;
            const bool twoDimensionalView = desc.dimension == TextureViewDimension::texture2D ||
                                            desc.dimension == TextureViewDimension::texture2DArray;
            const bool twoDimensionalAttachmentView = attachmentView && twoDimensionalView;
            if (multisampledTexture && !twoDimensionalAttachmentView)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A multisampled attachment requires a 2D or 2D-array attachment view.",
                });
            }
            return {};
        }

        [[nodiscard]] Result<std::uint32_t> resolveMipCount(const TextureViewSource& source,
                                                            const TextureViewDesc& desc)
        {
            const bool singleMip = desc.usage != TextureViewUsage::sampled;
            std::uint32_t mipCount = desc.mipLevelCount;
            if (mipCount == remainingSubresources)
            {
                mipCount = singleMip ? 1U : source.desc.mipLevels - std::min(desc.firstMipLevel, source.desc.mipLevels);
            }

            const bool mipRangeEmptyOrOutsideTexture = desc.firstMipLevel >= source.desc.mipLevels || mipCount == 0 ||
                                                       mipCount > source.desc.mipLevels - desc.firstMipLevel;
            const bool multipleMipsForSingleMipUsage = singleMip && mipCount != 1;
            if (mipRangeEmptyOrOutsideTexture || multipleMipsForSingleMipUsage)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The texture view mip range is invalid for its usage.",
                });
            }
            return mipCount;
        }

        [[nodiscard]] Result<std::uint32_t> resolveArrayCount(const TextureViewSource& source,
                                                              const TextureViewDesc& desc)
        {
            const std::uint32_t textureArraySize =
                source.desc.dimension == TextureDimension::texture3D ? 1U : source.desc.depthOrArrayLayers;
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
            const std::uint32_t arrayCount =
                desc.arrayLayerCount == remainingSubresources ? defaultArrayCount : desc.arrayLayerCount;

            const bool layerRangeEmptyOrOutsideTexture = desc.firstArrayLayer >= textureArraySize || arrayCount == 0 ||
                                                         arrayCount > textureArraySize - desc.firstArrayLayer;
            const bool singleLayerViewOfMultipleLayers = singleLayer && (textureArraySize != 1 || arrayCount != 1);
            const bool incompleteCube =
                desc.dimension == TextureViewDimension::cube && (desc.firstArrayLayer != 0 || arrayCount != 6);
            const bool misalignedCubeArray = desc.dimension == TextureViewDimension::cubeArray &&
                                             (desc.firstArrayLayer % 6 != 0 || arrayCount % 6 != 0);
            const bool layerOffsetIntoVolume =
                source.desc.dimension == TextureDimension::texture3D && desc.firstArrayLayer != 0;
            if (layerRangeEmptyOrOutsideTexture || singleLayerViewOfMultipleLayers || incompleteCube ||
                misalignedCubeArray || layerOffsetIntoVolume)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The texture view array or cube-face range is invalid.",
                });
            }
            return arrayCount;
        }

        [[nodiscard]] Result<ResolvedViewDesc> resolveViewDesc(ID3D12Device& device,
                                                               const TextureViewSource& source,
                                                               const TextureViewDesc& desc)
        {
            auto componentsValid = requireValidComponentMapping(desc);
            if (!componentsValid)
            {
                return std::unexpected(std::move(componentsValid.error()));
            }
            auto allowedByTexture = requireViewAllowedByTexture(source, desc);
            if (!allowedByTexture)
            {
                return std::unexpected(std::move(allowedByTexture.error()));
            }
            auto resolvedFormat = resolveSupportedViewFormat(device, source, desc);
            if (!resolvedFormat)
            {
                return std::unexpected(std::move(resolvedFormat.error()));
            }
            auto formatAllowedForUsage = requireFormatAllowedForUsage(desc, *resolvedFormat);
            if (!formatAllowedForUsage)
            {
                return std::unexpected(std::move(formatAllowedForUsage.error()));
            }
            auto dimensionAllowedForUsage = requireDimensionAllowedForUsage(source, desc);
            if (!dimensionAllowedForUsage)
            {
                return std::unexpected(std::move(dimensionAllowedForUsage.error()));
            }
            auto mipCount = resolveMipCount(source, desc);
            if (!mipCount)
            {
                return std::unexpected(std::move(mipCount.error()));
            }
            auto arrayCount = resolveArrayCount(source, desc);
            if (!arrayCount)
            {
                return std::unexpected(std::move(arrayCount.error()));
            }

            return ResolvedViewDesc{desc, *resolvedFormat, *mipCount, *arrayCount};
        }

        struct TextureViewRecord final
        {
            TextureViewSource source;
            TextureViewDesc desc;
            std::string label;
            std::shared_ptr<CpuDescriptorLease> allocation;
            bool externallyOwned = false;
        };

        void createSampledView(ID3D12Device& device,
                               const TextureViewSource& source,
                               const ResolvedViewDesc& view,
                               D3D12_CPU_DESCRIPTOR_HANDLE destination)
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC native{};
            native.Format = nativeViewFormat(view.format, view.desc.usage, view.desc.aspect);
            native.Shader4ComponentMapping = nativeComponentMapping(view.desc.components);
            switch (view.desc.dimension)
            {
            case TextureViewDimension::texture1D:
                native.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
                native.Texture1D = {view.desc.firstMipLevel, view.mipCount, 0.0F};
                break;
            case TextureViewDimension::texture1DArray:
                native.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
                native.Texture1DArray = {
                    view.desc.firstMipLevel,
                    view.mipCount,
                    view.desc.firstArrayLayer,
                    view.arrayCount,
                    0.0F,
                };
                break;
            case TextureViewDimension::texture2D:
                native.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                native.Texture2D = {
                    view.desc.firstMipLevel,
                    view.mipCount,
                    planeSlice(view.format, view.desc.aspect),
                    0.0F,
                };
                break;
            case TextureViewDimension::texture2DArray:
                native.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                native.Texture2DArray = {
                    view.desc.firstMipLevel,
                    view.mipCount,
                    view.desc.firstArrayLayer,
                    view.arrayCount,
                    planeSlice(view.format, view.desc.aspect),
                    0.0F,
                };
                break;
            case TextureViewDimension::texture3D:
                native.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
                native.Texture3D = {view.desc.firstMipLevel, view.mipCount, 0.0F};
                break;
            case TextureViewDimension::cube:
                native.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
                native.TextureCube = {view.desc.firstMipLevel, view.mipCount, 0.0F};
                break;
            case TextureViewDimension::cubeArray:
                native.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
                native.TextureCubeArray = {
                    view.desc.firstMipLevel,
                    view.mipCount,
                    view.desc.firstArrayLayer,
                    view.arrayCount / 6U,
                    0.0F,
                };
                break;
            }
            device.CreateShaderResourceView(source.resource, &native, destination);
        }

        void createStorageView(ID3D12Device& device,
                               const TextureViewSource& source,
                               const ResolvedViewDesc& view,
                               D3D12_CPU_DESCRIPTOR_HANDLE destination)
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC native{};
            native.Format = nativeViewFormat(view.format, view.desc.usage, view.desc.aspect);
            switch (view.desc.dimension)
            {
            case TextureViewDimension::texture1D:
                native.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
                native.Texture1D.MipSlice = view.desc.firstMipLevel;
                break;
            case TextureViewDimension::texture1DArray:
                native.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
                native.Texture1DArray = {
                    view.desc.firstMipLevel,
                    view.desc.firstArrayLayer,
                    view.arrayCount,
                };
                break;
            case TextureViewDimension::texture2D:
                native.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
                native.Texture2D = {view.desc.firstMipLevel, 0};
                break;
            case TextureViewDimension::texture2DArray:
                native.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                native.Texture2DArray = {
                    view.desc.firstMipLevel,
                    view.desc.firstArrayLayer,
                    view.arrayCount,
                    0,
                };
                break;
            case TextureViewDimension::texture3D:
                native.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
                native.Texture3D = {
                    view.desc.firstMipLevel,
                    0,
                    std::max(1U, source.desc.depthOrArrayLayers >> view.desc.firstMipLevel),
                };
                break;
            default:
                break;
            }
            device.CreateUnorderedAccessView(source.resource, nullptr, &native, destination);
        }

        void createRenderTargetView(ID3D12Device& device,
                                    const TextureViewSource& source,
                                    const ResolvedViewDesc& view,
                                    D3D12_CPU_DESCRIPTOR_HANDLE destination)
        {
            D3D12_RENDER_TARGET_VIEW_DESC native{};
            native.Format = nativeViewFormat(view.format, view.desc.usage, view.desc.aspect);
            switch (view.desc.dimension)
            {
            case TextureViewDimension::texture1D:
                native.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE1D;
                native.Texture1D.MipSlice = view.desc.firstMipLevel;
                break;
            case TextureViewDimension::texture1DArray:
                native.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE1DARRAY;
                native.Texture1DArray = {
                    view.desc.firstMipLevel,
                    view.desc.firstArrayLayer,
                    view.arrayCount,
                };
                break;
            case TextureViewDimension::texture2D:
                if (source.desc.sampleCount == 1)
                {
                    native.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
                    native.Texture2D = {view.desc.firstMipLevel, 0};
                }
                else
                {
                    native.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
                }
                break;
            case TextureViewDimension::texture2DArray:
                if (source.desc.sampleCount == 1)
                {
                    native.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
                    native.Texture2DArray = {
                        view.desc.firstMipLevel,
                        view.desc.firstArrayLayer,
                        view.arrayCount,
                        0,
                    };
                }
                else
                {
                    native.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY;
                    native.Texture2DMSArray = {view.desc.firstArrayLayer, view.arrayCount};
                }
                break;
            case TextureViewDimension::texture3D:
                native.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE3D;
                native.Texture3D = {
                    view.desc.firstMipLevel,
                    0,
                    std::max(1U, source.desc.depthOrArrayLayers >> view.desc.firstMipLevel),
                };
                break;
            default:
                break;
            }
            device.CreateRenderTargetView(source.resource, &native, destination);
        }

        void createDepthStencilView(ID3D12Device& device,
                                    const TextureViewSource& source,
                                    const ResolvedViewDesc& view,
                                    D3D12_CPU_DESCRIPTOR_HANDLE destination)
        {
            D3D12_DEPTH_STENCIL_VIEW_DESC native{};
            native.Format = nativeViewFormat(view.format, view.desc.usage, view.desc.aspect);
            native.Flags = D3D12_DSV_FLAG_NONE;
            switch (view.desc.dimension)
            {
            case TextureViewDimension::texture1D:
                native.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE1D;
                native.Texture1D.MipSlice = view.desc.firstMipLevel;
                break;
            case TextureViewDimension::texture1DArray:
                native.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE1DARRAY;
                native.Texture1DArray = {
                    view.desc.firstMipLevel,
                    view.desc.firstArrayLayer,
                    view.arrayCount,
                };
                break;
            case TextureViewDimension::texture2D:
                if (source.desc.sampleCount == 1)
                {
                    native.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
                    native.Texture2D.MipSlice = view.desc.firstMipLevel;
                }
                else
                {
                    native.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMS;
                }
                break;
            case TextureViewDimension::texture2DArray:
                if (source.desc.sampleCount == 1)
                {
                    native.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
                    native.Texture2DArray = {
                        view.desc.firstMipLevel,
                        view.desc.firstArrayLayer,
                        view.arrayCount,
                    };
                }
                else
                {
                    native.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY;
                    native.Texture2DMSArray = {view.desc.firstArrayLayer, view.arrayCount};
                }
                break;
            default:
                break;
            }
            device.CreateDepthStencilView(source.resource, &native, destination);
        }

    } // namespace

    struct ViewManager::Impl final
    {
        ID3D12Device* device = nullptr;
        std::mutex mutex;
        CpuDescriptorPool resourceDescriptors{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV};
        CpuDescriptorPool renderTargetDescriptors{D3D12_DESCRIPTOR_HEAP_TYPE_RTV};
        CpuDescriptorPool depthStencilDescriptors{D3D12_DESCRIPTOR_HEAP_TYPE_DSV};
        detail::HandleTable<std::shared_ptr<TextureViewRecord>, TextureViewHandle> views;

        [[nodiscard]] CpuDescriptorPool& pool(TextureViewUsage usage) noexcept
        {
            switch (usage)
            {
            case TextureViewUsage::renderTarget:
                return renderTargetDescriptors;
            case TextureViewUsage::depthStencil:
                return depthStencilDescriptors;
            default:
                return resourceDescriptors;
            }
        }
    };

    ViewManager::ViewManager() : impl_(std::make_unique<Impl>())
    {
    }

    ViewManager::~ViewManager() = default;

    Status ViewManager::initialize(ID3D12Device& device)
    {
        impl_->device = &device;
        return {};
    }

    DescriptorAllocationStatistics ViewManager::descriptorAllocationStatistics() const noexcept
    {
        auto result = impl_->resourceDescriptors.statistics();
        result += impl_->renderTargetDescriptors.statistics();
        result += impl_->depthStencilDescriptors.statistics();
        return result;
    }

    Result<TextureViewHandle> ViewManager::createTextureView(ResourceManager& resources,
                                                             const TextureViewDesc& desc,
                                                             bool externallyOwned)
    {
        auto source = resources.textureViewSource(desc.texture);
        if (!source)
        {
            return std::unexpected(std::move(source.error()));
        }
        auto resolved = resolveViewDesc(*impl_->device, *source, desc);
        if (!resolved)
        {
            return std::unexpected(std::move(resolved.error()));
        }

        const std::scoped_lock lock(impl_->mutex);
        auto allocation = impl_->pool(desc.usage).allocate(*impl_->device);
        if (!allocation)
        {
            return std::unexpected(std::move(allocation.error()));
        }

        switch (desc.usage)
        {
        case TextureViewUsage::sampled:
            createSampledView(*impl_->device, *source, *resolved, (*allocation)->handle());
            break;
        case TextureViewUsage::storage:
            createStorageView(*impl_->device, *source, *resolved, (*allocation)->handle());
            break;
        case TextureViewUsage::renderTarget:
            createRenderTargetView(*impl_->device, *source, *resolved, (*allocation)->handle());
            break;
        case TextureViewUsage::depthStencil:
            createDepthStencilView(*impl_->device, *source, *resolved, (*allocation)->handle());
            break;
        }

        auto record = std::make_shared<TextureViewRecord>();
        record->source = std::move(*source);
        record->desc = desc;
        record->label = desc.label;
        record->desc.label = record->label;
        record->allocation = std::move(*allocation);
        record->externallyOwned = externallyOwned;
        return impl_->views.emplace(std::move(record));
    }

    Status ViewManager::destroyTextureView(TextureViewHandle view, bool externallyOwned)
    {
        const std::scoped_lock lock(impl_->mutex);
        auto* stored = impl_->views.find(view);
        if (stored == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The texture view handle is invalid or stale.",
            });
        }
        if ((*stored)->externallyOwned != externallyOwned)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                externallyOwned ? "The texture view is owned by the application."
                                : "The texture view is borrowed from a swapchain.",
            });
        }
        static_cast<void>(impl_->views.erase(view));
        return {};
    }

    bool ViewManager::hasApplicationTextureViews(TextureHandle texture) const
    {
        const std::scoped_lock lock(impl_->mutex);
        bool found = false;
        impl_->views.forEach([&](TextureViewHandle, const std::shared_ptr<TextureViewRecord>& record) {
            found = found || (!record->externallyOwned && record->desc.texture == texture);
        });
        return found;
    }

    Result<NativeTextureView> ViewManager::nativeTextureView(TextureViewHandle view)
    {
        const std::scoped_lock lock(impl_->mutex);
        const auto* stored = impl_->views.find(view);
        if (stored == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The texture view handle is invalid or stale.",
            });
        }
        return NativeTextureView{
            .owner = *stored,
            .texture = (*stored)->source,
            .descriptor = (*stored)->allocation->handle(),
            .viewDesc = (*stored)->desc,
            .usage = (*stored)->desc.usage,
        };
    }

} // namespace rachis::d3d12
