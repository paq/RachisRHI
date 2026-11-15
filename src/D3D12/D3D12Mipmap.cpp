#include "D3D12/D3D12Mipmap.h"

#include "D3D12/D3D12NativeHelpers.h"

#include "RachisGenerateMipmaps.h"
#include "RachisGenerateMipmaps1D.h"
#include "RachisGenerateMipmaps3D.h"

#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rachis::d3d12
{
    namespace
    {

        using Microsoft::WRL::ComPtr;

        struct ThreadGroupSize final
        {
            std::uint32_t width = 1;
            std::uint32_t height = 1;
            std::uint32_t depth = 1;
        };

        [[nodiscard]] constexpr std::size_t dimensionIndex(TextureDimension dimension) noexcept
        {
            return static_cast<std::size_t>(dimension);
        }

        [[nodiscard]] constexpr ThreadGroupSize threadGroupSize(TextureDimension dimension) noexcept
        {
            switch (dimension)
            {
            case TextureDimension::texture1D:
                return {64, 1, 1};
            case TextureDimension::texture2D:
                return {8, 8, 1};
            case TextureDimension::texture3D:
                return {4, 4, 4};
            }
            return {};
        }

        struct BarrierScope final
        {
            PipelineStage stages = PipelineStage::none;
            Access access = Access::none;
        };

        [[nodiscard]] constexpr bool layoutCompatible(TextureUsage usage, TextureLayout layout) noexcept
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

        [[nodiscard]] constexpr PipelineStage shaderReadStages(QueueType queueType) noexcept
        {
            constexpr auto computeQueueStages = PipelineStage::computeShader | PipelineStage::rayTracingShader;
            if (queueType != QueueType::graphics)
            {
                return computeQueueStages;
            }
            return computeQueueStages | PipelineStage::vertexShader | PipelineStage::pixelShader |
                   PipelineStage::geometryShader | PipelineStage::hullShader | PipelineStage::domainShader |
                   PipelineStage::taskShader | PipelineStage::meshShader;
        }

        [[nodiscard]] constexpr std::optional<BarrierScope> barrierScope(TextureLayout layout,
                                                                         QueueType queueType) noexcept
        {
            switch (layout)
            {
            case TextureLayout::common:
                return BarrierScope{PipelineStage::all, Access::common};
            case TextureLayout::shaderResource:
                return BarrierScope{shaderReadStages(queueType), Access::shaderResource};
            case TextureLayout::unorderedAccess:
                return BarrierScope{PipelineStage::computeShader, Access::unorderedAccess};
            case TextureLayout::renderTarget:
                return BarrierScope{PipelineStage::renderTarget, Access::renderTarget};
            case TextureLayout::copySource:
                return BarrierScope{PipelineStage::copy, Access::copySource};
            case TextureLayout::copyDestination:
                return BarrierScope{PipelineStage::copy, Access::copyDestination};
            case TextureLayout::resolveSource:
                return BarrierScope{PipelineStage::resolve, Access::resolveSource};
            case TextureLayout::resolveDestination:
                return BarrierScope{PipelineStage::resolve, Access::resolveDestination};
            default:
                return std::nullopt;
            }
        }

        [[nodiscard]] Result<std::uint32_t> selectedCount(std::uint32_t first,
                                                          std::uint32_t requested,
                                                          std::uint32_t available,
                                                          std::string_view subject)
        {
            if (first >= available)
            {
                return std::unexpected(
                    Error{ErrorCode::invalidArgument, 0, std::string(subject) + " starts out of range."});
            }
            const std::uint32_t count = requested == remainingSubresources ? available - first : requested;
            if (count == 0 || count > available - first)
            {
                return std::unexpected(
                    Error{ErrorCode::invalidArgument, 0, std::string(subject) + " has an invalid count."});
            }
            return count;
        }

        [[nodiscard]] TextureBarrier textureBarrier(TextureHandle texture,
                                                    TextureSubresourceRange range,
                                                    TextureLayout before,
                                                    BarrierScope beforeScope,
                                                    TextureLayout after,
                                                    BarrierScope afterScope)
        {
            return TextureBarrier{
                .texture = texture,
                .stagesBefore = beforeScope.stages,
                .stagesAfter = afterScope.stages,
                .accessBefore = beforeScope.access,
                .accessAfter = afterScope.access,
                .layoutBefore = before,
                .layoutAfter = after,
                .subresources = range,
            };
        }

        struct MipmapFormats final
        {
            DXGI_FORMAT sampled = DXGI_FORMAT_UNKNOWN;
            DXGI_FORMAT storage = DXGI_FORMAT_UNKNOWN;
            bool encodeSrgb = false;
        };

        [[nodiscard]] constexpr Format linearViewFormat(Format format) noexcept
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

        [[nodiscard]] constexpr DXGI_FORMAT nativeLinearViewFormat(Format format) noexcept
        {
            switch (format)
            {
            case Format::rgba8Unorm:
                return DXGI_FORMAT_R8G8B8A8_UNORM;
            case Format::bgra8Unorm:
                return DXGI_FORMAT_B8G8R8A8_UNORM;
            default:
                return DXGI_FORMAT_UNKNOWN;
            }
        }

        [[nodiscard]] Result<MipmapFormats> mipmapFormats(const TextureViewSource& texture)
        {
            const auto linearFormat = linearViewFormat(texture.desc.format);
            if (linearFormat == Format::unknown)
            {
                return MipmapFormats{texture.format, texture.format, false};
            }
            if (std::ranges::find(texture.desc.viewFormats, linearFormat) == texture.desc.viewFormats.end())
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "sRGB mipmap generation requires the compatible linear format in TextureDesc::viewFormats.",
                });
            }
            return MipmapFormats{texture.format, nativeLinearViewFormat(linearFormat), true};
        }

        [[nodiscard]] Status validateFormats(ID3D12Device& device, const MipmapFormats& formats)
        {
            D3D12_FEATURE_DATA_FORMAT_SUPPORT sampledSupport{
                .Format = formats.sampled,
                .Support1 = {},
                .Support2 = {},
            };
            HRESULT result =
                device.CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &sampledSupport, sizeof(sampledSupport));
            if (FAILED(result))
            {
                return std::unexpected(
                    nativeError(ErrorCode::nativeFailure, result, "D3D12 mipmap sampled-format query"));
            }
            D3D12_FEATURE_DATA_FORMAT_SUPPORT storageSupport{
                .Format = formats.storage,
                .Support1 = {},
                .Support2 = {},
            };
            result = device.CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &storageSupport, sizeof(storageSupport));
            if (FAILED(result))
            {
                return std::unexpected(
                    nativeError(ErrorCode::nativeFailure, result, "D3D12 mipmap storage-format query"));
            }
            if ((sampledSupport.Support1 & D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE) == 0 ||
                (storageSupport.Support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW) == 0 ||
                (storageSupport.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE) == 0)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Mipmap generation requires filterable sampling and a compatible typed storage-write view.",
                });
            }
            return {};
        }

        [[nodiscard]] constexpr std::array<D3D12_DESCRIPTOR_RANGE, 2> mipViewDescriptorRanges() noexcept
        {
            return std::array{
                D3D12_DESCRIPTOR_RANGE{
                    .RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                    .NumDescriptors = 1,
                    .BaseShaderRegister = 0,
                    .RegisterSpace = 0,
                    .OffsetInDescriptorsFromTableStart = 0,
                },
                D3D12_DESCRIPTOR_RANGE{
                    .RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
                    .NumDescriptors = 1,
                    .BaseShaderRegister = 0,
                    .RegisterSpace = 0,
                    .OffsetInDescriptorsFromTableStart = 1,
                },
            };
        }

        [[nodiscard]] constexpr D3D12_STATIC_SAMPLER_DESC linearClampSampler() noexcept
        {
            return D3D12_STATIC_SAMPLER_DESC{
                .Filter = D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT,
                .AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                .AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                .AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                .MipLODBias = 0.0F,
                .MaxAnisotropy = 1,
                .ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS,
                .BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK,
                .MinLOD = 0.0F,
                .MaxLOD = D3D12_FLOAT32_MAX,
                .ShaderRegister = 0,
                .RegisterSpace = 0,
                .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL,
            };
        }

        [[nodiscard]] Result<ComPtr<ID3DBlob>> serializeMipmapRootSignature()
        {
            const auto ranges = mipViewDescriptorRanges();
            const D3D12_ROOT_PARAMETER descriptorTable{
                .ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                .DescriptorTable =
                    {
                        .NumDescriptorRanges = static_cast<UINT>(ranges.size()),
                        .pDescriptorRanges = ranges.data(),
                    },
                .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL,
            };
            const D3D12_ROOT_PARAMETER colorSpaceConstant{
                .ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                .Constants =
                    {
                        .ShaderRegister = 0,
                        .RegisterSpace = 0,
                        .Num32BitValues = 1,
                    },
                .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL,
            };
            const std::array parameters{descriptorTable, colorSpaceConstant};
            const auto sampler = linearClampSampler();
            const D3D12_ROOT_SIGNATURE_DESC rootDesc{
                .NumParameters = static_cast<UINT>(parameters.size()),
                .pParameters = parameters.data(),
                .NumStaticSamplers = 1,
                .pStaticSamplers = &sampler,
                .Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE,
            };

            ComPtr<ID3DBlob> serialized;
            ComPtr<ID3DBlob> diagnostics;
            const HRESULT serializeResult =
                D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &diagnostics);
            if (FAILED(serializeResult))
            {
                return std::unexpected(nativeError(
                    ErrorCode::validationError, serializeResult, "D3D12SerializeRootSignature(mipmap generation)"));
            }
            return serialized;
        }

        [[nodiscard]] Status validateCommandBufferReadyForMipmapGeneration(
            const detail::BackendCommandBuffer& commandBuffer)
        {
            if ((commandBuffer.queueType() != QueueType::graphics && commandBuffer.queueType() != QueueType::compute) ||
                commandBuffer.graphicsState().renderingOwner != nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Mipmap generation requires a graphics or compute command buffer outside a rendering scope.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateMipmapEligibility(const TextureDesc& desc, TextureAspect aspect)
        {
            if (desc.sampleCount != 1 || !hasAny(desc.usage, TextureUsage::sampled) ||
                !hasAny(desc.usage, TextureUsage::storage) || aspect != TextureAspect::color)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Mipmap generation requires a single-sampled color texture with sampled and storage usage.",
                });
            }
            return {};
        }

        [[nodiscard]] Result<TextureSubresourceRange> resolveMipChainRange(const TextureDesc& desc,
                                                                           const TextureSubresourceRange& requested)
        {
            auto mipCount =
                selectedCount(requested.firstMipLevel, requested.mipLevelCount, desc.mipLevels, "The mipmap range");
            const auto availableLayerCount =
                desc.dimension == TextureDimension::texture3D ? 1U : desc.depthOrArrayLayers;
            auto layerCount = selectedCount(
                requested.firstArrayLayer, requested.arrayLayerCount, availableLayerCount, "The mipmap layer range");
            if (!mipCount)
            {
                return std::unexpected(std::move(mipCount.error()));
            }
            if (!layerCount)
            {
                return std::unexpected(std::move(layerCount.error()));
            }
            if (*mipCount < 2)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument, 0, "Mipmap generation requires at least two selected mip levels."});
            }
            return TextureSubresourceRange{
                .firstMipLevel = requested.firstMipLevel,
                .mipLevelCount = *mipCount,
                .firstArrayLayer = requested.firstArrayLayer,
                .arrayLayerCount = *layerCount,
                .aspect = TextureAspect::color,
            };
        }

        struct BoundaryLayoutScopes final
        {
            BarrierScope baseLayout;
            BarrierScope finalLayout;
        };

        [[nodiscard]] Result<BoundaryLayoutScopes> resolveBoundaryLayoutScopes(
            TextureUsage usage, const TextureMipmapGeneration& generation, QueueType queueType)
        {
            const auto baseScope = barrierScope(generation.baseLayout, queueType);
            const auto finalScope = barrierScope(generation.finalLayout, queueType);
            if (!baseScope || !finalScope || !layoutCompatible(usage, generation.baseLayout) ||
                !layoutCompatible(usage, generation.finalLayout))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The mipmap seed or final layout is undefined or incompatible with the texture usage.",
                });
            }
            return BoundaryLayoutScopes{.baseLayout = *baseScope, .finalLayout = *finalScope};
        }

        struct MipChainPlan final
        {
            TextureSubresourceRange range;
            BoundaryLayoutScopes boundaryScopes;
            MipmapFormats formats;
        };

        [[nodiscard]] Result<MipChainPlan> planMipChain(ID3D12Device& device,
                                                        const TextureViewSource& texture,
                                                        const TextureMipmapGeneration& generation,
                                                        QueueType queueType)
        {
            auto eligible = validateMipmapEligibility(texture.desc, generation.subresources.aspect);
            if (!eligible)
            {
                return std::unexpected(std::move(eligible.error()));
            }
            auto range = resolveMipChainRange(texture.desc, generation.subresources);
            if (!range)
            {
                return std::unexpected(std::move(range.error()));
            }
            auto boundaryScopes = resolveBoundaryLayoutScopes(texture.desc.usage, generation, queueType);
            if (!boundaryScopes)
            {
                return std::unexpected(std::move(boundaryScopes.error()));
            }
            auto formats = mipmapFormats(texture);
            if (!formats)
            {
                return std::unexpected(std::move(formats.error()));
            }
            auto formatSupport = validateFormats(device, *formats);
            if (!formatSupport)
            {
                return std::unexpected(std::move(formatSupport.error()));
            }
            return MipChainPlan{.range = *range, .boundaryScopes = *boundaryScopes, .formats = *formats};
        }

        [[nodiscard]] TextureSubresourceRange singleMipRange(const TextureSubresourceRange& chain,
                                                             std::uint32_t mipLevel)
        {
            return TextureSubresourceRange{
                .firstMipLevel = mipLevel,
                .mipLevelCount = 1,
                .firstArrayLayer = chain.firstArrayLayer,
                .arrayLayerCount = chain.arrayLayerCount,
                .aspect = TextureAspect::color,
            };
        }

        [[nodiscard]] Result<std::vector<TextureBarrier>> orderedMipChainBarriers(
            const TextureMipmapGeneration& generation, const MipChainPlan& plan)
        {
            constexpr BarrierScope earlierWork{PipelineStage::all, Access::none};
            constexpr BarrierScope shaderRead{PipelineStage::computeShader, Access::shaderResource};
            constexpr BarrierScope shaderWrite{PipelineStage::computeShader, Access::unorderedAccess};
            const auto& chain = plan.range;
            std::vector<TextureBarrier> orderedBarriers;
            try
            {
                orderedBarriers.reserve((chain.mipLevelCount - 1U) * 2U + 2U);
                orderedBarriers.push_back(textureBarrier(generation.texture,
                                                         singleMipRange(chain, chain.firstMipLevel),
                                                         generation.baseLayout,
                                                         plan.boundaryScopes.baseLayout,
                                                         TextureLayout::shaderResource,
                                                         shaderRead));
                for (std::uint32_t mipOffset = 1; mipOffset < chain.mipLevelCount; ++mipOffset)
                {
                    const auto destinationRange = singleMipRange(chain, chain.firstMipLevel + mipOffset);
                    orderedBarriers.push_back(textureBarrier(generation.texture,
                                                             destinationRange,
                                                             TextureLayout::undefined,
                                                             earlierWork,
                                                             TextureLayout::unorderedAccess,
                                                             shaderWrite));
                    orderedBarriers.push_back(textureBarrier(generation.texture,
                                                             destinationRange,
                                                             TextureLayout::unorderedAccess,
                                                             shaderWrite,
                                                             TextureLayout::shaderResource,
                                                             shaderRead));
                }
                orderedBarriers.push_back(textureBarrier(generation.texture,
                                                         chain,
                                                         TextureLayout::shaderResource,
                                                         shaderRead,
                                                         generation.finalLayout,
                                                         plan.boundaryScopes.finalLayout));
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to prepare the ordered mipmap barrier program.",
                });
            }
            return orderedBarriers;
        }

        [[nodiscard]] Result<NativeTextureBarrierSequence> prepareMipChainBarriers(
            ResourceManager& resources,
            QueueManager& queues,
            detail::BackendCommandBuffer& commandBuffer,
            const TextureMipmapGeneration& generation,
            const MipChainPlan& plan)
        {
            auto orderedBarriers = orderedMipChainBarriers(generation, plan);
            if (!orderedBarriers)
            {
                return std::unexpected(std::move(orderedBarriers.error()));
            }
            return resources.prepareNativeTextureBarrierSequence(queues, commandBuffer, *orderedBarriers);
        }

        [[nodiscard]] D3D12_SHADER_RESOURCE_VIEW_DESC sourceMipViewDesc(const TextureDesc& desc,
                                                                        DXGI_FORMAT format,
                                                                        const TextureSubresourceRange& chain,
                                                                        std::uint32_t sourceMip)
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC sourceView{
                .Format = format,
                .ViewDimension = {},
                .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                .Buffer = {},
            };
            switch (desc.dimension)
            {
            case TextureDimension::texture1D:
                sourceView.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
                sourceView.Texture1DArray = {
                    .MostDetailedMip = sourceMip,
                    .MipLevels = 1,
                    .FirstArraySlice = chain.firstArrayLayer,
                    .ArraySize = chain.arrayLayerCount,
                    .ResourceMinLODClamp = 0.0F,
                };
                break;
            case TextureDimension::texture2D:
                sourceView.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                sourceView.Texture2DArray = {
                    .MostDetailedMip = sourceMip,
                    .MipLevels = 1,
                    .FirstArraySlice = chain.firstArrayLayer,
                    .ArraySize = chain.arrayLayerCount,
                    .PlaneSlice = 0,
                    .ResourceMinLODClamp = 0.0F,
                };
                break;
            case TextureDimension::texture3D:
                sourceView.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
                sourceView.Texture3D = {
                    .MostDetailedMip = sourceMip,
                    .MipLevels = 1,
                    .ResourceMinLODClamp = 0.0F,
                };
                break;
            }
            return sourceView;
        }

        [[nodiscard]] D3D12_UNORDERED_ACCESS_VIEW_DESC destinationMipViewDesc(const TextureDesc& desc,
                                                                              DXGI_FORMAT format,
                                                                              const TextureSubresourceRange& chain,
                                                                              std::uint32_t destinationMip)
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC destinationView{
                .Format = format,
                .ViewDimension = {},
                .Buffer = {},
            };
            switch (desc.dimension)
            {
            case TextureDimension::texture1D:
                destinationView.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
                destinationView.Texture1DArray = {
                    .MipSlice = destinationMip,
                    .FirstArraySlice = chain.firstArrayLayer,
                    .ArraySize = chain.arrayLayerCount,
                };
                break;
            case TextureDimension::texture2D:
                destinationView.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                destinationView.Texture2DArray = {
                    .MipSlice = destinationMip,
                    .FirstArraySlice = chain.firstArrayLayer,
                    .ArraySize = chain.arrayLayerCount,
                    .PlaneSlice = 0,
                };
                break;
            case TextureDimension::texture3D:
                destinationView.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
                destinationView.Texture3D = {
                    .MipSlice = destinationMip,
                    .FirstWSlice = 0,
                    .WSize = mipDimension(desc.depthOrArrayLayers, destinationMip),
                };
                break;
            }
            return destinationView;
        }

        [[nodiscard]] constexpr std::uint32_t mipViewPairOffset(std::uint32_t mipOffset) noexcept
        {
            return (mipOffset - 1U) * 2U;
        }

        [[nodiscard]] Result<std::shared_ptr<GpuDescriptorLease>> createMipViewDescriptors(
            BindingManager& bindings, ID3D12Device& device, const TextureViewSource& texture, const MipChainPlan& plan)
        {
            const auto descriptorCount = (plan.range.mipLevelCount - 1U) * 2U;
            auto descriptors = bindings.allocateResourceDescriptors(descriptorCount);
            if (!descriptors)
            {
                return std::unexpected(std::move(descriptors.error()));
            }

            for (std::uint32_t mipOffset = 1; mipOffset < plan.range.mipLevelCount; ++mipOffset)
            {
                const auto sourceMip = plan.range.firstMipLevel + mipOffset - 1U;
                const auto destinationMip = sourceMip + 1U;
                const auto descriptorOffset = mipViewPairOffset(mipOffset);
                const auto sourceView = sourceMipViewDesc(texture.desc, plan.formats.sampled, plan.range, sourceMip);
                const auto destinationView =
                    destinationMipViewDesc(texture.desc, plan.formats.storage, plan.range, destinationMip);
                device.CreateShaderResourceView(
                    texture.resource, &sourceView, (*descriptors)->cpuHandle(descriptorOffset));
                device.CreateUnorderedAccessView(
                    texture.resource, nullptr, &destinationView, (*descriptors)->cpuHandle(descriptorOffset + 1U));
            }
            return descriptors;
        }

        [[nodiscard]] Status retainMipmapDependencies(const QueueManager& queues,
                                                      detail::BackendCommandBuffer& commandBuffer,
                                                      const TextureViewSource& texture,
                                                      const std::shared_ptr<GpuDescriptorLease>& descriptors,
                                                      const std::optional<SparseBindingSnapshot>& sparseSnapshot)
        {
            std::array<QueueManager::ResourceRetention, 3> dependencies{
                QueueManager::ResourceRetention{
                    .owner = texture.owner,
                    .residency = texture.residency,
                },
                QueueManager::ResourceRetention{
                    .owner = descriptors,
                    .internal = true,
                },
            };
            std::size_t dependencyCount = 2;
            std::array<TimelinePoint, 1> sparseDependencies{};
            std::size_t sparseDependencyCount = 0;
            if (sparseSnapshot.has_value())
            {
                dependencies[dependencyCount++] = QueueManager::ResourceRetention{
                    .owner = sparseSnapshot->owner,
                    .internal = true,
                };
                if (sparseSnapshot->publication.isValid())
                {
                    sparseDependencies[sparseDependencyCount++] = sparseSnapshot->publication;
                }
            }
            return queues.retainCommandDependencies(commandBuffer,
                                                    std::span{dependencies}.first(dependencyCount),
                                                    {},
                                                    std::span{sparseDependencies}.first(sparseDependencyCount));
        }

        void dispatchMipLevel(ID3D12GraphicsCommandList& commandList,
                              const TextureDesc& desc,
                              std::uint32_t layerCount,
                              std::uint32_t destinationMip)
        {
            const auto width = mipDimension(desc.width, destinationMip);
            const auto height = mipDimension(desc.height, destinationMip);
            const auto depth = desc.dimension == TextureDimension::texture3D
                                   ? mipDimension(desc.depthOrArrayLayers, destinationMip)
                                   : layerCount;
            const auto groupSize = threadGroupSize(desc.dimension);
            commandList.Dispatch((width + groupSize.width - 1U) / groupSize.width,
                                 (height + groupSize.height - 1U) / groupSize.height,
                                 (depth + groupSize.depth - 1U) / groupSize.depth);
        }

    } // namespace

    struct MipmapGenerator::Impl final
    {
        ID3D12Device* device = nullptr;
        ComPtr<ID3D12RootSignature> rootSignature;
        std::array<ComPtr<ID3D12PipelineState>, 3> pipelineStates;

        [[nodiscard]] Status createRootSignature(ID3D12Device& nativeDevice);
        [[nodiscard]] Status createPipelineStates(ID3D12Device& nativeDevice);
        [[nodiscard]] bool initialized() const;
        void bindPipeline(ID3D12GraphicsCommandList& commandList,
                          const NativeDescriptorHeaps& heaps,
                          TextureDimension dimension,
                          bool encodeSrgb) const;
    };

    MipmapGenerator::MipmapGenerator() : impl_(std::make_unique<Impl>())
    {
    }

    MipmapGenerator::~MipmapGenerator() = default;

    Status MipmapGenerator::Impl::createRootSignature(ID3D12Device& nativeDevice)
    {
        auto serialized = serializeMipmapRootSignature();
        if (!serialized)
        {
            return std::unexpected(std::move(serialized.error()));
        }
        const HRESULT result = nativeDevice.CreateRootSignature(
            0, (*serialized)->GetBufferPointer(), (*serialized)->GetBufferSize(), IID_PPV_ARGS(&rootSignature));
        if (FAILED(result))
        {
            return std::unexpected(
                nativeError(result == E_OUTOFMEMORY ? ErrorCode::outOfMemory : ErrorCode::nativeFailure,
                            result,
                            "ID3D12Device::CreateRootSignature(mipmap generation)"));
        }
        return {};
    }

    Status MipmapGenerator::Impl::createPipelineStates(ID3D12Device& nativeDevice)
    {
        struct ShaderProgram final
        {
            const void* bytecode = nullptr;
            std::size_t size = 0;
            const wchar_t* label = nullptr;
        };
        const std::array shaders{
            ShaderProgram{g_rachisGenerateMipmaps1DDxil,
                          sizeof(g_rachisGenerateMipmaps1DDxil),
                          L"RHI 1D mipmap generation pipeline"},
            ShaderProgram{
                g_rachisGenerateMipmapsDxil, sizeof(g_rachisGenerateMipmapsDxil), L"RHI 2D mipmap generation pipeline"},
            ShaderProgram{g_rachisGenerateMipmaps3DDxil,
                          sizeof(g_rachisGenerateMipmaps3DDxil),
                          L"RHI 3D mipmap generation pipeline"},
        };
        for (std::size_t index = 0; index < shaders.size(); ++index)
        {
            const D3D12_COMPUTE_PIPELINE_STATE_DESC pipelineDesc{
                .pRootSignature = rootSignature.Get(),
                .CS = {.pShaderBytecode = shaders[index].bytecode, .BytecodeLength = shaders[index].size},
                .NodeMask = 0,
                .CachedPSO = {},
                .Flags = D3D12_PIPELINE_STATE_FLAG_NONE,
            };
            const HRESULT result =
                nativeDevice.CreateComputePipelineState(&pipelineDesc, IID_PPV_ARGS(&pipelineStates[index]));
            if (FAILED(result))
            {
                return std::unexpected(
                    nativeError(result == E_OUTOFMEMORY ? ErrorCode::outOfMemory : ErrorCode::nativeFailure,
                                result,
                                "ID3D12Device::CreateComputePipelineState(mipmap generation)"));
            }
            static_cast<void>(pipelineStates[index]->SetName(shaders[index].label));
        }
        return {};
    }

    Status MipmapGenerator::initialize(ID3D12Device& device)
    {
        auto rootSignatureCreated = impl_->createRootSignature(device);
        if (!rootSignatureCreated)
        {
            return rootSignatureCreated;
        }
        auto pipelineStatesCreated = impl_->createPipelineStates(device);
        if (!pipelineStatesCreated)
        {
            return pipelineStatesCreated;
        }
        static_cast<void>(impl_->rootSignature->SetName(L"RHI mipmap generation root signature"));
        impl_->device = &device;
        return {};
    }

    bool MipmapGenerator::Impl::initialized() const
    {
        return device != nullptr && rootSignature != nullptr &&
               std::ranges::none_of(pipelineStates, [](const auto& pipeline) { return pipeline == nullptr; });
    }

    void MipmapGenerator::Impl::bindPipeline(ID3D12GraphicsCommandList& commandList,
                                             const NativeDescriptorHeaps& heaps,
                                             TextureDimension dimension,
                                             bool encodeSrgb) const
    {
        ID3D12DescriptorHeap* const nativeHeaps[]{heaps.resources, heaps.samplers};
        commandList.SetDescriptorHeaps(static_cast<UINT>(std::size(nativeHeaps)), nativeHeaps);
        commandList.SetComputeRootSignature(rootSignature.Get());
        commandList.SetPipelineState(pipelineStates[dimensionIndex(dimension)].Get());
        commandList.SetComputeRoot32BitConstant(1, encodeSrgb ? 1U : 0U, 0);
    }

    Status MipmapGenerator::generate(QueueManager& queues,
                                     ResourceManager& resources,
                                     BindingManager& bindings,
                                     detail::BackendCommandBuffer& commandBuffer,
                                     const TextureMipmapGeneration& generation)
    {
        if (!impl_->initialized())
        {
            return std::unexpected(
                Error{ErrorCode::unsupported, 0, "Mipmap generation is unavailable on this device."});
        }
        auto readyForMipmapGeneration = validateCommandBufferReadyForMipmapGeneration(commandBuffer);
        if (!readyForMipmapGeneration)
        {
            return readyForMipmapGeneration;
        }

        auto texture = resources.textureViewSource(generation.texture);
        if (!texture)
        {
            return std::unexpected(std::move(texture.error()));
        }
        auto plan = planMipChain(*impl_->device, *texture, generation, commandBuffer.queueType());
        if (!plan)
        {
            return std::unexpected(std::move(plan.error()));
        }
        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto nativeBarriers = prepareMipChainBarriers(resources, queues, commandBuffer, generation, *plan);
        if (!nativeBarriers)
        {
            return std::unexpected(std::move(nativeBarriers.error()));
        }
        auto descriptors = createMipViewDescriptors(bindings, *impl_->device, *texture, *plan);
        if (!descriptors)
        {
            return std::unexpected(std::move(descriptors.error()));
        }
        auto sparseSnapshot = resources.captureSparseTextureBinding(*texture);
        if (!sparseSnapshot)
        {
            return std::unexpected(std::move(sparseSnapshot.error()));
        }
        auto retained = retainMipmapDependencies(queues, commandBuffer, *texture, *descriptors, *sparseSnapshot);
        if (!retained)
        {
            return retained;
        }

        const auto& desc = texture->desc;
        const auto& chain = plan->range;
        std::size_t barrierIndex = 0;
        nativeBarriers->record(barrierIndex++, **commandList);

        commandBuffer.computeState() = {};
        commandBuffer.rayTracingState() = {};
        impl_->bindPipeline(**commandList, bindings.nativeDescriptorHeaps(), desc.dimension, plan->formats.encodeSrgb);

        for (std::uint32_t mipOffset = 1; mipOffset < chain.mipLevelCount; ++mipOffset)
        {
            nativeBarriers->record(barrierIndex++, **commandList);
            (*commandList)->SetComputeRootDescriptorTable(0, (*descriptors)->gpuHandle(mipViewPairOffset(mipOffset)));
            dispatchMipLevel(**commandList, desc, chain.arrayLayerCount, chain.firstMipLevel + mipOffset);
            nativeBarriers->record(barrierIndex++, **commandList);
        }

        nativeBarriers->record(barrierIndex++, **commandList);
        assert(barrierIndex == nativeBarriers->size());
        return {};
    }
} // namespace rachis::d3d12
