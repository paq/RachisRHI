#include "D3D12/D3D12ResourceInternal.h"

#include "D3D12/D3D12BarrierTranslation.h"

#include <map>
#include <optional>

namespace rachis::d3d12
{

    using namespace resource_detail;

    namespace
    {

        [[nodiscard]] D3D12_BARRIER_SYNC toNativeBarrierSync(EnhancedBarrierSync sync) noexcept
        {
            if (hasAny(sync, EnhancedBarrierSync::all))
            {
                return D3D12_BARRIER_SYNC_ALL;
            }

            D3D12_BARRIER_SYNC result = D3D12_BARRIER_SYNC_NONE;
            if (hasAny(sync, EnhancedBarrierSync::draw))
            {
                result |= D3D12_BARRIER_SYNC_DRAW;
            }
            if (hasAny(sync, EnhancedBarrierSync::indexInput))
            {
                result |= D3D12_BARRIER_SYNC_INDEX_INPUT;
            }
            if (hasAny(sync, EnhancedBarrierSync::vertexShading))
            {
                result |= D3D12_BARRIER_SYNC_VERTEX_SHADING;
            }
            if (hasAny(sync, EnhancedBarrierSync::pixelShading))
            {
                result |= D3D12_BARRIER_SYNC_PIXEL_SHADING;
            }
            if (hasAny(sync, EnhancedBarrierSync::depthStencil))
            {
                result |= D3D12_BARRIER_SYNC_DEPTH_STENCIL;
            }
            if (hasAny(sync, EnhancedBarrierSync::renderTarget))
            {
                result |= D3D12_BARRIER_SYNC_RENDER_TARGET;
            }
            if (hasAny(sync, EnhancedBarrierSync::computeShading))
            {
                result |= D3D12_BARRIER_SYNC_COMPUTE_SHADING;
            }
            if (hasAny(sync, EnhancedBarrierSync::raytracing))
            {
                result |= D3D12_BARRIER_SYNC_RAYTRACING;
            }
            if (hasAny(sync, EnhancedBarrierSync::copy))
            {
                result |= D3D12_BARRIER_SYNC_COPY;
            }
            if (hasAny(sync, EnhancedBarrierSync::resolve))
            {
                result |= D3D12_BARRIER_SYNC_RESOLVE;
            }
            if (hasAny(sync, EnhancedBarrierSync::executeIndirect))
            {
                result |= D3D12_BARRIER_SYNC_EXECUTE_INDIRECT;
            }
            if (hasAny(sync, EnhancedBarrierSync::predication))
            {
                result |= D3D12_BARRIER_SYNC_PREDICATION;
            }
            if (hasAny(sync, EnhancedBarrierSync::buildRaytracingAccelerationStructure))
            {
                result |= D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE;
            }
            if (hasAny(sync, EnhancedBarrierSync::copyRaytracingAccelerationStructure))
            {
                result |= D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE;
            }
            if (hasAny(sync, EnhancedBarrierSync::clearUnorderedAccessView))
            {
                result |= D3D12_BARRIER_SYNC_CLEAR_UNORDERED_ACCESS_VIEW;
            }
            return result;
        }

        [[nodiscard]] D3D12_BARRIER_SYNC nativeSync(PipelineStage stages, Access access) noexcept
        {
            return toNativeBarrierSync(translateEnhancedSync(stages, access));
        }

        [[nodiscard]] D3D12_BARRIER_ACCESS toNativeBarrierAccess(EnhancedBarrierAccess access) noexcept
        {
            if (hasAny(access, EnhancedBarrierAccess::noAccess))
            {
                return D3D12_BARRIER_ACCESS_NO_ACCESS;
            }

            D3D12_BARRIER_ACCESS result = D3D12_BARRIER_ACCESS_COMMON;
            if (hasAny(access, EnhancedBarrierAccess::vertexBuffer))
            {
                result |= D3D12_BARRIER_ACCESS_VERTEX_BUFFER;
            }
            if (hasAny(access, EnhancedBarrierAccess::constantBuffer))
            {
                result |= D3D12_BARRIER_ACCESS_CONSTANT_BUFFER;
            }
            if (hasAny(access, EnhancedBarrierAccess::indexBuffer))
            {
                result |= D3D12_BARRIER_ACCESS_INDEX_BUFFER;
            }
            if (hasAny(access, EnhancedBarrierAccess::renderTarget))
            {
                result |= D3D12_BARRIER_ACCESS_RENDER_TARGET;
            }
            if (hasAny(access, EnhancedBarrierAccess::unorderedAccess))
            {
                result |= D3D12_BARRIER_ACCESS_UNORDERED_ACCESS;
            }
            if (hasAny(access, EnhancedBarrierAccess::depthStencilWrite))
            {
                result |= D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE;
            }
            if (hasAny(access, EnhancedBarrierAccess::depthStencilRead))
            {
                result |= D3D12_BARRIER_ACCESS_DEPTH_STENCIL_READ;
            }
            if (hasAny(access, EnhancedBarrierAccess::shaderResource))
            {
                result |= D3D12_BARRIER_ACCESS_SHADER_RESOURCE;
            }
            if (hasAny(access, EnhancedBarrierAccess::streamOutput))
            {
                result |= D3D12_BARRIER_ACCESS_STREAM_OUTPUT;
            }
            if (hasAny(access, EnhancedBarrierAccess::indirectArgument))
            {
                result |= D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT;
            }
            if (hasAny(access, EnhancedBarrierAccess::predication))
            {
                result |= D3D12_BARRIER_ACCESS_PREDICATION;
            }
            if (hasAny(access, EnhancedBarrierAccess::copyDest))
            {
                result |= D3D12_BARRIER_ACCESS_COPY_DEST;
            }
            if (hasAny(access, EnhancedBarrierAccess::copySource))
            {
                result |= D3D12_BARRIER_ACCESS_COPY_SOURCE;
            }
            if (hasAny(access, EnhancedBarrierAccess::resolveDest))
            {
                result |= D3D12_BARRIER_ACCESS_RESOLVE_DEST;
            }
            if (hasAny(access, EnhancedBarrierAccess::resolveSource))
            {
                result |= D3D12_BARRIER_ACCESS_RESOLVE_SOURCE;
            }
            if (hasAny(access, EnhancedBarrierAccess::shadingRateSource))
            {
                result |= D3D12_BARRIER_ACCESS_SHADING_RATE_SOURCE;
            }
            return result;
        }

        [[nodiscard]] D3D12_BARRIER_ACCESS nativeAccess(Access access) noexcept
        {
            return toNativeBarrierAccess(translateEnhancedAccess(access));
        }

        [[nodiscard]] constexpr D3D12_BARRIER_LAYOUT toNativeBarrierLayout(EnhancedBarrierLayout layout) noexcept
        {
            switch (layout)
            {
            case EnhancedBarrierLayout::undefined:
                return D3D12_BARRIER_LAYOUT_UNDEFINED;
            case EnhancedBarrierLayout::common:
                return D3D12_BARRIER_LAYOUT_COMMON;
            case EnhancedBarrierLayout::present:
                return D3D12_BARRIER_LAYOUT_PRESENT;
            case EnhancedBarrierLayout::shaderResource:
                return D3D12_BARRIER_LAYOUT_SHADER_RESOURCE;
            case EnhancedBarrierLayout::unorderedAccess:
                return D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS;
            case EnhancedBarrierLayout::renderTarget:
                return D3D12_BARRIER_LAYOUT_RENDER_TARGET;
            case EnhancedBarrierLayout::depthStencilWrite:
                return D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE;
            case EnhancedBarrierLayout::depthStencilRead:
                return D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ;
            case EnhancedBarrierLayout::resolveSource:
                return D3D12_BARRIER_LAYOUT_RESOLVE_SOURCE;
            case EnhancedBarrierLayout::resolveDest:
                return D3D12_BARRIER_LAYOUT_RESOLVE_DEST;
            case EnhancedBarrierLayout::shadingRateSource:
                return D3D12_BARRIER_LAYOUT_SHADING_RATE_SOURCE;
            }
            return D3D12_BARRIER_LAYOUT_COMMON;
        }

        [[nodiscard]] constexpr D3D12_BARRIER_LAYOUT nativeLayout(TextureLayout layout, QueueType queueType) noexcept
        {
            return toNativeBarrierLayout(translateEnhancedLayout(layout, queueType));
        }

        [[nodiscard]] D3D12_RESOURCE_STATES legacyTextureState(TextureLayout layout,
                                                               Access access,
                                                               QueueType queueType) noexcept
        {
            return toNativeResourceStates(translateLegacyTextureState(layout, access, queueType));
        }

        [[nodiscard]] constexpr D3D12_TEXTURE_BARRIER_FLAGS toNativeTextureBarrierFlags(
            EnhancedTextureBarrierFlags flags) noexcept
        {
            return flags == EnhancedTextureBarrierFlags::discard ? D3D12_TEXTURE_BARRIER_FLAG_DISCARD
                                                                 : D3D12_TEXTURE_BARRIER_FLAG_NONE;
        }

        struct NativeTextureRange final
        {
            std::uint32_t firstMip = 0;
            std::uint32_t mipCount = 0;
            std::uint32_t firstArray = 0;
            std::uint32_t arrayCount = 0;
            std::uint32_t plane = 0;
            std::uint32_t planeCount = 1;
        };

        [[nodiscard]] Result<NativeTextureRange> nativeTextureRange(const TextureRecord& texture,
                                                                    const TextureSubresourceRange& range)
        {
            const auto info = format::info(texture.desc.format);
            const auto plane = info.has_value() ? aspectPlane(*info, range.aspect) : std::nullopt;
            const std::uint32_t arraySize =
                texture.desc.dimension == TextureDimension::texture3D ? 1U : texture.desc.depthOrArrayLayers;
            const std::uint32_t mipCount =
                range.mipLevelCount == remainingSubresources && range.firstMipLevel < texture.desc.mipLevels
                    ? texture.desc.mipLevels - range.firstMipLevel
                    : range.mipLevelCount;
            const std::uint32_t arrayCount =
                range.arrayLayerCount == remainingSubresources && range.firstArrayLayer < arraySize
                    ? arraySize - range.firstArrayLayer
                    : range.arrayLayerCount;
            if (!plane.has_value() || range.firstMipLevel >= texture.desc.mipLevels || mipCount == 0 ||
                mipCount > texture.desc.mipLevels - range.firstMipLevel || range.firstArrayLayer >= arraySize ||
                arrayCount == 0 || arrayCount > arraySize - range.firstArrayLayer)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The texture barrier range has an invalid mip, layer, count, or aspect.",
                });
            }
            return NativeTextureRange{
                range.firstMipLevel,
                mipCount,
                range.firstArrayLayer,
                arrayCount,
                *plane,
            };
        }

        [[nodiscard]] Result<NativeTextureRange> nativeWholeTextureRange(const TextureRecord& texture)
        {
            const auto info = format::info(texture.desc.format);
            const bool depth = info.has_value() && info->depth;
            auto range = nativeTextureRange(texture, {.aspect = depth ? TextureAspect::depth : TextureAspect::color});
            if (range && info.has_value())
            {
                range->planeCount = info->planeCount;
            }
            return range;
        }

        [[nodiscard]] constexpr std::size_t subresourceCount(const NativeTextureRange& range) noexcept
        {
            return static_cast<std::size_t>(range.mipCount) * range.arrayCount * range.planeCount;
        }

        template<typename Visit>
        void forEachTextureSubresource(const TextureRecord& texture, const NativeTextureRange& range, Visit&& visit)
        {
            const std::uint32_t arraySize =
                texture.desc.dimension == TextureDimension::texture3D ? 1U : texture.desc.depthOrArrayLayers;
            for (std::uint32_t planeOffset = 0; planeOffset < range.planeCount; ++planeOffset)
            {
                for (std::uint32_t arrayOffset = 0; arrayOffset < range.arrayCount; ++arrayOffset)
                {
                    for (std::uint32_t mipOffset = 0; mipOffset < range.mipCount; ++mipOffset)
                    {
                        visit(range.firstMip + mipOffset + (range.firstArray + arrayOffset) * texture.desc.mipLevels +
                              (range.plane + planeOffset) * texture.desc.mipLevels * arraySize);
                    }
                }
            }
        }

        [[nodiscard]] D3D12_RESOURCE_BARRIER subresourceTransition(ID3D12Resource* resource,
                                                                   std::uint32_t subresource,
                                                                   D3D12_RESOURCE_STATES before,
                                                                   D3D12_RESOURCE_STATES after) noexcept
        {
            return D3D12_RESOURCE_BARRIER{
                .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
                .Transition =
                    {
                        .pResource = resource,
                        .Subresource = subresource,
                        .StateBefore = before,
                        .StateAfter = after,
                    },
            };
        }

        using LegacyBarrierList = SmallVector<D3D12_RESOURCE_BARRIER, 2>;

        void appendTextureTransitions(LegacyBarrierList& barriers,
                                      ID3D12Resource& resource,
                                      const TextureRecord& texture,
                                      const NativeTextureRange& range,
                                      D3D12_RESOURCE_STATES before,
                                      D3D12_RESOURCE_STATES after)
        {
            if (before == after)
            {
                return;
            }

            const auto info = *format::info(texture.desc.format);
            const std::uint32_t arraySize =
                texture.desc.dimension == TextureDimension::texture3D ? 1U : texture.desc.depthOrArrayLayers;
            const bool everySubresource = range.firstMip == 0 && range.mipCount == texture.desc.mipLevels &&
                                          range.firstArray == 0 && range.arrayCount == arraySize && range.plane == 0 &&
                                          range.planeCount == info.planeCount;
            if (everySubresource)
            {
                barriers.push_back(
                    subresourceTransition(&resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after));
                return;
            }

            forEachTextureSubresource(texture, range, [&](std::uint32_t subresource) {
                barriers.push_back(subresourceTransition(&resource, subresource, before, after));
            });
        }

        void appendTrackedTextureTransitions(LegacyBarrierList& barriers,
                                             ID3D12Resource& resource,
                                             const TextureRecord& texture,
                                             const NativeTextureRange& range,
                                             std::span<const D3D12_RESOURCE_STATES> before,
                                             D3D12_RESOURCE_STATES after)
        {
            if (std::ranges::adjacent_find(before, std::ranges::not_equal_to{}) == before.end())
            {
                appendTextureTransitions(barriers, resource, texture, range, before.front(), after);
                return;
            }

            std::size_t offset = 0;
            forEachTextureSubresource(texture, range, [&](std::uint32_t subresource) {
                const auto state = before[offset++];
                if (state != after)
                {
                    barriers.push_back(subresourceTransition(&resource, subresource, state, after));
                }
            });
        }

        void discardTextureRange(ID3D12GraphicsCommandList& commandList,
                                 ID3D12Resource& resource,
                                 const TextureRecord& texture,
                                 const NativeTextureRange& range)
        {
            const std::uint32_t arraySize =
                texture.desc.dimension == TextureDimension::texture3D ? 1U : texture.desc.depthOrArrayLayers;
            for (std::uint32_t planeOffset = 0; planeOffset < range.planeCount; ++planeOffset)
            {
                for (std::uint32_t arrayOffset = 0; arrayOffset < range.arrayCount; ++arrayOffset)
                {
                    const D3D12_DISCARD_REGION region{
                        .NumRects = 0,
                        .pRects = nullptr,
                        .FirstSubresource = range.firstMip + (range.firstArray + arrayOffset) * texture.desc.mipLevels +
                                            (range.plane + planeOffset) * texture.desc.mipLevels * arraySize,
                        .NumSubresources = range.mipCount,
                    };
                    commandList.DiscardResource(&resource, &region);
                }
            }
        }

        [[nodiscard]] D3D12_BARRIER_SUBRESOURCE_RANGE toNativeSubresourceRange(const NativeTextureRange& range) noexcept
        {
            return D3D12_BARRIER_SUBRESOURCE_RANGE{
                .IndexOrFirstMipLevel = range.firstMip,
                .NumMipLevels = range.mipCount,
                .FirstArraySlice = range.firstArray,
                .NumArraySlices = range.arrayCount,
                .FirstPlane = range.plane,
                .NumPlanes = range.planeCount,
            };
        }

        [[nodiscard]] std::array<D3D12_TEXTURE_BARRIER, 2> enhancedAttachmentInitializationBarriers(
            ID3D12Resource& resource,
            const TextureRecord& texture,
            const NativeTextureRange& range,
            const TextureBarrier& barrier) noexcept
        {
            const bool depth = hasAny(texture.desc.usage, TextureUsage::depthStencil);
            const auto initializationSync = depth ? D3D12_BARRIER_SYNC_DEPTH_STENCIL : D3D12_BARRIER_SYNC_RENDER_TARGET;
            const auto initializationAccess =
                depth ? D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE : D3D12_BARRIER_ACCESS_RENDER_TARGET;
            const auto initializationLayout =
                depth ? D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE : D3D12_BARRIER_LAYOUT_RENDER_TARGET;
            const auto subresources = toNativeSubresourceRange(range);
            return std::array{
                D3D12_TEXTURE_BARRIER{
                    .SyncBefore = nativeSync(barrier.stagesBefore, barrier.accessBefore),
                    .SyncAfter = initializationSync,
                    .AccessBefore = nativeAccess(barrier.accessBefore),
                    .AccessAfter = initializationAccess,
                    .LayoutBefore = D3D12_BARRIER_LAYOUT_UNDEFINED,
                    .LayoutAfter = initializationLayout,
                    .pResource = &resource,
                    .Subresources = subresources,
                    .Flags = D3D12_TEXTURE_BARRIER_FLAG_DISCARD,
                },
                D3D12_TEXTURE_BARRIER{
                    .SyncBefore = initializationSync,
                    .SyncAfter = nativeSync(barrier.stagesAfter, barrier.accessAfter),
                    .AccessBefore = initializationAccess,
                    .AccessAfter = nativeAccess(barrier.accessAfter),
                    .LayoutBefore = initializationLayout,
                    .LayoutAfter = nativeLayout(barrier.layoutAfter, QueueType::graphics),
                    .pResource = &resource,
                    .Subresources = subresources,
                    .Flags = {},
                },
            };
        }

        void initializeEnhancedAttachment(ID3D12GraphicsCommandList7& commandList,
                                          ID3D12Resource& resource,
                                          const TextureRecord& texture,
                                          const NativeTextureRange& range,
                                          const TextureBarrier& barrier)
        {
            const auto [initializationBarrier, finalBarrier] =
                enhancedAttachmentInitializationBarriers(resource, texture, range, barrier);
            D3D12_BARRIER_GROUP group{
                .Type = D3D12_BARRIER_TYPE_TEXTURE,
                .NumBarriers = 1,
                .pTextureBarriers = &initializationBarrier,
            };
            commandList.Barrier(1, &group);
            discardTextureRange(commandList, resource, texture, range);

            group.pTextureBarriers = &finalBarrier;
            commandList.Barrier(1, &group);
        }

        struct PreparedBufferBarrier final
        {
            BufferBarrier barrier;
            std::shared_ptr<BufferRecord> resource;
            std::uint64_t resolvedSize = 0;
        };

        struct PreparedTextureBarrier final
        {
            TextureBarrier barrier;
            std::shared_ptr<TextureRecord> resource;
            NativeTextureRange range;
        };

        struct PreparedSamplerFeedbackBarrier final
        {
            SamplerFeedbackBarrier barrier;
            std::shared_ptr<SamplerFeedbackMapRecord> resource;
        };

    } // namespace

    struct PreparedResourceBarrierBatch final
    {
        SmallVector<GlobalBarrier, 1> global;
        SmallVector<PreparedBufferBarrier, 2> buffers;
        SmallVector<PreparedTextureBarrier, 2> textures;
        SmallVector<PreparedSamplerFeedbackBarrier, 1> samplerFeedbackMaps;
        SmallVector<std::shared_ptr<ResidencyRecord>, 4> residencies;
        SparseMappingCommandLeases sparseMappings;
    };

    namespace
    {

        struct SamplerFeedbackBarrierRetention final
        {
            std::vector<std::shared_ptr<SamplerFeedbackMapRecord>> resources;
        };

        struct PreparedBufferAliasActivation final
        {
            std::shared_ptr<BufferRecord> before = {};
            std::shared_ptr<BufferRecord> after = {};
            std::array<std::shared_ptr<ResidencyRecord>, 2> residencies = {};
        };

        struct PreparedTextureAliasActivation final
        {
            std::shared_ptr<TextureRecord> before = {};
            std::shared_ptr<TextureRecord> after = {};
            NativeTextureRange beforeRange = {};
            NativeTextureRange afterRange = {};
            std::array<std::shared_ptr<ResidencyRecord>, 2> residencies = {};
        };

        [[nodiscard]] bool placementsAlias(const std::shared_ptr<PlacedResourceLease>& before,
                                           const std::shared_ptr<PlacedResourceLease>& after) noexcept
        {
            return before != nullptr && after != nullptr && before->heap == after->heap &&
                   hasAny(before->heap->desc.flags, ResourceHeapFlag::allowAliasing) &&
                   before->offset < after->offset + after->size && after->offset < before->offset + before->size;
        }

        [[nodiscard]] constexpr bool textureRangesOverlap(const NativeTextureRange& left,
                                                          const NativeTextureRange& right) noexcept
        {
            const bool mipOverlap =
                left.firstMip < right.firstMip + right.mipCount && right.firstMip < left.firstMip + left.mipCount;
            const bool arrayOverlap = left.firstArray < right.firstArray + right.arrayCount &&
                                      right.firstArray < left.firstArray + left.arrayCount;
            const bool planeOverlap =
                left.plane < right.plane + right.planeCount && right.plane < left.plane + left.planeCount;
            return planeOverlap && mipOverlap && arrayOverlap;
        }

        [[nodiscard]] constexpr bool exceedsNativeBarrierCount(std::size_t count) noexcept
        {
            return count > std::numeric_limits<UINT>::max();
        }

        [[nodiscard]] bool isEmptyBarrierBatch(const ResourceBarrierBatch& barriers) noexcept
        {
            return barriers.globalBarriers.empty() && barriers.bufferBarriers.empty() &&
                   barriers.textureBarriers.empty() && barriers.samplerFeedbackBarriers.empty();
        }

        [[nodiscard]] bool anyBarrierCountExceedsNativeLimit(const ResourceBarrierBatch& barriers) noexcept
        {
            return exceedsNativeBarrierCount(barriers.globalBarriers.size()) ||
                   exceedsNativeBarrierCount(barriers.bufferBarriers.size()) ||
                   exceedsNativeBarrierCount(barriers.textureBarriers.size()) ||
                   exceedsNativeBarrierCount(barriers.samplerFeedbackBarriers.size());
        }

        [[nodiscard]] Status prepareGlobalBarriers(PreparedResourceBarrierBatch& prepared,
                                                   std::span<const GlobalBarrier> barriers,
                                                   QueueType queueType)
        {
            for (const auto& barrier : barriers)
            {
                if (!validBarrierScope(barrier.stagesBefore, barrier.accessBefore, queueType) ||
                    !validBarrierScope(barrier.stagesAfter, barrier.accessAfter, queueType) ||
                    (barrier.accessBefore == Access::common && barrier.accessAfter != Access::common))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A global barrier contains an invalid or queue-incompatible stage/access scope.",
                    });
                }
                prepared.global.push_back(barrier);
            }
            return {};
        }

        template<typename RecordLookup>
        [[nodiscard]] auto lookUpUnderSparseMappingLock(std::unique_lock<std::mutex>& sparseMappingLock,
                                                        const RecordLookup& lookUp)
        {
            auto record = lookUp();
            if (record && !sparseMappingLock.owns_lock() && (*record)->sparse)
            {
                sparseMappingLock.lock();
                record = lookUp();
            }
            return record;
        }

        [[nodiscard]] constexpr bool validOwnershipTransfer(OwnershipTransfer ownership) noexcept
        {
            return ownership == OwnershipTransfer::none || ownership == OwnershipTransfer::releaseExternal ||
                   ownership == OwnershipTransfer::acquireExternal;
        }

        [[nodiscard]] Status validateBufferOwnershipTransfer(const BufferBarrier& barrier, const BufferRecord& buffer)
        {
            if (!validOwnershipTransfer(barrier.ownership))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A buffer barrier contains an unknown ownership transfer.",
                });
            }
            if (barrier.ownership != OwnershipTransfer::none && !buffer.shared)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Buffer ownership transfers require a buffer created or imported through the shared-buffer API.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateTextureOwnershipTransfer(const TextureBarrier& barrier,
                                                              const TextureRecord& texture)
        {
            if (!validOwnershipTransfer(barrier.ownership))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A texture barrier contains an unknown ownership transfer.",
                });
            }
            if (barrier.ownership == OwnershipTransfer::none)
            {
                return {};
            }
            if (!texture.shared)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Texture ownership transfers require a texture created or imported through the shared-texture "
                    "API.",
                });
            }
            if (barrier.layoutBefore != TextureLayout::common || barrier.layoutAfter != TextureLayout::common)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Texture ownership transfers require the common layout before and after the barrier.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateBufferBarrierTarget(const BufferBarrier& barrier, const BufferRecord& buffer)
        {
            if (auto owned = validateBufferOwnershipTransfer(barrier, buffer); !owned)
            {
                return owned;
            }
            if (!supportsBufferAccess(buffer.desc.usage, barrier.accessBefore) ||
                !supportsBufferAccess(buffer.desc.usage, barrier.accessAfter))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A buffer barrier access is incompatible with the buffer usage.",
                });
            }
            if (!validRange(buffer.desc.size, barrier.offset, barrier.size))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A buffer barrier range is empty or out of bounds.",
                });
            }
            if (barrier.offset != 0 || (barrier.size != wholeResource && barrier.size != buffer.desc.size))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "D3D12 buffer barriers currently support only the complete buffer range.",
                });
            }
            return {};
        }

        [[nodiscard]] constexpr bool usesPresentOutsideNoAccessScope(TextureLayout layout,
                                                                     PipelineStage stages,
                                                                     Access access) noexcept
        {
            return layout == TextureLayout::present && (stages != PipelineStage::none || access != Access::none);
        }

        [[nodiscard]] bool validTextureBarrierScope(const TextureBarrier& barrier, QueueType queueType) noexcept
        {
            return validResourceBarrierScope(barrier.stagesBefore, barrier.accessBefore, queueType) &&
                   validResourceBarrierScope(barrier.stagesAfter, barrier.accessAfter, queueType) &&
                   validTextureLayout(barrier.layoutBefore) && validTextureLayout(barrier.layoutAfter) &&
                   barrier.layoutAfter != TextureLayout::undefined &&
                   !usesPresentOutsideNoAccessScope(barrier.layoutBefore, barrier.stagesBefore, barrier.accessBefore) &&
                   !usesPresentOutsideNoAccessScope(barrier.layoutAfter, barrier.stagesAfter, barrier.accessAfter);
        }

        [[nodiscard]] bool textureSupportsLayout(const TextureRecord& texture, TextureLayout layout) noexcept
        {
            return supportsLayout(texture.desc.usage, layout) ||
                   (texture.externallyOwned && layout == TextureLayout::present);
        }

        [[nodiscard]] bool textureBarrierLayoutsSupported(const TextureBarrier& barrier,
                                                          const TextureRecord& texture,
                                                          QueueType queueType) noexcept
        {
            return textureSupportsLayout(texture, barrier.layoutBefore) &&
                   textureSupportsLayout(texture, barrier.layoutAfter) &&
                   supportsTextureLayoutOnQueue(barrier.layoutBefore, queueType) &&
                   supportsTextureLayoutOnQueue(barrier.layoutAfter, queueType);
        }

        [[nodiscard]] bool textureBarrierAccessesSupported(const TextureBarrier& barrier,
                                                           const TextureRecord& texture) noexcept
        {
            return supportsTextureAccess(texture.desc.usage, barrier.accessBefore) &&
                   supportsTextureAccess(texture.desc.usage, barrier.accessAfter) &&
                   accessMatchesTextureLayout(barrier.layoutBefore, barrier.accessBefore) &&
                   accessMatchesTextureLayout(barrier.layoutAfter, barrier.accessAfter);
        }

        [[nodiscard]] Status validateTextureBarrierTarget(const TextureBarrier& barrier,
                                                          const TextureRecord& texture,
                                                          QueueType queueType)
        {
            if (!textureBarrierLayoutsSupported(barrier, texture, queueType))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A texture barrier layout is incompatible with the texture usage or command queue.",
                });
            }
            if (!textureBarrierAccessesSupported(barrier, texture))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A texture barrier access is incompatible with the texture usage or layout.",
                });
            }
            return validateTextureOwnershipTransfer(barrier, texture);
        }

        [[nodiscard]] bool anyTextureBarriersOverlap(std::span<const PreparedTextureBarrier> textures)
        {
            return anyOverlappingPair(
                textures.size(),
                [&](std::size_t index) {
                    const auto& range = textures[index].range;
                    return BatchSweepKey{
                        .group = textures[index].barrier.texture.identity,
                        .primary =
                            {
                                .begin = range.firstArray,
                                .end = static_cast<std::uint64_t>(range.firstArray) + range.arrayCount,
                            },
                        .secondary =
                            layeredBatchSweepInterval(range.plane, range.planeCount, range.firstMip, range.mipCount),
                    };
                },
                [&](std::size_t earlier, std::size_t later) {
                    return textures[earlier].barrier.texture == textures[later].barrier.texture &&
                           textureRangesOverlap(textures[earlier].range, textures[later].range);
                });
        }

        [[nodiscard]] constexpr bool validSamplerFeedbackLayoutAfter(TextureLayout layout) noexcept
        {
            return layout == TextureLayout::common || layout == TextureLayout::unorderedAccess ||
                   layout == TextureLayout::resolveSource || layout == TextureLayout::resolveDestination;
        }

        [[nodiscard]] constexpr bool validSamplerFeedbackLayoutBefore(TextureLayout layout) noexcept
        {
            return validSamplerFeedbackLayoutAfter(layout) || layout == TextureLayout::undefined;
        }

        [[nodiscard]] constexpr bool samplerFeedbackAccessMatchesLayout(TextureLayout layout, Access access) noexcept
        {
            if (layout == TextureLayout::undefined)
            {
                return access == Access::none;
            }
            if (access == Access::none || access == Access::common)
            {
                return true;
            }
            if (layout == TextureLayout::unorderedAccess)
            {
                return access == Access::unorderedAccess;
            }
            if (layout == TextureLayout::resolveSource)
            {
                return access == Access::resolveSource;
            }
            if (layout == TextureLayout::resolveDestination)
            {
                return access == Access::resolveDestination;
            }
            return false;
        }

        [[nodiscard]] bool validSamplerFeedbackBarrierScope(const SamplerFeedbackBarrier& barrier,
                                                            QueueType queueType) noexcept
        {
            return validResourceBarrierScope(barrier.stagesBefore, barrier.accessBefore, queueType) &&
                   validResourceBarrierScope(barrier.stagesAfter, barrier.accessAfter, queueType) &&
                   validSamplerFeedbackLayoutBefore(barrier.layoutBefore) &&
                   validSamplerFeedbackLayoutAfter(barrier.layoutAfter) &&
                   supportsTextureLayoutOnQueue(barrier.layoutBefore, queueType) &&
                   supportsTextureLayoutOnQueue(barrier.layoutAfter, queueType) &&
                   samplerFeedbackAccessMatchesLayout(barrier.layoutBefore, barrier.accessBefore) &&
                   samplerFeedbackAccessMatchesLayout(barrier.layoutAfter, barrier.accessAfter);
        }

        [[nodiscard]] Status validateLegacyGlobalBarriers(std::span<const GlobalBarrier> barriers)
        {
            for (const auto& barrier : barriers)
            {
                if (barrier.accessBefore != Access::unorderedAccess || barrier.accessAfter != Access::unorderedAccess)
                {
                    return std::unexpected(Error{
                        ErrorCode::unsupported,
                        0,
                        "Legacy D3D12 can represent only global unordered-access memory barriers.",
                    });
                }
            }
            return {};
        }

        [[nodiscard]] Status validateLegacyBufferBarriers(std::span<const PreparedBufferBarrier> buffers)
        {
            for (const auto& item : buffers)
            {
                if (combinesWriteWithOtherAccess(item.barrier.accessBefore) ||
                    combinesWriteWithOtherAccess(item.barrier.accessAfter))
                {
                    return std::unexpected(Error{
                        ErrorCode::unsupported,
                        0,
                        "Legacy D3D12 cannot represent a buffer barrier access that combines a write with another "
                        "access.",
                    });
                }
            }
            return {};
        }

        [[nodiscard]] Status validateLegacyTextureBarriers(std::span<const PreparedTextureBarrier> textures,
                                                           QueueType queueType)
        {
            for (const auto& item : textures)
            {
                const auto& barrier = item.barrier;
                if (!legacyStateRepresentsTextureAccess(barrier.layoutBefore, barrier.accessBefore, queueType) ||
                    !legacyStateRepresentsTextureAccess(barrier.layoutAfter, barrier.accessAfter, queueType))
                {
                    return std::unexpected(Error{
                        ErrorCode::unsupported,
                        0,
                        "Legacy D3D12 cannot represent a common-layout texture barrier access that combines a write "
                        "with another access.",
                    });
                }
            }
            return {};
        }

        [[nodiscard]] Status validateLegacyBarrierBatch(const PreparedResourceBarrierBatch& prepared,
                                                        QueueType queueType)
        {
            auto global = validateLegacyGlobalBarriers(prepared.global);
            if (!global)
            {
                return global;
            }
            auto buffers = validateLegacyBufferBarriers(prepared.buffers);
            if (!buffers)
            {
                return buffers;
            }
            return validateLegacyTextureBarriers(prepared.textures, queueType);
        }

        [[nodiscard]] Result<SparseMappingDependencies> captureSparseMappings(PreparedResourceBarrierBatch& prepared)
        {
            SparseMappingDependencies dependencies;
            for (const auto& item : prepared.buffers)
            {
                auto captured = captureSparseBufferMapping(item.resource, prepared.sparseMappings, dependencies);
                if (!captured)
                {
                    return std::unexpected(std::move(captured.error()));
                }
            }
            for (const auto& item : prepared.textures)
            {
                auto captured = captureSparseTextureMapping(item.resource, prepared.sparseMappings, dependencies);
                if (!captured)
                {
                    return std::unexpected(std::move(captured.error()));
                }
            }
            return dependencies;
        }

        struct BarrierCommandDependencies final
        {
            std::shared_ptr<void> preparedOwner;
            SparseMappingDependencies sparseMappingDependencies;
            SmallVector<std::shared_ptr<SwapchainImageLeaseState>, 2> swapchainLeases;
            std::shared_ptr<SamplerFeedbackBarrierRetention> samplerFeedbackOwner;
        };

        [[nodiscard]] Result<std::shared_ptr<SamplerFeedbackBarrierRetention>> stageSamplerFeedbackOwnership(
            std::span<const PreparedSamplerFeedbackBarrier> samplerFeedbackMaps)
        {
            if (samplerFeedbackMaps.empty())
            {
                return std::shared_ptr<SamplerFeedbackBarrierRetention>{};
            }
            try
            {
                auto retention = std::make_shared<SamplerFeedbackBarrierRetention>();
                retention->resources.reserve(samplerFeedbackMaps.size());
                for (const auto& item : samplerFeedbackMaps)
                {
                    retention->resources.push_back(item.resource);
                }
                return retention;
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to stage sampler-feedback barrier ownership.",
                });
            }
        }

        [[nodiscard]] Result<BarrierCommandDependencies> stageBarrierCommandDependencies(
            const std::shared_ptr<PreparedResourceBarrierBatch>& prepared,
            SparseMappingDependencies sparseMappingDependencies)
        {
            BarrierCommandDependencies dependencies;
            dependencies.sparseMappingDependencies = std::move(sparseMappingDependencies);
            for (const auto& item : prepared->textures)
            {
                if (item.resource->swapchainLease != nullptr)
                {
                    dependencies.swapchainLeases.push_back(item.resource->swapchainLease);
                }
            }
            if (!prepared->buffers.empty() || !prepared->textures.empty())
            {
                dependencies.preparedOwner = prepared;
            }
            auto samplerFeedbackOwner = stageSamplerFeedbackOwnership(prepared->samplerFeedbackMaps);
            if (!samplerFeedbackOwner)
            {
                return std::unexpected(std::move(samplerFeedbackOwner.error()));
            }
            dependencies.samplerFeedbackOwner = std::move(*samplerFeedbackOwner);
            return dependencies;
        }

        [[nodiscard]] Status retainBarrierCommandDependencies(QueueManager& queues,
                                                              detail::BackendCommandBuffer& commandBuffer,
                                                              const PreparedResourceBarrierBatch& prepared,
                                                              const BarrierCommandDependencies& dependencies)
        {
            return retainResourceCommandDependencies(queues,
                                                     commandBuffer,
                                                     dependencies.preparedOwner,
                                                     prepared.residencies,
                                                     prepared.sparseMappings,
                                                     dependencies.sparseMappingDependencies,
                                                     dependencies.swapchainLeases,
                                                     dependencies.samplerFeedbackOwner);
        }

        [[nodiscard]] D3D12_GLOBAL_BARRIER enhancedGlobalBarrier(const GlobalBarrier& barrier) noexcept
        {
            return D3D12_GLOBAL_BARRIER{
                .SyncBefore = nativeSync(barrier.stagesBefore, barrier.accessBefore),
                .SyncAfter = nativeSync(barrier.stagesAfter, barrier.accessAfter),
                .AccessBefore = nativeAccess(barrier.accessBefore),
                .AccessAfter = nativeAccess(barrier.accessAfter),
            };
        }

        [[nodiscard]] D3D12_BUFFER_BARRIER enhancedBufferBarrier(const PreparedBufferBarrier& prepared)
        {
            return D3D12_BUFFER_BARRIER{
                .SyncBefore = nativeSync(prepared.barrier.stagesBefore, prepared.barrier.accessBefore),
                .SyncAfter = nativeSync(prepared.barrier.stagesAfter, prepared.barrier.accessAfter),
                .AccessBefore = nativeAccess(prepared.barrier.accessBefore),
                .AccessAfter = nativeAccess(prepared.barrier.accessAfter),
                .pResource = prepared.resource->resource.Get(),
                .Offset = prepared.barrier.offset,
                .Size = prepared.resolvedSize,
            };
        }

        [[nodiscard]] bool requiresAttachmentInitialization(const TextureBarrier& barrier,
                                                            const TextureRecord& texture,
                                                            QueueType queueType) noexcept
        {
            const bool attachment = hasAny(texture.desc.usage, TextureUsage::renderTarget | TextureUsage::depthStencil);
            return barrier.layoutBefore == TextureLayout::undefined && attachment && queueType == QueueType::graphics;
        }

        [[nodiscard]] D3D12_TEXTURE_BARRIER enhancedTextureBarrier(const TextureBarrier& barrier,
                                                                   const TextureRecord& texture,
                                                                   const NativeTextureRange& range,
                                                                   QueueType queueType)
        {
            return D3D12_TEXTURE_BARRIER{
                .SyncBefore = nativeSync(barrier.stagesBefore, barrier.accessBefore),
                .SyncAfter = nativeSync(barrier.stagesAfter, barrier.accessAfter),
                .AccessBefore = nativeAccess(barrier.accessBefore),
                .AccessAfter = nativeAccess(barrier.accessAfter),
                .LayoutBefore = nativeLayout(barrier.layoutBefore, queueType),
                .LayoutAfter = nativeLayout(barrier.layoutAfter, queueType),
                .pResource = texture.resource.Get(),
                .Subresources = toNativeSubresourceRange(range),
                .Flags = barrier.layoutBefore == TextureLayout::undefined ? D3D12_TEXTURE_BARRIER_FLAG_DISCARD
                                                                          : D3D12_TEXTURE_BARRIER_FLAG_NONE,
            };
        }

        [[nodiscard]] D3D12_TEXTURE_BARRIER enhancedSamplerFeedbackBarrier(
            const PreparedSamplerFeedbackBarrier& prepared, QueueType queueType)
        {
            return D3D12_TEXTURE_BARRIER{
                .SyncBefore = nativeSync(prepared.barrier.stagesBefore, prepared.barrier.accessBefore),
                .SyncAfter = nativeSync(prepared.barrier.stagesAfter, prepared.barrier.accessAfter),
                .AccessBefore = nativeAccess(prepared.barrier.accessBefore),
                .AccessAfter = nativeAccess(prepared.barrier.accessAfter),
                .LayoutBefore = nativeLayout(prepared.barrier.layoutBefore, queueType),
                .LayoutAfter = nativeLayout(prepared.barrier.layoutAfter, queueType),
                .pResource = prepared.resource->resource.Get(),
                .Subresources =
                    {
                        .IndexOrFirstMipLevel = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        .NumMipLevels = {},
                        .FirstArraySlice = {},
                        .NumArraySlices = {},
                        .FirstPlane = {},
                        .NumPlanes = {},
                    },
                .Flags = prepared.barrier.layoutBefore == TextureLayout::undefined ? D3D12_TEXTURE_BARRIER_FLAG_DISCARD
                                                                                   : D3D12_TEXTURE_BARRIER_FLAG_NONE,
            };
        }

        struct EnhancedBarrierBatch final
        {
            SmallVector<D3D12_GLOBAL_BARRIER, 1> global;
            SmallVector<D3D12_BUFFER_BARRIER, 2> buffers;
            SmallVector<D3D12_TEXTURE_BARRIER, 2> textures;
            SmallVector<std::size_t, 2> attachmentInitializations;
        };

        [[nodiscard]] EnhancedBarrierBatch buildEnhancedBarrierBatch(const PreparedResourceBarrierBatch& prepared,
                                                                     QueueType queueType)
        {
            EnhancedBarrierBatch native;
            native.global.reserve(prepared.global.size());
            native.buffers.reserve(prepared.buffers.size());
            native.textures.reserve(prepared.textures.size() + prepared.samplerFeedbackMaps.size());
            for (const auto& barrier : prepared.global)
            {
                native.global.push_back(enhancedGlobalBarrier(barrier));
            }
            for (const auto& item : prepared.buffers)
            {
                native.buffers.push_back(enhancedBufferBarrier(item));
            }
            for (std::size_t index = 0; index < prepared.textures.size(); ++index)
            {
                const auto& item = prepared.textures[index];
                if (requiresAttachmentInitialization(item.barrier, *item.resource, queueType))
                {
                    native.attachmentInitializations.push_back(index);
                    continue;
                }
                native.textures.push_back(enhancedTextureBarrier(item.barrier, *item.resource, item.range, queueType));
            }
            for (const auto& item : prepared.samplerFeedbackMaps)
            {
                native.textures.push_back(enhancedSamplerFeedbackBarrier(item, queueType));
            }
            return native;
        }

        void recordEnhancedBarrierGroups(ID3D12GraphicsCommandList7& commandList, const EnhancedBarrierBatch& native)
        {
            std::array<D3D12_BARRIER_GROUP, 3> nativeGroups{};
            std::uint32_t groupCount = 0;
            if (!native.global.empty())
            {
                nativeGroups[groupCount++] = D3D12_BARRIER_GROUP{
                    .Type = D3D12_BARRIER_TYPE_GLOBAL,
                    .NumBarriers = static_cast<std::uint32_t>(native.global.size()),
                    .pGlobalBarriers = native.global.data(),
                };
            }
            if (!native.buffers.empty())
            {
                nativeGroups[groupCount++] = D3D12_BARRIER_GROUP{
                    .Type = D3D12_BARRIER_TYPE_BUFFER,
                    .NumBarriers = static_cast<std::uint32_t>(native.buffers.size()),
                    .pBufferBarriers = native.buffers.data(),
                };
            }
            if (!native.textures.empty())
            {
                nativeGroups[groupCount++] = D3D12_BARRIER_GROUP{
                    .Type = D3D12_BARRIER_TYPE_TEXTURE,
                    .NumBarriers = static_cast<std::uint32_t>(native.textures.size()),
                    .pTextureBarriers = native.textures.data(),
                };
            }
            if (groupCount != 0)
            {
                commandList.Barrier(groupCount, nativeGroups.data());
            }
        }

        [[nodiscard]] Status recordEnhancedBarrierBatch(QueueManager& queues,
                                                        detail::BackendCommandBuffer& commandBuffer,
                                                        ID3D12GraphicsCommandList& commandList,
                                                        const PreparedResourceBarrierBatch& prepared,
                                                        const BarrierCommandDependencies& dependencies)
        {
            ComPtr<ID3D12GraphicsCommandList7> enhanced;
            const HRESULT result = commandList.QueryInterface(IID_PPV_ARGS(&enhanced));
            if (FAILED(result))
            {
                return std::unexpected(
                    nativeError(ErrorCode::nativeFailure, result, "ID3D12GraphicsCommandList7 query"));
            }

            const auto native = buildEnhancedBarrierBatch(prepared, commandBuffer.queueType());
            auto retained = retainBarrierCommandDependencies(queues, commandBuffer, prepared, dependencies);
            if (!retained)
            {
                return retained;
            }

            recordEnhancedBarrierGroups(*enhanced.Get(), native);
            for (const auto index : native.attachmentInitializations)
            {
                const auto& item = prepared.textures[index];
                initializeEnhancedAttachment(
                    *enhanced.Get(), *item.resource->resource.Get(), *item.resource, item.range, item.barrier);
            }
            test::noteResourceCommandRecorded();
            return {};
        }

        [[nodiscard]] D3D12_RESOURCE_BARRIER unorderedAccessBarrier(ID3D12Resource* resource) noexcept
        {
            return D3D12_RESOURCE_BARRIER{
                .Type = D3D12_RESOURCE_BARRIER_TYPE_UAV,
                .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
                .UAV = {.pResource = resource},
            };
        }

        [[nodiscard]] D3D12_RESOURCE_BARRIER wholeResourceTransition(ID3D12Resource* resource,
                                                                     D3D12_RESOURCE_STATES before,
                                                                     D3D12_RESOURCE_STATES after) noexcept
        {
            return D3D12_RESOURCE_BARRIER{
                .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
                .Transition =
                    {
                        .pResource = resource,
                        .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        .StateBefore = before,
                        .StateAfter = after,
                    },
            };
        }

        void appendLegacyBufferBarrier(LegacyBarrierList& barriers,
                                       const PreparedBufferBarrier& prepared,
                                       QueueType queueType)
        {
            const auto plan = planLegacyBufferBarrier(prepared.resource->desc.memoryType,
                                                      prepared.barrier.accessBefore,
                                                      prepared.barrier.accessAfter,
                                                      queueType);
            if (plan.kind == LegacyBufferBarrierKind::uav)
            {
                barriers.push_back(unorderedAccessBarrier(prepared.resource->resource.Get()));
            }
            else if (plan.kind == LegacyBufferBarrierKind::transition)
            {
                barriers.push_back(wholeResourceTransition(prepared.resource->resource.Get(),
                                                           toNativeResourceStates(plan.stateBefore),
                                                           toNativeResourceStates(plan.stateAfter)));
            }
        }

        inline constexpr std::size_t untrackedLegacyTexture = std::numeric_limits<std::size_t>::max();

        struct LegacyStateChange final
        {
            D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_COMMON;
            D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_COMMON;
            SmallVector<D3D12_RESOURCE_STATES, 1> trackedBefore = {};
            std::size_t use = untrackedLegacyTexture;
            bool discard = false;

            [[nodiscard]] std::span<const D3D12_RESOURCE_STATES> befores() const noexcept
            {
                return trackedBefore.empty() ? std::span<const D3D12_RESOURCE_STATES>(&before, 1)
                                             : std::span<const D3D12_RESOURCE_STATES>(trackedBefore);
            }
        };

        [[nodiscard]] LegacyStateChange legacyStateChange(const TextureBarrier& barrier, QueueType queueType) noexcept
        {
            return LegacyStateChange{
                .before = legacyTextureState(barrier.layoutBefore, barrier.accessBefore, queueType),
                .after = legacyTextureState(barrier.layoutAfter, barrier.accessAfter, queueType),
            };
        }

        [[nodiscard]] constexpr bool discardsTrackedLegacyState(const TextureBarrier& barrier,
                                                                QueueType queueType) noexcept
        {
            return barrier.layoutBefore == TextureLayout::undefined && queueType != QueueType::copy;
        }

        [[nodiscard]] LegacyStateChange trackedLegacyStateChange(const TextureBarrier& barrier,
                                                                 const TextureRecord& texture,
                                                                 const NativeTextureRange& range,
                                                                 QueueType queueType,
                                                                 const LegacyTextureStateUse& use,
                                                                 std::size_t useIndex)
        {
            auto change = legacyStateChange(barrier, queueType);
            change.use = useIndex;
            change.discard = discardsTrackedLegacyState(barrier, queueType);
            if (change.discard)
            {
                const auto queueStates = legacyQueueResourceStates(queueType);
                change.trackedBefore.reserve(subresourceCount(range));
                const std::scoped_lock lock(use.tracker->mutex());
                forEachTextureSubresource(texture, range, [&](std::uint32_t subresource) {
                    change.trackedBefore.push_back(use.stateBeforeLocked(subresource, queueStates));
                });
            }
            return change;
        }

        void trackLegacyStateChange(LegacyTextureStateUse& use,
                                    const TextureRecord& texture,
                                    const NativeTextureRange& range,
                                    const LegacyStateChange& change) noexcept
        {
            const auto before = change.befores();
            std::size_t offset = 0;
            forEachTextureSubresource(texture, range, [&](std::uint32_t subresource) {
                use.transition(subresource, before[before.size() == 1 ? 0 : offset], change.after, change.discard);
                ++offset;
            });
        }

        void appendLegacyStateChange(LegacyBarrierList& barriers,
                                     const TextureRecord& texture,
                                     const NativeTextureRange& range,
                                     const LegacyStateChange& change)
        {
            const auto before = change.befores();
            appendTrackedTextureTransitions(barriers, *texture.resource.Get(), texture, range, before, change.after);
            if (change.after == D3D12_RESOURCE_STATE_UNORDERED_ACCESS &&
                std::ranges::find(before, D3D12_RESOURCE_STATE_UNORDERED_ACCESS) != before.end())
            {
                barriers.push_back(unorderedAccessBarrier(texture.resource.Get()));
            }
        }

        void appendLegacyAttachmentInitializationTransitions(LegacyBarrierList& beforeDiscard,
                                                             LegacyBarrierList& afterDiscard,
                                                             const TextureRecord& texture,
                                                             const NativeTextureRange& range,
                                                             const LegacyStateChange& change)
        {
            const auto initializationState = hasAny(texture.desc.usage, TextureUsage::depthStencil)
                                                 ? D3D12_RESOURCE_STATE_DEPTH_WRITE
                                                 : D3D12_RESOURCE_STATE_RENDER_TARGET;
            appendTrackedTextureTransitions(
                beforeDiscard, *texture.resource.Get(), texture, range, change.befores(), initializationState);
            appendTextureTransitions(
                afterDiscard, *texture.resource.Get(), texture, range, initializationState, change.after);
        }

        struct LegacyAttachmentInitialization final
        {
            std::size_t textureIndex = 0;
            LegacyBarrierList before = {};
            LegacyBarrierList after = {};
        };

        struct LegacyBarrierBatch final
        {
            LegacyBarrierList barriers;
            SmallVector<LegacyAttachmentInitialization, 1> attachmentInitializations;
            SmallVector<LegacyStateChange, 2> textureChanges;
        };

        void appendLegacyTextureBarrier(LegacyBarrierBatch& native,
                                        std::size_t textureIndex,
                                        const PreparedTextureBarrier& prepared,
                                        QueueType queueType,
                                        LegacyTextureStateRecording& recording)
        {
            const auto& tracker = prepared.resource->legacyStates;
            if (tracker == nullptr)
            {
                native.textureChanges.push_back(legacyStateChange(prepared.barrier, queueType));
            }
            else
            {
                const auto useIndex = recording.use(tracker);
                native.textureChanges.push_back(trackedLegacyStateChange(
                    prepared.barrier, *prepared.resource, prepared.range, queueType, recording[useIndex], useIndex));
            }
            const auto& change = native.textureChanges.back();
            if (!requiresAttachmentInitialization(prepared.barrier, *prepared.resource, queueType))
            {
                appendLegacyStateChange(native.barriers, *prepared.resource, prepared.range, change);
                return;
            }

            LegacyAttachmentInitialization initialization{.textureIndex = textureIndex};
            appendLegacyAttachmentInitializationTransitions(
                initialization.before, initialization.after, *prepared.resource, prepared.range, change);
            native.attachmentInitializations.push_back(std::move(initialization));
        }

        void appendLegacySamplerFeedbackBarrier(LegacyBarrierList& barriers,
                                                const PreparedSamplerFeedbackBarrier& prepared,
                                                QueueType queueType)
        {
            const auto before =
                legacyTextureState(prepared.barrier.layoutBefore, prepared.barrier.accessBefore, queueType);
            const auto after =
                legacyTextureState(prepared.barrier.layoutAfter, prepared.barrier.accessAfter, queueType);
            if (before != after)
            {
                barriers.push_back(wholeResourceTransition(prepared.resource->resource.Get(), before, after));
            }
            else if (after == D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            {
                barriers.push_back(unorderedAccessBarrier(prepared.resource->resource.Get()));
            }
        }

        [[nodiscard]] LegacyBarrierBatch buildLegacyBarrierBatch(const PreparedResourceBarrierBatch& prepared,
                                                                 QueueType queueType,
                                                                 LegacyTextureStateRecording& recording)
        {
            LegacyBarrierBatch native;
            native.barriers.reserve(prepared.global.size() + prepared.buffers.size() + prepared.textures.size() +
                                    prepared.samplerFeedbackMaps.size());
            native.textureChanges.reserve(prepared.textures.size());
            for (std::size_t index = 0; index < prepared.global.size(); ++index)
            {
                native.barriers.push_back(unorderedAccessBarrier(nullptr));
            }
            for (const auto& item : prepared.buffers)
            {
                appendLegacyBufferBarrier(native.barriers, item, queueType);
            }
            for (std::size_t index = 0; index < prepared.textures.size(); ++index)
            {
                appendLegacyTextureBarrier(native, index, prepared.textures[index], queueType, recording);
            }
            for (const auto& item : prepared.samplerFeedbackMaps)
            {
                appendLegacySamplerFeedbackBarrier(native.barriers, item, queueType);
            }
            return native;
        }

        [[nodiscard]] Status validateLegacyBarrierCounts(const LegacyBarrierBatch& native)
        {
            if (exceedsNativeBarrierCount(native.barriers.size()))
            {
                return std::unexpected(Error{
                    ErrorCode::resourceExhausted,
                    0,
                    "A resource-barrier batch expands beyond the native legacy barrier-count limit.",
                });
            }
            for (const auto& initialization : native.attachmentInitializations)
            {
                if (exceedsNativeBarrierCount(initialization.before.size()) ||
                    exceedsNativeBarrierCount(initialization.after.size()))
                {
                    return std::unexpected(Error{
                        ErrorCode::resourceExhausted,
                        0,
                        "An attachment initialization expands beyond the native legacy barrier-count limit.",
                    });
                }
            }
            return {};
        }

        void recordLegacyBarriers(ID3D12GraphicsCommandList& commandList,
                                  std::span<const D3D12_RESOURCE_BARRIER> barriers)
        {
            if (!barriers.empty())
            {
                commandList.ResourceBarrier(static_cast<std::uint32_t>(barriers.size()), barriers.data());
            }
        }

        void trackLegacyTextureChanges(LegacyTextureStateRecording& recording,
                                       const PreparedResourceBarrierBatch& prepared,
                                       std::span<const LegacyStateChange> changes) noexcept
        {
            for (std::size_t index = 0; index < changes.size(); ++index)
            {
                if (changes[index].use != untrackedLegacyTexture)
                {
                    const auto& item = prepared.textures[index];
                    trackLegacyStateChange(recording[changes[index].use], *item.resource, item.range, changes[index]);
                }
            }
        }

        [[nodiscard]] Result<LegacyBarrierBatch> buildTrackedLegacyBarrierBatch(
            const PreparedResourceBarrierBatch& prepared, QueueType queueType, LegacyTextureStateRecording& recording)
        {
            try
            {
                return buildLegacyBarrierBatch(prepared, queueType, recording);
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to resolve the legacy texture states of a resource-barrier batch.",
                });
            }
        }

        [[nodiscard]] Status recordLegacyBarrierBatch(QueueManager& queues,
                                                      detail::BackendCommandBuffer& commandBuffer,
                                                      ID3D12GraphicsCommandList& commandList,
                                                      const PreparedResourceBarrierBatch& prepared,
                                                      const BarrierCommandDependencies& dependencies)
        {
            auto recording = queues.legacyTextureStates(commandBuffer);
            if (!recording)
            {
                return std::unexpected(std::move(recording.error()));
            }
            auto built = buildTrackedLegacyBarrierBatch(prepared, commandBuffer.queueType(), **recording);
            if (!built)
            {
                return std::unexpected(std::move(built.error()));
            }
            const auto& native = *built;
            auto withinLimits = validateLegacyBarrierCounts(native);
            if (!withinLimits)
            {
                return withinLimits;
            }
            auto retained = retainBarrierCommandDependencies(queues, commandBuffer, prepared, dependencies);
            if (!retained)
            {
                return retained;
            }

            recordLegacyBarriers(commandList, native.barriers);
            for (const auto& initialization : native.attachmentInitializations)
            {
                const auto& item = prepared.textures[initialization.textureIndex];
                recordLegacyBarriers(commandList, initialization.before);
                discardTextureRange(commandList, *item.resource->resource.Get(), *item.resource, item.range);
                recordLegacyBarriers(commandList, initialization.after);
            }
            trackLegacyTextureChanges(**recording, prepared, native.textureChanges);
            test::noteResourceCommandRecorded();
            return {};
        }

        [[nodiscard]] Result<SmallVector<std::size_t, 2>> prepareTextureBarrierLayouts(
            LegacyTextureStateRecording& recording, std::span<const PreparedTextureBarrier> textures)
        {
            SmallVector<std::size_t, 2> uses;
            try
            {
                uses.reserve(textures.size());
                for (const auto& item : textures)
                {
                    const auto& tracker = item.resource->legacyStates;
                    if (tracker == nullptr)
                    {
                        uses.push_back(untrackedLegacyTexture);
                        continue;
                    }
                    if (item.barrier.layoutBefore != TextureLayout::undefined)
                    {
                        if (const auto* use = recording.find(tracker.get()); use != nullptr)
                        {
                            std::optional<Error> mismatch;
                            forEachTextureSubresource(*item.resource, item.range, [&](std::uint32_t subresource) {
                                const auto left = use->layoutLeft(subresource);
                                if (!mismatch.has_value() && left.has_value() && *left != item.barrier.layoutBefore)
                                {
                                    mismatch = textureLayoutMismatch(item.barrier.layoutBefore,
                                                                     *left,
                                                                     subresource,
                                                                     "this command buffer leaves it in");
                                }
                            });
                            if (mismatch.has_value())
                            {
                                return std::unexpected(std::move(*mismatch));
                            }
                        }
                    }
                    uses.push_back(recording.use(tracker));
                }
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to track the texture layouts of a resource-barrier batch.",
                });
            }
            return uses;
        }

        void recordTextureBarrierLayout(LegacyTextureStateUse& use,
                                        const TextureRecord& texture,
                                        const NativeTextureRange& range,
                                        const TextureBarrier& barrier) noexcept
        {
            forEachTextureSubresource(texture, range, [&](std::uint32_t subresource) {
                use.recordLayout(subresource, barrier.layoutBefore, barrier.layoutAfter);
            });
        }

        void recordTextureBarrierLayouts(LegacyTextureStateRecording& recording,
                                         std::span<const PreparedTextureBarrier> textures,
                                         std::span<const std::size_t> uses) noexcept
        {
            for (std::size_t index = 0; index < textures.size(); ++index)
            {
                if (uses[index] != untrackedLegacyTexture)
                {
                    recordTextureBarrierLayout(recording[uses[index]],
                                               *textures[index].resource,
                                               textures[index].range,
                                               textures[index].barrier);
                }
            }
        }

        struct OrderedTextureBarrier final
        {
            std::shared_ptr<TextureRecord> resource = {};
            NativeTextureRange range = {};
            std::array<D3D12_TEXTURE_BARRIER, 2> enhanced{};
            std::uint32_t enhancedCount = 0;
            LegacyBarrierList legacyBefore = {};
            LegacyBarrierList legacyAfter = {};
            LegacyStateChange legacyChange = {};
            bool discard = false;
            TextureBarrier layoutBarrier = {};
            std::size_t layoutUse = untrackedLegacyTexture;
        };

        void recordOrderedTextureBarrierLayout(LegacyTextureStateRecording* recording,
                                               const OrderedTextureBarrier& barrier) noexcept
        {
            if (recording != nullptr && barrier.layoutUse != untrackedLegacyTexture)
            {
                recordTextureBarrierLayout(
                    (*recording)[barrier.layoutUse], *barrier.resource, barrier.range, barrier.layoutBarrier);
            }
        }

        [[nodiscard]] Status validateOrderedTextureBarrierLayouts(const LegacyTextureStateRecording& recording,
                                                                  std::span<const OrderedTextureBarrier> barriers)
        {
            std::map<std::pair<const LegacyTextureStateTracker*, std::uint32_t>, TextureLayout> sequence;
            for (const auto& ordered : barriers)
            {
                if (ordered.layoutUse == untrackedLegacyTexture)
                {
                    continue;
                }
                const auto* tracker = ordered.resource->legacyStates.get();
                const auto* use = recording.find(tracker);
                std::optional<Error> mismatch;
                forEachTextureSubresource(*ordered.resource, ordered.range, [&](std::uint32_t subresource) {
                    const auto key = std::pair{tracker, subresource};
                    const auto earlier = sequence.find(key);
                    const auto left = earlier != sequence.end() ? std::optional{earlier->second}
                                      : use != nullptr          ? use->layoutLeft(subresource)
                                                                : std::nullopt;
                    const auto before = ordered.layoutBarrier.layoutBefore;
                    if (!mismatch.has_value() && before != TextureLayout::undefined && left.has_value() &&
                        *left != before)
                    {
                        mismatch =
                            textureLayoutMismatch(before, *left, subresource, "this command buffer leaves it in");
                    }
                    sequence[key] = ordered.layoutBarrier.layoutAfter;
                });
                if (mismatch.has_value())
                {
                    return std::unexpected(std::move(*mismatch));
                }
            }
            return {};
        }

        void fillEnhancedOrderedBarrier(OrderedTextureBarrier& ordered,
                                        const TextureBarrier& barrier,
                                        QueueType queueType)
        {
            if (requiresAttachmentInitialization(barrier, *ordered.resource, queueType))
            {
                ordered.enhanced = enhancedAttachmentInitializationBarriers(
                    *ordered.resource->resource.Get(), *ordered.resource, ordered.range, barrier);
                ordered.enhancedCount = 2;
                ordered.discard = true;
                return;
            }
            ordered.enhanced[0] = enhancedTextureBarrier(barrier, *ordered.resource, ordered.range, queueType);
            ordered.enhancedCount = 1;
        }

        [[nodiscard]] Status fillLegacyOrderedBarrier(OrderedTextureBarrier& ordered,
                                                      const TextureBarrier& barrier,
                                                      QueueType queueType,
                                                      LegacyStateChange change)
        {
            ordered.legacyChange = std::move(change);
            const auto& legacyChange = ordered.legacyChange;
            if (requiresAttachmentInitialization(barrier, *ordered.resource, queueType))
            {
                appendLegacyAttachmentInitializationTransitions(
                    ordered.legacyBefore, ordered.legacyAfter, *ordered.resource, ordered.range, legacyChange);
                ordered.discard = true;
            }
            else
            {
                appendLegacyStateChange(ordered.legacyBefore, *ordered.resource, ordered.range, legacyChange);
            }
            if (exceedsNativeBarrierCount(ordered.legacyBefore.size()) ||
                exceedsNativeBarrierCount(ordered.legacyAfter.size()))
            {
                return std::unexpected(Error{
                    ErrorCode::resourceExhausted,
                    0,
                    "An ordered texture barrier expands beyond the native legacy barrier-count limit.",
                });
            }
            return {};
        }

        class LegacyOrderedStateResolver final
        {
        public:
            explicit LegacyOrderedStateResolver(LegacyTextureStateRecording& recording) noexcept
                : recording_(&recording)
            {
            }

            [[nodiscard]] LegacyStateChange resolve(const TextureBarrier& barrier,
                                                    const TextureRecord& texture,
                                                    const NativeTextureRange& range,
                                                    QueueType queueType)
            {
                if (texture.legacyStates == nullptr)
                {
                    return legacyStateChange(barrier, queueType);
                }
                const auto useIndex = recording_->use(texture.legacyStates);
                auto found = pending_.find(useIndex);
                if (found == pending_.end())
                {
                    found = pending_.emplace(useIndex, (*recording_)[useIndex]).first;
                }
                auto change = trackedLegacyStateChange(barrier, texture, range, queueType, found->second, useIndex);
                trackLegacyStateChange(found->second, texture, range, change);
                return change;
            }

        private:
            LegacyTextureStateRecording* recording_ = nullptr;
            std::unordered_map<std::size_t, LegacyTextureStateUse> pending_;
        };

        [[nodiscard]] Result<OrderedTextureBarrier> prepareOrderedTextureBarrier(const TextureBarrier& barrier,
                                                                                 std::shared_ptr<TextureRecord> texture,
                                                                                 QueueType queueType,
                                                                                 bool enhancedBarriers,
                                                                                 LegacyOrderedStateResolver* legacy)
        {
            auto range = nativeTextureRange(*texture, barrier.subresources);
            if (!range)
            {
                return std::unexpected(std::move(range.error()));
            }
            if (!textureBarrierLayoutsSupported(barrier, *texture, queueType) ||
                !textureBarrierAccessesSupported(barrier, *texture))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "An ordered texture barrier is incompatible with its texture usage or command queue.",
                });
            }

            OrderedTextureBarrier ordered{
                .resource = std::move(texture),
                .range = *range,
            };
            if (enhancedBarriers)
            {
                fillEnhancedOrderedBarrier(ordered, barrier, queueType);
                return ordered;
            }
            auto change = legacy != nullptr ? legacy->resolve(barrier, *ordered.resource, ordered.range, queueType)
                                            : legacyStateChange(barrier, queueType);
            auto filled = fillLegacyOrderedBarrier(ordered, barrier, queueType, std::move(change));
            if (!filled)
            {
                return std::unexpected(std::move(filled.error()));
            }
            return ordered;
        }

        [[nodiscard]] bool validBufferAliasActivationScope(const BufferAliasActivation& activation,
                                                           QueueType queueType) noexcept
        {
            return activation.before != activation.after &&
                   validResourceBarrierScope(activation.stagesBefore, activation.accessBefore, queueType) &&
                   validResourceBarrierScope(activation.stagesAfter, activation.accessAfter, queueType) &&
                   hasWriteAccess(activation.accessAfter);
        }

        [[nodiscard]] Status validateBufferAliasTargets(const BufferAliasActivation& activation,
                                                        const BufferRecord& before,
                                                        const BufferRecord& after)
        {
            if (!placementsAlias(before.placement, after.placement))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Buffer alias activation requires overlapping placed resources from the same aliasable heap.",
                });
            }
            if (!supportsBufferAccess(before.desc.usage, activation.accessBefore) ||
                !supportsBufferAccess(after.desc.usage, activation.accessAfter))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A buffer alias activation access is incompatible with its buffer usage.",
                });
            }
            return {};
        }

        void recordBufferAliasActivation(ID3D12GraphicsCommandList7& commandList,
                                         const PreparedBufferAliasActivation& prepared,
                                         const BufferAliasActivation& activation)
        {
            const D3D12_BUFFER_BARRIER deactivation{
                .SyncBefore = nativeSync(activation.stagesBefore, activation.accessBefore),
                .SyncAfter = nativeSync(activation.stagesAfter, Access::none),
                .AccessBefore = nativeAccess(activation.accessBefore),
                .AccessAfter = D3D12_BARRIER_ACCESS_NO_ACCESS,
                .pResource = prepared.before->resource.Get(),
                .Offset = 0,
                .Size = prepared.before->desc.size,
            };
            D3D12_BARRIER_GROUP group{
                .Type = D3D12_BARRIER_TYPE_BUFFER,
                .NumBarriers = 1,
                .pBufferBarriers = &deactivation,
            };
            commandList.Barrier(1, &group);

            const D3D12_BUFFER_BARRIER activationBarrier{
                .SyncBefore = nativeSync(activation.stagesBefore, Access::none),
                .SyncAfter = nativeSync(activation.stagesAfter, activation.accessAfter),
                .AccessBefore = D3D12_BARRIER_ACCESS_NO_ACCESS,
                .AccessAfter = nativeAccess(activation.accessAfter),
                .pResource = prepared.after->resource.Get(),
                .Offset = 0,
                .Size = prepared.after->desc.size,
            };
            group.pBufferBarriers = &activationBarrier;
            commandList.Barrier(1, &group);
        }

        [[nodiscard]] bool validTextureAliasActivationScope(const TextureAliasActivation& activation,
                                                            QueueType queueType) noexcept
        {
            return activation.before != activation.after && activation.layoutBefore != TextureLayout::undefined &&
                   activation.layoutAfter != TextureLayout::undefined && validTextureLayout(activation.layoutBefore) &&
                   validTextureLayout(activation.layoutAfter) &&
                   validResourceBarrierScope(activation.stagesBefore, activation.accessBefore, queueType) &&
                   validResourceBarrierScope(activation.stagesAfter, activation.accessAfter, queueType) &&
                   hasWriteAccess(activation.accessAfter);
        }

        [[nodiscard]] bool textureAliasScopesSupported(const TextureAliasActivation& activation,
                                                       const TextureRecord& before,
                                                       const TextureRecord& after,
                                                       QueueType queueType) noexcept
        {
            const bool beforeLayoutSupported = supportsLayout(before.desc.usage, activation.layoutBefore) &&
                                               supportsTextureLayoutOnQueue(activation.layoutBefore, queueType);
            const bool afterLayoutSupported = supportsLayout(after.desc.usage, activation.layoutAfter) &&
                                              supportsTextureLayoutOnQueue(activation.layoutAfter, queueType);
            return beforeLayoutSupported && afterLayoutSupported &&
                   supportsTextureAccess(before.desc.usage, activation.accessBefore) &&
                   supportsTextureAccess(after.desc.usage, activation.accessAfter) &&
                   accessMatchesTextureLayout(activation.layoutBefore, activation.accessBefore) &&
                   accessMatchesTextureLayout(activation.layoutAfter, activation.accessAfter);
        }

        [[nodiscard]] Status validateTextureAliasTargets(const TextureAliasActivation& activation,
                                                         const TextureRecord& before,
                                                         const TextureRecord& after,
                                                         QueueType queueType)
        {
            if (!placementsAlias(before.placement, after.placement))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Texture alias activation requires overlapping placed resources from the same aliasable heap.",
                });
            }
            if (!textureAliasScopesSupported(activation, before, after, queueType))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A texture alias activation scope is incompatible with its texture usage, layout, or command "
                    "queue.",
                });
            }
            const bool attachment = hasAny(after.desc.usage, TextureUsage::renderTarget | TextureUsage::depthStencil);
            if (attachment && queueType != QueueType::graphics)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Aliased attachment metadata must be initialized on a graphics queue.",
                });
            }
            return {};
        }

        [[nodiscard]] Result<std::shared_ptr<PreparedTextureAliasActivation>> prepareTextureAliasActivation(
            std::shared_ptr<TextureRecord> before, std::shared_ptr<TextureRecord> after)
        {
            auto beforeRange = nativeWholeTextureRange(*before);
            if (!beforeRange)
            {
                return std::unexpected(std::move(beforeRange.error()));
            }
            auto afterRange = nativeWholeTextureRange(*after);
            if (!afterRange)
            {
                return std::unexpected(std::move(afterRange.error()));
            }
            auto prepared = std::make_shared<PreparedTextureAliasActivation>(PreparedTextureAliasActivation{
                .before = std::move(before),
                .after = std::move(after),
                .beforeRange = *beforeRange,
                .afterRange = *afterRange,
            });
            prepared->residencies = {prepared->before->residency, prepared->after->residency};
            return prepared;
        }

        void recordTextureAliasActivation(ID3D12GraphicsCommandList7& commandList,
                                          const PreparedTextureAliasActivation& prepared,
                                          const EnhancedTextureAliasPlan& plan)
        {
            for (std::uint32_t stepIndex = 0; stepIndex < plan.stepCount; ++stepIndex)
            {
                const auto& step = plan.steps[stepIndex];
                const bool targetBefore = step.target == EnhancedTextureAliasTarget::before;
                const auto& targetTexture = targetBefore ? *prepared.before : *prepared.after;
                const auto& targetRange = targetBefore ? prepared.beforeRange : prepared.afterRange;
                const D3D12_TEXTURE_BARRIER nativeBarrier{
                    .SyncBefore = toNativeBarrierSync(step.syncBefore),
                    .SyncAfter = toNativeBarrierSync(step.syncAfter),
                    .AccessBefore = toNativeBarrierAccess(step.accessBefore),
                    .AccessAfter = toNativeBarrierAccess(step.accessAfter),
                    .LayoutBefore = toNativeBarrierLayout(step.layoutBefore),
                    .LayoutAfter = toNativeBarrierLayout(step.layoutAfter),
                    .pResource = targetTexture.resource.Get(),
                    .Subresources = toNativeSubresourceRange(targetRange),
                    .Flags = toNativeTextureBarrierFlags(step.flags),
                };
                const D3D12_BARRIER_GROUP group{
                    .Type = D3D12_BARRIER_TYPE_TEXTURE,
                    .NumBarriers = 1,
                    .pTextureBarriers = &nativeBarrier,
                };
                commandList.Barrier(1, &group);
                if (step.discardAfterBarrier)
                {
                    discardTextureRange(commandList, *targetTexture.resource.Get(), targetTexture, targetRange);
                }
            }
        }

    } // namespace

    Result<TextureLayoutUse> resource_detail::prepareTextureLayoutUse(QueueManager& queues,
                                                                      detail::BackendCommandBuffer& commandBuffer,
                                                                      const TextureRecord& texture,
                                                                      const TextureSubresourceRange& subresources,
                                                                      TextureLayout layout,
                                                                      Access access,
                                                                      bool enhancedBarriers,
                                                                      std::string_view operation)
    {
        auto range = nativeTextureRange(texture, subresources);
        if (!range)
        {
            return std::unexpected(std::move(range.error()));
        }
        if (texture.legacyStates == nullptr)
        {
            return TextureLayoutUse{.layout = layout};
        }
        auto recording = queues.legacyTextureStates(commandBuffer);
        if (!recording)
        {
            return std::unexpected(std::move(recording.error()));
        }
        try
        {
            TextureLayoutUse prepared{
                .recording = *recording,
                .layout = layout,
            };
            if (!enhancedBarriers)
            {
                prepared.legacyState = legacyTextureState(layout, access, commandBuffer.queueType());
            }
            prepared.subresources.reserve(subresourceCount(*range));
            forEachTextureSubresource(
                texture, *range, [&](std::uint32_t subresource) { prepared.subresources.push_back(subresource); });
            if (const auto* use = (*recording)->find(texture.legacyStates.get()); use != nullptr)
            {
                for (const auto subresource : prepared.subresources)
                {
                    const auto left = use->layoutLeft(subresource);
                    if (left.has_value() && *left != layout)
                    {
                        return std::unexpected(Error{
                            ErrorCode::invalidState,
                            0,
                            std::string(operation) + " requires the " + std::string(textureLayoutName(layout)) +
                                " layout, but this command buffer leaves subresource " + std::to_string(subresource) +
                                " in " + std::string(textureLayoutName(*left)) + ".",
                        });
                    }
                }
            }
            prepared.use = (*recording)->use(texture.legacyStates);
            return prepared;
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to track the texture layout of a recorded command.",
            });
        }
    }

    void resource_detail::commitTextureLayoutUse(const TextureLayoutUse& use) noexcept
    {
        if (use.recording == nullptr)
        {
            return;
        }
        auto& tracked = (*use.recording)[use.use];
        for (const auto subresource : use.subresources)
        {
            tracked.recordLayout(subresource, use.layout, use.layout);
            if (use.legacyState.has_value())
            {
                tracked.transition(subresource, *use.legacyState, *use.legacyState, false);
            }
        }
    }

    struct NativeTextureBarrierSequence::Impl final
    {
        ComPtr<ID3D12GraphicsCommandList7> enhancedCommandList;
        std::vector<OrderedTextureBarrier> barriers;
        LegacyTextureStateRecording* legacyStates = nullptr;
        LegacyTextureStateRecording* layoutStates = nullptr;
    };

    NativeTextureBarrierSequence::NativeTextureBarrierSequence() : impl_(std::make_unique<Impl>())
    {
    }

    NativeTextureBarrierSequence::~NativeTextureBarrierSequence() = default;
    NativeTextureBarrierSequence::NativeTextureBarrierSequence(NativeTextureBarrierSequence&&) noexcept = default;
    NativeTextureBarrierSequence& NativeTextureBarrierSequence::operator=(NativeTextureBarrierSequence&&) noexcept =
        default;

    std::size_t NativeTextureBarrierSequence::size() const noexcept
    {
        return impl_->barriers.size();
    }

    void NativeTextureBarrierSequence::record(std::size_t index, ID3D12GraphicsCommandList& commandList) const noexcept
    {
        assert(index < impl_->barriers.size());
        if (index >= impl_->barriers.size())
        {
            return;
        }

        const auto& barrier = impl_->barriers[index];
        if (impl_->enhancedCommandList != nullptr)
        {
            assert(barrier.enhancedCount != 0);
            const D3D12_BARRIER_GROUP beginGroup{
                .Type = D3D12_BARRIER_TYPE_TEXTURE,
                .NumBarriers = 1,
                .pTextureBarriers = barrier.enhanced.data(),
            };
            impl_->enhancedCommandList->Barrier(1, &beginGroup);
            if (barrier.discard)
            {
                discardTextureRange(commandList, *barrier.resource->resource.Get(), *barrier.resource, barrier.range);
            }
            if (barrier.enhancedCount == 2)
            {
                const D3D12_BARRIER_GROUP endGroup{
                    .Type = D3D12_BARRIER_TYPE_TEXTURE,
                    .NumBarriers = 1,
                    .pTextureBarriers = barrier.enhanced.data() + 1,
                };
                impl_->enhancedCommandList->Barrier(1, &endGroup);
            }
            recordOrderedTextureBarrierLayout(impl_->layoutStates, barrier);
            return;
        }

        if (!barrier.legacyBefore.empty())
        {
            commandList.ResourceBarrier(static_cast<UINT>(barrier.legacyBefore.size()), barrier.legacyBefore.data());
        }
        if (barrier.discard)
        {
            discardTextureRange(commandList, *barrier.resource->resource.Get(), *barrier.resource, barrier.range);
        }
        if (!barrier.legacyAfter.empty())
        {
            commandList.ResourceBarrier(static_cast<UINT>(barrier.legacyAfter.size()), barrier.legacyAfter.data());
        }
        if (impl_->legacyStates != nullptr && barrier.legacyChange.use != untrackedLegacyTexture)
        {
            trackLegacyStateChange((*impl_->legacyStates)[barrier.legacyChange.use],
                                   *barrier.resource,
                                   barrier.range,
                                   barrier.legacyChange);
        }
        recordOrderedTextureBarrierLayout(impl_->layoutStates, barrier);
    }

    Status ResourceManager::bufferBarrier(QueueManager& queues,
                                          detail::BackendCommandBuffer& commandBuffer,
                                          const BufferBarrier& barrier)
    {
        const std::array barriers{barrier};
        return resourceBarriers(queues, commandBuffer, ResourceBarrierBatch{.bufferBarriers = barriers});
    }

    Status ResourceManager::activateBufferAlias(QueueManager& queues,
                                                detail::BackendCommandBuffer& commandBuffer,
                                                const BufferAliasActivation& activation)
    {
        if (!impl_->enhancedBarriers)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Placed-buffer alias activation requires D3D12 enhanced barriers.",
            });
        }
        if (!validBufferAliasActivationScope(activation, commandBuffer.queueType()))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Buffer alias activation requires distinct aliases plus compatible scopes and a destination write.",
            });
        }

        auto before = impl_->buffer(activation.before);
        if (!before)
        {
            return std::unexpected(std::move(before.error()));
        }
        auto after = impl_->buffer(activation.after);
        if (!after)
        {
            return std::unexpected(std::move(after.error()));
        }
        auto compatible = validateBufferAliasTargets(activation, **before, **after);
        if (!compatible)
        {
            return compatible;
        }

        auto prepared = std::make_shared<PreparedBufferAliasActivation>(PreparedBufferAliasActivation{
            .before = std::move(*before),
            .after = std::move(*after),
        });
        prepared->residencies = {prepared->before->residency, prepared->after->residency};
        auto commandList = queues.commandList7(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto retained = queues.retainResource(commandBuffer, prepared, prepared->residencies);
        if (!retained)
        {
            return retained;
        }

        recordBufferAliasActivation(**commandList, *prepared, activation);
        return {};
    }

    [[nodiscard]] Status ResourceManager::Impl::prepareBufferBarriers(
        PreparedResourceBarrierBatch& prepared,
        std::span<const BufferBarrier> barriers,
        QueueType queueType,
        std::unique_lock<std::mutex>& sparseMappingLock) const
    {
        const auto repeatedBuffer =
            firstRepeatedKeyIndex(barriers.size(), [&](std::size_t index) { return barriers[index].buffer.identity; });
        for (std::size_t index = 0; index < barriers.size(); ++index)
        {
            const auto& barrier = barriers[index];
            if (!validResourceBarrierScope(barrier.stagesBefore, barrier.accessBefore, queueType) ||
                !validResourceBarrierScope(barrier.stagesAfter, barrier.accessAfter, queueType))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A buffer barrier contains an invalid or queue-incompatible stage/access scope.",
                });
            }
            auto record = lookUpUnderSparseMappingLock(sparseMappingLock, [&] { return buffer(barrier.buffer); });
            if (!record)
            {
                return std::unexpected(std::move(record.error()));
            }
            auto valid = validateBufferBarrierTarget(barrier, **record);
            if (!valid)
            {
                return valid;
            }
            if (index == repeatedBuffer)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A resource-barrier batch cannot contain multiple transitions for one buffer.",
                });
            }
            const auto resolvedSize = resolveSize((*record)->desc.size, barrier.offset, barrier.size);
            prepared.residencies.push_back((*record)->residency);
            prepared.buffers.push_back(PreparedBufferBarrier{
                .barrier = barrier,
                .resource = std::move(*record),
                .resolvedSize = resolvedSize,
            });
        }
        return {};
    }

    [[nodiscard]] Status ResourceManager::Impl::prepareTextureBarriers(
        PreparedResourceBarrierBatch& prepared,
        std::span<const TextureBarrier> barriers,
        QueueType queueType,
        std::unique_lock<std::mutex>& sparseMappingLock) const
    {
        const auto prepareTexture = [&](const TextureBarrier& barrier) -> Status {
            if (!validTextureBarrierScope(barrier, queueType))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A texture barrier contains an unknown stage, access, or layout value, ends in the undefined "
                    "layout, or uses present outside its no-access endpoint scope.",
                });
            }
            auto record = lookUpUnderSparseMappingLock(sparseMappingLock, [&] { return texture(barrier.texture); });
            if (!record)
            {
                return std::unexpected(std::move(record.error()));
            }
            auto range = nativeTextureRange(**record, barrier.subresources);
            if (!range)
            {
                return std::unexpected(std::move(range.error()));
            }
            auto valid = validateTextureBarrierTarget(barrier, **record, queueType);
            if (!valid)
            {
                return valid;
            }
            prepared.residencies.push_back((*record)->residency);
            prepared.textures.push_back(PreparedTextureBarrier{
                .barrier = barrier,
                .resource = std::move(*record),
                .range = *range,
            });
            return {};
        };

        Status firstRejected = {};
        for (const auto& barrier : barriers)
        {
            firstRejected = prepareTexture(barrier);
            if (!firstRejected)
            {
                break;
            }
        }
        if (anyTextureBarriersOverlap(prepared.textures))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A resource-barrier batch cannot contain overlapping transitions for one texture.",
            });
        }
        return firstRejected;
    }

    [[nodiscard]] Status ResourceManager::Impl::prepareSamplerFeedbackBarriers(
        PreparedResourceBarrierBatch& prepared,
        std::span<const SamplerFeedbackBarrier> barriers,
        QueueType queueType) const
    {
        const auto repeatedFeedbackMap = firstRepeatedKeyIndex(
            barriers.size(), [&](std::size_t index) { return barriers[index].feedbackMap.identity; });
        for (std::size_t index = 0; index < barriers.size(); ++index)
        {
            const auto& barrier = barriers[index];
            if (!validSamplerFeedbackBarrierScope(barrier, queueType))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A sampler-feedback barrier contains an invalid or queue-incompatible stage, access, or layout.",
                });
            }
            auto record = samplerFeedbackMap(barrier.feedbackMap);
            if (!record)
            {
                return std::unexpected(std::move(record.error()));
            }
            if (index == repeatedFeedbackMap)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A resource-barrier batch cannot contain multiple transitions for one sampler-feedback map.",
                });
            }
            prepared.samplerFeedbackMaps.push_back({
                .barrier = barrier,
                .resource = std::move(*record),
            });
        }
        return {};
    }

    [[nodiscard]] Result<std::shared_ptr<PreparedResourceBarrierBatch>> ResourceManager::Impl::
        prepareResourceBarrierBatch(const ResourceBarrierBatch& barriers,
                                    QueueType queueType,
                                    std::unique_lock<std::mutex>& sparseMappingLock) const
    {
        auto prepared = std::make_shared<PreparedResourceBarrierBatch>();
        prepared->global.reserve(barriers.globalBarriers.size());
        prepared->buffers.reserve(barriers.bufferBarriers.size());
        prepared->textures.reserve(barriers.textureBarriers.size());
        prepared->samplerFeedbackMaps.reserve(barriers.samplerFeedbackBarriers.size());
        prepared->residencies.reserve(barriers.bufferBarriers.size() + barriers.textureBarriers.size());

        auto globalPrepared = prepareGlobalBarriers(*prepared, barriers.globalBarriers, queueType);
        if (!globalPrepared)
        {
            return std::unexpected(std::move(globalPrepared.error()));
        }
        auto buffersPrepared = prepareBufferBarriers(*prepared, barriers.bufferBarriers, queueType, sparseMappingLock);
        if (!buffersPrepared)
        {
            return std::unexpected(std::move(buffersPrepared.error()));
        }
        auto texturesPrepared =
            prepareTextureBarriers(*prepared, barriers.textureBarriers, queueType, sparseMappingLock);
        if (!texturesPrepared)
        {
            return std::unexpected(std::move(texturesPrepared.error()));
        }
        auto samplerFeedbackPrepared =
            prepareSamplerFeedbackBarriers(*prepared, barriers.samplerFeedbackBarriers, queueType);
        if (!samplerFeedbackPrepared)
        {
            return std::unexpected(std::move(samplerFeedbackPrepared.error()));
        }
        return prepared;
    }

    Status ResourceManager::resourceBarriers(QueueManager& queues,
                                             detail::BackendCommandBuffer& commandBuffer,
                                             const ResourceBarrierBatch& barriers)
    {
        if (isEmptyBarrierBatch(barriers) || anyBarrierCountExceedsNativeLimit(barriers))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A resource-barrier batch must be non-empty and fit within native barrier-count limits.",
            });
        }

        std::unique_lock<std::mutex> mappingLock(impl_->sparseMappingsMutex, std::defer_lock);
        if (impl_->activeSparseResourceCount.load(std::memory_order_acquire) != 0)
        {
            mappingLock.lock();
        }
        auto prepared = impl_->prepareResourceBarrierBatch(barriers, commandBuffer.queueType(), mappingLock);
        if (!prepared)
        {
            return std::unexpected(std::move(prepared.error()));
        }
        if (!impl_->enhancedBarriers)
        {
            auto representable = validateLegacyBarrierBatch(**prepared, commandBuffer.queueType());
            if (!representable)
            {
                return representable;
            }
        }
        auto layoutRecording = queues.legacyTextureStates(commandBuffer);
        if (!layoutRecording)
        {
            return std::unexpected(std::move(layoutRecording.error()));
        }
        auto layoutUses = prepareTextureBarrierLayouts(**layoutRecording, (*prepared)->textures);
        if (!layoutUses)
        {
            return std::unexpected(std::move(layoutUses.error()));
        }

        auto sparseMappingDependencies = captureSparseMappings(**prepared);
        if (!sparseMappingDependencies)
        {
            return std::unexpected(std::move(sparseMappingDependencies.error()));
        }
        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto dependencies = stageBarrierCommandDependencies(*prepared, std::move(*sparseMappingDependencies));
        if (!dependencies)
        {
            return std::unexpected(std::move(dependencies.error()));
        }

        auto recorded =
            impl_->enhancedBarriers
                ? recordEnhancedBarrierBatch(queues, commandBuffer, **commandList, **prepared, *dependencies)
                : recordLegacyBarrierBatch(queues, commandBuffer, **commandList, **prepared, *dependencies);
        if (recorded)
        {
            recordTextureBarrierLayouts(**layoutRecording, (*prepared)->textures, *layoutUses);
        }
        return recorded;
    }

    Status ResourceManager::textureBarrier(QueueManager& queues,
                                           detail::BackendCommandBuffer& commandBuffer,
                                           const TextureBarrier& barrier)
    {
        const std::array barriers{barrier};
        return resourceBarriers(queues, commandBuffer, ResourceBarrierBatch{.textureBarriers = barriers});
    }

    Result<NativeTextureBarrierSequence> ResourceManager::prepareNativeTextureBarrierSequence(
        QueueManager& queues, detail::BackendCommandBuffer& commandBuffer, std::span<const TextureBarrier> barriers)
    {
        if (barriers.empty() || exceedsNativeBarrierCount(barriers.size()))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "An ordered native texture-barrier sequence must be non-empty and fit within native limits.",
            });
        }

        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }

        NativeTextureBarrierSequence result;
        std::optional<LegacyOrderedStateResolver> legacyStates;
        auto layoutRecording = queues.legacyTextureStates(commandBuffer);
        if (!layoutRecording)
        {
            return std::unexpected(std::move(layoutRecording.error()));
        }
        result.impl_->layoutStates = *layoutRecording;
        try
        {
            result.impl_->barriers.reserve(barriers.size());
            if (impl_->enhancedBarriers)
            {
                auto enhanced = queues.commandList7(commandBuffer);
                if (!enhanced)
                {
                    return std::unexpected(std::move(enhanced.error()));
                }
                result.impl_->enhancedCommandList = *enhanced;
            }
            else
            {
                auto recording = queues.legacyTextureStates(commandBuffer);
                if (!recording)
                {
                    return std::unexpected(std::move(recording.error()));
                }
                result.impl_->legacyStates = *recording;
                legacyStates.emplace(**recording);
            }

            for (const auto& barrier : barriers)
            {
                if (!validTextureBarrierScope(barrier, commandBuffer.queueType()))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "An ordered texture barrier contains an invalid or queue-incompatible scope or layout.",
                    });
                }
                auto texture = impl_->texture(barrier.texture);
                if (!texture)
                {
                    return std::unexpected(std::move(texture.error()));
                }
                auto ordered = prepareOrderedTextureBarrier(barrier,
                                                            std::move(*texture),
                                                            commandBuffer.queueType(),
                                                            impl_->enhancedBarriers,
                                                            legacyStates ? &*legacyStates : nullptr);
                if (!ordered)
                {
                    return std::unexpected(std::move(ordered.error()));
                }
                ordered->layoutBarrier = barrier;
                if (ordered->resource->legacyStates != nullptr)
                {
                    ordered->layoutUse = (**layoutRecording).use(ordered->resource->legacyStates);
                }
                result.impl_->barriers.push_back(std::move(*ordered));
            }
            if (auto layouts = validateOrderedTextureBarrierLayouts(**layoutRecording, result.impl_->barriers);
                !layouts)
            {
                return std::unexpected(std::move(layouts.error()));
            }
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to prepare an ordered native texture-barrier sequence.",
            });
        }
        return result;
    }

    Status ResourceManager::samplerFeedbackBarrier(QueueManager& queues,
                                                   detail::BackendCommandBuffer& commandBuffer,
                                                   const SamplerFeedbackBarrier& barrier)
    {
        const std::array barriers{barrier};
        return resourceBarriers(queues, commandBuffer, ResourceBarrierBatch{.samplerFeedbackBarriers = barriers});
    }

    Status ResourceManager::activateTextureAlias(QueueManager& queues,
                                                 detail::BackendCommandBuffer& commandBuffer,
                                                 const TextureAliasActivation& activation)
    {
        if (!impl_->enhancedBarriers)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Placed-texture alias activation requires D3D12 enhanced barriers.",
            });
        }
        if (!validTextureAliasActivationScope(activation, commandBuffer.queueType()))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Texture alias activation requires distinct aliases, defined layouts, compatible scopes, and a "
                "destination write.",
            });
        }

        auto before = impl_->texture(activation.before);
        if (!before)
        {
            return std::unexpected(std::move(before.error()));
        }
        auto after = impl_->texture(activation.after);
        if (!after)
        {
            return std::unexpected(std::move(after.error()));
        }
        auto compatible = validateTextureAliasTargets(activation, **before, **after, commandBuffer.queueType());
        if (!compatible)
        {
            return compatible;
        }

        auto prepared = prepareTextureAliasActivation(std::move(*before), std::move(*after));
        if (!prepared)
        {
            return std::unexpected(std::move(prepared.error()));
        }
        auto commandList = queues.commandList7(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        const auto& afterTexture = *(*prepared)->after;
        auto layoutRecording = queues.legacyTextureStates(commandBuffer);
        if (!layoutRecording)
        {
            return std::unexpected(std::move(layoutRecording.error()));
        }
        auto afterRange = nativeWholeTextureRange(afterTexture);
        if (!afterRange)
        {
            return std::unexpected(std::move(afterRange.error()));
        }
        std::size_t afterUse = untrackedLegacyTexture;
        try
        {
            if (afterTexture.legacyStates != nullptr)
            {
                afterUse = (**layoutRecording).use(afterTexture.legacyStates);
            }
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to track the layout of an activated texture alias.",
            });
        }
        auto retained = queues.retainResource(commandBuffer, *prepared, (*prepared)->residencies);
        if (!retained)
        {
            return retained;
        }

        const auto plan =
            planEnhancedTextureAliasActivation(activation, (*prepared)->after->desc.usage, commandBuffer.queueType());
        recordTextureAliasActivation(**commandList, **prepared, plan);
        if (afterUse != untrackedLegacyTexture)
        {
            recordTextureBarrierLayout((**layoutRecording)[afterUse],
                                       afterTexture,
                                       *afterRange,
                                       TextureBarrier{.texture = activation.after,
                                                      .layoutBefore = TextureLayout::undefined,
                                                      .layoutAfter = activation.layoutAfter});
        }
        return {};
    }

} // namespace rachis::d3d12
