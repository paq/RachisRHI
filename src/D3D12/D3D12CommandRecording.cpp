#include "D3D12/D3D12PipelineInternal.h"

#include "Internal/PortableValidation.h"

namespace rachis::d3d12
{
    using Microsoft::WRL::ComPtr;
    using namespace pipeline_detail;

    namespace
    {

        [[nodiscard]] Status validateBindingSetBatchRange(std::span<const NativePipelineLayoutSlot> slots,
                                                          std::uint32_t firstSet,
                                                          std::span<const BindingSetHandle> bindingSets,
                                                          std::span<const std::uint32_t> dynamicOffsets)
        {
            if (bindingSets.empty() || firstSet >= slots.size() || bindingSets.size() > slots.size() - firstSet ||
                bindingSets.size() > maxBindingSets)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A binding-set batch must be non-empty and fit contiguously in the active pipeline layout.",
                });
            }

            std::size_t requiredDynamicOffsetCount = 0;
            for (std::size_t index = 0; index < bindingSets.size(); ++index)
            {
                requiredDynamicOffsetCount += slots[firstSet + index].dynamicBufferCount;
            }
            if (dynamicOffsets.size() != requiredDynamicOffsetCount)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A binding-set batch requires exactly one offset for each dynamic buffer across every set.",
                });
            }
            return {};
        }

        void mergeBindingSetResidencies(BindingSetBatchOwner& batch, const NativeBindingSet& nativeSet)
        {
            if (!batch.usesMergedResidencies)
            {
                batch.usesMergedResidencies = true;
                for (const auto& previousSet : batch.activeSets())
                {
                    for (const auto& residency : previousSet.staticResidencies)
                    {
                        batch.mergedResidencies.push_back(residency);
                    }
                }
            }
            for (const auto& residency : nativeSet.staticResidencies)
            {
                batch.mergedResidencies.push_back(residency);
            }
            for (const auto& residency : nativeSet.extraResidencies)
            {
                batch.mergedResidencies.push_back(residency);
            }
        }

        template<typename T, std::size_t Capacity, std::size_t SourceCapacity>
        void appendUnique(SmallVector<T, Capacity>& destination, const SmallVector<T, SourceCapacity>& values)
        {
            for (const auto& value : values)
            {
                if (std::ranges::find(destination.span(), value) == destination.span().end())
                {
                    destination.push_back(value);
                }
            }
        }

        void transferRetainedOwners(BindingSetBatchInternalOwner& internal,
                                    std::size_t index,
                                    NativeBindingSet& nativeSet)
        {
            internal.poolRetentions[index] = std::move(nativeSet.bindingPoolRetention);
            internal.resourceDescriptors[index] = std::move(nativeSet.resourceDescriptorOwner);
            internal.samplerDescriptors[index] = std::move(nativeSet.samplerDescriptorOwner);
            ++internal.setCount;
            for (auto& sparseOwner : nativeSet.sparseMappingOwners)
            {
                internal.sparseMappingOwners.push_back(std::move(sparseOwner));
            }
            for (auto& asLease : nativeSet.accelerationStructureLeases)
            {
                internal.accelerationStructureLeases.push_back(std::move(asLease));
            }
        }

        [[nodiscard]] Result<std::shared_ptr<DynamicDescriptorArena>> recordingDynamicDescriptorArena(
            QueueManager& queues, detail::BackendCommandBuffer& commandBuffer)
        {
            auto existing = queues.recordingDescriptorArena(commandBuffer);
            if (!existing)
            {
                return std::unexpected(std::move(existing.error()));
            }
            if (*existing != nullptr)
            {
                return std::static_pointer_cast<DynamicDescriptorArena>(std::move(*existing));
            }
            auto created = std::make_shared<DynamicDescriptorArena>();
            auto attached = queues.setRecordingDescriptorArena(commandBuffer, created);
            if (!attached)
            {
                return std::unexpected(std::move(attached.error()));
            }
            return created;
        }

    } // namespace

    namespace pipeline_detail
    {

        Result<std::shared_ptr<BindingSetBatchOwner>> prepareBindingSetBatch(
            QueueManager& queues,
            detail::BackendCommandBuffer& commandBuffer,
            BindingManager& bindings,
            ResourceManager& resources,
            std::span<const NativePipelineLayoutSlot> slots,
            std::uint32_t firstSet,
            std::span<const BindingSetHandle> bindingSets,
            std::span<const std::uint32_t> dynamicOffsets)
        {
            auto validRange = validateBindingSetBatchRange(slots, firstSet, bindingSets, dynamicOffsets);
            if (!validRange)
            {
                return std::unexpected(std::move(validRange.error()));
            }
            auto recordingId = queues.recordingId(commandBuffer);
            if (!recordingId)
            {
                return std::unexpected(std::move(recordingId.error()));
            }

            try
            {
                auto owner = std::make_shared<BindingSetBatchOwner>();
                auto internal = std::make_shared<BindingSetBatchInternalOwner>();
                const bool singleSetFastPath = bindingSets.size() == 1;
                std::size_t dynamicOffsetCursor = 0;
                for (std::size_t index = 0; index < bindingSets.size(); ++index)
                {
                    const auto& slot = slots[firstSet + index];
                    const auto offsets = dynamicOffsets.subspan(dynamicOffsetCursor, slot.dynamicBufferCount);
                    dynamicOffsetCursor += slot.dynamicBufferCount;
                    if (slot.rootPlan.dynamicTableDescriptorCount != 0 && internal->dynamicDescriptorArena == nullptr)
                    {
                        auto arena = recordingDynamicDescriptorArena(queues, commandBuffer);
                        if (!arena)
                        {
                            return std::unexpected(std::move(arena.error()));
                        }
                        internal->dynamicDescriptorArena = std::move(*arena);
                    }
                    auto nativeSet = bindings.nativeBindingSet(resources,
                                                               bindingSets[index],
                                                               offsets,
                                                               &slot.rootPlan,
                                                               slot.bindingLayoutOwner.get(),
                                                               *recordingId,
                                                               internal->dynamicDescriptorArena.get());
                    if (!nativeSet)
                    {
                        return std::unexpected(std::move(nativeSet.error()));
                    }
                    if (nativeSet->bindingLayoutOwner != slot.bindingLayoutOwner)
                    {
                        return std::unexpected(Error{
                            ErrorCode::invalidArgument,
                            0,
                            "A binding set layout does not match its active pipeline layout batch slot.",
                        });
                    }

                    if (!singleSetFastPath || !nativeSet->extraResidencies.empty())
                    {
                        mergeBindingSetResidencies(*owner, *nativeSet);
                    }
                    appendUnique(owner->swapchainImageLeases, nativeSet->swapchainImageLeases);
                    transferRetainedOwners(*internal, index, *nativeSet);
                    appendUnique(owner->sparseMappingDependencies, nativeSet->sparseMappingDependencies);
                    owner->sets[owner->setCount++] = std::move(*nativeSet);
                }
                if (owner->usesMergedResidencies)
                {
                    deduplicateSharedPointers(owner->mergedResidencies);
                }
                owner->internalOwner = std::move(internal);
                return owner;
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to allocate D3D12 binding-set batch retention state.",
                });
            }
        }

        Status retainBindingSetBatchDependencies(QueueManager& queues,
                                                 detail::BackendCommandBuffer& commandBuffer,
                                                 const std::shared_ptr<BindingSetBatchOwner>& prepared)
        {
            const std::array dependencies{
                QueueManager::ResourceRetention{.owner = prepared},
                QueueManager::ResourceRetention{
                    .owner = prepared->internalOwner,
                    .residencySet = prepared->residencySpan(),
                    .internal = true,
                },
            };
            std::array<InPlacePublicationReference, maxBindingSets> inPlacePublications{};
            std::size_t inPlaceResidencyCount = 0;
            for (const auto& set : prepared->activeSets())
            {
                if (set.inPlacePublication != nullptr)
                {
                    inPlacePublications[inPlaceResidencyCount++] = set.inPlacePublication;
                }
            }
            return queues.retainCommandDependencies(commandBuffer,
                                                    dependencies,
                                                    prepared->swapchainImageLeases.span(),
                                                    prepared->sparseMappingDependencies.span(),
                                                    std::span{inPlacePublications}.first(inPlaceResidencyCount));
        }

    } // namespace pipeline_detail

    namespace
    {

        [[nodiscard]] Status retainIndirectBufferDependencies(QueueManager& queues,
                                                              detail::BackendCommandBuffer& commandBuffer,
                                                              const BufferBindingSource& arguments,
                                                              const std::optional<BufferBindingSource>& count)
        {
            std::array dependencies{
                QueueManager::ResourceRetention{
                    .owner = arguments.owner,
                    .residency = arguments.residency,
                },
                QueueManager::ResourceRetention{},
            };
            std::size_t dependencyCount = 1;
            if (count.has_value() && count->owner != arguments.owner)
            {
                dependencies[dependencyCount++] = QueueManager::ResourceRetention{
                    .owner = count->owner,
                    .residency = count->residency,
                };
            }
            return queues.retainCommandDependencies(commandBuffer, std::span{dependencies}.first(dependencyCount), {});
        }

        struct IndirectCommandLayout final
        {
            BufferHandle argumentBuffer = {};
            std::uint64_t argumentOffset = 0;
            std::uint32_t maxCommandCount = 0;
            std::uint32_t recordSize = 0;
            std::uint32_t stride = 0;
            BufferHandle countBuffer = {};
            std::uint64_t countOffset = 0;
        };

        template<typename IndirectDesc>
        [[nodiscard]] IndirectCommandLayout indirectCommandLayout(const IndirectDesc& desc,
                                                                  std::uint32_t maxCommandCount,
                                                                  std::uint32_t recordSize) noexcept
        {
            return {
                .argumentBuffer = desc.argumentBuffer,
                .argumentOffset = desc.argumentOffset,
                .maxCommandCount = maxCommandCount,
                .recordSize = recordSize,
                .stride = desc.stride == 0 ? recordSize : desc.stride,
                .countBuffer = desc.countBuffer,
                .countOffset = desc.countOffset,
            };
        }

        struct IndirectCommandErrorMessages final
        {
            std::string_view invalidLayout;
            std::string_view invalidArgumentBuffer;
            std::string_view argumentSpanOverflow;
            std::string_view argumentRecordsOutOfBounds;
            std::string_view invalidCountBuffer;
        };

        struct ResolvedIndirectCommand final
        {
            IndirectCommandLayout layout;
            BufferBindingSource arguments;
            std::optional<BufferBindingSource> count;
        };

        [[nodiscard]] Status validateIndirectArgumentSpan(const IndirectCommandLayout& layout,
                                                          std::uint64_t argumentBufferSize,
                                                          const IndirectCommandErrorMessages& errorMessages)
        {
            if (layout.maxCommandCount == 0)
            {
                return {};
            }
            const auto precedingRecords = static_cast<std::uint64_t>(layout.maxCommandCount - 1U);
            if (precedingRecords > (std::numeric_limits<std::uint64_t>::max() - layout.recordSize) / layout.stride)
            {
                return std::unexpected(
                    Error{ErrorCode::invalidArgument, 0, std::string{errorMessages.argumentSpanOverflow}});
            }
            const auto argumentBytes = precedingRecords * layout.stride + layout.recordSize;
            if (argumentBytes > argumentBufferSize - layout.argumentOffset)
            {
                return std::unexpected(
                    Error{ErrorCode::invalidArgument, 0, std::string{errorMessages.argumentRecordsOutOfBounds}});
            }
            return {};
        }

        [[nodiscard]] Result<std::optional<BufferBindingSource>> resolveIndirectCountBuffer(
            ResourceManager& resources,
            const IndirectCommandLayout& layout,
            const IndirectCommandErrorMessages& errorMessages)
        {
            if (!layout.countBuffer.isValid())
            {
                return std::optional<BufferBindingSource>{};
            }
            auto source = resources.bufferBindingSource(layout.countBuffer);
            if (!source)
            {
                return std::unexpected(std::move(source.error()));
            }
            if (!hasAny(source->desc.usage, BufferUsage::indirect) ||
                layout.countOffset % indirectArgumentAlignment != 0 || layout.countOffset > source->desc.size ||
                sizeof(std::uint32_t) > source->desc.size - layout.countOffset)
            {
                return std::unexpected(
                    Error{ErrorCode::invalidArgument, 0, std::string{errorMessages.invalidCountBuffer}});
            }
            return std::optional<BufferBindingSource>{std::move(*source)};
        }

        [[nodiscard]] Result<ResolvedIndirectCommand> resolveIndirectCommand(
            ResourceManager& resources,
            const IndirectCommandLayout& layout,
            const IndirectCommandErrorMessages& errorMessages)
        {
            if (!layout.argumentBuffer.isValid() || layout.argumentOffset % indirectArgumentAlignment != 0 ||
                layout.stride < layout.recordSize || layout.stride % indirectArgumentAlignment != 0 ||
                (!layout.countBuffer.isValid() && layout.countOffset != 0))
            {
                return std::unexpected(Error{ErrorCode::invalidArgument, 0, std::string{errorMessages.invalidLayout}});
            }

            auto arguments = resources.bufferBindingSource(layout.argumentBuffer);
            if (!arguments)
            {
                return std::unexpected(std::move(arguments.error()));
            }
            if (!hasAny(arguments->desc.usage, BufferUsage::indirect) || layout.argumentOffset > arguments->desc.size)
            {
                return std::unexpected(
                    Error{ErrorCode::invalidArgument, 0, std::string{errorMessages.invalidArgumentBuffer}});
            }
            auto validSpan = validateIndirectArgumentSpan(layout, arguments->desc.size, errorMessages);
            if (!validSpan)
            {
                return std::unexpected(std::move(validSpan.error()));
            }

            auto count = resolveIndirectCountBuffer(resources, layout, errorMessages);
            if (!count)
            {
                return std::unexpected(std::move(count.error()));
            }
            return ResolvedIndirectCommand{
                .layout = layout,
                .arguments = std::move(*arguments),
                .count = std::move(*count),
            };
        }

        [[nodiscard]] Status recordIndirectExecution(QueueManager& queues,
                                                     detail::BackendCommandBuffer& commandBuffer,
                                                     ID3D12GraphicsCommandList& commandList,
                                                     ID3D12CommandSignature* signature,
                                                     const ResolvedIndirectCommand& command,
                                                     std::uint32_t drawIndexRootParameter = invalidRootParameter)
        {
            auto retained = retainIndirectBufferDependencies(queues, commandBuffer, command.arguments, command.count);
            if (!retained)
            {
                return retained;
            }
            test::noteIndirectExecutionRecorded();
            commandList.ExecuteIndirect(signature,
                                        command.layout.maxCommandCount,
                                        command.arguments.resource,
                                        command.layout.argumentOffset,
                                        command.count.has_value() ? command.count->resource : nullptr,
                                        command.count.has_value() ? command.layout.countOffset : 0);
            if (drawIndexRootParameter != invalidRootParameter)
            {
                commandList.SetGraphicsRoot32BitConstant(drawIndexRootParameter, 0, 0);
            }
            return {};
        }

        [[nodiscard]] Status validateDrawIndexPrefix(const GraphicsPipelineRecord& pipeline,
                                                     bool drawIndexPrefix,
                                                     std::uint32_t maxCommandCount)
        {
            if (!drawIndexPrefix && maxCommandCount > 1 && pipeline.readsDrawIndex)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The bound pipeline reads the draw index, so an indirect call with more than one record needs "
                    "drawIndexPrefix.",
                });
            }
            return {};
        }

        using PipelineRecord = std::variant<std::shared_ptr<ComputePipelineRecord>,
                                            std::shared_ptr<GraphicsPipelineRecord>,
                                            std::shared_ptr<RayTracingPipelineRecord>>;

        struct RenderingRecord final
        {
            struct DiscardAttachment final
            {
                NativeTextureView view;
                TextureAspect aspect = TextureAspect::color;
            };

            std::vector<QueueManager::ResourceRetention> dependencies;
            std::vector<std::shared_ptr<SwapchainImageLeaseState>> attachmentSwapchainLeases;
            std::vector<DiscardAttachment> storeDiscards;
            std::vector<D3D12_CPU_DESCRIPTOR_HANDLE> colorDescriptors;
            std::optional<D3D12_CPU_DESCRIPTOR_HANDLE> depthStencilDescriptor;
            std::vector<Format> colorFormats;
            Format depthStencilFormat = Format::unknown;
            std::uint32_t sampleCount = 1;
            std::uint32_t width = 0;
            std::uint32_t height = 0;
            std::uint32_t layerCount = 1;
            std::uint32_t viewMask = 0;
        };

        [[nodiscard]] bool hasCompleteViewportState(const detail::GraphicsCommandState& state) noexcept
        {
            return state.viewportCount != 0 && state.viewportCount == state.scissorCount;
        }

        [[nodiscard]] bool isMissingRequiredDynamicState(const detail::GraphicsCommandState& state) noexcept
        {
            return (state.blendConstantsRequired && !state.blendConstantsSet) ||
                   (state.depthBiasRequired && !state.depthBiasSet) ||
                   (state.depthBoundsRequired && !state.depthBoundsSet) ||
                   (state.stencilReferenceRequired && !state.stencilReferenceSet);
        }

        [[nodiscard]] bool isReadyForVertexDraw(const detail::GraphicsCommandState& state,
                                                const GraphicsPipelineRecord* pipeline) noexcept
        {
            if (state.renderingOwner == nullptr || pipeline == nullptr || pipeline->meshShader != nullptr)
            {
                return false;
            }
            const bool viewportStateComplete = !pipeline->rasterizationEnabled || hasCompleteViewportState(state);
            return viewportStateComplete && !isMissingRequiredDynamicState(state) && state.boundBindingSets.all() &&
                   state.boundVertexBuffers.all() && state.boundStreamOutputBuffers.all();
        }

        [[nodiscard]] bool isReadyForMeshDispatch(const detail::GraphicsCommandState& state,
                                                  const GraphicsPipelineRecord* pipeline) noexcept
        {
            return state.renderingOwner != nullptr && pipeline != nullptr && pipeline->meshShader != nullptr &&
                   hasCompleteViewportState(state) && !isMissingRequiredDynamicState(state) &&
                   state.boundBindingSets.all();
        }

        [[nodiscard]] constexpr bool isValidLoadOperation(LoadOperation operation) noexcept
        {
            return operation == LoadOperation::load || operation == LoadOperation::clear ||
                   operation == LoadOperation::discard;
        }

        [[nodiscard]] constexpr bool isValidStoreOperation(StoreOperation operation) noexcept
        {
            return operation == StoreOperation::store || operation == StoreOperation::discard;
        }

        [[nodiscard]] constexpr std::optional<D3D12_SHADING_RATE> nativeShadingRate(ShadingRate rate) noexcept
        {
            switch (rate)
            {
            case ShadingRate::oneByOne:
                return D3D12_SHADING_RATE_1X1;
            case ShadingRate::oneByTwo:
                return D3D12_SHADING_RATE_1X2;
            case ShadingRate::twoByOne:
                return D3D12_SHADING_RATE_2X1;
            case ShadingRate::twoByTwo:
                return D3D12_SHADING_RATE_2X2;
            case ShadingRate::twoByFour:
                return D3D12_SHADING_RATE_2X4;
            case ShadingRate::fourByTwo:
                return D3D12_SHADING_RATE_4X2;
            case ShadingRate::fourByFour:
                return D3D12_SHADING_RATE_4X4;
            default:
                return std::nullopt;
            }
        }

        [[nodiscard]] constexpr bool isAdditionalShadingRate(ShadingRate rate) noexcept
        {
            return rate == ShadingRate::twoByFour || rate == ShadingRate::fourByTwo || rate == ShadingRate::fourByFour;
        }

        [[nodiscard]] constexpr std::optional<D3D12_SHADING_RATE_COMBINER> nativeShadingRateCombiner(
            ShadingRateCombiner combiner) noexcept
        {
            switch (combiner)
            {
            case ShadingRateCombiner::passthrough:
                return D3D12_SHADING_RATE_COMBINER_PASSTHROUGH;
            case ShadingRateCombiner::override:
                return D3D12_SHADING_RATE_COMBINER_OVERRIDE;
            case ShadingRateCombiner::minimum:
                return D3D12_SHADING_RATE_COMBINER_MIN;
            case ShadingRateCombiner::maximum:
                return D3D12_SHADING_RATE_COMBINER_MAX;
            case ShadingRateCombiner::sum:
                return D3D12_SHADING_RATE_COMBINER_SUM;
            default:
                return std::nullopt;
            }
        }

        [[nodiscard]] constexpr bool supportsSampleCount(ShadingRate rate, std::uint32_t sampleCount) noexcept
        {
            switch (rate)
            {
            case ShadingRate::oneByOne:
                return true;
            case ShadingRate::oneByTwo:
            case ShadingRate::twoByOne:
            case ShadingRate::twoByTwo:
                return sampleCount <= 4;
            case ShadingRate::twoByFour:
                return sampleCount <= 2;
            case ShadingRate::fourByTwo:
            case ShadingRate::fourByFour:
                return sampleCount == 1;
            default:
                return false;
            }
        }

        [[nodiscard]] Result<std::vector<D3D12_RECT>> nativeClearRectangles(std::span<const ScissorRect> rectangles,
                                                                            std::uint32_t targetWidth,
                                                                            std::uint32_t targetHeight)
        {
            if (rectangles.size() > std::numeric_limits<std::uint32_t>::max())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The attachment clear has too many rectangles.",
                });
            }

            std::vector<D3D12_RECT> result;
            result.reserve(rectangles.size());
            for (const auto& rectangle : rectangles)
            {
                if (rectangle.x < 0 || rectangle.y < 0 || rectangle.width == 0 || rectangle.height == 0)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Attachment clear rectangles must have a non-negative origin and non-zero extent.",
                    });
                }
                const auto left = static_cast<std::uint32_t>(rectangle.x);
                const auto top = static_cast<std::uint32_t>(rectangle.y);
                if (left > targetWidth || rectangle.width > targetWidth - left || top > targetHeight ||
                    rectangle.height > targetHeight - top)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "An attachment clear rectangle is outside the active rendering extent.",
                    });
                }
                result.push_back({
                    rectangle.x,
                    rectangle.y,
                    static_cast<LONG>(left + rectangle.width),
                    static_cast<LONG>(top + rectangle.height),
                });
            }
            return result;
        }

        [[nodiscard]] constexpr std::uint32_t viewArrayCount(const NativeTextureView& view) noexcept
        {
            switch (view.viewDesc.dimension)
            {
            case TextureViewDimension::texture1D:
            case TextureViewDimension::texture2D:
            case TextureViewDimension::texture3D:
                return 1;
            case TextureViewDimension::cube:
                return 6;
            case TextureViewDimension::texture1DArray:
            case TextureViewDimension::texture2DArray:
            case TextureViewDimension::cubeArray:
                return view.viewDesc.arrayLayerCount == remainingSubresources
                           ? view.texture.desc.depthOrArrayLayers - view.viewDesc.firstArrayLayer
                           : view.viewDesc.arrayLayerCount;
            }
            return 0;
        }

        void discardAttachment(ID3D12GraphicsCommandList& commandList,
                               const NativeTextureView& view,
                               TextureAspect aspect) noexcept
        {
            const std::uint32_t arrayCount = viewArrayCount(view);
            const std::uint32_t resourceArraySize =
                view.texture.desc.dimension == TextureDimension::texture3D ? 1U : view.texture.desc.depthOrArrayLayers;
            const std::uint32_t plane = aspect == TextureAspect::stencil ? 1U : 0U;
            for (std::uint32_t index = 0; index < arrayCount; ++index)
            {
                const std::uint32_t arrayLayer = view.texture.desc.dimension == TextureDimension::texture3D
                                                     ? 0U
                                                     : view.viewDesc.firstArrayLayer + index;
                const std::uint32_t subresource = view.viewDesc.firstMipLevel +
                                                  arrayLayer * view.texture.desc.mipLevels +
                                                  plane * view.texture.desc.mipLevels * resourceArraySize;
                const D3D12_DISCARD_REGION region{
                    .NumRects = 0,
                    .pRects = nullptr,
                    .FirstSubresource = subresource,
                    .NumSubresources = 1,
                };
                commandList.DiscardResource(view.texture.resource, &region);
            }
        }

        [[nodiscard]] Status validateRenderingScope(QueueType queueType,
                                                    const detail::GraphicsCommandState& state,
                                                    const RenderingDesc& desc,
                                                    std::uint32_t maxMultiviewViewCount)
        {
            const bool hasAttachment = !desc.colorAttachments.empty() || desc.depthStencilAttachment.has_value();
            const bool requiresAttachment = desc.viewMask != 0 || desc.timestampWrites.has_value();
            if (queueType != QueueType::graphics || state.renderingOwner != nullptr ||
                desc.colorAttachments.size() > maxColorAttachments || (!hasAttachment && requiresAttachment))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Rendering requires a graphics command buffer, a non-nested scope, and a valid attachment, "
                    "timestamp, and multiview configuration.",
                });
            }
            if (!isContiguousViewMask(desc.viewMask))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A multiview rendering mask must contain one contiguous run of view bits.",
                });
            }
            if (desc.viewMask != 0 && multiviewLayerCount(desc.viewMask) > maxMultiviewViewCount)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The rendering view mask exceeds the device's multiview support.",
                });
            }
            return {};
        }

        struct PreparedRendering final
        {
            std::shared_ptr<RenderingRecord> record;
            std::vector<RenderingRecord::DiscardAttachment> loadDiscards;
            bool extentEstablished = false;
        };

        [[nodiscard]] bool acceptAttachmentExtent(PreparedRendering& prepared,
                                                  const NativeTextureView& view,
                                                  std::uint32_t maxFramebufferLayers) noexcept
        {
            auto& rendering = *prepared.record;
            const auto width = mipExtent(view.texture.desc.width, view.viewDesc.firstMipLevel);
            const auto height = mipExtent(view.texture.desc.height, view.viewDesc.firstMipLevel);
            const auto layerCount = viewArrayCount(view);
            if (layerCount == 0 || layerCount > maxFramebufferLayers ||
                (rendering.viewMask != 0 && layerCount < multiviewLayerCount(rendering.viewMask)))
            {
                return false;
            }
            if (!prepared.extentEstablished)
            {
                rendering.width = width;
                rendering.height = height;
                rendering.sampleCount = view.texture.desc.sampleCount;
                rendering.layerCount = layerCount;
                prepared.extentEstablished = true;
                return true;
            }
            return rendering.width == width && rendering.height == height &&
                   rendering.sampleCount == view.texture.desc.sampleCount && rendering.layerCount == layerCount;
        }

        [[nodiscard]] Format attachmentFormat(const NativeTextureView& view) noexcept
        {
            return view.viewDesc.format == Format::unknown ? view.texture.desc.format : view.viewDesc.format;
        }

        [[nodiscard]] Status validateColorAttachmentOperations(const ColorAttachment& attachment)
        {
            if (!isValidLoadOperation(attachment.loadOperation) || !isValidStoreOperation(attachment.storeOperation))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Color attachments require valid load and store operations.",
                });
            }
            const auto& clear = attachment.clearValue;
            if (attachment.loadOperation == LoadOperation::clear &&
                (!std::isfinite(clear.red) || !std::isfinite(clear.green) || !std::isfinite(clear.blue) ||
                 !std::isfinite(clear.alpha)))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Color clear values must be finite.",
                });
            }
            return {};
        }

        [[nodiscard]] Status addColorAttachment(PreparedRendering& prepared,
                                                ViewManager& views,
                                                const ColorAttachment& attachment,
                                                std::uint32_t maxFramebufferLayers)
        {
            auto validOperations = validateColorAttachmentOperations(attachment);
            if (!validOperations)
            {
                return validOperations;
            }
            auto view = views.nativeTextureView(attachment.view);
            if (!view)
            {
                return std::unexpected(std::move(view.error()));
            }
            if (view->usage != TextureViewUsage::renderTarget ||
                !acceptAttachmentExtent(prepared, *view, maxFramebufferLayers))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Color attachments require render-target views with matching extent, sample count, and layer count "
                    "within the device limit.",
                });
            }
            const auto format = attachmentFormat(*view);
            if (!isColorTargetFormat(format))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A rendering color attachment has an unsupported format.",
                });
            }
            auto& rendering = *prepared.record;
            rendering.dependencies.push_back({view->owner, view->texture.residency});
            rendering.attachmentSwapchainLeases.push_back(view->texture.swapchainLease);
            if (attachment.loadOperation == LoadOperation::discard)
            {
                prepared.loadDiscards.push_back({*view, TextureAspect::color});
            }
            if (attachment.storeOperation == StoreOperation::discard)
            {
                rendering.storeDiscards.push_back({*view, TextureAspect::color});
            }
            rendering.colorFormats.push_back(format);
            rendering.colorDescriptors.push_back(view->descriptor);
            return {};
        }

        [[nodiscard]] Status validateDepthStencilAttachmentOperations(const DepthStencilAttachment& attachment)
        {
            if (!isValidLoadOperation(attachment.depthLoadOperation) ||
                !isValidStoreOperation(attachment.depthStoreOperation) ||
                !isValidLoadOperation(attachment.stencilLoadOperation) ||
                !isValidStoreOperation(attachment.stencilStoreOperation))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The depth-stencil attachment requires valid depth and stencil load and store operations.",
                });
            }
            return {};
        }

        [[nodiscard]] Status validateStencilOperationsForFormat(const DepthStencilAttachment& attachment,
                                                                bool hasStencil)
        {
            if (!hasStencil &&
                (attachment.stencilLoadOperation != LoadOperation::load ||
                 attachment.stencilStoreOperation != StoreOperation::store || attachment.stencilClearValue != 0))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Stencil attachment operations require the portable d24UnormS8Uint format.",
                });
            }
            if (attachment.stencilClearValue > std::numeric_limits<std::uint8_t>::max())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Stencil clear values must fit in the portable 8-bit stencil range.",
                });
            }
            return {};
        }

        void collectDepthStencilDiscards(PreparedRendering& prepared,
                                         const DepthStencilAttachment& attachment,
                                         const NativeTextureView& view,
                                         bool hasStencil)
        {
            auto& storeDiscards = prepared.record->storeDiscards;
            if (attachment.depthLoadOperation == LoadOperation::discard)
            {
                prepared.loadDiscards.push_back({view, TextureAspect::depth});
            }
            if (attachment.depthStoreOperation == StoreOperation::discard)
            {
                storeDiscards.push_back({view, TextureAspect::depth});
            }
            if (hasStencil && attachment.stencilLoadOperation == LoadOperation::discard)
            {
                prepared.loadDiscards.push_back({view, TextureAspect::stencil});
            }
            if (hasStencil && attachment.stencilStoreOperation == StoreOperation::discard)
            {
                storeDiscards.push_back({view, TextureAspect::stencil});
            }
        }

        [[nodiscard]] Status validateDepthClearValue(const DepthStencilAttachment& attachment)
        {
            const auto clearDepth = attachment.depthClearValue;
            if (attachment.depthLoadOperation == LoadOperation::clear &&
                (!std::isfinite(clearDepth) || clearDepth < 0.0F || clearDepth > 1.0F))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Depth clear values must be finite and between zero and one.",
                });
            }
            return {};
        }

        [[nodiscard]] Status addDepthStencilAttachment(PreparedRendering& prepared,
                                                       ViewManager& views,
                                                       const DepthStencilAttachment& attachment,
                                                       std::uint32_t maxFramebufferLayers)
        {
            auto validOperations = validateDepthStencilAttachmentOperations(attachment);
            if (!validOperations)
            {
                return validOperations;
            }
            auto view = views.nativeTextureView(attachment.view);
            if (!view)
            {
                return std::unexpected(std::move(view.error()));
            }
            if (view->usage != TextureViewUsage::depthStencil ||
                !acceptAttachmentExtent(prepared, *view, maxFramebufferLayers))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The depth attachment requires a depth-stencil view matching the color extent, sample count, and "
                    "layer count within the device limit.",
                });
            }
            auto& rendering = *prepared.record;
            rendering.depthStencilFormat = attachmentFormat(*view);
            if (rendering.depthStencilFormat != Format::d32Float &&
                rendering.depthStencilFormat != Format::d24UnormS8Uint)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A rendering depth-stencil attachment has an unsupported format.",
                });
            }
            const bool hasStencil = rendering.depthStencilFormat == Format::d24UnormS8Uint;
            auto validStencilOperations = validateStencilOperationsForFormat(attachment, hasStencil);
            if (!validStencilOperations)
            {
                return validStencilOperations;
            }
            rendering.dependencies.push_back({view->owner, view->texture.residency});
            rendering.attachmentSwapchainLeases.push_back(view->texture.swapchainLease);
            collectDepthStencilDiscards(prepared, attachment, *view, hasStencil);
            rendering.depthStencilDescriptor = view->descriptor;
            return validateDepthClearValue(attachment);
        }

        [[nodiscard]] Result<PreparedRendering> resolveRenderingAttachments(ViewManager& views,
                                                                            const RenderingDesc& desc,
                                                                            std::uint32_t maxFramebufferLayers)
        {
            PreparedRendering prepared{.record = std::make_shared<RenderingRecord>()};
            prepared.record->viewMask = desc.viewMask;
            for (const auto& attachment : desc.colorAttachments)
            {
                auto added = addColorAttachment(prepared, views, attachment, maxFramebufferLayers);
                if (!added)
                {
                    return std::unexpected(std::move(added.error()));
                }
            }
            if (desc.depthStencilAttachment.has_value())
            {
                auto added =
                    addDepthStencilAttachment(prepared, views, *desc.depthStencilAttachment, maxFramebufferLayers);
                if (!added)
                {
                    return std::unexpected(std::move(added.error()));
                }
            }
            return prepared;
        }

        [[nodiscard]] Status appendTimestampDependency(RenderingRecord& rendering,
                                                       const std::optional<PreparedRenderingTimestamps>& timestamps)
        {
            if (!timestamps.has_value())
            {
                return {};
            }
            try
            {
                rendering.dependencies.push_back({timestamps->owner, nullptr});
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to allocate render-pass timestamp retention.",
                });
            }
            return {};
        }

        void resetShadingRate(ID3D12GraphicsCommandList5& commandList, bool supportsShadingRateImage) noexcept
        {
            commandList.RSSetShadingRate(D3D12_SHADING_RATE_1X1, nullptr);
            if (supportsShadingRateImage)
            {
                commandList.RSSetShadingRateImage(nullptr);
            }
        }

        void applyColorLoadClears(ID3D12GraphicsCommandList& commandList,
                                  std::span<const ColorAttachment> attachments,
                                  std::span<const D3D12_CPU_DESCRIPTOR_HANDLE> descriptors) noexcept
        {
            for (std::size_t index = 0; index < attachments.size(); ++index)
            {
                if (attachments[index].loadOperation == LoadOperation::clear)
                {
                    const auto& clear = attachments[index].clearValue;
                    const float value[]{clear.red, clear.green, clear.blue, clear.alpha};
                    commandList.ClearRenderTargetView(descriptors[index], value, 0, nullptr);
                }
            }
        }

        void applyDepthStencilLoadClear(ID3D12GraphicsCommandList& commandList,
                                        const DepthStencilAttachment& attachment,
                                        D3D12_CPU_DESCRIPTOR_HANDLE descriptor) noexcept
        {
            std::uint32_t clearFlags = 0;
            if (attachment.depthLoadOperation == LoadOperation::clear)
            {
                clearFlags |= static_cast<std::uint32_t>(D3D12_CLEAR_FLAG_DEPTH);
            }
            if (attachment.stencilLoadOperation == LoadOperation::clear)
            {
                clearFlags |= static_cast<std::uint32_t>(D3D12_CLEAR_FLAG_STENCIL);
            }
            if (clearFlags != 0)
            {
                commandList.ClearDepthStencilView(descriptor,
                                                  static_cast<D3D12_CLEAR_FLAGS>(clearFlags),
                                                  attachment.depthClearValue,
                                                  static_cast<std::uint8_t>(attachment.stencilClearValue),
                                                  0,
                                                  nullptr);
            }
        }

        void bindAttachmentsAndApplyLoadOperations(
            ID3D12GraphicsCommandList& commandList,
            const RenderingDesc& desc,
            const RenderingRecord& rendering,
            std::span<const RenderingRecord::DiscardAttachment> loadDiscards) noexcept
        {
            for (const auto& attachment : loadDiscards)
            {
                discardAttachment(commandList, attachment.view, attachment.aspect);
            }
            const D3D12_CPU_DESCRIPTOR_HANDLE* depthStencilDescriptor =
                rendering.depthStencilDescriptor.has_value() ? &*rendering.depthStencilDescriptor : nullptr;
            commandList.OMSetRenderTargets(static_cast<UINT>(desc.colorAttachments.size()),
                                           desc.colorAttachments.empty() ? nullptr : rendering.colorDescriptors.data(),
                                           FALSE,
                                           depthStencilDescriptor);
            applyColorLoadClears(commandList, desc.colorAttachments, rendering.colorDescriptors);
            if (desc.depthStencilAttachment.has_value())
            {
                applyDepthStencilLoadClear(
                    commandList, *desc.depthStencilAttachment, *rendering.depthStencilDescriptor);
            }
        }

        struct StreamOutputWritableRange final
        {
            ID3D12Resource* resource = nullptr;
            std::uint64_t beginning = 0;
            std::uint64_t end = 0;
        };

        struct PreparedStreamOutputBindings final
        {
            std::vector<BufferBindingSource> sources;
            std::vector<std::shared_ptr<ResidencyRecord>> residencies;
            std::vector<D3D12_STREAM_OUTPUT_BUFFER_VIEW> views;
            std::vector<StreamOutputWritableRange> writableRanges;
        };

        [[nodiscard]] bool isValidStreamOutputBinding(const StreamOutputBufferBinding& binding,
                                                      const BufferDesc& data,
                                                      const BufferDesc& counter,
                                                      std::uint32_t requiredStride) noexcept
        {
            const bool hasStreamOutputUsage =
                hasAny(data.usage, BufferUsage::streamOutput) && hasAny(counter.usage, BufferUsage::streamOutput);
            const bool dataRangeValid = binding.size >= requiredStride && binding.offset % sizeof(std::uint32_t) == 0 &&
                                        binding.size % sizeof(std::uint32_t) == 0 && binding.offset <= data.size &&
                                        binding.size <= data.size - binding.offset;
            const bool counterRangeValid = binding.filledSizeOffset % sizeof(std::uint32_t) == 0 &&
                                           binding.filledSizeOffset <= counter.size &&
                                           sizeof(std::uint64_t) <= counter.size - binding.filledSizeOffset;
            return hasStreamOutputUsage && dataRangeValid && counterRangeValid;
        }

        [[nodiscard]] bool overlapsWritableRange(std::span<const StreamOutputWritableRange> existingRanges,
                                                 const StreamOutputWritableRange& range) noexcept
        {
            return std::ranges::any_of(existingRanges, [&](const StreamOutputWritableRange& existing) {
                return range.resource == existing.resource && range.beginning < existing.end &&
                       existing.beginning < range.end;
            });
        }

        [[nodiscard]] Status appendStreamOutputBinding(PreparedStreamOutputBindings& prepared,
                                                       ResourceManager& resources,
                                                       const StreamOutputBufferBinding& binding,
                                                       std::uint32_t requiredStride)
        {
            auto data = resources.bufferBindingSource(binding.buffer);
            auto counter = resources.bufferBindingSource(binding.filledSizeBuffer);
            if (!data || !counter)
            {
                return std::unexpected(data ? std::move(counter.error()) : std::move(data.error()));
            }
            if (!isValidStreamOutputBinding(binding, data->desc, counter->desc, requiredStride))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Stream-output data and counters require stream-output usage and aligned, in-bounds ranges.",
                });
            }
            const std::array newRanges{
                StreamOutputWritableRange{data->resource, binding.offset, binding.offset + binding.size},
                StreamOutputWritableRange{
                    counter->resource, binding.filledSizeOffset, binding.filledSizeOffset + sizeof(std::uint64_t)},
            };
            for (const auto& range : newRanges)
            {
                if (overlapsWritableRange(prepared.writableRanges, range))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Stream-output data and counter ranges may not overlap within one binding batch.",
                    });
                }
                prepared.writableRanges.push_back(range);
            }
            prepared.views.push_back({
                .BufferLocation = data->resource->GetGPUVirtualAddress() + binding.offset,
                .SizeInBytes = binding.size,
                .BufferFilledSizeLocation = counter->resource->GetGPUVirtualAddress() + binding.filledSizeOffset,
            });
            prepared.residencies.push_back(data->residency);
            prepared.residencies.push_back(counter->residency);
            prepared.sources.push_back(std::move(*data));
            prepared.sources.push_back(std::move(*counter));
            return {};
        }
    } // namespace

    bool NativePipelineContinuation::hasPipeline() const noexcept
    {
        return computePipeline_ != nullptr || rayTracingPipeline_ != nullptr;
    }

    void NativePipelineContinuation::restore(ID3D12GraphicsCommandList& commandList) const noexcept
    {
        if (computePipeline_ != nullptr)
        {
            commandList.SetPipelineState(computePipeline_);
        }
        else if (rayTracingPipeline_ != nullptr)
        {
            rayTracingCommandList_->SetPipelineState1(rayTracingPipeline_);
        }
    }

    Result<NativePipelineContinuation> PipelineManager::prepareNativePipelineContinuation(
        QueueManager& queues, detail::BackendCommandBuffer& commandBuffer) const
    {
        const auto& computeState = commandBuffer.computeState();
        const auto& rayTracingState = commandBuffer.rayTracingState();
        if (computeState.pipelineOwner != nullptr && rayTracingState.pipelineOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Compute and ray-tracing pipelines cannot both be active on one command buffer.",
            });
        }

        NativePipelineContinuation continuation;
        if (computeState.pipelineOwner != nullptr)
        {
            auto* const pipeline = static_cast<ComputePipelineRecord*>(computeState.pipelineOwner.get());
            continuation.computePipeline_ = pipeline->pipelineState.Get();
        }
        else if (rayTracingState.pipelineOwner != nullptr)
        {
            auto commandList = queues.commandList4(commandBuffer);
            if (!commandList)
            {
                return std::unexpected(std::move(commandList.error()));
            }
            auto* const pipeline = static_cast<RayTracingPipelineRecord*>(rayTracingState.pipelineOwner.get());
            continuation.rayTracingCommandList_ = *commandList;
            continuation.rayTracingPipeline_ = pipeline->stateObject.Get();
        }
        return continuation;
    }

    Status PipelineManager::bindComputePipeline(QueueManager& queues,
                                                BindingManager& bindings,
                                                detail::BackendCommandBuffer& commandBuffer,
                                                PipelineHandle pipeline)
    {
        if (commandBuffer.queueType() == QueueType::copy || commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A compute pipeline requires a graphics or compute command buffer outside a rendering scope.",
            });
        }

        std::shared_ptr<ComputePipelineRecord> record;
        {
            const std::scoped_lock lock(impl_->mutex);
            const auto* found = impl_->pipelines.find(pipeline);
            if (found == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The pipeline handle is invalid or stale.",
                });
            }
            const auto* compute = std::get_if<std::shared_ptr<ComputePipelineRecord>>(found);
            if (compute == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The pipeline is not a compute pipeline.",
                });
            }
            record = *compute;
        }

        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto retained = queues.retainResource(commandBuffer, record);
        if (!retained)
        {
            return retained;
        }

        const auto heaps = bindings.nativeDescriptorHeaps();
        ID3D12DescriptorHeap* const nativeHeaps[]{heaps.resources, heaps.samplers};
        (*commandList)->SetDescriptorHeaps(static_cast<UINT>(std::size(nativeHeaps)), nativeHeaps);
        (*commandList)->SetComputeRootSignature(record->layout->rootSignature.Get());
        (*commandList)->SetPipelineState(record->pipelineState.Get());

        auto& state = commandBuffer.computeState();
        commandBuffer.rayTracingState() = {};
        state.pipelineOwner = std::move(record);
        state.boundBindingSets.clear();
        auto* const activePipeline = static_cast<ComputePipelineRecord*>(state.pipelineOwner.get());
        state.boundBindingSets.reserve(activePipeline->layout->slots.size());
        for (const auto& slot : activePipeline->layout->slots)
        {
            state.boundBindingSets.push_back(!slot.requiresBindingSet);
        }
        return {};
    }

    Status PipelineManager::bindComputeBindingSet(QueueManager& queues,
                                                  BindingManager& bindings,
                                                  ResourceManager& resources,
                                                  detail::BackendCommandBuffer& commandBuffer,
                                                  std::uint32_t setIndex,
                                                  BindingSetHandle bindingSet,
                                                  std::span<const std::uint32_t> dynamicOffsets)
    {
        const std::array bindingSets{bindingSet};
        return bindComputeBindingSets(
            queues, bindings, resources, commandBuffer, setIndex, bindingSets, dynamicOffsets);
    }

    Status PipelineManager::bindComputeBindingSets(QueueManager& queues,
                                                   BindingManager& bindings,
                                                   ResourceManager& resources,
                                                   detail::BackendCommandBuffer& commandBuffer,
                                                   std::uint32_t firstSet,
                                                   std::span<const BindingSetHandle> bindingSets,
                                                   std::span<const std::uint32_t> dynamicOffsets)
    {
        auto& state = commandBuffer.computeState();
        if (state.pipelineOwner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A compute pipeline must be bound before binding sets.",
            });
        }
        auto* const pipeline = static_cast<ComputePipelineRecord*>(state.pipelineOwner.get());
        auto prepared = prepareBindingSetBatch(
            queues, commandBuffer, bindings, resources, pipeline->layout->slots, firstSet, bindingSets, dynamicOffsets);
        if (!prepared)
        {
            return std::unexpected(std::move(prepared.error()));
        }

        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto retained = retainBindingSetBatchDependencies(queues, commandBuffer, *prepared);
        if (!retained)
        {
            return retained;
        }
        const auto activeSets = (*prepared)->activeSets();
        for (std::size_t index = 0; index < activeSets.size(); ++index)
        {
            const auto& slot = pipeline->layout->slots[firstSet + index];
            applyComputeRootBindings(**commandList, slot, activeSets[index]);
            state.boundBindingSets[firstSet + index] = true;
        }
        return {};
    }

    Status PipelineManager::setComputePushConstants(QueueManager& queues,
                                                    detail::BackendCommandBuffer& commandBuffer,
                                                    std::uint32_t offset,
                                                    std::span<const std::byte> data)
    {
        auto& state = commandBuffer.computeState();
        if (state.pipelineOwner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A compute pipeline must be bound before setting push constants.",
            });
        }
        if (data.empty() || offset % sizeof(std::uint32_t) != 0 || data.size() % sizeof(std::uint32_t) != 0 ||
            data.size() > maxPushConstantBytes || offset > maxPushConstantBytes - data.size())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Compute push constants require a non-empty, 4-byte aligned range within 128 bytes.",
            });
        }

        auto* const pipeline = static_cast<ComputePipelineRecord*>(state.pipelineOwner.get());
        const auto end = static_cast<std::uint64_t>(offset) + data.size();
        const bool declared = std::ranges::any_of(pipeline->layout->pushConstantRanges, [&](const auto& range) {
            return hasAny(range.visibility, ShaderStage::compute) && range.offset <= offset &&
                   end <= static_cast<std::uint64_t>(range.offset) + range.size;
        });
        if (!declared || pipeline->layout->pushConstantRootParameter == invalidRootParameter)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The push-constant write is not contained in a compute-visible pipeline layout range.",
            });
        }

        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        (*commandList)
            ->SetComputeRoot32BitConstants(pipeline->layout->pushConstantRootParameter,
                                           static_cast<UINT>(data.size() / sizeof(std::uint32_t)),
                                           data.data(),
                                           offset / sizeof(std::uint32_t));
        return {};
    }

    Status PipelineManager::dispatch(QueueManager& queues,
                                     detail::BackendCommandBuffer& commandBuffer,
                                     std::uint32_t groupCountX,
                                     std::uint32_t groupCountY,
                                     std::uint32_t groupCountZ)
    {
        const auto& state = commandBuffer.computeState();
        if (state.pipelineOwner == nullptr || commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Dispatch requires a bound compute pipeline outside a rendering scope.",
            });
        }
        if (!state.boundBindingSets.all())
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Every required compute binding set must be bound before dispatch.",
            });
        }
        if (groupCountX == 0 || groupCountY == 0 || groupCountZ == 0 ||
            groupCountX > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION ||
            groupCountY > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION ||
            groupCountZ > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Dispatch group counts must be between one and the D3D12 per-dimension limit.",
            });
        }

        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        (*commandList)->Dispatch(groupCountX, groupCountY, groupCountZ);
        return {};
    }

    Status PipelineManager::dispatchIndirect(QueueManager& queues,
                                             ResourceManager& resources,
                                             detail::BackendCommandBuffer& commandBuffer,
                                             const IndirectDispatchDesc& desc)
    {
        const auto& state = commandBuffer.computeState();
        if (state.pipelineOwner == nullptr || commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Indirect dispatch requires a bound compute pipeline outside a rendering scope.",
            });
        }
        if (!state.boundBindingSets.all())
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Every required compute binding set must be bound before indirect dispatch.",
            });
        }

        constexpr std::uint32_t recordSize = sizeof(DispatchIndirectArguments);
        constexpr IndirectCommandErrorMessages errorMessages{
            .invalidLayout =
                "Indirect dispatch requires an argument buffer, aligned offsets, and an aligned 12-byte record.",
            .invalidArgumentBuffer =
                "The indirect dispatch argument buffer requires indirect usage and an in-bounds offset.",
            .argumentSpanOverflow = "The indirect dispatch argument span overflows.",
            .argumentRecordsOutOfBounds = "The indirect dispatch records exceed the argument buffer.",
            .invalidCountBuffer =
                "The indirect dispatch count buffer requires indirect usage and an aligned in-bounds uint32 count.",
        };
        auto command = resolveIndirectCommand(
            resources, indirectCommandLayout(desc, desc.maxDispatchCount, recordSize), errorMessages);
        if (!command)
        {
            return std::unexpected(std::move(command.error()));
        }

        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        if (desc.maxDispatchCount == 0)
        {
            return {};
        }
        auto signature = impl_->indirectDispatchSignature(command->layout.stride);
        if (!signature)
        {
            return std::unexpected(std::move(signature.error()));
        }
        return recordIndirectExecution(queues, commandBuffer, **commandList, signature->Get(), *command);
    }

    Status PipelineManager::beginRendering(QueueManager& queues,
                                           ViewManager& views,
                                           QueryManager& queries,
                                           detail::BackendCommandBuffer& commandBuffer,
                                           const RenderingDesc& desc)
    {
        auto& state = commandBuffer.graphicsState();
        auto validScope = validateRenderingScope(commandBuffer.queueType(), state, desc, impl_->maxMultiviewViewCount);
        if (!validScope)
        {
            return validScope;
        }
        auto preparedRendering = resolveRenderingAttachments(views, desc, impl_->maxFramebufferLayers);
        if (!preparedRendering)
        {
            return std::unexpected(std::move(preparedRendering.error()));
        }
        auto& rendering = *preparedRendering->record;

        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        ID3D12GraphicsCommandList5* commandList5 = nullptr;
        if (impl_->variableRateShading)
        {
            auto queriedCommandList5 = queues.commandList5(commandBuffer);
            if (!queriedCommandList5)
            {
                return std::unexpected(std::move(queriedCommandList5.error()));
            }
            commandList5 = *queriedCommandList5;
        }
        auto preparedTimestamps = queries.prepareRenderingTimestamps(commandBuffer, desc.timestampWrites);
        if (!preparedTimestamps)
        {
            return std::unexpected(std::move(preparedTimestamps.error()));
        }
        auto timestampDependencyAppended = appendTimestampDependency(rendering, *preparedTimestamps);
        if (!timestampDependencyAppended)
        {
            return timestampDependencyAppended;
        }
        auto retainedDependencies = queues.retainCommandDependencies(
            commandBuffer, rendering.dependencies, rendering.attachmentSwapchainLeases);
        if (!retainedDependencies)
        {
            return retainedDependencies;
        }

        const auto commitRendering = [&]() noexcept {
            if (preparedTimestamps->has_value())
            {
                queries.commitRenderingTimestamps(**commandList, commandBuffer, std::move(**preparedTimestamps));
            }
            state = {};
            state.renderingOwner = std::move(preparedRendering->record);
            commandBuffer.computeState() = {};
            commandBuffer.rayTracingState() = {};

            if (commandList5 != nullptr)
            {
                resetShadingRate(*commandList5, impl_->variableRateShadingTier2);
            }
            bindAttachmentsAndApplyLoadOperations(**commandList, desc, rendering, preparedRendering->loadDiscards);
        };
        commitRendering();
        return {};
    }

    Status PipelineManager::endRendering(QueueManager& queues,
                                         QueryManager& queries,
                                         detail::BackendCommandBuffer& commandBuffer)
    {
        auto& state = commandBuffer.graphicsState();
        if (state.renderingOwner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "No rendering scope is active.",
            });
        }
        if (auto queriesEnded = detail::validateRenderingQueriesEnded(commandBuffer); !queriesEnded)
        {
            return queriesEnded;
        }
        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        const auto rendering = std::static_pointer_cast<RenderingRecord>(state.renderingOwner);

        const auto commitEndRendering = [&]() noexcept {
            for (const auto& attachment : rendering->storeDiscards)
            {
                discardAttachment(**commandList, attachment.view, attachment.aspect);
            }
            queries.endRenderingTimestamps(**commandList, commandBuffer);
            state = {};
        };
        commitEndRendering();
        return {};
    }

    Status PipelineManager::clearColorAttachment(QueueManager& queues,
                                                 detail::BackendCommandBuffer& commandBuffer,
                                                 std::uint32_t attachmentIndex,
                                                 const ClearColor& clearValue,
                                                 std::span<const ScissorRect> rectangles)
    {
        const auto& state = commandBuffer.graphicsState();
        if (state.renderingOwner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A color attachment can be cleared only inside an active rendering scope.",
            });
        }
        auto* const rendering = static_cast<RenderingRecord*>(state.renderingOwner.get());
        if (attachmentIndex >= rendering->colorDescriptors.size())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The color attachment clear index is outside the active attachment list.",
            });
        }
        if (!std::isfinite(clearValue.red) || !std::isfinite(clearValue.green) || !std::isfinite(clearValue.blue) ||
            !std::isfinite(clearValue.alpha))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Color attachment clear values must be finite.",
            });
        }
        auto nativeRectangles = nativeClearRectangles(rectangles, rendering->width, rendering->height);
        if (!nativeRectangles)
        {
            return std::unexpected(std::move(nativeRectangles.error()));
        }
        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }

        const float values[]{clearValue.red, clearValue.green, clearValue.blue, clearValue.alpha};
        (*commandList)
            ->ClearRenderTargetView(rendering->colorDescriptors[attachmentIndex],
                                    values,
                                    static_cast<UINT>(nativeRectangles->size()),
                                    nativeRectangles->empty() ? nullptr : nativeRectangles->data());
        return {};
    }

    Status PipelineManager::clearDepthStencilAttachment(QueueManager& queues,
                                                        detail::BackendCommandBuffer& commandBuffer,
                                                        DepthStencilClearAspect aspects,
                                                        float depth,
                                                        std::uint32_t stencil,
                                                        std::span<const ScissorRect> rectangles)
    {
        const auto validAspects = DepthStencilClearAspect::depth | DepthStencilClearAspect::stencil;
        const auto aspectBits = static_cast<std::uint8_t>(aspects);
        if (aspects == DepthStencilClearAspect::none || (aspectBits & ~static_cast<std::uint8_t>(validAspects)) != 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A depth-stencil clear requires at least one valid aspect.",
            });
        }

        const auto& state = commandBuffer.graphicsState();
        if (state.renderingOwner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A depth-stencil attachment can be cleared only inside an active rendering scope.",
            });
        }
        auto* const rendering = static_cast<RenderingRecord*>(state.renderingOwner.get());
        if (!rendering->depthStencilDescriptor.has_value())
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The active rendering scope has no depth-stencil attachment.",
            });
        }
        if (hasAny(aspects, DepthStencilClearAspect::depth) && (!std::isfinite(depth) || depth < 0.0F || depth > 1.0F))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A depth clear value must be finite and between zero and one.",
            });
        }
        if (hasAny(aspects, DepthStencilClearAspect::stencil))
        {
            if (rendering->depthStencilFormat != Format::d24UnormS8Uint)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Stencil clears require the portable d24UnormS8Uint format.",
                });
            }
            if (stencil > std::numeric_limits<std::uint8_t>::max())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A stencil clear value must fit in the portable 8-bit range.",
                });
            }
        }
        auto nativeRectangles = nativeClearRectangles(rectangles, rendering->width, rendering->height);
        if (!nativeRectangles)
        {
            return std::unexpected(std::move(nativeRectangles.error()));
        }
        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }

        std::uint32_t clearFlags = 0;
        if (hasAny(aspects, DepthStencilClearAspect::depth))
        {
            clearFlags |= D3D12_CLEAR_FLAG_DEPTH;
        }
        if (hasAny(aspects, DepthStencilClearAspect::stencil))
        {
            clearFlags |= D3D12_CLEAR_FLAG_STENCIL;
        }
        (*commandList)
            ->ClearDepthStencilView(*rendering->depthStencilDescriptor,
                                    static_cast<D3D12_CLEAR_FLAGS>(clearFlags),
                                    depth,
                                    static_cast<std::uint8_t>(stencil),
                                    static_cast<UINT>(nativeRectangles->size()),
                                    nativeRectangles->empty() ? nullptr : nativeRectangles->data());
        return {};
    }

    Status PipelineManager::bindGraphicsPipeline(QueueManager& queues,
                                                 BindingManager& bindings,
                                                 detail::BackendCommandBuffer& commandBuffer,
                                                 PipelineHandle pipeline)
    {
        auto& state = commandBuffer.graphicsState();
        if (commandBuffer.queueType() != QueueType::graphics || state.renderingOwner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A graphics pipeline can be bound only inside a graphics rendering scope.",
            });
        }

        std::shared_ptr<GraphicsPipelineRecord> record;
        {
            const std::scoped_lock lock(impl_->mutex);
            const auto* found = impl_->pipelines.find(pipeline);
            const auto* graphics =
                found != nullptr ? std::get_if<std::shared_ptr<GraphicsPipelineRecord>>(found) : nullptr;
            if (graphics == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The pipeline handle is invalid, stale, or not a graphics pipeline.",
                });
            }
            record = *graphics;
        }
        auto* const rendering = static_cast<RenderingRecord*>(state.renderingOwner.get());
        if (record->colorFormats != rendering->colorFormats ||
            record->depthStencilFormat != rendering->depthStencilFormat ||
            record->sampleCount != rendering->sampleCount || record->viewMask != rendering->viewMask)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The graphics pipeline attachment formats, sample count, or view mask do not match the active "
                "rendering scope.",
            });
        }

        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        ComPtr<ID3D12GraphicsCommandList1> commandList1;
        if (record->viewMask != 0)
        {
            const HRESULT queryResult = (*commandList)->QueryInterface(IID_PPV_ARGS(&commandList1));
            if (FAILED(queryResult))
            {
                return std::unexpected(
                    nativeError(ErrorCode::nativeFailure, queryResult, "ID3D12GraphicsCommandList::QueryInterface(1)"));
            }
        }
        auto retained = queues.retainResource(commandBuffer, record);
        if (!retained)
        {
            return retained;
        }
        const auto heaps = bindings.nativeDescriptorHeaps();
        ID3D12DescriptorHeap* const nativeHeaps[]{heaps.resources, heaps.samplers};
        (*commandList)->SetDescriptorHeaps(static_cast<UINT>(std::size(nativeHeaps)), nativeHeaps);
        (*commandList)->SetGraphicsRootSignature(record->layout->rootSignature.Get());
        if (record->layout->drawIndexRootParameter != invalidRootParameter)
        {
            (*commandList)->SetGraphicsRoot32BitConstant(record->layout->drawIndexRootParameter, 0, 0);
        }
        (*commandList)->SetPipelineState(record->pipelineState.Get());
        if (commandList1 != nullptr)
        {
            commandList1->SetViewInstanceMask(record->viewMask);
        }
        if (record->meshShader == nullptr)
        {
            (*commandList)->IASetPrimitiveTopology(record->topology);
        }
        const std::array<D3D12_STREAM_OUTPUT_BUFFER_VIEW, maxStreamOutputBuffers> emptyStreamOutputViews{};
        (*commandList)
            ->SOSetTargets(0, static_cast<UINT>(emptyStreamOutputViews.size()), emptyStreamOutputViews.data());

        state.pipelineOwner = std::move(record);
        auto* const activePipeline = static_cast<GraphicsPipelineRecord*>(state.pipelineOwner.get());
        state.boundBindingSets.clear();
        for (const auto& slot : activePipeline->layout->slots)
        {
            state.boundBindingSets.push_back(!slot.requiresBindingSet);
        }
        state.boundVertexBuffers.assign(activePipeline->vertexStrides.size(), false);
        state.boundStreamOutputBuffers.assign(activePipeline->streamOutputStrides.size(), false);
        state.indexBufferBound = false;
        state.blendConstantsRequired = activePipeline->usesBlendConstants;
        state.depthBiasRequired = activePipeline->dynamicDepthBias;
        state.depthBiasSet = false;
        state.depthBoundsRequired = activePipeline->depthBoundsTest;
        state.depthBoundsSet = false;
        state.stencilReferenceRequired = activePipeline->stencilEnabled;
        state.stencilReferenceSet = false;
        state.stencilReferenceFacesSet = 0;
        return {};
    }

    Status PipelineManager::bindGraphicsBindingSet(QueueManager& queues,
                                                   BindingManager& bindings,
                                                   ResourceManager& resources,
                                                   detail::BackendCommandBuffer& commandBuffer,
                                                   std::uint32_t setIndex,
                                                   BindingSetHandle bindingSet,
                                                   std::span<const std::uint32_t> dynamicOffsets)
    {
        const std::array bindingSets{bindingSet};
        return bindGraphicsBindingSets(
            queues, bindings, resources, commandBuffer, setIndex, bindingSets, dynamicOffsets);
    }

    Status PipelineManager::bindGraphicsBindingSets(QueueManager& queues,
                                                    BindingManager& bindings,
                                                    ResourceManager& resources,
                                                    detail::BackendCommandBuffer& commandBuffer,
                                                    std::uint32_t firstSet,
                                                    std::span<const BindingSetHandle> bindingSets,
                                                    std::span<const std::uint32_t> dynamicOffsets)
    {
        auto& state = commandBuffer.graphicsState();
        if (state.renderingOwner == nullptr || state.pipelineOwner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A graphics pipeline must be bound before binding graphics sets.",
            });
        }
        auto* const pipeline = static_cast<GraphicsPipelineRecord*>(state.pipelineOwner.get());
        auto prepared = prepareBindingSetBatch(
            queues, commandBuffer, bindings, resources, pipeline->layout->slots, firstSet, bindingSets, dynamicOffsets);
        if (!prepared)
        {
            return std::unexpected(std::move(prepared.error()));
        }
        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto retained = retainBindingSetBatchDependencies(queues, commandBuffer, *prepared);
        if (!retained)
        {
            return retained;
        }
        const auto activeSets = (*prepared)->activeSets();
        for (std::size_t index = 0; index < activeSets.size(); ++index)
        {
            const auto& slot = pipeline->layout->slots[firstSet + index];
            applyGraphicsRootBindings(**commandList, slot, activeSets[index]);
            state.boundBindingSets[firstSet + index] = true;
        }
        return {};
    }

    Status PipelineManager::setGraphicsPushConstants(QueueManager& queues,
                                                     detail::BackendCommandBuffer& commandBuffer,
                                                     ShaderStage visibility,
                                                     std::uint32_t offset,
                                                     std::span<const std::byte> data)
    {
        auto& state = commandBuffer.graphicsState();
        if (state.renderingOwner == nullptr || state.pipelineOwner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Graphics push constants require an active rendering scope and bound graphics pipeline.",
            });
        }
        const auto visibilityValue = static_cast<std::uint16_t>(visibility);
        if (visibilityValue == 0 || (visibilityValue & ~static_cast<std::uint16_t>(ShaderStage::allGraphics)) != 0 ||
            data.empty() || offset % sizeof(std::uint32_t) != 0 || data.size() % sizeof(std::uint32_t) != 0 ||
            data.size() > maxPushConstantBytes || offset > maxPushConstantBytes - data.size())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Graphics push constants require graphics visibility and an aligned range.",
            });
        }
        auto* const pipeline = static_cast<GraphicsPipelineRecord*>(state.pipelineOwner.get());
        const auto end = static_cast<std::uint64_t>(offset) + data.size();
        const bool declared = std::ranges::any_of(pipeline->layout->pushConstantRanges, [&](const auto& range) {
            return (range.visibility & visibility) == visibility && range.offset <= offset &&
                   end <= static_cast<std::uint64_t>(range.offset) + range.size;
        });
        if (!declared || pipeline->layout->pushConstantRootParameter == invalidRootParameter)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The push-constant write is not contained in a matching graphics-visible layout range.",
            });
        }
        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        (*commandList)
            ->SetGraphicsRoot32BitConstants(pipeline->layout->pushConstantRootParameter,
                                            static_cast<UINT>(data.size() / sizeof(std::uint32_t)),
                                            data.data(),
                                            offset / sizeof(std::uint32_t));
        return {};
    }

    Status PipelineManager::setViewports(QueueManager& queues,
                                         detail::BackendCommandBuffer& commandBuffer,
                                         std::span<const Viewport> viewports)
    {
        auto& state = commandBuffer.graphicsState();
        if (state.renderingOwner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Viewports can be set only inside an active rendering scope.",
            });
        }
        if (viewports.empty())
        {
            return std::unexpected(Error{ErrorCode::invalidArgument, 0, "At least one viewport must be supplied."});
        }
        if (viewports.size() > D3D12_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The viewport count exceeds the D3D12 per-pipeline limit.",
            });
        }

        std::array<D3D12_VIEWPORT, D3D12_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE> native{};
        for (std::size_t index = 0; index < viewports.size(); ++index)
        {
            const auto& viewport = viewports[index];
            const auto right = static_cast<double>(viewport.x) + viewport.width;
            const auto bottom = static_cast<double>(viewport.y) + viewport.height;
            if (!std::isfinite(viewport.x) || !std::isfinite(viewport.y) || !std::isfinite(viewport.width) ||
                !std::isfinite(viewport.height) || !std::isfinite(viewport.minDepth) ||
                !std::isfinite(viewport.maxDepth) || viewport.width <= 0.0F || viewport.height <= 0.0F ||
                viewport.minDepth < 0.0F || viewport.maxDepth > 1.0F || viewport.minDepth > viewport.maxDepth ||
                viewport.x < D3D12_VIEWPORT_BOUNDS_MIN || viewport.y < D3D12_VIEWPORT_BOUNDS_MIN ||
                right > D3D12_VIEWPORT_BOUNDS_MAX || bottom > D3D12_VIEWPORT_BOUNDS_MAX)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Every viewport requires finite positive in-bounds dimensions and an ordered depth range within "
                    "zero and one.",
                });
            }
            native[index] = {
                viewport.x,
                viewport.y,
                viewport.width,
                viewport.height,
                viewport.minDepth,
                viewport.maxDepth,
            };
        }

        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        (*commandList)->RSSetViewports(static_cast<UINT>(viewports.size()), native.data());
        state.viewportCount = static_cast<std::uint32_t>(viewports.size());
        return {};
    }

    Status PipelineManager::setScissors(QueueManager& queues,
                                        detail::BackendCommandBuffer& commandBuffer,
                                        std::span<const ScissorRect> scissors)
    {
        auto& state = commandBuffer.graphicsState();
        if (state.renderingOwner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Scissor rectangles can be set only inside an active rendering scope.",
            });
        }
        if (scissors.empty())
        {
            return std::unexpected(
                Error{ErrorCode::invalidArgument, 0, "At least one scissor rectangle must be supplied."});
        }
        if (scissors.size() > D3D12_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The scissor-rectangle count exceeds the D3D12 per-pipeline limit.",
            });
        }

        std::array<D3D12_RECT, D3D12_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE> native{};
        for (std::size_t index = 0; index < scissors.size(); ++index)
        {
            const auto& scissor = scissors[index];
            const auto right = static_cast<std::int64_t>(scissor.x) + scissor.width;
            const auto bottom = static_cast<std::int64_t>(scissor.y) + scissor.height;
            if (scissor.width == 0 || scissor.height == 0 || right > std::numeric_limits<LONG>::max() ||
                bottom > std::numeric_limits<LONG>::max())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Every scissor rectangle requires positive dimensions representable by D3D12.",
                });
            }
            native[index] = {scissor.x, scissor.y, static_cast<LONG>(right), static_cast<LONG>(bottom)};
        }

        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        (*commandList)->RSSetScissorRects(static_cast<UINT>(scissors.size()), native.data());
        state.scissorCount = static_cast<std::uint32_t>(scissors.size());
        return {};
    }

    Status PipelineManager::setBlendConstants(QueueManager& queues,
                                              detail::BackendCommandBuffer& commandBuffer,
                                              const BlendConstants& constants)
    {
        auto& state = commandBuffer.graphicsState();
        if (state.renderingOwner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Blend constants can be set only inside an active rendering scope.",
            });
        }
        if (!std::isfinite(constants.red) || !std::isfinite(constants.green) || !std::isfinite(constants.blue) ||
            !std::isfinite(constants.alpha))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Blend constants require four finite values.",
            });
        }
        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        const float native[]{constants.red, constants.green, constants.blue, constants.alpha};
        (*commandList)->OMSetBlendFactor(native);
        state.blendConstantsSet = true;
        return {};
    }

    Status PipelineManager::setDepthBias(QueueManager& queues,
                                         detail::BackendCommandBuffer& commandBuffer,
                                         const DepthBias& bias)
    {
        auto& state = commandBuffer.graphicsState();
        if (state.renderingOwner == nullptr || state.pipelineOwner == nullptr || !state.depthBiasRequired)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Dynamic depth bias requires an active graphics pipeline that opted into the state.",
            });
        }
        if (!std::isfinite(bias.constantFactor) || !std::isfinite(bias.clamp) || !std::isfinite(bias.slopeScale))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Dynamic depth-bias values must be finite.",
            });
        }
        auto commandList = queues.commandList9(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        (*commandList)->RSSetDepthBias(bias.constantFactor, bias.clamp, bias.slopeScale);
        state.depthBiasSet = true;
        return {};
    }

    Status PipelineManager::setDepthBounds(QueueManager& queues,
                                           detail::BackendCommandBuffer& commandBuffer,
                                           const DepthBounds& bounds)
    {
        auto& state = commandBuffer.graphicsState();
        if (state.renderingOwner == nullptr || state.pipelineOwner == nullptr || !state.depthBoundsRequired)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Dynamic depth bounds require an active graphics pipeline that opted into the test.",
            });
        }
        if (!std::isfinite(bounds.minimum) || !std::isfinite(bounds.maximum) || bounds.minimum < 0.0F ||
            bounds.maximum > 1.0F || bounds.minimum > bounds.maximum)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Depth bounds must be a finite ordered range within [0, 1].",
            });
        }
        auto commandList = queues.commandList1(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        (*commandList)->OMSetDepthBounds(bounds.minimum, bounds.maximum);
        state.depthBoundsSet = true;
        return {};
    }

    Status PipelineManager::setStencilReference(QueueManager& queues,
                                                detail::BackendCommandBuffer& commandBuffer,
                                                StencilFace face,
                                                std::uint32_t reference)
    {
        constexpr std::uint8_t frontFaceMask = 1U << 0U;
        constexpr std::uint8_t backFaceMask = 1U << 1U;
        constexpr std::uint8_t bothFacesMask = frontFaceMask | backFaceMask;
        auto& state = commandBuffer.graphicsState();
        if (state.renderingOwner == nullptr || reference > std::numeric_limits<std::uint8_t>::max() ||
            (face != StencilFace::front && face != StencilFace::back && face != StencilFace::frontAndBack))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A stencil reference requires an active rendering scope, a valid face selector, and an 8-bit value.",
            });
        }

        const auto nativeReference = static_cast<std::uint8_t>(reference);
        if (face == StencilFace::frontAndBack)
        {
            auto commandList = queues.commandList(commandBuffer);
            if (!commandList)
            {
                return std::unexpected(std::move(commandList.error()));
            }
            (*commandList)->OMSetStencilRef(reference);
            state.frontStencilReference = nativeReference;
            state.backStencilReference = nativeReference;
            state.stencilReferenceFacesSet = bothFacesMask;
        }
        else
        {
            auto commandList = queues.commandList8(commandBuffer);
            if (!commandList)
            {
                return std::unexpected(std::move(commandList.error()));
            }
            const auto frontReference = face == StencilFace::front ? nativeReference : state.frontStencilReference;
            const auto backReference = face == StencilFace::back ? nativeReference : state.backStencilReference;
            (*commandList)->OMSetFrontAndBackStencilRef(frontReference, backReference);
            state.frontStencilReference = frontReference;
            state.backStencilReference = backReference;
            state.stencilReferenceFacesSet |= face == StencilFace::front ? frontFaceMask : backFaceMask;
        }
        state.stencilReferenceSet = state.stencilReferenceFacesSet == bothFacesMask;
        return {};
    }

    Status PipelineManager::setShadingRate(QueueManager& queues,
                                           detail::BackendCommandBuffer& commandBuffer,
                                           ShadingRate rate,
                                           const ShadingRateCombiners& combiners)
    {
        const auto nativeRate = nativeShadingRate(rate);
        const auto nativePrimitiveCombiner = nativeShadingRateCombiner(combiners.primitive);
        const auto nativeImageCombiner = nativeShadingRateCombiner(combiners.image);
        auto& state = commandBuffer.graphicsState();
        if (state.renderingOwner == nullptr || !nativeRate.has_value() || !nativePrimitiveCombiner.has_value() ||
            !nativeImageCombiner.has_value())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A shading rate requires an active rendering scope and a valid rate.",
            });
        }
        if (!impl_->variableRateShading || (isAdditionalShadingRate(rate) && !impl_->additionalShadingRates))
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The requested shading rate is unavailable on this adapter.",
            });
        }
        const bool nonDefaultCombiners = combiners.primitive != ShadingRateCombiner::passthrough ||
                                         combiners.image != ShadingRateCombiner::passthrough;
        if (nonDefaultCombiners && !impl_->variableRateShadingTier2)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Non-default shading-rate combiners require Tier 2 variable-rate shading.",
            });
        }
        auto* const rendering = static_cast<RenderingRecord*>(state.renderingOwner.get());
        if (!supportsSampleCount(rate, rendering->sampleCount))
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "The requested shading rate is unavailable for the active rendering sample count.",
            });
        }
        auto commandList = queues.commandList5(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        const std::array nativeCombiners{*nativePrimitiveCombiner, *nativeImageCombiner};
        (*commandList)->RSSetShadingRate(*nativeRate, nonDefaultCombiners ? nativeCombiners.data() : nullptr);
        return {};
    }

    Status PipelineManager::setShadingRateImage(QueueManager& queues,
                                                ResourceManager& resources,
                                                detail::BackendCommandBuffer& commandBuffer,
                                                TextureHandle image)
    {
        auto& state = commandBuffer.graphicsState();
        if (state.renderingOwner == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A shading-rate image requires an active rendering scope.",
            });
        }
        if (!impl_->variableRateShadingTier2)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Shading-rate images require Tier 2 variable-rate shading.",
            });
        }
        auto commandList = queues.commandList5(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        if (!image)
        {
            (*commandList)->RSSetShadingRateImage(nullptr);
            return {};
        }

        auto source = resources.textureViewSource(image);
        if (!source)
        {
            return std::unexpected(std::move(source.error()));
        }
        if (!hasAny(source->desc.usage, TextureUsage::shadingRateSource) || source->desc.format != Format::r8Uint ||
            source->desc.dimension != TextureDimension::texture2D || source->desc.depthOrArrayLayers != 1 ||
            source->desc.mipLevels != 1 || source->desc.sampleCount != 1)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A shading-rate image must be a live single-layer, single-mip, single-sample r8Uint 2D texture with "
                "shading-rate-source usage.",
            });
        }
        auto retained = queues.retainResource(commandBuffer, source->owner, source->residency);
        if (!retained)
        {
            return retained;
        }
        (*commandList)->RSSetShadingRateImage(source->resource);
        return {};
    }

    Status PipelineManager::bindVertexBuffers(QueueManager& queues,
                                              ResourceManager& resources,
                                              detail::BackendCommandBuffer& commandBuffer,
                                              std::uint32_t firstSlot,
                                              std::span<const VertexBufferBinding> bindings)
    {
        auto& state = commandBuffer.graphicsState();
        if (state.renderingOwner == nullptr || state.pipelineOwner == nullptr)
        {
            return std::unexpected(Error{ErrorCode::invalidState, 0, "A graphics pipeline must be bound first."});
        }
        auto* const pipeline = static_cast<GraphicsPipelineRecord*>(state.pipelineOwner.get());
        if (pipeline->meshShader != nullptr || bindings.empty() || firstSlot >= pipeline->vertexStrides.size() ||
            bindings.size() > pipeline->vertexStrides.size() - firstSlot)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Vertex-buffer batches must be non-empty and fit contiguous declared slots.",
            });
        }
        if (auto aligned = detail::validateVertexBufferOffsetAlignment(pipeline->vertexAlignments, firstSlot, bindings);
            !aligned)
        {
            return aligned;
        }

        struct PreparedVertexBufferBindings final
        {
            std::vector<BufferBindingSource> sources;
            std::vector<std::shared_ptr<ResidencyRecord>> residencies;
            std::vector<D3D12_VERTEX_BUFFER_VIEW> views;
        };
        auto prepared = std::make_shared<PreparedVertexBufferBindings>();
        prepared->sources.reserve(bindings.size());
        prepared->residencies.reserve(bindings.size());
        prepared->views.reserve(bindings.size());
        for (std::size_t index = 0; index < bindings.size(); ++index)
        {
            const auto& binding = bindings[index];
            auto source = resources.bufferBindingSource(binding.buffer);
            if (!source)
            {
                return std::unexpected(std::move(source.error()));
            }
            if (!hasAny(source->desc.usage, BufferUsage::vertex) || binding.offset >= source->desc.size)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Vertex buffers require vertex usage and an in-bounds offset.",
                });
            }
            const auto remaining = (std::min)(source->desc.size - binding.offset, static_cast<std::uint64_t>(UINT_MAX));
            prepared->views.push_back(D3D12_VERTEX_BUFFER_VIEW{
                source->resource->GetGPUVirtualAddress() + binding.offset,
                static_cast<UINT>(remaining),
                pipeline->vertexStrides[firstSlot + index],
            });
            prepared->residencies.push_back(source->residency);
            prepared->sources.push_back(std::move(*source));
        }

        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto retained = queues.retainResource(commandBuffer, prepared, prepared->residencies);
        if (!retained)
        {
            return retained;
        }
        (*commandList)
            ->IASetVertexBuffers(firstSlot, static_cast<UINT>(prepared->views.size()), prepared->views.data());
        for (std::size_t index = 0; index < bindings.size(); ++index)
        {
            state.boundVertexBuffers[firstSlot + index] = true;
        }
        return {};
    }

    Status PipelineManager::bindStreamOutputBuffers(QueueManager& queues,
                                                    ResourceManager& resources,
                                                    detail::BackendCommandBuffer& commandBuffer,
                                                    std::uint32_t firstSlot,
                                                    std::span<const StreamOutputBufferBinding> bindings)
    {
        auto& state = commandBuffer.graphicsState();
        if (state.renderingOwner == nullptr || state.pipelineOwner == nullptr)
        {
            return std::unexpected(Error{ErrorCode::invalidState, 0, "A graphics pipeline must be bound first."});
        }
        auto* const pipeline = static_cast<GraphicsPipelineRecord*>(state.pipelineOwner.get());
        if (pipeline->meshShader != nullptr || bindings.empty() || firstSlot >= pipeline->streamOutputStrides.size() ||
            bindings.size() > pipeline->streamOutputStrides.size() - firstSlot)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Stream-output batches must be non-empty and fit contiguous declared slots.",
            });
        }

        auto prepared = std::make_shared<PreparedStreamOutputBindings>();
        prepared->sources.reserve(bindings.size() * 2);
        prepared->residencies.reserve(bindings.size() * 2);
        prepared->views.reserve(bindings.size());
        prepared->writableRanges.reserve(bindings.size() * 2);
        for (std::size_t index = 0; index < bindings.size(); ++index)
        {
            const auto requiredStride = pipeline->streamOutputStrides[firstSlot + index];
            auto appended = appendStreamOutputBinding(*prepared, resources, bindings[index], requiredStride);
            if (!appended)
            {
                return appended;
            }
        }

        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto retained = queues.retainResource(commandBuffer, prepared, prepared->residencies);
        if (!retained)
        {
            return retained;
        }
        (*commandList)->SOSetTargets(firstSlot, static_cast<UINT>(prepared->views.size()), prepared->views.data());
        for (std::size_t index = 0; index < bindings.size(); ++index)
        {
            state.boundStreamOutputBuffers[firstSlot + index] = true;
        }
        return {};
    }

    Status PipelineManager::bindIndexBuffer(QueueManager& queues,
                                            ResourceManager& resources,
                                            detail::BackendCommandBuffer& commandBuffer,
                                            BufferHandle buffer,
                                            IndexFormat format,
                                            std::uint64_t offset)
    {
        auto& state = commandBuffer.graphicsState();
        if (state.renderingOwner == nullptr || state.pipelineOwner == nullptr || !isValidIndexFormat(format))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Index buffers require a bound graphics pipeline and a valid index format.",
            });
        }
        auto* const pipeline = static_cast<GraphicsPipelineRecord*>(state.pipelineOwner.get());
        if (pipeline->meshShader != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Mesh pipelines do not consume input-assembler index buffers.",
            });
        }
        if (pipeline->stripIndexFormat.has_value() && pipeline->stripIndexFormat != format)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The index-buffer format must match the graphics pipeline primitive-restart format.",
            });
        }
        auto source = resources.bufferBindingSource(buffer);
        const auto alignment = format == IndexFormat::uint16 ? 2U : 4U;
        if (!source)
        {
            return std::unexpected(std::move(source.error()));
        }
        if (!hasAny(source->desc.usage, BufferUsage::index) || offset >= source->desc.size || offset % alignment != 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Index buffers require a bound graphics pipeline, index usage, and an aligned in-bounds offset.",
            });
        }
        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        auto retained = queues.retainResource(commandBuffer, source->owner, source->residency);
        if (!retained)
        {
            return retained;
        }
        const auto remaining = (std::min)(source->desc.size - offset, static_cast<std::uint64_t>(UINT_MAX));
        const D3D12_INDEX_BUFFER_VIEW native{
            source->resource->GetGPUVirtualAddress() + offset,
            static_cast<UINT>(remaining),
            format == IndexFormat::uint16 ? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_R32_UINT,
        };
        (*commandList)->IASetIndexBuffer(&native);
        state.indexBufferBound = true;
        return {};
    }

    Status PipelineManager::draw(QueueManager& queues,
                                 detail::BackendCommandBuffer& commandBuffer,
                                 std::uint32_t vertexCount,
                                 std::uint32_t instanceCount,
                                 std::uint32_t firstVertex,
                                 std::uint32_t firstInstance)
    {
        const auto& state = commandBuffer.graphicsState();
        auto* const pipeline = static_cast<GraphicsPipelineRecord*>(state.pipelineOwner.get());
        if (!isReadyForVertexDraw(state, pipeline) || vertexCount == 0 || instanceCount == 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Draw requires complete rendering, pipeline, binding, vertex, viewport, and scissor state.",
            });
        }
        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        (*commandList)->DrawInstanced(vertexCount, instanceCount, firstVertex, firstInstance);
        return {};
    }

    Status PipelineManager::drawIndexed(QueueManager& queues,
                                        detail::BackendCommandBuffer& commandBuffer,
                                        std::uint32_t indexCount,
                                        std::uint32_t instanceCount,
                                        std::uint32_t firstIndex,
                                        std::int32_t baseVertex,
                                        std::uint32_t firstInstance)
    {
        const auto& state = commandBuffer.graphicsState();
        auto* const pipeline = static_cast<GraphicsPipelineRecord*>(state.pipelineOwner.get());
        if (!isReadyForVertexDraw(state, pipeline) || !state.indexBufferBound || indexCount == 0 || instanceCount == 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Indexed draw requires complete graphics state, a bound index buffer, and non-zero draw counts.",
            });
        }
        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        (*commandList)->DrawIndexedInstanced(indexCount, instanceCount, firstIndex, baseVertex, firstInstance);
        return {};
    }

    Status PipelineManager::dispatchMesh(QueueManager& queues,
                                         detail::BackendCommandBuffer& commandBuffer,
                                         std::uint32_t groupCountX,
                                         std::uint32_t groupCountY,
                                         std::uint32_t groupCountZ)
    {
        const auto& state = commandBuffer.graphicsState();
        auto* const pipeline = static_cast<GraphicsPipelineRecord*>(state.pipelineOwner.get());
        if (!isReadyForMeshDispatch(state, pipeline))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Mesh dispatch requires complete rendering, mesh-pipeline, binding, viewport, and scissor state.",
            });
        }
        const auto& limits = impl_->meshShaderCapabilities;
        const auto totalWorkgroups = static_cast<std::uint64_t>(groupCountX) * groupCountY * groupCountZ;
        if (groupCountX == 0 || groupCountY == 0 || groupCountZ == 0 || groupCountX > limits.maxWorkgroupCount[0] ||
            groupCountY > limits.maxWorkgroupCount[1] || groupCountZ > limits.maxWorkgroupCount[2] ||
            totalWorkgroups > limits.maxTotalWorkgroups)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Mesh dispatch counts must fit every dimension and the total device workgroup limit.",
            });
        }

        auto commandList = queues.commandList6(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        (*commandList)->DispatchMesh(groupCountX, groupCountY, groupCountZ);
        return {};
    }

    Status PipelineManager::dispatchMeshIndirect(QueueManager& queues,
                                                 ResourceManager& resources,
                                                 detail::BackendCommandBuffer& commandBuffer,
                                                 const IndirectMeshDispatchDesc& desc)
    {
        const auto& state = commandBuffer.graphicsState();
        auto* const pipeline = static_cast<GraphicsPipelineRecord*>(state.pipelineOwner.get());
        if (!isReadyForMeshDispatch(state, pipeline))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Indirect mesh dispatch requires complete rendering, mesh-pipeline, binding, viewport, and scissor "
                "state.",
            });
        }

        if (auto prefix = validateDrawIndexPrefix(*pipeline, desc.drawIndexPrefix, desc.maxDispatchCount); !prefix)
        {
            return prefix;
        }

        const std::uint32_t recordSize =
            sizeof(DispatchIndirectArguments) + (desc.drawIndexPrefix ? drawIndexPrefixSize : 0U);
        constexpr IndirectCommandErrorMessages errorMessages{
            .invalidLayout = "Indirect mesh dispatch requires an argument buffer, aligned offsets, and an aligned "
                             "record-sized stride.",
            .invalidArgumentBuffer =
                "The mesh-dispatch argument buffer requires indirect usage and an in-bounds offset.",
            .argumentSpanOverflow = "The indirect mesh-dispatch argument span overflows.",
            .argumentRecordsOutOfBounds = "The indirect mesh-dispatch records exceed the argument buffer.",
            .invalidCountBuffer =
                "The indirect mesh count buffer requires indirect usage and an aligned in-bounds uint32 count.",
        };
        auto command = resolveIndirectCommand(
            resources, indirectCommandLayout(desc, desc.maxDispatchCount, recordSize), errorMessages);
        if (!command)
        {
            return std::unexpected(std::move(command.error()));
        }

        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        if (desc.maxDispatchCount == 0)
        {
            return {};
        }
        const auto drawIndexRootParameter =
            desc.drawIndexPrefix ? pipeline->layout->drawIndexRootParameter : invalidRootParameter;
        auto signature = drawIndexRootParameter != invalidRootParameter
                             ? impl_->drawIndexSignature(
                                   *pipeline->layout, DrawIndexCommandKind::dispatchMesh, command->layout.stride)
                             : impl_->indirectMeshSignature(command->layout.stride);
        if (!signature)
        {
            return std::unexpected(std::move(signature.error()));
        }
        if (desc.drawIndexPrefix && drawIndexRootParameter == invalidRootParameter)
        {
            command->layout.argumentOffset += drawIndexPrefixSize;
        }
        return recordIndirectExecution(
            queues, commandBuffer, **commandList, signature->Get(), *command, drawIndexRootParameter);
    }

    Status PipelineManager::drawIndirect(QueueManager& queues,
                                         ResourceManager& resources,
                                         detail::BackendCommandBuffer& commandBuffer,
                                         const IndirectDrawDesc& desc)
    {
        return executeIndirect(queues, resources, commandBuffer, desc, false);
    }

    Status PipelineManager::drawIndexedIndirect(QueueManager& queues,
                                                ResourceManager& resources,
                                                detail::BackendCommandBuffer& commandBuffer,
                                                const IndirectDrawDesc& desc)
    {
        return executeIndirect(queues, resources, commandBuffer, desc, true);
    }

    Status PipelineManager::executeIndirect(QueueManager& queues,
                                            ResourceManager& resources,
                                            detail::BackendCommandBuffer& commandBuffer,
                                            const IndirectDrawDesc& desc,
                                            bool indexed)
    {
        const auto& state = commandBuffer.graphicsState();
        auto* const pipeline = static_cast<GraphicsPipelineRecord*>(state.pipelineOwner.get());
        if (!isReadyForVertexDraw(state, pipeline) || (indexed && !state.indexBufferBound))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                indexed ? "Indexed indirect draw requires complete graphics state and a bound index buffer."
                        : "Indirect draw requires complete rendering, pipeline, binding, vertex, viewport, and scissor "
                          "state.",
            });
        }

        if (auto prefix = validateDrawIndexPrefix(*pipeline, desc.drawIndexPrefix, desc.maxDrawCount); !prefix)
        {
            return prefix;
        }

        const std::uint32_t recordSize =
            (indexed ? sizeof(DrawIndexedIndirectArguments) : sizeof(DrawIndirectArguments)) +
            (desc.drawIndexPrefix ? drawIndexPrefixSize : 0U);
        constexpr IndirectCommandErrorMessages errorMessages{
            .invalidLayout =
                "Indirect draw requires an argument buffer, aligned offsets, and a record-sized aligned stride.",
            .invalidArgumentBuffer = "The indirect argument buffer requires indirect usage and an in-bounds offset.",
            .argumentSpanOverflow = "The indirect argument span overflows.",
            .argumentRecordsOutOfBounds = "The indirect argument records exceed the argument buffer.",
            .invalidCountBuffer =
                "The indirect count buffer requires indirect usage and an aligned in-bounds uint32 count.",
        };
        auto command = resolveIndirectCommand(
            resources, indirectCommandLayout(desc, desc.maxDrawCount, recordSize), errorMessages);
        if (!command)
        {
            return std::unexpected(std::move(command.error()));
        }

        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        if (desc.maxDrawCount == 0)
        {
            return {};
        }
        const auto drawIndexRootParameter =
            desc.drawIndexPrefix ? pipeline->layout->drawIndexRootParameter : invalidRootParameter;
        auto signature =
            drawIndexRootParameter != invalidRootParameter
                ? impl_->drawIndexSignature(*pipeline->layout,
                                            indexed ? DrawIndexCommandKind::drawIndexed : DrawIndexCommandKind::draw,
                                            command->layout.stride)
                : impl_->indirectSignature(indexed, command->layout.stride);
        if (!signature)
        {
            return std::unexpected(std::move(signature.error()));
        }
        if (desc.drawIndexPrefix && drawIndexRootParameter == invalidRootParameter)
        {
            command->layout.argumentOffset += drawIndexPrefixSize;
        }
        return recordIndirectExecution(
            queues, commandBuffer, **commandList, signature->Get(), *command, drawIndexRootParameter);
    }

} // namespace rachis::d3d12
