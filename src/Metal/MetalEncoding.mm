#include "Metal/MetalInternal.h"

namespace rachis::metal
{
    namespace
    {

        [[nodiscard]] bool metalBindingLayoutsCompatible(const MetalBindingLayoutRecord& expected,
                                                         const MetalBindingLayoutRecord& actual)
        {
            return &expected == &actual ||
                   std::ranges::equal(
                       expected.entries, actual.entries, [](const BindingLayoutEntry& a, const BindingLayoutEntry& b) {
                           return a.binding == b.binding && a.type == b.type && a.visibility == b.visibility &&
                                  a.count == b.count && a.flags == b.flags;
                       });
        }

        [[nodiscard]] Status metalValidateBindingSetBatch(
            const MetalPipelineRecord& pipeline,
            std::uint32_t firstSet,
            std::span<const std::shared_ptr<MetalBindingSetVersion>> versions,
            const char* rangeMessage,
            const char* mismatchMessage)
        {
            const std::size_t setCount = pipeline.layout == nullptr ? 0 : pipeline.layout->sets.size();
            if (versions.empty() || firstSet >= setCount || versions.size() > setCount - firstSet)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument, rangeMessage));
            }
            for (std::size_t index = 0; index < versions.size(); ++index)
            {
                const auto& expected = pipeline.layout->sets[firstSet + index];
                const auto& actual = versions[index]->layout;
                if (expected == nullptr || actual == nullptr || !metalBindingLayoutsCompatible(*expected, *actual))
                {
                    return std::unexpected(metalSelectionError(ErrorCode::invalidArgument, mismatchMessage));
                }
            }
            return {};
        }

        [[nodiscard]] id<MTL4ArgumentTable> metalFindReusableSingleSetTable(
            const MetalResolvedStageBindings& stageBindings,
            std::span<const std::shared_ptr<MetalBindingSetVersion>> versions,
            std::span<const MetalBoundBuffer> vertexBuffers)
        {
            bool anyVertexBuffer = false;
            for (const auto& bound : vertexBuffers)
            {
                anyVertexBuffer = anyVertexBuffer || bound.buffer != nil;
            }

            if (stageBindings.usesPushConstants || stageBindings.usesDrawIndex || anyVertexBuffer ||
                stageBindings.resources.empty() ||
                std::ranges::any_of(stageBindings.resources,
                                    [](const MetalResolvedResourceBinding& res) { return res.argumentBuffer; }))
            {
                return nil;
            }
            const std::uint32_t onlySet = stageBindings.resources.front().set;
            const bool reusableSingleSet = onlySet < versions.size() && versions[onlySet] != nullptr &&
                                           versions[onlySet]->table != nil &&
                                           std::ranges::all_of(
                                               stageBindings.resources,
                                               [&](const MetalResolvedResourceBinding& res) {
                                                   return res.set == onlySet && res.metalIndex == res.binding;
                                               });
            if (reusableSingleSet)
            {
                return versions[onlySet]->table;
            }
            return nil;
        }

        struct MetalArgumentTableCapacity final
        {
            std::uint32_t maxBuffer = 0;
            std::uint32_t maxTexture = 0;
            std::uint32_t maxSampler = 0;
            bool anyBuffer = false;
            bool anyTexture = false;
            bool anySampler = false;
        };

        [[nodiscard]] Status metalMeasurePushConstantSlot(MetalArgumentTableCapacity& capacity,
                                                          const MetalResolvedStageBindings& stageBindings,
                                                          std::optional<std::uint64_t> pushConstantGpuAddress)
        {
            if (!stageBindings.usesPushConstants || !pushConstantGpuAddress.has_value())
            {
                return {};
            }
            if (stageBindings.pushConstantMetalIndex >= metalMaxBufferArguments)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "The push-constant slot exceeds Metal's argument-table capacity."));
            }
            capacity.anyBuffer = true;
            capacity.maxBuffer = (std::max)(capacity.maxBuffer, stageBindings.pushConstantMetalIndex + 1U);
            return {};
        }

        [[nodiscard]] Status metalMeasureVertexBufferSlots(MetalArgumentTableCapacity& capacity,
                                                           std::span<const MetalBoundBuffer> vertexBuffers,
                                                           std::uint32_t vertexBufferBaseIndex)
        {
            for (std::uint32_t slot = 0; slot < vertexBuffers.size(); ++slot)
            {
                if (vertexBuffers[slot].buffer == nil)
                {
                    continue;
                }
                const std::uint32_t targetIndex = vertexBufferBaseIndex + slot;
                if (targetIndex >= metalMaxBufferArguments)
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument, "A vertex-buffer slot exceeds Metal's argument-table capacity."));
                }
                capacity.anyBuffer = true;
                capacity.maxBuffer = (std::max)(capacity.maxBuffer, targetIndex + 1U);
            }
            return {};
        }

        void metalIncludeBindingSlot(MetalArgumentTableCapacity& capacity, BindingType type, std::uint32_t endIndex)
        {
            if (metalIsBufferType(type))
            {
                capacity.anyBuffer = true;
                capacity.maxBuffer = (std::max)(capacity.maxBuffer, endIndex);
            }
            else if (type == BindingType::sampledTexture || type == BindingType::storageTexture)
            {
                capacity.anyTexture = true;
                capacity.maxTexture = (std::max)(capacity.maxTexture, endIndex);
            }
            else if (type == BindingType::sampler)
            {
                capacity.anySampler = true;
                capacity.maxSampler = (std::max)(capacity.maxSampler, endIndex);
            }
        }

        [[nodiscard]] Status metalMeasureResourceSlots(
            id<MTLDevice> device,
            MetalArgumentTableCapacity& capacity,
            const MetalResolvedStageBindings& stageBindings,
            std::span<const std::shared_ptr<MetalBindingSetVersion>> versions)
        {
            std::uint64_t argumentSamplers = 0;
            for (const auto& res : stageBindings.resources)
            {
                if (res.set >= versions.size() || versions[res.set] == nullptr)
                {
                    return std::unexpected(
                        metalSelectionError(ErrorCode::invalidState, "A required Metal binding set is missing."));
                }
                if (res.argumentBuffer)
                {
                    const auto& version = *versions[res.set];
                    if (version.argumentBuffer == nil)
                    {
                        return std::unexpected(metalSelectionError(
                            ErrorCode::invalidState,
                            "The bound Metal binding set has no binding array for an unbounded shader array."));
                    }
                    if (version.variableDescriptorCount > res.argumentCapacity)
                    {
                        return std::unexpected(metalSelectionError(
                            ErrorCode::invalidArgument,
                            "The bound binding array has more elements than the shader's argument array holds."));
                    }
                    if (res.type == ShaderResourceType::sampler)
                    {
                        argumentSamplers += version.variableDescriptorCount;
                    }
                    metalIncludeBindingSlot(capacity, BindingType::storageBuffer, res.metalIndex + 1U);
                    continue;
                }
                for (const auto& slot : versions[res.set]->slots)
                {
                    if (slot.binding != res.binding || slot.arrayElement >= res.count)
                    {
                        continue;
                    }
                    const std::uint32_t targetIndex = res.metalIndex + slot.arrayElement;
                    if (targetIndex >= metalArgumentLimit(slot.type))
                    {
                        return std::unexpected(metalSelectionError(
                            ErrorCode::invalidArgument, "The resource index exceeds Metal's argument-table capacity."));
                    }
                    metalIncludeBindingSlot(capacity, slot.type, targetIndex + 1U);
                }
            }
            if (argumentSamplers != 0 && argumentSamplers > metalArgumentBufferSamplerLimit(device))
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "The binding arrays of samplers bound to one shader stage exceed "
                                        "maxVariableSamplerDescriptors."));
            }
            return {};
        }

        void metalWriteBindingSlot(id<MTL4ArgumentTable> table, const MetalBindingSlot& slot, std::uint32_t targetIndex)
        {
            if (metalIsBufferType(slot.type) && slot.buffer != nil)
            {
                [table setAddress:(slot.gpuAddress + slot.offset) atIndex:targetIndex];
            }
            else if ((slot.type == BindingType::sampledTexture || slot.type == BindingType::storageTexture) &&
                     slot.texture != nil)
            {
                [table setTexture:slot.texture.gpuResourceID atIndex:targetIndex];
            }
            else if (slot.type == BindingType::sampler && slot.sampler != nil)
            {
                [table setSamplerState:slot.sampler.gpuResourceID atIndex:targetIndex];
            }
        }

        void metalWriteResourceSlots(id<MTL4ArgumentTable> table,
                                     const MetalResolvedStageBindings& stageBindings,
                                     std::span<const std::shared_ptr<MetalBindingSetVersion>> versions)
        {
            for (const auto& res : stageBindings.resources)
            {
                if (res.argumentBuffer)
                {
                    [table setAddress:versions[res.set]->argumentBuffer.gpuAddress atIndex:res.metalIndex];
                    continue;
                }
                for (const auto& slot : versions[res.set]->slots)
                {
                    if (slot.binding != res.binding || slot.arrayElement >= res.count ||
                        std::holds_alternative<std::monostate>(slot.resource))
                    {
                        continue;
                    }
                    metalWriteBindingSlot(table, slot, res.metalIndex + slot.arrayElement);
                }
            }
        }

        void metalWriteVertexBufferSlots(id<MTL4ArgumentTable> table,
                                         std::span<const MetalBoundBuffer> vertexBuffers,
                                         std::uint32_t vertexBufferBaseIndex)
        {
            for (std::uint32_t slot = 0; slot < vertexBuffers.size(); ++slot)
            {
                const auto& bound = vertexBuffers[slot];
                if (bound.buffer == nil)
                {
                    continue;
                }
                [table setAddress:(bound.buffer.gpuAddress + bound.offset) atIndex:(vertexBufferBaseIndex + slot)];
            }
        }

        [[nodiscard]] Result<id<MTL4ArgumentTable>> metalBuildStageArgumentTable(
            id<MTLDevice> device,
            MetalCommandEntry& entry,
            MetalArgumentTableStage stage,
            const MetalResolvedStageBindings& stageBindings,
            std::span<const std::shared_ptr<MetalBindingSetVersion>> versions,
            std::optional<std::uint64_t> pushConstantGpuAddress,
            std::span<const MetalBoundBuffer> vertexBuffers,
            std::uint32_t vertexBufferBaseIndex)
        {
            if (auto reusable = metalFindReusableSingleSetTable(stageBindings, versions, vertexBuffers);
                reusable != nil)
            {
                return reusable;
            }

            MetalArgumentTableCapacity capacity{};
            if (auto measured = metalMeasurePushConstantSlot(capacity, stageBindings, pushConstantGpuAddress);
                !measured)
            {
                return std::unexpected(std::move(measured.error()));
            }
            if (auto measured = metalMeasureVertexBufferSlots(capacity, vertexBuffers, vertexBufferBaseIndex);
                !measured)
            {
                return std::unexpected(std::move(measured.error()));
            }
            if (auto measured = metalMeasureResourceSlots(device, capacity, stageBindings, versions); !measured)
            {
                return std::unexpected(std::move(measured.error()));
            }
            if (stageBindings.usesDrawIndex)
            {
                metalIncludeBindingSlot(capacity, BindingType::storageBuffer, stageBindings.drawIndexMetalIndex + 1U);
            }

            if (!capacity.anyBuffer && !capacity.anyTexture && !capacity.anySampler)
            {
                return nil;
            }

            auto acquired = acquireMetalArgumentTable(device,
                                                      entry,
                                                      stage,
                                                      capacity.anyBuffer ? capacity.maxBuffer : 0U,
                                                      capacity.anyTexture ? capacity.maxTexture : 0U,
                                                      capacity.anySampler ? capacity.maxSampler : 0U);
            if (!acquired)
            {
                return std::unexpected(std::move(acquired.error()));
            }
            id<MTL4ArgumentTable> const table = *acquired;

            metalWriteResourceSlots(table, stageBindings, versions);

            if (stageBindings.usesPushConstants && pushConstantGpuAddress.has_value())
            {
                [table setAddress:*pushConstantGpuAddress atIndex:stageBindings.pushConstantMetalIndex];
            }

            metalWriteVertexBufferSlots(table, vertexBuffers, vertexBufferBaseIndex);
            if (stageBindings.usesDrawIndex)
            {
                [table setAddress:entry.drawIndexAddress atIndex:stageBindings.drawIndexMetalIndex];
            }

            return table;
        }

        [[nodiscard]] Status metalApplyComputeDispatchState(id<MTLDevice> device,
                                                            MetalCommandEntry& entry,
                                                            const MetalPipelineRecord& pipeline)
        {
            if (entry.computeEncoder == nil)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidState, "Compute dispatch requires an active Metal compute encoder."));
            }
            if (entry.activeComputePipeline != &pipeline)
            {
                [entry.computeEncoder setComputePipelineState:pipeline.compute];
                entry.activeComputePipeline = &pipeline;
                entry.computeTableDirty = true;
            }

            std::optional<std::uint64_t> pushConstantGpuAddress;
            if (pipeline.computeStageBindings.usesPushConstants)
            {
                const auto copyBytes = std::max(entry.computePushConstantBytes, pipeline.pushConstantBytes);
                if (entry.computePushConstantsDirty || entry.activeComputePushSlice.buffer == nil ||
                    entry.activeComputePushSlice.byteLength < copyBytes)
                {
                    const std::span<const std::byte> sourceBytes{entry.computePushConstants.data(),
                                                                 entry.computePushConstantBytes};
                    auto captured = metalCapturePushConstantSlice(device, entry, sourceBytes, copyBytes);
                    if (!captured)
                    {
                        return std::unexpected(std::move(captured.error()));
                    }
                    entry.activeComputePushSlice = *captured;
                    entry.computePushConstantsDirty = false;
                    entry.computeTableDirty = true;
                    metalRetainResidency(entry, entry.activeComputePushSlice.buffer);
                }
                pushConstantGpuAddress = entry.activeComputePushSlice.gpuAddress;
            }

            if (!entry.computeTableDirty)
            {
#ifdef RACHIS_TESTING
                if (entry.activeComputeTable != nil)
                {
                    recordMetalArgumentTableEncoderReuse();
                }
#endif
                return {};
            }

            auto table = metalBuildStageArgumentTable(device,
                                                      entry,
                                                      MetalArgumentTableStage::compute,
                                                      pipeline.computeStageBindings,
                                                      entry.boundComputeVersions,
                                                      pushConstantGpuAddress,
                                                      {},
                                                      0);
            if (!table)
            {
                return std::unexpected(std::move(table.error()));
            }
            if (*table != nil)
            {
                [entry.computeEncoder setArgumentTable:*table];
            }
            entry.activeComputeTable = *table;
            entry.computeTableDirty = false;
            return {};
        }

        [[nodiscard]] constexpr bool metalIsValidLoadOperation(LoadOperation operation) noexcept
        {
            return operation == LoadOperation::load || operation == LoadOperation::clear ||
                   operation == LoadOperation::discard;
        }

        [[nodiscard]] constexpr bool metalIsValidStoreOperation(StoreOperation operation) noexcept
        {
            return operation == StoreOperation::store || operation == StoreOperation::discard;
        }

        [[nodiscard]] bool metalIsPortableDepthValue(float value) noexcept
        {
            return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
        }

        [[nodiscard]] constexpr MTLLoadAction metalLoadAction(LoadOperation operation) noexcept
        {
            switch (operation)
            {
            case LoadOperation::load:
                return MTLLoadActionLoad;
            case LoadOperation::clear:
                return MTLLoadActionClear;
            case LoadOperation::discard:
                return MTLLoadActionDontCare;
            }
            return MTLLoadActionDontCare;
        }

        [[nodiscard]] constexpr MTLStoreAction metalStoreAction(StoreOperation operation) noexcept
        {
            switch (operation)
            {
            case StoreOperation::store:
                return MTLStoreActionStore;
            case StoreOperation::discard:
                return MTLStoreActionDontCare;
            }
            return MTLStoreActionDontCare;
        }

        [[nodiscard]] constexpr MTLPrimitiveType metalPrimitiveType(PrimitiveTopology topology) noexcept
        {
            switch (topology)
            {
            case PrimitiveTopology::pointList:
                return MTLPrimitiveTypePoint;
            case PrimitiveTopology::lineList:
                return MTLPrimitiveTypeLine;
            case PrimitiveTopology::lineStrip:
                return MTLPrimitiveTypeLineStrip;
            case PrimitiveTopology::triangleList:
                return MTLPrimitiveTypeTriangle;
            case PrimitiveTopology::triangleStrip:
                return MTLPrimitiveTypeTriangleStrip;
            case PrimitiveTopology::patchList:
            case PrimitiveTopology::triangleFan:
                break;
            }
            return MTLPrimitiveTypeTriangle;
        }

        [[nodiscard]] Status metalValidateRestartIndexFormat(const MetalGraphicsPipelineState& raster,
                                                             IndexFormat format)
        {
            if (raster.stripIndexFormat.has_value() && *raster.stripIndexFormat != format)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "The index-buffer format must match the graphics pipeline primitive-restart format."));
            }
            return {};
        }

        [[nodiscard]] Status metalValidateIndexedRestart(const MetalGraphicsPipelineState& raster, IndexFormat format)
        {
            const bool strip =
                raster.topology == PrimitiveTopology::lineStrip || raster.topology == PrimitiveTopology::triangleStrip;
            if (strip && !raster.stripIndexFormat.has_value())
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::unsupported,
                    "Metal always restarts indexed strips at the maximum index, so indexed strip draws require a "
                    "pipeline strip index format."));
            }
            return metalValidateRestartIndexFormat(raster, format);
        }

        [[nodiscard]] constexpr MTLCullMode metalCullMode(CullMode mode) noexcept
        {
            switch (mode)
            {
            case CullMode::none:
                return MTLCullModeNone;
            case CullMode::front:
                return MTLCullModeFront;
            case CullMode::back:
                return MTLCullModeBack;
            }
            return MTLCullModeNone;
        }

        [[nodiscard]] constexpr MTLWinding metalWinding(FrontFace face) noexcept
        {
            return face == FrontFace::clockwise ? MTLWindingClockwise : MTLWindingCounterClockwise;
        }

        [[nodiscard]] constexpr MTLTriangleFillMode metalFillMode(FillMode mode) noexcept
        {
            return mode == FillMode::wireframe ? MTLTriangleFillModeLines : MTLTriangleFillModeFill;
        }

        [[nodiscard]] constexpr bool metalHasCompleteViewportState(const detail::GraphicsCommandState& state) noexcept
        {
            return state.viewportCount != 0 && state.viewportCount == state.scissorCount;
        }

        [[nodiscard]] bool metalDrawScissoredOut(const MetalCommandEntry& entry,
                                                 const MetalPipelineRecord& pipeline) noexcept
        {
            return entry.scissorsEmpty && pipeline.graphicsState.has_value() &&
                   pipeline.graphicsState->rasterizationEnabled;
        }

        [[nodiscard]] std::optional<MTLScissorRect> metalClipScissor(const ScissorRect& scissor,
                                                                     const MetalRenderingScope& scope) noexcept
        {
            const auto left = static_cast<std::uint64_t>(scissor.x);
            const auto top = static_cast<std::uint64_t>(scissor.y);
            const auto right = std::min(left + scissor.width, static_cast<std::uint64_t>(scope.width));
            const auto bottom = std::min(top + scissor.height, static_cast<std::uint64_t>(scope.height));
            if (left >= right || top >= bottom)
            {
                return std::nullopt;
            }
            return MTLScissorRect{static_cast<NSUInteger>(left),
                                  static_cast<NSUInteger>(top),
                                  static_cast<NSUInteger>(right - left),
                                  static_cast<NSUInteger>(bottom - top)};
        }

        template<std::size_t MaxSlots>
        [[nodiscard]] constexpr bool metalAllBound(const detail::FixedBoolMask<MaxSlots>& values) noexcept
        {
            return values.all();
        }

        void metalConfigureColorAttachment(MTLRenderPassColorAttachmentDescriptor* attachment,
                                           const MetalRenderingAttachment& source,
                                           bool forceClear,
                                           const ClearColor& clearValue) noexcept
        {
            attachment.texture = source.texture;
            attachment.level = 0;
            attachment.slice = 0;
            attachment.loadAction = forceClear ? MTLLoadActionClear : metalLoadAction(source.loadOperation);
            attachment.storeAction = MTLStoreActionUnknown;
            const auto& clear = forceClear ? clearValue : source.clearColor;
            attachment.clearColor = MTLClearColorMake(clear.red, clear.green, clear.blue, clear.alpha);
        }

        void metalDeferScopeStoreActions(MetalCommandEntry& entry, const MetalRenderingScope& scope) noexcept
        {
            auto& stores = entry.renderStoreActions;
            stores = {};
            stores.colorCount = static_cast<std::uint32_t>(scope.colors.size());
            for (std::uint32_t index = 0; index < stores.colorCount; ++index)
            {
                stores.colors[index] = metalStoreAction(scope.colors[index].storeOperation);
            }
            stores.hasDepth = scope.hasDepth;
            stores.depth = metalStoreAction(scope.depth.storeOperation);
            stores.hasStencil = scope.hasStencil;
            stores.stencil = metalStoreAction(scope.depth.stencilStoreOperation);
            stores.pending = true;
        }

        void metalStoreAttachmentsForRestart(MetalCommandEntry& entry,
                                             bool clearColor,
                                             std::uint32_t clearColorIndex,
                                             bool clearDepth,
                                             bool clearStencil) noexcept
        {
            auto& stores = entry.renderStoreActions;
            for (std::uint32_t index = 0; index < stores.colorCount; ++index)
            {
                stores.colors[index] =
                    clearColor && index == clearColorIndex ? MTLStoreActionDontCare : MTLStoreActionStore;
            }
            stores.depth = clearDepth ? MTLStoreActionDontCare : MTLStoreActionStore;
            stores.stencil = clearStencil ? MTLStoreActionDontCare : MTLStoreActionStore;
        }

        [[nodiscard]] Status metalBeginRenderEncoder(MetalCommandEntry& entry,
                                                     const MetalRenderingScope& scope,
                                                     bool clearColor,
                                                     std::uint32_t clearColorIndex,
                                                     const ClearColor& colorValue,
                                                     bool clearDepth,
                                                     float depthValue,
                                                     bool clearStencil,
                                                     std::uint32_t stencilValue)
        {
            if (entry.command == nil)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Rendering requires a begun Metal command buffer."));
            }
            const bool restarting = entry.renderEncoder != nil;
            if (restarting)
            {
                metalStoreAttachmentsForRestart(entry, clearColor, clearColorIndex, clearDepth, clearStencil);
            }
            endMetalEncoders(entry);
            MTL4RenderPassDescriptor* const pass = [MTL4RenderPassDescriptor new];
            pass.renderTargetWidth = scope.width;
            pass.renderTargetHeight = scope.height;
            pass.renderTargetArrayLength = scope.layerCount;
            for (std::uint32_t index = 0; index < scope.colors.size(); ++index)
            {
                const bool forceClear = clearColor && index == clearColorIndex;
                metalConfigureColorAttachment(pass.colorAttachments[index],
                                              scope.colors[index],
                                              forceClear,
                                              forceClear ? colorValue : ClearColor{});
                if (restarting && !forceClear)
                {
                    pass.colorAttachments[index].loadAction = MTLLoadActionLoad;
                }
                metalRetainResidency(entry, scope.colors[index].texture);
            }
            if (scope.hasDepth)
            {
                auto* const depth = pass.depthAttachment;
                depth.texture = scope.depth.texture;
                depth.loadAction = restarting ? MTLLoadActionLoad : metalLoadAction(scope.depth.loadOperation);
                if (clearDepth)
                {
                    depth.loadAction = MTLLoadActionClear;
                }
                depth.storeAction = MTLStoreActionUnknown;
                depth.clearDepth = clearDepth ? depthValue : scope.depth.depthClear;
                metalRetainResidency(entry, scope.depth.texture);
            }
            if (scope.hasStencil)
            {
                auto* const stencil = pass.stencilAttachment;
                stencil.texture = scope.depth.texture;
                stencil.loadAction = restarting ? MTLLoadActionLoad : metalLoadAction(scope.depth.stencilLoadOperation);
                if (clearStencil)
                {
                    stencil.loadAction = MTLLoadActionClear;
                }
                stencil.storeAction = MTLStoreActionUnknown;
                stencil.clearStencil = clearStencil ? stencilValue : scope.depth.stencilClear;
                metalRetainResidency(entry, scope.depth.texture);
            }
            entry.renderEncoder = [entry.command renderCommandEncoderWithDescriptor:pass];
            if (entry.renderEncoder == nil)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::nativeFailure, "Metal could not create a render command encoder."));
            }
            metalDeferScopeStoreActions(entry, scope);
            entry.renderEncoder.label = @"RHI.Metal.Render";
            entry.encoder = MetalEncoderKind::render;
            if (restarting)
            {
                [entry.renderEncoder barrierAfterQueueStages:MTLStageFragment
                                                beforeStages:MTLStageFragment
                                           visibilityOptions:MTL4VisibilityOptionDevice];
            }
            flushMetalRenderPendingBarriers(entry);
            return {};
        }

        void metalApplyGraphicsRasterState(MetalCommandEntry& entry, const MetalPipelineRecord& pipeline) noexcept
        {
            if (entry.renderEncoder == nil || !pipeline.graphicsState.has_value())
            {
                return;
            }
            const auto& raster = *pipeline.graphicsState;
            [entry.renderEncoder setCullMode:metalCullMode(raster.cullMode)];
            [entry.renderEncoder setFrontFacingWinding:metalWinding(raster.frontFace)];
            [entry.renderEncoder setTriangleFillMode:metalFillMode(raster.fillMode)];
            [entry.renderEncoder
                setDepthClipMode:raster.depthClipEnabled ? MTLDepthClipModeClip : MTLDepthClipModeClamp];
            [entry.renderEncoder setDepthBias:static_cast<float>(raster.depthBias)
                                   slopeScale:raster.depthBiasSlopeScale
                                        clamp:raster.depthBiasClamp];
            if (pipeline.depthStencil != nil)
            {
                [entry.renderEncoder setDepthStencilState:pipeline.depthStencil];
            }
        }

        void metalApplyGraphicsDynamicState(MetalCommandEntry& entry,
                                            const detail::GraphicsCommandState& state) noexcept
        {
            if (entry.renderEncoder == nil)
            {
                return;
            }
            if (state.viewportCount != 0)
            {
                std::array<MTLViewport, maxViewportCount> viewports{};
                for (std::uint32_t index = 0; index < state.viewportCount; ++index)
                {
                    const auto& viewport = entry.viewports[index];
                    viewports[index] = MTLViewport{
                        viewport.x, viewport.y, viewport.width, viewport.height, viewport.minDepth, viewport.maxDepth};
                }
                [entry.renderEncoder setViewports:viewports.data() count:state.viewportCount];
            }
            const auto* scope = static_cast<const MetalRenderingScope*>(state.renderingOwner.get());
            if (state.scissorCount != 0 && scope != nullptr)
            {
                std::array<MTLScissorRect, maxViewportCount> scissors{};
                bool empty = false;
                for (std::uint32_t index = 0; index < state.scissorCount && !empty; ++index)
                {
                    const auto clipped = metalClipScissor(entry.scissors[index], *scope);
                    empty = !clipped.has_value();
                    scissors[index] = clipped.value_or(MTLScissorRect{});
                }
                entry.scissorsEmpty = empty;
                if (!empty)
                {
                    [entry.renderEncoder setScissorRects:scissors.data() count:state.scissorCount];
                }
            }
            [entry.renderEncoder setBlendColorRed:entry.blendConstants.red
                                            green:entry.blendConstants.green
                                             blue:entry.blendConstants.blue
                                            alpha:entry.blendConstants.alpha];
            [entry.renderEncoder setStencilReferenceValue:entry.stencilReference];
        }

        [[nodiscard]] Result<std::optional<std::uint64_t>> metalRefreshGraphicsPushConstants(
            id<MTLDevice> device, MetalCommandEntry& entry, const MetalPipelineRecord& pipeline)
        {
            std::optional<std::uint64_t> pushConstantGpuAddress;
            const bool geometryPushConstants =
                pipeline.mesh ? (pipeline.meshStageBindings.usesPushConstants ||
                                 (pipeline.hasObjectShader && pipeline.objectStageBindings.usesPushConstants))
                              : pipeline.vertexStageBindings.usesPushConstants;
            const bool anyStagePushConstants =
                geometryPushConstants || (pipeline.hasPixelShader && pipeline.pixelStageBindings.usesPushConstants);
            if (!anyStagePushConstants)
            {
                return pushConstantGpuAddress;
            }
            const auto copyBytes = std::max(entry.graphicsPushConstantBytes, pipeline.pushConstantBytes);
            if (entry.graphicsPushConstantsDirty || entry.activeGraphicsPushSlice.buffer == nil ||
                entry.activeGraphicsPushSlice.byteLength < copyBytes)
            {
                const std::span<const std::byte> sourceBytes{entry.graphicsPushConstants.data(),
                                                             entry.graphicsPushConstantBytes};
                auto captured = metalCapturePushConstantSlice(device, entry, sourceBytes, copyBytes);
                if (!captured)
                {
                    return std::unexpected(std::move(captured.error()));
                }
                entry.activeGraphicsPushSlice = *captured;
                entry.graphicsPushConstantsDirty = false;
                if (geometryPushConstants)
                {
                    entry.vertexTableDirty = true;
                }
                if (pipeline.hasPixelShader && pipeline.pixelStageBindings.usesPushConstants)
                {
                    entry.fragmentTableDirty = true;
                }
                metalRetainResidency(entry, entry.activeGraphicsPushSlice.buffer);
            }
            pushConstantGpuAddress = entry.activeGraphicsPushSlice.gpuAddress;
            return pushConstantGpuAddress;
        }

        [[nodiscard]] Status metalApplyVertexArgumentTable(id<MTLDevice> device,
                                                           MetalCommandEntry& entry,
                                                           const MetalPipelineRecord& pipeline,
                                                           std::optional<std::uint64_t> pushConstantGpuAddress)
        {
            if (!entry.vertexTableDirty)
            {
#ifdef RACHIS_TESTING
                if (entry.activeVertexTable != nil)
                {
                    recordMetalArgumentTableEncoderReuse();
                }
#endif
                return {};
            }

            const auto vertexCount =
                pipeline.graphicsState.has_value() ? pipeline.graphicsState->vertexBufferCount : 0U;
            const auto vertexBase =
                pipeline.graphicsState.has_value() ? pipeline.graphicsState->vertexBufferBaseIndex : 0U;
            for (std::uint32_t slot = 0; slot < vertexCount; ++slot)
            {
                if (entry.vertexBuffers[slot].buffer != nil)
                {
                    metalRetainResidency(entry, entry.vertexBuffers[slot].buffer);
                }
            }

            const std::span<const MetalBoundBuffer> activeVertexBuffers{entry.vertexBuffers.data(), vertexCount};
            auto vertexTable = metalBuildStageArgumentTable(
                device,
                entry,
                MetalArgumentTableStage::vertex,
                pipeline.vertexStageBindings,
                entry.boundGraphicsVersions,
                pipeline.vertexStageBindings.usesPushConstants ? pushConstantGpuAddress : std::nullopt,
                activeVertexBuffers,
                vertexBase);
            if (!vertexTable)
            {
                return std::unexpected(std::move(vertexTable.error()));
            }
            if (*vertexTable != nil)
            {
                [entry.renderEncoder setArgumentTable:*vertexTable atStages:MTLRenderStageVertex];
            }
            entry.activeVertexTable = *vertexTable;
            entry.vertexTableDirty = false;
            return {};
        }

        [[nodiscard]] Status metalApplyMeshStageArgumentTables(id<MTLDevice> device,
                                                               MetalCommandEntry& entry,
                                                               const MetalPipelineRecord& pipeline,
                                                               std::optional<std::uint64_t> pushConstantGpuAddress)
        {
            if (!entry.vertexTableDirty)
            {
                return {};
            }
            if (pipeline.hasObjectShader)
            {
                auto objectTable = metalBuildStageArgumentTable(
                    device,
                    entry,
                    MetalArgumentTableStage::object,
                    pipeline.objectStageBindings,
                    entry.boundGraphicsVersions,
                    pipeline.objectStageBindings.usesPushConstants ? pushConstantGpuAddress : std::nullopt,
                    {},
                    0);
                if (!objectTable)
                {
                    return std::unexpected(std::move(objectTable.error()));
                }
                if (*objectTable != nil)
                {
                    [entry.renderEncoder setArgumentTable:*objectTable atStages:MTLRenderStageObject];
                }
            }
            auto meshTable = metalBuildStageArgumentTable(
                device,
                entry,
                MetalArgumentTableStage::mesh,
                pipeline.meshStageBindings,
                entry.boundGraphicsVersions,
                pipeline.meshStageBindings.usesPushConstants ? pushConstantGpuAddress : std::nullopt,
                {},
                0);
            if (!meshTable)
            {
                return std::unexpected(std::move(meshTable.error()));
            }
            if (*meshTable != nil)
            {
                [entry.renderEncoder setArgumentTable:*meshTable atStages:MTLRenderStageMesh];
            }
            entry.activeVertexTable = nil;
            entry.vertexTableDirty = false;
            return {};
        }

        [[nodiscard]] Status metalApplyFragmentArgumentTable(id<MTLDevice> device,
                                                             MetalCommandEntry& entry,
                                                             const MetalPipelineRecord& pipeline,
                                                             std::optional<std::uint64_t> pushConstantGpuAddress)
        {
            if (!entry.fragmentTableDirty)
            {
#ifdef RACHIS_TESTING
                if (entry.activeFragmentTable != nil)
                {
                    recordMetalArgumentTableEncoderReuse();
                }
#endif
                return {};
            }
            auto fragmentTable = metalBuildStageArgumentTable(
                device,
                entry,
                MetalArgumentTableStage::fragment,
                pipeline.pixelStageBindings,
                entry.boundGraphicsVersions,
                pipeline.pixelStageBindings.usesPushConstants ? pushConstantGpuAddress : std::nullopt,
                {},
                0);
            if (!fragmentTable)
            {
                return std::unexpected(std::move(fragmentTable.error()));
            }
            if (*fragmentTable != nil)
            {
                [entry.renderEncoder setArgumentTable:*fragmentTable atStages:MTLRenderStageFragment];
            }
            entry.activeFragmentTable = *fragmentTable;
            entry.fragmentTableDirty = false;
            return {};
        }

        constexpr std::uint32_t kMetalDrawIndexTableMinimum = 1024;

        [[nodiscard]] Status metalPrepareDrawIndex(id<MTLDevice> device, MetalCommandEntry& entry)
        {
            if (entry.drawIndex >= entry.drawIndexTableCapacity)
            {
                std::uint32_t capacity = std::max(kMetalDrawIndexTableMinimum, entry.drawIndexTableCapacity);
                while (capacity <= entry.drawIndex && capacity < (1U << 31U))
                {
                    capacity *= 2U;
                }
                capacity = std::max(capacity, entry.drawIndex + 1U);
                id<MTLBuffer> table =
                    [device newBufferWithLength:static_cast<NSUInteger>(capacity) * sizeof(std::uint32_t)
                                        options:MTLResourceStorageModeShared];
#ifdef RACHIS_TESTING
                if (capacity > kMetalDrawIndexTableMinimum && consumeInjectedMetalDrawIndexTableGrowthFailure())
                {
                    table = nil;
                }
#endif
                if (table == nil)
                {
                    return std::unexpected(
                        metalSelectionError(ErrorCode::outOfMemory, "Metal could not allocate the draw-index table."));
                }
                table.label = @"RHI.Metal.DrawIndexTable";
                auto* const values = static_cast<std::uint32_t*>(table.contents);
                for (std::uint32_t index = 0; index < capacity; ++index)
                {
                    values[index] = index;
                }
                if (entry.drawIndexTable != nil)
                {
                    try
                    {
                        entry.indirectScratchBuffers.push_back(entry.drawIndexTable);
                    }
                    catch (const std::bad_alloc&)
                    {
                        return std::unexpected(metalSelectionError(ErrorCode::outOfMemory,
                                                                   "Failed to retain the Metal draw-index table."));
                    }
                }
                entry.drawIndexTable = table;
                entry.drawIndexTableCapacity = capacity;
            }
            metalRetainResidency(entry, entry.drawIndexTable);
            const auto address =
                entry.drawIndexTable.gpuAddress + static_cast<std::uint64_t>(entry.drawIndex) * sizeof(std::uint32_t);
            if (address != entry.drawIndexAddress)
            {
                entry.drawIndexAddress = address;
                entry.vertexTableDirty = true;
            }
            return {};
        }

        void metalResetDrawIndex(MetalCommandEntry& entry) noexcept
        {
            if (entry.drawIndex != 0)
            {
                entry.drawIndex = 0;
                entry.vertexTableDirty = true;
            }
        }

        class MetalDrawIndexRestore final
        {
        public:
            explicit MetalDrawIndexRestore(MetalCommandEntry& entry) noexcept : entry_(&entry)
            {
            }

            ~MetalDrawIndexRestore()
            {
                metalResetDrawIndex(*entry_);
            }

            MetalDrawIndexRestore(const MetalDrawIndexRestore&) = delete;
            MetalDrawIndexRestore& operator=(const MetalDrawIndexRestore&) = delete;
            MetalDrawIndexRestore(MetalDrawIndexRestore&&) = delete;
            MetalDrawIndexRestore& operator=(MetalDrawIndexRestore&&) = delete;

        private:
            MetalCommandEntry* entry_ = nullptr;
        };

        [[nodiscard]] Status metalReserveDrawIndices(id<MTLDevice> device,
                                                     MetalCommandEntry& entry,
                                                     const MetalPipelineRecord& pipeline,
                                                     std::uint32_t maxRecords)
        {
            if (!pipeline.readsDrawIndex || maxRecords <= 1)
            {
                return {};
            }
            const MetalDrawIndexRestore restore(entry);
            entry.drawIndex = maxRecords - 1U;
            return metalPrepareDrawIndex(device, entry);
        }

        [[nodiscard]] Status metalApplyGraphicsDrawState(id<MTLDevice> device,
                                                         MetalCommandEntry& entry,
                                                         const MetalPipelineRecord& pipeline)
        {
            if (entry.renderEncoder == nil)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Draw requires an active Metal render encoder."));
            }
            if (pipeline.readsDrawIndex)
            {
                if (auto prepared = metalPrepareDrawIndex(device, entry); !prepared)
                {
                    return prepared;
                }
            }
            if (entry.activeGraphicsPipeline != &pipeline)
            {
                [entry.renderEncoder setRenderPipelineState:pipeline.graphics];
                metalApplyGraphicsRasterState(entry, pipeline);
                entry.activeGraphicsPipeline = &pipeline;
                entry.vertexTableDirty = true;
                entry.fragmentTableDirty = true;
            }

            auto pushConstantGpuAddress = metalRefreshGraphicsPushConstants(device, entry, pipeline);
            if (!pushConstantGpuAddress)
            {
                return std::unexpected(std::move(pushConstantGpuAddress.error()));
            }
            auto applied = pipeline.mesh
                               ? metalApplyMeshStageArgumentTables(device, entry, pipeline, *pushConstantGpuAddress)
                               : metalApplyVertexArgumentTable(device, entry, pipeline, *pushConstantGpuAddress);
            if (!applied)
            {
                return applied;
            }
            if (!pipeline.hasPixelShader)
            {
                return {};
            }
            return metalApplyFragmentArgumentTable(device, entry, pipeline, *pushConstantGpuAddress);
        }

        [[nodiscard]] bool metalClearRectanglesCoverTarget(const MetalRenderingScope& scope,
                                                           std::span<const ScissorRect> rectangles)
        {
            const auto coversTarget = [&](const ScissorRect& rect) {
                return rect.x == 0 && rect.y == 0 && rect.width == scope.width && rect.height == scope.height;
            };
            if (rectangles.empty() || std::ranges::any_of(rectangles, coversTarget))
            {
                return true;
            }
            std::vector<std::uint64_t> columns{0, scope.width};
            columns.reserve(rectangles.size() * 2 + 2);
            for (const auto& rect : rectangles)
            {
                columns.push_back(static_cast<std::uint64_t>(rect.x));
                columns.push_back(static_cast<std::uint64_t>(rect.x) + rect.width);
            }
            std::ranges::sort(columns);
            const auto duplicates = std::ranges::unique(columns);
            columns.erase(duplicates.begin(), duplicates.end());
            std::vector<std::pair<std::uint64_t, std::uint64_t>> rows;
            rows.reserve(rectangles.size());
            for (std::size_t column = 0; column + 1 < columns.size(); ++column)
            {
                rows.clear();
                for (const auto& rect : rectangles)
                {
                    const auto left = static_cast<std::uint64_t>(rect.x);
                    if (left <= columns[column] && left + rect.width >= columns[column + 1])
                    {
                        const auto top = static_cast<std::uint64_t>(rect.y);
                        rows.emplace_back(top, top + rect.height);
                    }
                }
                std::ranges::sort(rows);
                std::uint64_t covered = 0;
                for (const auto& [top, bottom] : rows)
                {
                    if (top > covered)
                    {
                        return false;
                    }
                    covered = std::max(covered, bottom);
                }
                if (covered < scope.height)
                {
                    return false;
                }
            }
            return true;
        }

        [[nodiscard]] Status metalValidateClearRectangles(const MetalRenderingScope& scope,
                                                          std::span<const ScissorRect> rectangles)
        {
            for (const auto& rect : rectangles)
            {
                if (rect.x < 0 || rect.y < 0 || rect.width == 0 || rect.height == 0)
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument,
                        "Attachment clear rectangles must have a non-negative origin and non-zero extent."));
                }
                if (static_cast<std::uint64_t>(rect.x) + rect.width > scope.width ||
                    static_cast<std::uint64_t>(rect.y) + rect.height > scope.height)
                {
                    return std::unexpected(
                        metalSelectionError(ErrorCode::invalidArgument,
                                            "An attachment clear rectangle is outside the active rendering extent."));
                }
            }
            if (!metalClearRectanglesCoverTarget(scope, rectangles))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::unsupported,
                    "Metal attachment clears inside a rendering scope must cover the whole render target."));
            }
            return {};
        }

    } // namespace

    Status MetalDevice::bindComputePipeline(detail::BackendCommandBuffer& commandBuffer, PipelineHandle pipeline)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        std::uint64_t deviceToken = 0;
        std::shared_ptr<MetalPipelineRecord> record;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                           "Metal bindComputePipeline requires an active device."));
            }
            deviceToken = native_->deviceToken;
            auto* found = native_->pipelines.find(pipeline);
            if (found == nullptr || *found == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, "The pipeline handle is invalid or stale."));
            }
            if ((*found)->kind != MetalPipelineKind::compute || (*found)->compute == nil)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "bindComputePipeline requires a live compute pipeline."));
            }
            record = *found;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        auto kept = retainMetalApplicationResource(*native, std::static_pointer_cast<void>(record));
        if (!kept)
        {
            return kept;
        }
        const std::scoped_lock lock(native->owner()->mutex);
        auto* entry = native->entry();
        if (entry == nullptr || entry->command == nil || !entry->begun)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState, "A compute pipeline bind requires a begun Metal command buffer."));
        }
        if (entry->encoder == MetalEncoderKind::render)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState, "Dispatch requires a bound compute pipeline outside a rendering scope."));
        }
        entry->boundComputePipeline = record;
        entry->boundComputeVersions.clear();
        native->computeState().pipelineOwner = record;
        entry->computeTableDirty = true;
        auto prepared = native->ensureComputeEncoder();
        if (!prepared)
        {
            return prepared;
        }
        if (entry->activeComputePipeline != record.get())
        {
            [entry->computeEncoder setComputePipelineState:record->compute];
            entry->activeComputePipeline = record.get();
        }
        return {};
    }

    Status MetalDevice::bindComputeBindingSet(detail::BackendCommandBuffer& commandBuffer,
                                              std::uint32_t setIndex,
                                              BindingSetHandle bindingSet,
                                              std::span<const std::uint32_t> dynamicOffsets)
    {
        const std::array sets{bindingSet};
        return bindComputeBindingSets(commandBuffer, setIndex, sets, dynamicOffsets);
    }

    Status MetalDevice::bindComputeBindingSets(detail::BackendCommandBuffer& commandBuffer,
                                               std::uint32_t firstSet,
                                               std::span<const BindingSetHandle> bindingSets,
                                               std::span<const std::uint32_t> dynamicOffsets)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (bindingSets.empty() || firstSet >= maxBindingSets || bindingSets.size() > maxBindingSets - firstSet)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument,
                                    "Compute binding-set indices must stay within the portable four-set limit."));
        }
        std::uint64_t deviceToken = 0;
        std::vector<std::shared_ptr<MetalBindingSetVersion>> versions;
        versions.reserve(bindingSets.size());
        std::vector<std::shared_ptr<MetalBindingSetCommandRetention>> retentions;
        retentions.reserve(bindingSets.size());
        std::uint32_t requiredDynamic = 0;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                           "Metal bindComputeBindingSets requires an active device."));
            }
            deviceToken = native_->deviceToken;
            for (const auto handle : bindingSets)
            {
                auto* found = native_->bindingSets.find(handle);
                if (found == nullptr || (*found)->version == nullptr)
                {
                    return std::unexpected(
                        metalSelectionError(ErrorCode::invalidArgument, "The binding set handle is invalid or stale."));
                }
                requiredDynamic += (*found)->layout->dynamicBufferCount;
                versions.push_back((*found)->version);
                retentions.push_back(metalRetainBindingSetForCommand(**found));
            }
        }
        if (dynamicOffsets.size() != requiredDynamic)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument, "Dynamic offset count must match the bound Metal binding sets."));
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        if (auto adjusted = metalApplyDynamicOffsets(versions, dynamicOffsets); !adjusted)
        {
            return adjusted;
        }
        const std::scoped_lock lock(native->owner()->mutex);
        auto* entry = native->entry();
        if (entry == nullptr || entry->command == nil || !entry->begun)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Binding sets require a begun Metal command buffer."));
        }
        if (entry->boundComputePipeline == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "A compute pipeline must be bound before binding sets."));
        }
        if (auto valid = metalValidateBindingSetBatch(*entry->boundComputePipeline,
                                                      firstSet,
                                                      versions,
                                                      "The compute binding-set batch is empty or exceeds the pipeline "
                                                      "layout.",
                                                      "A compute binding set does not match the pipeline layout.");
            !valid)
        {
            return valid;
        }
        for (auto& version : versions)
        {
            auto kept = retainMetalApplicationResource(*native, std::static_pointer_cast<void>(version));
            if (!kept)
            {
                return kept;
            }
        }
        if (auto counted = retainMetalBindingSetRetentions(*native, retentions); !counted)
        {
            return counted;
        }
        if (entry->boundComputeVersions.size() < firstSet + versions.size())
        {
            entry->boundComputeVersions.resize(firstSet + versions.size());
        }
        for (std::size_t index = 0; index < versions.size(); ++index)
        {
            auto& version = versions[index];
            entry->boundComputeVersions[firstSet + index] = version;
            metalRetainBindingSetResidency(*entry, *version);
        }
        entry->computeTableDirty = true;
        return {};
    }

    Status MetalDevice::setComputePushConstants(detail::BackendCommandBuffer& commandBuffer,
                                                std::uint32_t offset,
                                                std::span<const std::byte> data)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (data.empty() || offset % sizeof(std::uint32_t) != 0 || data.size() % sizeof(std::uint32_t) != 0 ||
            data.size() > maxPushConstantBytes || offset > maxPushConstantBytes - data.size())
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument,
                "Compute push constants require a non-empty, 4-byte aligned range within 128 bytes."));
        }
        std::uint64_t deviceToken = 0;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                           "Metal setComputePushConstants requires an active device."));
            }
            deviceToken = native_->deviceToken;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        const std::scoped_lock lock(native->owner()->mutex);
        auto* entry = native->entry();
        if (entry == nullptr || entry->command == nil || !entry->begun)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                       "Compute push constants require a begun Metal command buffer."));
        }
        if (entry->boundComputePipeline == nullptr)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState, "A compute pipeline must be bound before setting push constants."));
        }
        const auto& ranges = entry->boundComputePipeline->layout->pushConstantRanges;
        if (!metalPushConstantWriteCovered(ranges, ShaderStage::compute, offset, data.size()))
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument,
                "The push-constant write is not contained in compute-visible pipeline layout ranges."));
        }
        const auto end = static_cast<std::uint32_t>(offset + data.size());
        std::memcpy(entry->computePushConstants.data() + offset, data.data(), data.size());
        entry->computePushConstantBytes = std::max(entry->computePushConstantBytes, end);
        entry->computePushConstantsDirty = true;
        entry->computeTableDirty = true;
        return {};
    }

    Status MetalDevice::dispatch(detail::BackendCommandBuffer& commandBuffer,
                                 std::uint32_t groupCountX,
                                 std::uint32_t groupCountY,
                                 std::uint32_t groupCountZ)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        const auto& limits = capabilities_.limits.maxComputeWorkgroupCount;
        if (groupCountX == 0 || groupCountY == 0 || groupCountZ == 0 || groupCountX > limits[0] ||
            groupCountY > limits[1] || groupCountZ > limits[2])
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument,
                                    "Dispatch group counts must be between one and the device per-dimension limit."));
        }
        std::uint64_t deviceToken = 0;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal dispatch requires an active device."));
            }
            deviceToken = native_->deviceToken;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        const std::scoped_lock lock(native->owner()->mutex);
        auto* entry = native->entry();
        if (entry == nullptr || entry->command == nil || !entry->begun)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Dispatch requires a begun Metal command buffer."));
        }
        if (entry->boundComputePipeline == nullptr || entry->encoder == MetalEncoderKind::render)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState, "Dispatch requires a bound compute pipeline outside a rendering scope."));
        }
        const auto& pipeline = *entry->boundComputePipeline;
        for (std::size_t index = 0; index < pipeline.setRequired.size(); ++index)
        {
            if (pipeline.setRequired[index] == 0)
            {
                continue;
            }
            if (index >= entry->boundComputeVersions.size() || entry->boundComputeVersions[index] == nullptr)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidState, "Every required compute binding set must be bound before dispatch."));
            }
        }
        auto prepared = native->ensureComputeEncoder();
        if (!prepared)
        {
            return prepared;
        }
        auto applied = metalApplyComputeDispatchState(native->owner()->device, *entry, pipeline);
        if (!applied)
        {
            return applied;
        }
        const auto& workgroup = pipeline.workgroup;
        [entry->computeEncoder dispatchThreadgroups:MTLSizeMake(groupCountX, groupCountY, groupCountZ)
                              threadsPerThreadgroup:MTLSizeMake(workgroup.x, workgroup.y, workgroup.z)];
        return {};
    }

    namespace
    {

        struct MetalIndirectRecordLayout final
        {
            BufferHandle argumentBuffer = {};
            std::uint64_t argumentOffset = 0;
            std::uint32_t maxRecords = 0;
            std::uint32_t recordSize = 0;
            std::uint32_t stride = 0;
            BufferHandle countBuffer = {};
            std::uint64_t countOffset = 0;
            std::uint32_t prefixBytes = 0;
        };

        struct MetalIndirectRecordMessages final
        {
            std::string_view invalidLayout;
            std::string_view invalidArgumentBuffer;
            std::string_view recordsOutOfBounds;
            std::string_view invalidCountBuffer;
        };

        template<typename IndirectDesc>
        [[nodiscard]] MetalIndirectRecordLayout metalIndirectRecordLayout(const IndirectDesc& desc,
                                                                          std::uint32_t maxRecords,
                                                                          std::uint32_t recordSize) noexcept
        {
            return {
                .argumentBuffer = desc.argumentBuffer,
                .argumentOffset = desc.argumentOffset,
                .maxRecords = maxRecords,
                .recordSize = recordSize,
                .stride = desc.stride == 0 ? recordSize : desc.stride,
                .countBuffer = desc.countBuffer,
                .countOffset = desc.countOffset,
            };
        }

        template<typename IndirectDesc>
        [[nodiscard]] MetalIndirectRecordLayout metalPrefixedRecordLayout(const IndirectDesc& desc,
                                                                          std::uint32_t maxRecords,
                                                                          std::uint32_t argumentSize) noexcept
        {
            const std::uint32_t prefix = desc.drawIndexPrefix ? drawIndexPrefixSize : 0U;
            auto layout = metalIndirectRecordLayout(desc, maxRecords, argumentSize + prefix);
            layout.prefixBytes = prefix;
            return layout;
        }

        [[nodiscard]] Status metalRequireDrawIndexPrefix(const MetalPipelineRecord& pipeline,
                                                         const MetalIndirectRecordLayout& layout)
        {
            if (pipeline.readsDrawIndex && layout.prefixBytes == 0 && layout.maxRecords > 1)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "An indirect call with more than one record must set drawIndexPrefix when the bound pipeline "
                    "reads the draw index."));
            }
            return {};
        }

        [[nodiscard]] Status metalValidateIndirectRecordLayout(const MetalIndirectRecordLayout& layout,
                                                               const MetalIndirectRecordMessages& messages)
        {
            if (!layout.argumentBuffer.isValid() || layout.argumentOffset % indirectArgumentAlignment != 0 ||
                layout.stride < layout.recordSize || layout.stride % indirectArgumentAlignment != 0 ||
                (!layout.countBuffer.isValid() && layout.countOffset != 0))
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, std::string{messages.invalidLayout}));
            }
            return {};
        }

        struct MetalResolvedIndirectRecords final
        {
            __strong id<MTLBuffer> argument = nil;
            std::shared_ptr<void> argumentOwner;
            __strong id<MTLBuffer> count = nil;
            std::shared_ptr<void> countOwner;
        };

        [[nodiscard]] Result<MetalResolvedIndirectRecords> metalResolveIndirectRecords(
            MetalDeviceNative& native,
            const MetalIndirectRecordLayout& layout,
            const MetalIndirectRecordMessages& messages)
        {
            auto* record = native.buffers.find(layout.argumentBuffer);
            if (record == nullptr || record->buffer == nil || !hasAny(record->desc.usage, BufferUsage::indirect) ||
                layout.argumentOffset > record->desc.size)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, std::string{messages.invalidArgumentBuffer}));
            }
            const auto available = record->desc.size - layout.argumentOffset;
            if (layout.maxRecords != 0 &&
                (available < layout.recordSize ||
                 static_cast<std::uint64_t>(layout.maxRecords - 1U) > (available - layout.recordSize) / layout.stride))
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, std::string{messages.recordsOutOfBounds}));
            }
            MetalResolvedIndirectRecords resolved{.argument = record->buffer, .argumentOwner = record->owner};
            if (!layout.countBuffer.isValid())
            {
                return resolved;
            }
            auto* count = native.buffers.find(layout.countBuffer);
            if (count == nullptr || count->buffer == nil || !hasAny(count->desc.usage, BufferUsage::indirect) ||
                layout.countOffset % indirectArgumentAlignment != 0 || layout.countOffset > count->desc.size ||
                sizeof(std::uint32_t) > count->desc.size - layout.countOffset)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, std::string{messages.invalidCountBuffer}));
            }
            resolved.count = count->buffer;
            resolved.countOwner = count->owner;
            return resolved;
        }

        [[nodiscard]] Status metalRetainIndirectRecords(MetalCommandBuffer& native,
                                                        MetalCommandEntry& entry,
                                                        const MetalResolvedIndirectRecords& records)
        {
            if (auto kept = retainMetalApplicationResource(native, records.argumentOwner); !kept)
            {
                return kept;
            }
            metalRetainResidency(entry, records.argument);
            if (records.count == nil)
            {
                return {};
            }
            if (auto kept = retainMetalApplicationResource(native, records.countOwner); !kept)
            {
                return kept;
            }
            metalRetainResidency(entry, records.count);
            return {};
        }

        struct MetalIndirectRecordSpan final
        {
            std::uint64_t firstAddress = 0;
            std::uint64_t stride = 0;
        };

        [[nodiscard]] MetalIndirectClampRequest metalClampRequest(const MetalIndirectRecordLayout& layout,
                                                                  const MetalResolvedIndirectRecords& records,
                                                                  std::uint32_t zeroFirstWord,
                                                                  std::uint32_t zeroWordCount) noexcept
        {
            return {
                .sourceAddress = records.argument.gpuAddress + layout.argumentOffset + layout.prefixBytes,
                .countAddress = records.count.gpuAddress + layout.countOffset,
                .maxRecords = layout.maxRecords,
                .strideBytes = layout.stride,
                .recordWords =
                    (layout.recordSize - layout.prefixBytes) / static_cast<std::uint32_t>(sizeof(std::uint32_t)),
                .zeroFirstWord = zeroFirstWord,
                .zeroWordCount = zeroWordCount,
            };
        }

        [[nodiscard]] Result<MetalIndirectRecordSpan> metalClampComputeRecords(
            id<MTLDevice> device,
            MetalCommandEntry& entry,
            const MetalIndirectClampPipelines& pipelines,
            const MetalIndirectRecordLayout& layout,
            const MetalResolvedIndirectRecords& records)
        {
            if (records.count == nil)
            {
                return MetalIndirectRecordSpan{records.argument.gpuAddress + layout.argumentOffset + layout.prefixBytes,
                                               layout.stride};
            }
            auto clamped = metalPrepareClampedRecords(
                device, entry, metalClampRequest(layout, records, 0, sizeof(DispatchIndirectArguments) / 4U));
            if (!clamped)
            {
                return std::unexpected(std::move(clamped.error()));
            }
            if (auto encoded =
                    metalEncodeComputeRecordClamp(device, entry, pipelines.compute, *clamped, layout.maxRecords);
                !encoded)
            {
                return std::unexpected(std::move(encoded.error()));
            }
            return MetalIndirectRecordSpan{clamped->recordsAddress, clamped->recordBytes};
        }

        [[nodiscard]] Result<MetalIndirectRecordSpan> metalClampRenderRecords(
            id<MTLDevice> device,
            MetalCommandEntry& entry,
            const MetalIndirectClampPipelines& pipelines,
            const MetalIndirectRecordLayout& layout,
            const MetalResolvedIndirectRecords& records,
            std::uint32_t zeroFirstWord,
            std::uint32_t zeroWordCount)
        {
            if (records.count == nil)
            {
                return MetalIndirectRecordSpan{records.argument.gpuAddress + layout.argumentOffset + layout.prefixBytes,
                                               layout.stride};
            }
            auto clamped = metalPrepareClampedRecords(
                device, entry, metalClampRequest(layout, records, zeroFirstWord, zeroWordCount));
            if (!clamped)
            {
                return std::unexpected(std::move(clamped.error()));
            }
            if (auto encoded =
                    metalEncodeRenderRecordClamp(device, entry, pipelines.render, *clamped, layout.maxRecords);
                !encoded)
            {
                return std::unexpected(std::move(encoded.error()));
            }
            return MetalIndirectRecordSpan{clamped->recordsAddress, clamped->recordBytes};
        }

        [[nodiscard]] Result<MetalIndirectClampPipelines> metalClampPipelinesFor(
            std::mutex& deviceMutex,
            const std::shared_ptr<MetalDeviceNative>& native,
            const MetalIndirectRecordLayout& layout)
        {
            if (!layout.countBuffer.isValid() || layout.maxRecords == 0)
            {
                return MetalIndirectClampPipelines{};
            }
            return metalIndirectClampPipelines(deviceMutex, native);
        }

        [[nodiscard]] Status metalValidateIndirectDispatchRecording(const MetalCommandEntry* entry)
        {
            if (entry == nullptr || !entry->begun)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Dispatch requires a begun Metal command buffer."));
            }
            if (entry->boundComputePipeline == nullptr || entry->encoder == MetalEncoderKind::render)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidState, "Dispatch requires a bound compute pipeline outside a rendering scope."));
            }
            const auto& pipeline = *entry->boundComputePipeline;
            for (std::size_t index = 0; index < pipeline.setRequired.size(); ++index)
            {
                if (pipeline.setRequired[index] == 0)
                {
                    continue;
                }
                if (index >= entry->boundComputeVersions.size() || entry->boundComputeVersions[index] == nullptr)
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidState, "Every required compute binding set must be bound before dispatch."));
                }
            }
            return {};
        }

        constexpr MetalIndirectRecordMessages kMetalDispatchRecordMessages{
            .invalidLayout =
                "Indirect dispatch requires an argument buffer, aligned offsets, and an aligned 12-byte record.",
            .invalidArgumentBuffer =
                "The indirect dispatch argument buffer requires indirect usage and an in-bounds offset.",
            .recordsOutOfBounds = "The indirect dispatch records exceed the argument buffer.",
            .invalidCountBuffer =
                "The indirect dispatch count buffer requires indirect usage and an aligned in-bounds uint32 count.",
        };

    } // namespace

    Status MetalDevice::dispatchIndirect(detail::BackendCommandBuffer& commandBuffer, const IndirectDispatchDesc& desc)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        const auto layout = metalIndirectRecordLayout(desc, desc.maxDispatchCount, sizeof(DispatchIndirectArguments));
        if (auto validated = metalValidateIndirectRecordLayout(layout, kMetalDispatchRecordMessages); !validated)
        {
            return validated;
        }
        auto pipelines = metalClampPipelinesFor(mutex_, native_, layout);
        if (!pipelines)
        {
            return std::unexpected(std::move(pipelines.error()));
        }
        std::uint64_t deviceToken = 0;
        id<MTLDevice> device = nil;
        MetalResolvedIndirectRecords records{};
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal dispatchIndirect requires an active device."));
            }
            deviceToken = native_->deviceToken;
            device = native_->device;
            auto resolved = metalResolveIndirectRecords(*native_, layout, kMetalDispatchRecordMessages);
            if (!resolved)
            {
                return std::unexpected(std::move(resolved.error()));
            }
            records = std::move(*resolved);
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        const std::scoped_lock lock(native->owner()->mutex);
        auto* entry = native->entry();
        if (auto validated = metalValidateIndirectDispatchRecording(entry); !validated)
        {
            return validated;
        }
        if (layout.maxRecords == 0)
        {
            return {};
        }
        if (auto kept = metalRetainIndirectRecords(*native, *entry, records); !kept)
        {
            return kept;
        }
        if (auto prepared = native->ensureComputeEncoder(); !prepared)
        {
            return prepared;
        }
        auto span = metalClampComputeRecords(device, *entry, *pipelines, layout, records);
        if (!span)
        {
            return std::unexpected(std::move(span.error()));
        }
        const auto& pipeline = *entry->boundComputePipeline;
        if (auto applied = metalApplyComputeDispatchState(device, *entry, pipeline); !applied)
        {
            return applied;
        }
        const auto& workgroup = pipeline.workgroup;
        for (std::uint32_t record = 0; record < layout.maxRecords; ++record)
        {
            [entry->computeEncoder
                dispatchThreadgroupsWithIndirectBuffer:span->firstAddress + span->stride * record
                                 threadsPerThreadgroup:MTLSizeMake(workgroup.x, workgroup.y, workgroup.z)];
        }
        return {};
    }

    namespace
    {

        [[nodiscard]] Status metalValidateRenderingShape(const RenderingDesc& desc)
        {
            if (desc.colorAttachments.size() > maxColorAttachments ||
                (desc.colorAttachments.empty() && !desc.depthStencilAttachment.has_value()))
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState,
                                        "Rendering requires a graphics command buffer, a non-nested scope, and a valid "
                                        "attachment, timestamp, and multiview configuration."));
            }
            if (desc.viewMask != 0)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::unsupported, "The device does not support multiview rendering."));
            }
            return {};
        }

        struct MetalRenderingPassResources final
        {
            std::shared_ptr<MetalRenderingScope> scope = std::make_shared<MetalRenderingScope>();
            std::vector<std::shared_ptr<void>> retained{};
            std::vector<MetalWaitedDrawable> waitedDrawables{};
        };

        [[nodiscard]] Status metalAcceptAttachmentExtent(MetalRenderingScope& scope,
                                                         const DeviceCapabilities& capabilities,
                                                         id<MTLTexture> texture,
                                                         std::uint32_t layerCount)
        {
            if (texture == nil)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "A rendering attachment references an invalid or stale texture view."));
            }
            const auto width = static_cast<std::uint32_t>(texture.width);
            const auto height = static_cast<std::uint32_t>(texture.height);
            const auto sampleCount = static_cast<std::uint32_t>(texture.sampleCount);
            if (layerCount == 0 || layerCount > capabilities.limits.maxFramebufferLayers)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "Rendering attachments require matching extents and a supported layer count."));
            }
            if (scope.width == 0)
            {
                scope.width = width;
                scope.height = height;
                scope.sampleCount = sampleCount;
                scope.layerCount = layerCount;
                return {};
            }
            if (scope.width != width || scope.height != height || scope.sampleCount != sampleCount ||
                scope.layerCount != layerCount)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "Rendering attachments require matching extents and a supported layer count."));
            }
            return {};
        }

        [[nodiscard]] Status metalValidateColorAttachmentOperations(const ColorAttachment& attachment)
        {
            if (!metalIsValidLoadOperation(attachment.loadOperation) ||
                !metalIsValidStoreOperation(attachment.storeOperation))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "Color attachments require valid load and store operations."));
            }
            if (attachment.loadOperation == LoadOperation::clear &&
                (!std::isfinite(attachment.clearValue.red) || !std::isfinite(attachment.clearValue.green) ||
                 !std::isfinite(attachment.clearValue.blue) || !std::isfinite(attachment.clearValue.alpha)))
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, "Color clear values must be finite."));
            }
            return {};
        }

        [[nodiscard]] Status metalRecordWaitedDrawable(const MetalDeviceNative& deviceNative,
                                                       const MetalTextureRecord& texture,
                                                       std::vector<MetalWaitedDrawable>& passWaitedDrawables)
        {
            const bool alreadyRecorded = std::ranges::any_of(passWaitedDrawables, [&](const MetalWaitedDrawable& item) {
                return item.drawable == texture.drawable;
            });
            if (alreadyRecorded)
            {
                return {};
            }
            try
            {
                passWaitedDrawables.push_back(metalSwapchainImageReference(deviceNative, texture));
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(metalSelectionError(ErrorCode::outOfMemory,
                                                           "Failed to reserve Metal swapchain drawable tracking."));
            }
            return {};
        }

        [[nodiscard]] Status metalResolveColorAttachment(MetalDeviceNative& deviceNative,
                                                         const DeviceCapabilities& capabilities,
                                                         const ColorAttachment& attachment,
                                                         MetalRenderingPassResources& pass)
        {
            if (auto validated = metalValidateColorAttachmentOperations(attachment); !validated)
            {
                return validated;
            }
            auto* view = deviceNative.views.find(attachment.view);
            if (view == nullptr || view->view == nil || view->desc.usage != TextureViewUsage::renderTarget)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "Color attachments require a live render-target texture view."));
            }
            const auto layers = view->view.arrayLength == 0 ? 1U : static_cast<std::uint32_t>(view->view.arrayLength);
            if (auto accepted = metalAcceptAttachmentExtent(*pass.scope, capabilities, view->view, layers); !accepted)
            {
                return accepted;
            }
            MetalRenderingAttachment native{};
            native.texture = view->view;
            native.format = view->desc.format;
            if (auto* texture = deviceNative.textures.find(view->texture); texture != nullptr)
            {
                if (native.format == Format::unknown)
                {
                    native.format = texture->desc.format;
                }
                if (texture->drawable != nil)
                {
                    if (auto recorded = metalRecordWaitedDrawable(deviceNative, *texture, pass.waitedDrawables);
                        !recorded)
                    {
                        return recorded;
                    }
                }
            }
            native.loadOperation = attachment.loadOperation;
            native.storeOperation = attachment.storeOperation;
            native.clearColor = attachment.clearValue;
            pass.scope->colors.push_back(std::move(native));
            pass.retained.push_back(view->owner);
            return {};
        }

        [[nodiscard]] Status metalResolveDepthStencilAttachment(MetalDeviceNative& deviceNative,
                                                                const DeviceCapabilities& capabilities,
                                                                const DepthStencilAttachment& attachment,
                                                                MetalRenderingPassResources& pass)
        {
            if (!metalIsValidLoadOperation(attachment.depthLoadOperation) ||
                !metalIsValidStoreOperation(attachment.depthStoreOperation) ||
                !metalIsValidLoadOperation(attachment.stencilLoadOperation) ||
                !metalIsValidStoreOperation(attachment.stencilStoreOperation))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "Depth-stencil attachments require valid load and store operations."));
            }
            const bool clearsDepth = attachment.depthLoadOperation == LoadOperation::clear;
            if (clearsDepth && !metalIsPortableDepthValue(attachment.depthClearValue))
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "Depth clear values must be finite and inside the portable [0, 1] range."));
            }
            if (attachment.stencilClearValue > std::numeric_limits<std::uint8_t>::max())
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "Stencil clear values must fit in the portable 8-bit stencil range."));
            }
            auto* view = deviceNative.views.find(attachment.view);
            if (view == nullptr || view->view == nil || view->desc.usage != TextureViewUsage::depthStencil)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "Depth-stencil attachments require a live depth-stencil texture view."));
            }
            const auto layers = view->view.arrayLength == 0 ? 1U : static_cast<std::uint32_t>(view->view.arrayLength);
            if (auto accepted = metalAcceptAttachmentExtent(*pass.scope, capabilities, view->view, layers); !accepted)
            {
                return accepted;
            }
            Format format = view->desc.format;
            if (auto* texture = deviceNative.textures.find(view->texture);
                texture != nullptr && format == Format::unknown)
            {
                format = texture->desc.format;
            }
            auto& scope = *pass.scope;
            scope.depth.texture = view->view;
            scope.depth.format = format;
            scope.depth.loadOperation = attachment.depthLoadOperation;
            scope.depth.storeOperation = attachment.depthStoreOperation;
            scope.depth.depthClear = clearsDepth ? attachment.depthClearValue : 1.0F;
            scope.depth.stencilClear = attachment.stencilClearValue;
            scope.depth.stencilLoadOperation = attachment.stencilLoadOperation;
            scope.depth.stencilStoreOperation = attachment.stencilStoreOperation;
            scope.hasDepth = true;
            scope.hasStencil = metalFormatHasStencil(format);
            pass.retained.push_back(view->owner);
            return {};
        }

        [[nodiscard]] Status metalResolveRenderingAttachments(MetalDeviceNative& deviceNative,
                                                              const DeviceCapabilities& capabilities,
                                                              const RenderingDesc& desc,
                                                              MetalRenderingPassResources& pass)
        {
            pass.scope->colors.reserve(desc.colorAttachments.size());
            for (const auto& attachment : desc.colorAttachments)
            {
                if (auto resolved = metalResolveColorAttachment(deviceNative, capabilities, attachment, pass);
                    !resolved)
                {
                    return resolved;
                }
            }
            if (!desc.depthStencilAttachment.has_value())
            {
                return {};
            }
            return metalResolveDepthStencilAttachment(deviceNative, capabilities, *desc.depthStencilAttachment, pass);
        }

        [[nodiscard]] bool metalHasInvalidRenderPassTimestampTargets(const PassTimestampWrites& writes,
                                                                     const MetalQueryPoolRecord* timestampPool)
        {
            const bool writesBeginning = writes.beginningOfPassWriteIndex != timestampWriteUnused;
            const bool writesEnd = writes.endOfPassWriteIndex != timestampWriteUnused;
            return timestampPool == nullptr || timestampPool->kind != MetalQueryPoolKind::timestamp ||
                   timestampPool->queueType != QueueType::graphics || (!writesBeginning && !writesEnd) ||
                   (writesBeginning && writes.beginningOfPassWriteIndex >= timestampPool->queryCount) ||
                   (writesEnd && writes.endOfPassWriteIndex >= timestampPool->queryCount) ||
                   (writesBeginning && writesEnd && writes.beginningOfPassWriteIndex == writes.endOfPassWriteIndex);
        }

        [[nodiscard]] bool metalRenderPassTimestampsConflict(const MetalCommandBuffer& native,
                                                             const PassTimestampWrites& writes,
                                                             const MetalQueryPoolRecord& timestampPool)
        {
            const bool writesBeginning = writes.beginningOfPassWriteIndex != timestampWriteUnused;
            const bool writesEnd = writes.endOfPassWriteIndex != timestampWriteUnused;
            const auto& queryState = native.queryState();
            return queryState.activePassTimestampWrites.has_value() || queryState.activeGpuProfileScopeDepth != 0 ||
                   (writesBeginning &&
                    metalIsTimestampSlotPending(
                        native, writes.queryPool, timestampPool, writes.beginningOfPassWriteIndex)) ||
                   (writesEnd &&
                    metalIsTimestampSlotPending(native, writes.queryPool, timestampPool, writes.endOfPassWriteIndex));
        }

        [[nodiscard]] Status metalReserveRenderPassTimestamps(
            MetalCommandBuffer& native,
            const PassTimestampWrites& writes,
            const std::shared_ptr<MetalQueryPoolRecord>& timestampPool)
        {
            if (metalHasInvalidRenderPassTimestampTargets(writes, timestampPool.get()))
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "Render-pass timestamps require a graphics timestamp pool "
                                                           "and one or two distinct in-bounds indices."));
            }
            if (metalRenderPassTimestampsConflict(native, writes, *timestampPool))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidState,
                    "Render-pass timestamps cannot overlap a GPU profile scope or reuse unresolved query indices."));
            }
            const std::array queryIndices{writes.beginningOfPassWriteIndex, writes.endOfPassWriteIndex};
            auto reserved = metalReserveStagedQuerySlots(native.stagedQuerySlots(), writes.queryPool, queryIndices);
            if (!reserved)
            {
                return reserved;
            }
            return retainMetalApplicationResource(native, timestampPool);
        }

        [[nodiscard]] Status metalPrepareRenderingRecording(MetalCommandBuffer& native,
                                                            const std::vector<std::shared_ptr<void>>& retained)
        {
            auto ready = metalRecordingReady(native);
            if (!ready)
            {
                return ready;
            }
            for (auto& resource : retained)
            {
                auto kept = retainMetalApplicationResource(native, resource);
                if (!kept)
                {
                    return kept;
                }
            }
            return {};
        }

        [[nodiscard]] Status metalEncodeRenderPassTimestamps(MetalCommandBuffer& native,
                                                             MetalCommandEntry& entry,
                                                             const PassTimestampWrites& writes,
                                                             const std::shared_ptr<MetalQueryPoolRecord>& timestampPool)
        {
            auto& queryState = native.queryState();
            if (writes.beginningOfPassWriteIndex != timestampWriteUnused)
            {
                auto encoded =
                    metalEncodeTimestamp(entry, timestampPool, writes.beginningOfPassWriteIndex, MTLRenderStageVertex);
                if (!encoded)
                {
                    return encoded;
                }
                metalStageTimestampWrite(
                    native.stagedQuerySlots(), writes.queryPool, timestampPool, writes.beginningOfPassWriteIndex);
            }
            if (writes.endOfPassWriteIndex != timestampWriteUnused)
            {
                metalStageTimestampWrite(
                    native.stagedQuerySlots(), writes.queryPool, timestampPool, writes.endOfPassWriteIndex);
                queryState.activePassTimestampWrites = detail::ActivePassTimestampWrites{
                    .queryPool = writes.queryPool,
                    .endOfPassWriteIndex = writes.endOfPassWriteIndex,
                    .owner = timestampPool,
                };
            }
            return {};
        }

        [[nodiscard]] Status metalOpenRenderingScope(MetalCommandBuffer& native,
                                                     const RenderingDesc& desc,
                                                     const std::shared_ptr<MetalQueryPoolRecord>& timestampPool,
                                                     MetalRenderingPassResources& pass)
        {
            const std::scoped_lock lock(native.owner()->mutex);
            auto& state = native.graphicsState();
            if (state.renderingOwner != nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState,
                                        "Rendering requires a graphics command buffer, a non-nested scope, and a valid "
                                        "attachment, timestamp, and multiview configuration."));
            }
            if (auto ready = metalRecordingReady(native); !ready)
            {
                return ready;
            }
            auto* entry = native.entry();
            if (auto reserved = metalReserveWaitedDrawableTracking(entry, pass.waitedDrawables); !reserved)
            {
                return reserved;
            }
            auto started = metalBeginRenderEncoder(*entry, *pass.scope, false, 0, {}, false, 1.0F, false, 0);
            if (!started)
            {
                return started;
            }
            if (desc.timestampWrites.has_value() && timestampPool != nullptr)
            {
                if (auto encoded =
                        metalEncodeRenderPassTimestamps(native, *entry, *desc.timestampWrites, timestampPool);
                    !encoded)
                {
                    return encoded;
                }
            }
            metalMergeWaitedDrawables(entry, pass.waitedDrawables);
            state = {};
            state.renderingOwner = std::move(pass.scope);
            return {};
        }

        [[nodiscard]] bool metalTracksWaitedDrawable(const std::vector<MetalWaitedDrawable>& tracked,
                                                     const MetalWaitedDrawable& reference) noexcept
        {
            return std::ranges::any_of(tracked, [&](const MetalWaitedDrawable& item) {
                return item.drawable == reference.drawable && item.acquireId == reference.acquireId;
            });
        }

    } // namespace

    MetalWaitedDrawable metalSwapchainImageReference(const MetalDeviceNative& native, const MetalTextureRecord& texture)
    {
        if (texture.drawable == nil)
        {
            return {};
        }
        std::shared_ptr<MetalSwapchainRecord> swapchainRecord;
        if (const auto* swapchain = native.swapchains.find(texture.swapchain);
            swapchain != nullptr && *swapchain != nullptr)
        {
            swapchainRecord = *swapchain;
        }
        return MetalWaitedDrawable{
            .drawable = texture.drawable,
            .swapchain = std::move(swapchainRecord),
            .acquireId = texture.swapchainAcquireId,
        };
    }

    Status metalReserveWaitedDrawableTracking(MetalCommandEntry* entry, std::span<const MetalWaitedDrawable> references)
    {
        if (entry == nullptr)
        {
            return {};
        }
        const auto referenceCount = static_cast<std::size_t>(std::ranges::count_if(
            references, [](const MetalWaitedDrawable& reference) { return reference.drawable != nil; }));
        if (referenceCount == 0)
        {
            return {};
        }
        try
        {
            entry->waitedDrawables.reserve(entry->waitedDrawables.size() + referenceCount);
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::outOfMemory, "Failed to reserve Metal swapchain drawable tracking."));
        }
        return {};
    }

    void metalMergeWaitedDrawables(MetalCommandEntry* entry, std::span<const MetalWaitedDrawable> references) noexcept
    {
        if (entry == nullptr)
        {
            return;
        }
        for (const auto& reference : references)
        {
            if (reference.drawable != nil && !metalTracksWaitedDrawable(entry->waitedDrawables, reference) &&
                entry->waitedDrawables.size() < entry->waitedDrawables.capacity())
            {
                entry->waitedDrawables.push_back(reference);
            }
        }
    }

    Status MetalDevice::beginRendering(detail::BackendCommandBuffer& commandBuffer, const RenderingDesc& desc)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (auto validated = metalValidateRenderingShape(desc); !validated)
        {
            return validated;
        }

        MetalRenderingPassResources pass{};
        std::uint64_t deviceToken = 0;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal beginRendering requires an active device."));
            }
            deviceToken = native_->deviceToken;
            if (auto resolved = metalResolveRenderingAttachments(*native_, capabilities_, desc, pass); !resolved)
            {
                return resolved;
            }
        }

        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        if (native->queueType() != QueueType::graphics)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState,
                                    "Rendering requires a graphics command buffer, a non-nested scope, and a valid "
                                    "attachment, timestamp, and multiview configuration."));
        }
        std::shared_ptr<MetalQueryPoolRecord> timestampPool;
        if (desc.timestampWrites.has_value())
        {
            const auto& writes = *desc.timestampWrites;
            if (!graphicsQueue_.timestampQueries)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::unsupported, "The graphics queue does not support render-pass timestamp writes."));
            }
            {
                std::scoped_lock const lock(mutex_);
                auto* found = native_ == nullptr ? nullptr : native_->queryPools.find(writes.queryPool);
                timestampPool = found == nullptr ? nullptr : *found;
            }
            if (auto reserved = metalReserveRenderPassTimestamps(*native, writes, timestampPool); !reserved)
            {
                return reserved;
            }
        }
        if (auto prepared = metalPrepareRenderingRecording(*native, pass.retained); !prepared)
        {
            return prepared;
        }
        return metalOpenRenderingScope(*native, desc, timestampPool, pass);
    }

    Status MetalDevice::endRendering(detail::BackendCommandBuffer& commandBuffer)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        std::uint64_t deviceToken = 0;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal endRendering requires an active device."));
            }
            deviceToken = native_->deviceToken;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        const std::scoped_lock lock(native->owner()->mutex);
        auto& state = native->graphicsState();
        if (state.renderingOwner == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidState, "No rendering scope is active."));
        }
        auto* entry = native->entry();
        if (entry != nullptr)
        {
            auto& queryState = native->queryState();
            if (queryState.activePassTimestampWrites.has_value() &&
                queryState.activePassTimestampWrites->endOfPassWriteIndex != timestampWriteUnused)
            {
                const auto pool =
                    std::static_pointer_cast<MetalQueryPoolRecord>(queryState.activePassTimestampWrites->owner);
                auto encoded = metalEncodeTimestamp(
                    *entry, pool, queryState.activePassTimestampWrites->endOfPassWriteIndex, MTLRenderStageFragment);
                if (!encoded)
                {
                    return encoded;
                }
            }
            queryState.activePassTimestampWrites.reset();
            endMetalEncoders(*entry);
            entry->boundGraphicsVersions.clear();
            entry->boundGraphicsPipeline.reset();
            entry->vertexBuffers = {};
            entry->indexBuffer = {};
            entry->indexFormat = IndexFormat::uint16;
            entry->viewports = {};
            entry->scissors = {};
            entry->scissorsEmpty = false;
            entry->blendConstants = {};
            entry->stencilReference = 0;
        }
        state = {};
        return {};
    }

    Status MetalDevice::clearColorAttachment(detail::BackendCommandBuffer& commandBuffer,
                                             std::uint32_t attachmentIndex,
                                             const ClearColor& clearValue,
                                             std::span<const ScissorRect> rectangles)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (!std::isfinite(clearValue.red) || !std::isfinite(clearValue.green) || !std::isfinite(clearValue.blue) ||
            !std::isfinite(clearValue.alpha))
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "Color clear values must be finite."));
        }
        std::uint64_t deviceToken = 0;
        id<MTLDevice> device = nil;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                           "Metal clearColorAttachment requires an active device."));
            }
            deviceToken = native_->deviceToken;
            device = native_->device;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        const std::scoped_lock lock(native->owner()->mutex);
        auto& state = native->graphicsState();
        if (state.renderingOwner == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidState, "No rendering scope is active."));
        }
        auto* const scope = static_cast<MetalRenderingScope*>(state.renderingOwner.get());
        if (attachmentIndex >= scope->colors.size())
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument, "The color attachment clear index is outside the active attachment list."));
        }
        if (auto valid = metalValidateClearRectangles(*scope, rectangles); !valid)
        {
            return valid;
        }
        auto* entry = native->entry();
        auto restarted =
            metalBeginRenderEncoder(*entry, *scope, true, attachmentIndex, clearValue, false, 1.0F, false, 0);
        if (!restarted)
        {
            return restarted;
        }
        if (entry->boundGraphicsPipeline != nullptr)
        {
            metalApplyGraphicsRasterState(*entry, *entry->boundGraphicsPipeline);
            [entry->renderEncoder setRenderPipelineState:entry->boundGraphicsPipeline->graphics];
            entry->activeGraphicsPipeline = entry->boundGraphicsPipeline.get();
        }
        metalApplyGraphicsDynamicState(*entry, state);
        static_cast<void>(device);
        return {};
    }

    Status MetalDevice::clearDepthStencilAttachment(detail::BackendCommandBuffer& commandBuffer,
                                                    DepthStencilClearAspect aspects,
                                                    float depth,
                                                    std::uint32_t stencil,
                                                    std::span<const ScissorRect> rectangles)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        constexpr auto validAspects = DepthStencilClearAspect::depth | DepthStencilClearAspect::stencil;
        if (aspects == DepthStencilClearAspect::none ||
            (static_cast<std::uint8_t>(aspects) & ~static_cast<std::uint8_t>(validAspects)) != 0)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "Depth-stencil clears require at least one valid aspect."));
        }
        const bool clearDepth = hasAny(aspects, DepthStencilClearAspect::depth);
        const bool clearStencil = hasAny(aspects, DepthStencilClearAspect::stencil);
        if (clearDepth && !metalIsPortableDepthValue(depth))
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument, "Depth clear values must be finite and inside the portable [0, 1] range."));
        }
        if (clearStencil && stencil > std::numeric_limits<std::uint8_t>::max())
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument, "Stencil clear values must fit in the portable 8-bit stencil range."));
        }
        std::uint64_t deviceToken = 0;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidState, "Metal clearDepthStencilAttachment requires an active device."));
            }
            deviceToken = native_->deviceToken;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        const std::scoped_lock lock(native->owner()->mutex);
        auto& state = native->graphicsState();
        if (state.renderingOwner == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidState, "No rendering scope is active."));
        }
        auto* const scope = static_cast<MetalRenderingScope*>(state.renderingOwner.get());
        if (!scope->hasDepth)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument, "Depth-stencil clears require an active depth-stencil attachment."));
        }
        if (clearStencil && !scope->hasStencil)
        {
            return std::unexpected(metalSelectionError(ErrorCode::unsupported,
                                                       "Stencil clears require a packed depth-stencil attachment."));
        }
        if (auto valid = metalValidateClearRectangles(*scope, rectangles); !valid)
        {
            return valid;
        }
        auto* entry = native->entry();
        auto restarted =
            metalBeginRenderEncoder(*entry, *scope, false, 0, {}, clearDepth, depth, clearStencil, stencil);
        if (!restarted)
        {
            return restarted;
        }
        if (entry->boundGraphicsPipeline != nullptr)
        {
            metalApplyGraphicsRasterState(*entry, *entry->boundGraphicsPipeline);
            [entry->renderEncoder setRenderPipelineState:entry->boundGraphicsPipeline->graphics];
            entry->activeGraphicsPipeline = entry->boundGraphicsPipeline.get();
        }
        metalApplyGraphicsDynamicState(*entry, state);
        return {};
    }

    Status MetalDevice::bindGraphicsPipeline(detail::BackendCommandBuffer& commandBuffer, PipelineHandle pipeline)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        std::uint64_t deviceToken = 0;
        std::shared_ptr<MetalPipelineRecord> record;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                           "Metal bindGraphicsPipeline requires an active device."));
            }
            deviceToken = native_->deviceToken;
            auto* found = native_->pipelines.find(pipeline);
            if (found == nullptr || *found == nullptr || (*found)->kind != MetalPipelineKind::graphics ||
                (*found)->graphics == nil || !(*found)->graphicsState.has_value())
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "The pipeline handle is invalid, stale, or not a graphics pipeline."));
            }
            record = *found;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        auto kept = retainMetalApplicationResource(*native, std::static_pointer_cast<void>(record));
        if (!kept)
        {
            return kept;
        }
        const std::scoped_lock lock(native->owner()->mutex);
        auto& state = native->graphicsState();
        if (state.renderingOwner == nullptr)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState, "A graphics pipeline must be bound inside an active rendering scope."));
        }
        auto* const scope = static_cast<MetalRenderingScope*>(state.renderingOwner.get());
        const auto& raster = *record->graphicsState;
        std::vector<Format> scopeFormats;
        scopeFormats.reserve(scope->colors.size());
        for (const auto& color : scope->colors)
        {
            scopeFormats.push_back(color.format);
        }
        if (raster.colorFormats != scopeFormats || raster.depthStencilFormat != scope->depth.format ||
            raster.sampleCount != scope->sampleCount || raster.viewMask != 0)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The graphics pipeline attachment formats, sample count, or "
                                                       "view mask do not match the active rendering scope."));
        }
        auto* entry = native->entry();
        if (entry == nullptr || entry->renderEncoder == nil)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState, "A graphics pipeline bind requires an active Metal render encoder."));
        }
        entry->boundGraphicsPipeline = record;
        entry->vertexTableDirty = true;
        entry->fragmentTableDirty = true;
        if (entry->activeGraphicsPipeline != record.get())
        {
            [entry->renderEncoder setRenderPipelineState:record->graphics];
            metalApplyGraphicsRasterState(*entry, *record);
            entry->activeGraphicsPipeline = record.get();
        }
        state.pipelineOwner = record;
        state.boundBindingSets.clear();
        state.boundBindingSets.reserve(record->setRequired.size());
        for (const auto required : record->setRequired)
        {
            state.boundBindingSets.push_back(required == 0);
        }
        state.boundVertexBuffers.assign(raster.vertexBufferCount, false);
        state.boundStreamOutputBuffers.clear();
        state.indexBufferBound = false;
        state.blendConstantsRequired = raster.usesBlendConstants;
        state.blendConstantsSet = false;
        state.depthBiasRequired = false;
        state.depthBiasSet = false;
        state.depthBoundsRequired = false;
        state.depthBoundsSet = false;
        state.stencilReferenceRequired = raster.stencilEnabled;
        state.stencilReferenceSet = false;
        state.stencilReferenceFacesSet = 0;
        return {};
    }

    Status MetalDevice::bindGraphicsBindingSet(detail::BackendCommandBuffer& commandBuffer,
                                               std::uint32_t setIndex,
                                               BindingSetHandle bindingSet,
                                               std::span<const std::uint32_t> dynamicOffsets)
    {
        const std::array sets{bindingSet};
        return bindGraphicsBindingSets(commandBuffer, setIndex, sets, dynamicOffsets);
    }

    Status MetalDevice::bindGraphicsBindingSets(detail::BackendCommandBuffer& commandBuffer,
                                                std::uint32_t firstSet,
                                                std::span<const BindingSetHandle> bindingSets,
                                                std::span<const std::uint32_t> dynamicOffsets)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (bindingSets.empty() || firstSet >= maxBindingSets || bindingSets.size() > maxBindingSets - firstSet)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument,
                                    "Graphics binding-set indices must stay within the portable four-set limit."));
        }
        std::uint64_t deviceToken = 0;
        std::vector<std::shared_ptr<MetalBindingSetVersion>> versions;
        versions.reserve(bindingSets.size());
        std::vector<std::shared_ptr<MetalBindingSetCommandRetention>> retentions;
        retentions.reserve(bindingSets.size());
        std::uint32_t requiredDynamic = 0;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                           "Metal bindGraphicsBindingSets requires an active device."));
            }
            deviceToken = native_->deviceToken;
            for (const auto handle : bindingSets)
            {
                auto* found = native_->bindingSets.find(handle);
                if (found == nullptr || (*found)->version == nullptr)
                {
                    return std::unexpected(
                        metalSelectionError(ErrorCode::invalidArgument, "The binding set handle is invalid or stale."));
                }
                requiredDynamic += (*found)->layout->dynamicBufferCount;
                versions.push_back((*found)->version);
                retentions.push_back(metalRetainBindingSetForCommand(**found));
            }
        }
        if (dynamicOffsets.size() != requiredDynamic)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument, "Dynamic offset count must match the bound Metal binding sets."));
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        if (auto adjusted = metalApplyDynamicOffsets(versions, dynamicOffsets); !adjusted)
        {
            return adjusted;
        }
        const std::scoped_lock lock(native->owner()->mutex);
        auto& state = native->graphicsState();
        if (state.renderingOwner == nullptr || state.pipelineOwner == nullptr)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState, "A graphics pipeline must be bound before binding graphics sets."));
        }
        auto* entry = native->entry();
        if (entry == nullptr || entry->renderEncoder == nil)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState, "Graphics binding sets require an active Metal render encoder."));
        }
        auto* const pipeline = static_cast<MetalPipelineRecord*>(state.pipelineOwner.get());
        if (auto valid = metalValidateBindingSetBatch(*pipeline,
                                                      firstSet,
                                                      versions,
                                                      "The graphics binding-set batch is empty or exceeds the pipeline "
                                                      "layout.",
                                                      "A graphics binding set does not match the pipeline layout.");
            !valid)
        {
            return valid;
        }
        for (auto& version : versions)
        {
            auto kept = retainMetalApplicationResource(*native, std::static_pointer_cast<void>(version));
            if (!kept)
            {
                return kept;
            }
        }
        if (entry->boundGraphicsVersions.size() < firstSet + versions.size())
        {
            entry->boundGraphicsVersions.resize(firstSet + versions.size());
        }
        if (auto counted = retainMetalBindingSetRetentions(*native, retentions); !counted)
        {
            return counted;
        }
        for (std::size_t index = 0; index < versions.size(); ++index)
        {
            auto& version = versions[index];
            entry->boundGraphicsVersions[firstSet + index] = version;
            metalRetainBindingSetResidency(*entry, *version);
            state.boundBindingSets[firstSet + index] = true;
        }
        entry->vertexTableDirty = true;
        entry->fragmentTableDirty = true;
        return {};
    }

    Status MetalDevice::setGraphicsPushConstants(detail::BackendCommandBuffer& commandBuffer,
                                                 ShaderStage visibility,
                                                 std::uint32_t offset,
                                                 std::span<const std::byte> data)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        const auto visibilityMask = static_cast<std::uint16_t>(visibility);
        if (visibilityMask == 0 || (visibilityMask & ~static_cast<std::uint16_t>(ShaderStage::allGraphics)) != 0 ||
            data.empty() || offset % sizeof(std::uint32_t) != 0 || data.size() % sizeof(std::uint32_t) != 0 ||
            data.size() > maxPushConstantBytes || offset > maxPushConstantBytes - data.size())
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "Graphics push constants require graphics visibility and a "
                                                       "non-empty, 4-byte aligned range within 128 bytes."));
        }
        std::uint64_t deviceToken = 0;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidState, "Metal setGraphicsPushConstants requires an active device."));
            }
            deviceToken = native_->deviceToken;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        const std::scoped_lock lock(native->owner()->mutex);
        auto* entry = native->entry();
        if (entry == nullptr || entry->boundGraphicsPipeline == nullptr)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState, "A graphics pipeline must be bound before setting push constants."));
        }
        const auto& ranges = entry->boundGraphicsPipeline->layout->pushConstantRanges;
        if (!metalPushConstantWriteCovered(ranges, visibility, offset, data.size()))
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument,
                "The push-constant write is not contained in graphics-visible pipeline layout ranges."));
        }
        const auto end = static_cast<std::uint32_t>(offset + data.size());
        std::memcpy(entry->graphicsPushConstants.data() + offset, data.data(), data.size());
        entry->graphicsPushConstantBytes = std::max(entry->graphicsPushConstantBytes, end);
        entry->graphicsPushConstantsDirty = true;
        entry->vertexTableDirty = true;
        entry->fragmentTableDirty = true;
        return {};
    }

    Status MetalDevice::setViewports(detail::BackendCommandBuffer& commandBuffer, std::span<const Viewport> viewports)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (viewports.empty() || viewports.size() > capabilities_.limits.maxViewports)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "Viewport count must be between one and the device limit."));
        }
        for (const auto& viewport : viewports)
        {
            if (!std::isfinite(viewport.x) || !std::isfinite(viewport.y) || !std::isfinite(viewport.width) ||
                !std::isfinite(viewport.height) || !std::isfinite(viewport.minDepth) ||
                !std::isfinite(viewport.maxDepth) || viewport.width <= 0.0F || viewport.height <= 0.0F ||
                viewport.minDepth < 0.0F || viewport.maxDepth > 1.0F || viewport.minDepth > viewport.maxDepth)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "Viewports require a finite positive extent and an ordered [0, 1] depth range."));
            }
        }
        std::uint64_t deviceToken = 0;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal setViewports requires an active device."));
            }
            deviceToken = native_->deviceToken;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        const std::scoped_lock lock(native->owner()->mutex);
        auto& state = native->graphicsState();
        if (state.renderingOwner == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Viewports require an active rendering scope."));
        }
        auto* entry = native->entry();
        state.viewportCount = static_cast<std::uint32_t>(viewports.size());
        for (std::uint32_t index = 0; index < state.viewportCount; ++index)
        {
            entry->viewports[index] = viewports[index];
        }
        metalApplyGraphicsDynamicState(*entry, state);
        return {};
    }

    Status MetalDevice::setScissors(detail::BackendCommandBuffer& commandBuffer, std::span<const ScissorRect> scissors)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (scissors.empty() || scissors.size() > capabilities_.limits.maxViewports)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "Scissor count must be between one and the device limit."));
        }
        std::uint64_t deviceToken = 0;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal setScissors requires an active device."));
            }
            deviceToken = native_->deviceToken;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        const std::scoped_lock lock(native->owner()->mutex);
        auto& state = native->graphicsState();
        if (state.renderingOwner == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Scissors require an active rendering scope."));
        }
        for (const auto& scissor : scissors)
        {
            if (scissor.width == 0 || scissor.height == 0 || scissor.x < 0 || scissor.y < 0)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "Scissor rectangles must have non-negative origin and positive extent."));
            }
        }
        auto* entry = native->entry();
        state.scissorCount = static_cast<std::uint32_t>(scissors.size());
        for (std::uint32_t index = 0; index < state.scissorCount; ++index)
        {
            entry->scissors[index] = scissors[index];
        }
        metalApplyGraphicsDynamicState(*entry, state);
        return {};
    }

    Status MetalDevice::setBlendConstants(detail::BackendCommandBuffer& commandBuffer, const BlendConstants& constants)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (!std::isfinite(constants.red) || !std::isfinite(constants.green) || !std::isfinite(constants.blue) ||
            !std::isfinite(constants.alpha))
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument, "Blend constants must be finite."));
        }
        std::uint64_t deviceToken = 0;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal setBlendConstants requires an active device."));
            }
            deviceToken = native_->deviceToken;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        const std::scoped_lock lock(native->owner()->mutex);
        auto& state = native->graphicsState();
        if (state.renderingOwner == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Blend constants require an active rendering scope."));
        }
        auto* entry = native->entry();
        entry->blendConstants = constants;
        state.blendConstantsSet = true;
        metalApplyGraphicsDynamicState(*entry, state);
        return {};
    }

    Status MetalDevice::setStencilReference(detail::BackendCommandBuffer& commandBuffer,
                                            StencilFace face,
                                            std::uint32_t reference)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if ((face != StencilFace::front && face != StencilFace::back && face != StencilFace::frontAndBack) ||
            reference > std::numeric_limits<std::uint8_t>::max())
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument, "A stencil reference requires a valid face selector and an 8-bit value."));
        }
        std::uint64_t deviceToken = 0;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                           "Metal setStencilReference requires an active device."));
            }
            deviceToken = native_->deviceToken;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        const std::scoped_lock lock(native->owner()->mutex);
        auto& state = native->graphicsState();
        if (state.renderingOwner == nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState, "Stencil references require an active rendering scope."));
        }
        if (face != StencilFace::frontAndBack)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::unsupported, "The device does not support independent front and back stencil references."));
        }
        auto* entry = native->entry();
        entry->stencilReference = reference;
        state.stencilReferenceSet = true;
        state.stencilReferenceFacesSet = 0x3;
        metalApplyGraphicsDynamicState(*entry, state);
        return {};
    }

    Status MetalDevice::bindVertexBuffers(detail::BackendCommandBuffer& commandBuffer,
                                          std::uint32_t firstSlot,
                                          std::span<const VertexBufferBinding> bindings)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (bindings.empty() || firstSlot >= maxVertexBuffers || bindings.size() > maxVertexBuffers - firstSlot)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument, "Vertex-buffer slots must stay within the portable vertex-buffer limit."));
        }
        std::uint64_t deviceToken = 0;
        std::vector<MetalBoundBuffer> resolved;
        resolved.resize(bindings.size());
        std::vector<std::shared_ptr<void>> owners;
        owners.reserve(bindings.size());
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal bindVertexBuffers requires an active device."));
            }
            deviceToken = native_->deviceToken;
            for (std::size_t index = 0; index < bindings.size(); ++index)
            {
                const auto& binding = bindings[index];
                auto* found = native_->buffers.find(binding.buffer);
                if (found == nullptr || found->buffer == nil || !hasAny(found->desc.usage, BufferUsage::vertex) ||
                    binding.offset >= found->desc.size)
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument, "Vertex buffers require vertex usage and an in-bounds offset."));
                }
                resolved[index].buffer = found->buffer;
                resolved[index].offset = binding.offset;
                resolved[index].size = found->desc.size - binding.offset;
                owners.push_back(found->owner);
            }
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        for (auto& owner : owners)
        {
            auto kept = retainMetalApplicationResource(*native, owner);
            if (!kept)
            {
                return kept;
            }
        }
        const std::scoped_lock lock(native->owner()->mutex);
        auto& state = native->graphicsState();
        if (state.renderingOwner == nullptr || state.pipelineOwner == nullptr ||
            firstSlot + bindings.size() > state.boundVertexBuffers.size())
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState,
                                    "Vertex buffers require a bound graphics pipeline and matching vertex slots."));
        }
        auto* const pipeline = static_cast<MetalPipelineRecord*>(state.pipelineOwner.get());
        if (pipeline->graphicsState.has_value())
        {
            if (auto aligned = detail::validateVertexBufferOffsetAlignment(
                    pipeline->graphicsState->vertexBufferAlignments, firstSlot, bindings);
                !aligned)
            {
                return aligned;
            }
        }
        auto* entry = native->entry();
        for (std::size_t index = 0; index < resolved.size(); ++index)
        {
            entry->vertexBuffers[firstSlot + index] = resolved[index];
            metalRetainResidency(*entry, resolved[index].buffer);
            state.boundVertexBuffers[firstSlot + index] = true;
        }
        entry->vertexTableDirty = true;
        return {};
    }

    Status MetalDevice::bindIndexBuffer(detail::BackendCommandBuffer& commandBuffer,
                                        BufferHandle buffer,
                                        IndexFormat format,
                                        std::uint64_t offset)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (format != IndexFormat::uint16 && format != IndexFormat::uint32)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "Index buffers require a valid index format."));
        }
        const auto alignment = format == IndexFormat::uint16 ? 2U : 4U;
        std::uint64_t deviceToken = 0;
        MetalBoundBuffer resolved{};
        std::shared_ptr<void> owner;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal bindIndexBuffer requires an active device."));
            }
            deviceToken = native_->deviceToken;
            auto* found = native_->buffers.find(buffer);
            if (found == nullptr || found->buffer == nil || !hasAny(found->desc.usage, BufferUsage::index) ||
                offset >= found->desc.size || found->desc.size - offset < alignment || offset % alignment != 0)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "Index buffers require a bound graphics pipeline, index usage, and an aligned in-bounds offset."));
            }
            resolved.buffer = found->buffer;
            resolved.offset = offset;
            resolved.size = found->desc.size - offset;
            owner = found->owner;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        auto kept = retainMetalApplicationResource(*native, owner);
        if (!kept)
        {
            return kept;
        }
        const std::scoped_lock lock(native->owner()->mutex);
        auto& state = native->graphicsState();
        if (state.renderingOwner == nullptr || state.pipelineOwner == nullptr)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument,
                "Index buffers require a bound graphics pipeline, index usage, and an aligned in-bounds offset."));
        }
        auto* const pipeline = static_cast<MetalPipelineRecord*>(state.pipelineOwner.get());
        if (pipeline->graphicsState.has_value())
        {
            if (auto restart = metalValidateRestartIndexFormat(*pipeline->graphicsState, format); !restart)
            {
                return restart;
            }
        }
        auto* entry = native->entry();
        entry->indexBuffer = resolved;
        entry->indexFormat = format;
        metalRetainResidency(*entry, resolved.buffer);
        state.indexBufferBound = true;
        return {};
    }

    Status MetalDevice::draw(detail::BackendCommandBuffer& commandBuffer,
                             std::uint32_t vertexCount,
                             std::uint32_t instanceCount,
                             std::uint32_t firstVertex,
                             std::uint32_t firstInstance)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        std::uint64_t deviceToken = 0;
        id<MTLDevice> device = nil;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal draw requires an active device."));
            }
            deviceToken = native_->deviceToken;
            device = native_->device;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        const std::scoped_lock lock(native->owner()->mutex);
        auto& state = native->graphicsState();
        auto* const pipeline = static_cast<MetalPipelineRecord*>(state.pipelineOwner.get());
        if (state.renderingOwner == nullptr || pipeline == nullptr || pipeline->mesh ||
            !pipeline->graphicsState.has_value() ||
            (pipeline->graphicsState->rasterizationEnabled && !metalHasCompleteViewportState(state)) ||
            (state.blendConstantsRequired && !state.blendConstantsSet) ||
            (state.stencilReferenceRequired && !state.stencilReferenceSet) || !metalAllBound(state.boundBindingSets) ||
            !metalAllBound(state.boundVertexBuffers) || vertexCount == 0 || instanceCount == 0)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState,
                "Draw requires complete rendering, pipeline, binding, vertex, viewport, and scissor state."));
        }
        auto* entry = native->entry();
        if (metalDrawScissoredOut(*entry, *pipeline))
        {
            return {};
        }
        auto applied = metalApplyGraphicsDrawState(device, *entry, *pipeline);
        if (!applied)
        {
            return applied;
        }
        const auto primitive = metalPrimitiveType(pipeline->graphicsState->topology);
        [entry->renderEncoder drawPrimitives:primitive
                                 vertexStart:firstVertex
                                 vertexCount:vertexCount
                               instanceCount:instanceCount
                                baseInstance:firstInstance];
        return {};
    }

    Status MetalDevice::drawIndexed(detail::BackendCommandBuffer& commandBuffer,
                                    std::uint32_t indexCount,
                                    std::uint32_t instanceCount,
                                    std::uint32_t firstIndex,
                                    std::int32_t baseVertex,
                                    std::uint32_t firstInstance)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        std::uint64_t deviceToken = 0;
        id<MTLDevice> device = nil;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal drawIndexed requires an active device."));
            }
            deviceToken = native_->deviceToken;
            device = native_->device;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        const std::scoped_lock lock(native->owner()->mutex);
        auto& state = native->graphicsState();
        auto* const pipeline = static_cast<MetalPipelineRecord*>(state.pipelineOwner.get());
        if (state.renderingOwner == nullptr || pipeline == nullptr || pipeline->mesh ||
            !pipeline->graphicsState.has_value() ||
            (pipeline->graphicsState->rasterizationEnabled && !metalHasCompleteViewportState(state)) ||
            (state.blendConstantsRequired && !state.blendConstantsSet) ||
            (state.stencilReferenceRequired && !state.stencilReferenceSet) || !metalAllBound(state.boundBindingSets) ||
            !metalAllBound(state.boundVertexBuffers) || !state.indexBufferBound || indexCount == 0 ||
            instanceCount == 0)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                       "Indexed draw requires complete rendering, pipeline, binding, "
                                                       "vertex, index, viewport, and scissor state."));
        }
        auto* entry = native->entry();
        const auto indexStride = entry->indexFormat == IndexFormat::uint16 ? 2U : 4U;
        const auto availableIndices = entry->indexBuffer.size / indexStride;
        if (entry->indexBuffer.buffer == nil || firstIndex > availableIndices ||
            indexCount > availableIndices - firstIndex)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "Indexed draw index range is outside the bound index buffer."));
        }
        if (auto restart = metalValidateIndexedRestart(*pipeline->graphicsState, entry->indexFormat); !restart)
        {
            return restart;
        }
        if (metalDrawScissoredOut(*entry, *pipeline))
        {
            return {};
        }
        auto applied = metalApplyGraphicsDrawState(device, *entry, *pipeline);
        if (!applied)
        {
            return applied;
        }
        const auto indexOffset = static_cast<std::uint64_t>(firstIndex) * indexStride;
        const auto alignedRemaining = (availableIndices - firstIndex) * indexStride;
        const auto primitive = metalPrimitiveType(pipeline->graphicsState->topology);
        [entry->renderEncoder
            drawIndexedPrimitives:primitive
                       indexCount:indexCount
                        indexType:entry->indexFormat == IndexFormat::uint16 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32
                      indexBuffer:entry->indexBuffer.buffer.gpuAddress + entry->indexBuffer.offset + indexOffset
                indexBufferLength:static_cast<NSUInteger>(alignedRemaining)
                    instanceCount:instanceCount
                       baseVertex:baseVertex
                     baseInstance:firstInstance];
        return {};
    }

    namespace
    {

        constexpr MetalIndirectRecordMessages kMetalDrawRecordMessages{
            .invalidLayout =
                "Indirect draw requires an argument buffer, aligned offsets, and a record-sized aligned stride.",
            .invalidArgumentBuffer =
                "The indirect draw argument buffer requires indirect usage and an in-bounds offset.",
            .recordsOutOfBounds = "The indirect draw records exceed the argument buffer.",
            .invalidCountBuffer =
                "The indirect draw count buffer requires indirect usage and an aligned in-bounds uint32 count.",
        };

        struct MetalIndirectDrawInputs final
        {
            std::uint64_t deviceToken = 0;
            __strong id<MTLDevice> device = nil;
            MetalResolvedIndirectRecords records{};
            MetalIndirectClampPipelines pipelines{};
        };

        [[nodiscard]] Result<MetalIndirectDrawInputs> metalAcquireIndirectDrawInputs(
            std::mutex& deviceMutex,
            const std::shared_ptr<MetalDeviceNative>& native,
            const MetalIndirectRecordLayout& layout,
            const MetalIndirectRecordMessages& messages,
            std::string_view inactiveMessage)
        {
            if (auto validated = metalValidateIndirectRecordLayout(layout, messages); !validated)
            {
                return std::unexpected(std::move(validated.error()));
            }
            auto pipelines = metalClampPipelinesFor(deviceMutex, native, layout);
            if (!pipelines)
            {
                return std::unexpected(std::move(pipelines.error()));
            }
            std::scoped_lock const lock(deviceMutex);
            if (native == nullptr)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidState, std::string{inactiveMessage}));
            }
            auto resolved = metalResolveIndirectRecords(*native, layout, messages);
            if (!resolved)
            {
                return std::unexpected(std::move(resolved.error()));
            }
            return MetalIndirectDrawInputs{
                .deviceToken = native->deviceToken,
                .device = native->device,
                .records = std::move(*resolved),
                .pipelines = std::move(*pipelines),
            };
        }

    } // namespace

    Status MetalDevice::drawIndirect(detail::BackendCommandBuffer& commandBuffer, const IndirectDrawDesc& desc)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        const auto layout = metalPrefixedRecordLayout(desc, desc.maxDrawCount, sizeof(DrawIndirectArguments));
        auto inputs = metalAcquireIndirectDrawInputs(
            mutex_, native_, layout, kMetalDrawRecordMessages, "Metal drawIndirect requires an active device.");
        if (!inputs)
        {
            return std::unexpected(std::move(inputs.error()));
        }
        auto* native = asMetalCommand(commandBuffer, inputs->deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        const std::scoped_lock lock(native->owner()->mutex);
        auto& state = native->graphicsState();
        auto* const pipeline = static_cast<MetalPipelineRecord*>(state.pipelineOwner.get());
        auto* entry = native->entry();
        if (entry == nullptr || state.renderingOwner == nullptr || pipeline == nullptr || pipeline->mesh ||
            !pipeline->graphicsState.has_value() ||
            (pipeline->graphicsState->rasterizationEnabled && !metalHasCompleteViewportState(state)) ||
            (state.blendConstantsRequired && !state.blendConstantsSet) ||
            (state.stencilReferenceRequired && !state.stencilReferenceSet) || !metalAllBound(state.boundBindingSets) ||
            !metalAllBound(state.boundVertexBuffers))
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState,
                "Draw requires complete rendering, pipeline, binding, vertex, viewport, and scissor state."));
        }
        if (auto prefixed = metalRequireDrawIndexPrefix(*pipeline, layout); !prefixed)
        {
            return prefixed;
        }
        if (layout.maxRecords == 0 || metalDrawScissoredOut(*entry, *pipeline))
        {
            return {};
        }
        if (auto kept = metalRetainIndirectRecords(*native, *entry, inputs->records); !kept)
        {
            return kept;
        }
        if (auto reserved = metalReserveDrawIndices(inputs->device, *entry, *pipeline, layout.maxRecords); !reserved)
        {
            return reserved;
        }
        auto span = metalClampRenderRecords(inputs->device, *entry, inputs->pipelines, layout, inputs->records, 1, 1);
        if (!span)
        {
            return std::unexpected(std::move(span.error()));
        }
        if (auto applied = metalApplyGraphicsDrawState(inputs->device, *entry, *pipeline); !applied)
        {
            return applied;
        }
        const MetalDrawIndexRestore restore(*entry);
        const auto primitive = metalPrimitiveType(pipeline->graphicsState->topology);
        for (std::uint32_t record = 0; record < layout.maxRecords; ++record)
        {
            if (pipeline->readsDrawIndex)
            {
                entry->drawIndex = record;
                if (auto applied = metalApplyGraphicsDrawState(inputs->device, *entry, *pipeline); !applied)
                {
                    return applied;
                }
            }
            [entry->renderEncoder drawPrimitives:primitive indirectBuffer:span->firstAddress + span->stride * record];
        }
        return {};
    }

    Status MetalDevice::drawIndexedIndirect(detail::BackendCommandBuffer& commandBuffer, const IndirectDrawDesc& desc)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        const auto layout = metalPrefixedRecordLayout(desc, desc.maxDrawCount, sizeof(DrawIndexedIndirectArguments));
        auto inputs = metalAcquireIndirectDrawInputs(
            mutex_, native_, layout, kMetalDrawRecordMessages, "Metal drawIndexedIndirect requires an active device.");
        if (!inputs)
        {
            return std::unexpected(std::move(inputs.error()));
        }
        auto* native = asMetalCommand(commandBuffer, inputs->deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        const std::scoped_lock lock(native->owner()->mutex);
        auto& state = native->graphicsState();
        auto* const pipeline = static_cast<MetalPipelineRecord*>(state.pipelineOwner.get());
        auto* entry = native->entry();
        if (entry == nullptr || state.renderingOwner == nullptr || pipeline == nullptr || pipeline->mesh ||
            !pipeline->graphicsState.has_value() ||
            (pipeline->graphicsState->rasterizationEnabled && !metalHasCompleteViewportState(state)) ||
            (state.blendConstantsRequired && !state.blendConstantsSet) ||
            (state.stencilReferenceRequired && !state.stencilReferenceSet) || !metalAllBound(state.boundBindingSets) ||
            !metalAllBound(state.boundVertexBuffers) || !state.indexBufferBound || entry->indexBuffer.buffer == nil)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                       "Indexed draw requires complete rendering, pipeline, binding, "
                                                       "vertex, index, viewport, and scissor state."));
        }
        if (auto restart = metalValidateIndexedRestart(*pipeline->graphicsState, entry->indexFormat); !restart)
        {
            return restart;
        }
        if (auto prefixed = metalRequireDrawIndexPrefix(*pipeline, layout); !prefixed)
        {
            return prefixed;
        }
        if (layout.maxRecords == 0 || metalDrawScissoredOut(*entry, *pipeline))
        {
            return {};
        }
        if (auto kept = metalRetainIndirectRecords(*native, *entry, inputs->records); !kept)
        {
            return kept;
        }
        if (auto reserved = metalReserveDrawIndices(inputs->device, *entry, *pipeline, layout.maxRecords); !reserved)
        {
            return reserved;
        }
        auto span = metalClampRenderRecords(inputs->device, *entry, inputs->pipelines, layout, inputs->records, 1, 1);
        if (!span)
        {
            return std::unexpected(std::move(span.error()));
        }
        if (auto applied = metalApplyGraphicsDrawState(inputs->device, *entry, *pipeline); !applied)
        {
            return applied;
        }
        const MetalDrawIndexRestore restore(*entry);
        const auto remaining = entry->indexBuffer.size;
        const auto indexStride = entry->indexFormat == IndexFormat::uint16 ? 2U : 4U;
        const auto alignedRemaining = remaining - (remaining % indexStride);
        const auto primitive = metalPrimitiveType(pipeline->graphicsState->topology);
        const auto indexType = entry->indexFormat == IndexFormat::uint16 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
        const auto indexAddress = entry->indexBuffer.buffer.gpuAddress + entry->indexBuffer.offset;
        for (std::uint32_t record = 0; record < layout.maxRecords; ++record)
        {
            if (pipeline->readsDrawIndex)
            {
                entry->drawIndex = record;
                if (auto applied = metalApplyGraphicsDrawState(inputs->device, *entry, *pipeline); !applied)
                {
                    return applied;
                }
            }
            [entry->renderEncoder drawIndexedPrimitives:primitive
                                              indexType:indexType
                                            indexBuffer:indexAddress
                                      indexBufferLength:static_cast<NSUInteger>(alignedRemaining)
                                         indirectBuffer:span->firstAddress + span->stride * record];
        }
        return {};
    }

    namespace
    {

        [[nodiscard]] bool metalMeshDispatchReady(const detail::GraphicsCommandState& state,
                                                  const MetalPipelineRecord* pipeline,
                                                  const MetalCommandEntry* entry)
        {
            return entry != nullptr && state.renderingOwner != nullptr && pipeline != nullptr && pipeline->mesh &&
                   pipeline->graphicsState.has_value() &&
                   (!pipeline->graphicsState->rasterizationEnabled || metalHasCompleteViewportState(state)) &&
                   (!state.blendConstantsRequired || state.blendConstantsSet) &&
                   (!state.stencilReferenceRequired || state.stencilReferenceSet) &&
                   metalAllBound(state.boundBindingSets);
        }

    } // namespace

    Status MetalDevice::dispatchMesh(detail::BackendCommandBuffer& commandBuffer,
                                     std::uint32_t groupCountX,
                                     std::uint32_t groupCountY,
                                     std::uint32_t groupCountZ)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (!capabilities_.meshShaders)
        {
            return std::unexpected(unsupportedOperation("dispatchMesh"));
        }
        std::uint64_t deviceToken = 0;
        id<MTLDevice> device = nil;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal dispatchMesh requires an active device."));
            }
            deviceToken = native_->deviceToken;
            device = native_->device;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        const std::scoped_lock lock(native->owner()->mutex);
        auto& state = native->graphicsState();
        auto* const pipeline = static_cast<MetalPipelineRecord*>(state.pipelineOwner.get());
        auto* entry = native->entry();
        if (!metalMeshDispatchReady(state, pipeline, entry))
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState,
                "Mesh dispatch requires complete rendering, mesh-pipeline, binding, viewport, and scissor state."));
        }
        const auto& limits = capabilities_.meshShader;
        const auto totalWorkgroups = static_cast<std::uint64_t>(groupCountX) * groupCountY * groupCountZ;
        if (groupCountX == 0 || groupCountY == 0 || groupCountZ == 0 || groupCountX > limits.maxWorkgroupCount[0] ||
            groupCountY > limits.maxWorkgroupCount[1] || groupCountZ > limits.maxWorkgroupCount[2] ||
            totalWorkgroups > limits.maxTotalWorkgroups)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument,
                "Mesh dispatch counts must fit every dimension and the total device workgroup limit."));
        }
        if (metalDrawScissoredOut(*entry, *pipeline))
        {
            return {};
        }
        auto applied = metalApplyGraphicsDrawState(device, *entry, *pipeline);
        if (!applied)
        {
            return applied;
        }
        [entry->renderEncoder drawMeshThreadgroups:MTLSizeMake(groupCountX, groupCountY, groupCountZ)
                       threadsPerObjectThreadgroup:pipeline->objectThreadgroupSize
                         threadsPerMeshThreadgroup:pipeline->meshThreadgroupSize];
        return {};
    }

    Status MetalDevice::dispatchMeshIndirect(detail::BackendCommandBuffer& commandBuffer,
                                             const IndirectMeshDispatchDesc& desc)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        if (!capabilities_.meshShaders)
        {
            return std::unexpected(unsupportedOperation("dispatchMeshIndirect"));
        }
        constexpr MetalIndirectRecordMessages messages{
            .invalidLayout =
                "Indirect mesh dispatch requires an argument buffer, aligned offsets, and an aligned 12-byte record.",
            .invalidArgumentBuffer =
                "The mesh-dispatch argument buffer requires indirect usage and an in-bounds offset.",
            .recordsOutOfBounds = "The indirect mesh-dispatch records exceed the argument buffer.",
            .invalidCountBuffer =
                "The indirect mesh count buffer requires indirect usage and an aligned in-bounds uint32 count.",
        };
        const auto layout = metalPrefixedRecordLayout(desc, desc.maxDispatchCount, sizeof(DispatchIndirectArguments));
        auto inputs = metalAcquireIndirectDrawInputs(
            mutex_, native_, layout, messages, "Metal dispatchMeshIndirect requires an active device.");
        if (!inputs)
        {
            return std::unexpected(std::move(inputs.error()));
        }
        auto* native = asMetalCommand(commandBuffer, inputs->deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        const std::scoped_lock lock(native->owner()->mutex);
        auto& state = native->graphicsState();
        auto* const pipeline = static_cast<MetalPipelineRecord*>(state.pipelineOwner.get());
        auto* entry = native->entry();
        if (!metalMeshDispatchReady(state, pipeline, entry))
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                       "Indirect mesh dispatch requires complete rendering, "
                                                       "mesh-pipeline, binding, viewport, and scissor state."));
        }
        if (auto prefixed = metalRequireDrawIndexPrefix(*pipeline, layout); !prefixed)
        {
            return prefixed;
        }
        if (layout.maxRecords == 0 || metalDrawScissoredOut(*entry, *pipeline))
        {
            return {};
        }
        if (auto kept = metalRetainIndirectRecords(*native, *entry, inputs->records); !kept)
        {
            return kept;
        }
        if (auto reserved = metalReserveDrawIndices(inputs->device, *entry, *pipeline, layout.maxRecords); !reserved)
        {
            return reserved;
        }
        auto span = metalClampRenderRecords(inputs->device,
                                            *entry,
                                            inputs->pipelines,
                                            layout,
                                            inputs->records,
                                            0,
                                            sizeof(DispatchIndirectArguments) / 4U);
        if (!span)
        {
            return std::unexpected(std::move(span.error()));
        }
        if (auto applied = metalApplyGraphicsDrawState(inputs->device, *entry, *pipeline); !applied)
        {
            return applied;
        }
        const MetalDrawIndexRestore restore(*entry);
        for (std::uint32_t record = 0; record < layout.maxRecords; ++record)
        {
            if (pipeline->readsDrawIndex)
            {
                entry->drawIndex = record;
                if (auto applied = metalApplyGraphicsDrawState(inputs->device, *entry, *pipeline); !applied)
                {
                    return applied;
                }
            }
            [entry->renderEncoder drawMeshThreadgroupsWithIndirectBuffer:span->firstAddress + span->stride * record
                                             threadsPerObjectThreadgroup:pipeline->objectThreadgroupSize
                                               threadsPerMeshThreadgroup:pipeline->meshThreadgroupSize];
        }
        return {};
    }

} // namespace rachis::metal
