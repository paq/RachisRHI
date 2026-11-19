#include "Metal/MetalInternal.h"

namespace rachis::metal
{
    namespace
    {

        struct MetalTextureRange final
        {
            std::uint32_t firstMip = 0;
            std::uint32_t mipCount = 0;
            std::uint32_t firstArray = 0;
            std::uint32_t arrayCount = 0;
            std::uint32_t aspect = 0;
        };

        [[nodiscard]] constexpr bool validMetalBarrierRange(std::uint64_t resourceSize,
                                                            std::uint64_t offset,
                                                            std::uint64_t size) noexcept
        {
            return offset < resourceSize && size != 0 && (size == wholeResource || size <= resourceSize - offset);
        }

        [[nodiscard]] constexpr bool validMetalPipelineStages(PipelineStage stages) noexcept
        {
            constexpr auto allowed =
                static_cast<std::uint32_t>(PipelineStage::all) | static_cast<std::uint32_t>(PipelineStage::draw) |
                static_cast<std::uint32_t>(PipelineStage::vertexInput) |
                static_cast<std::uint32_t>(PipelineStage::vertexShader) |
                static_cast<std::uint32_t>(PipelineStage::pixelShader) |
                static_cast<std::uint32_t>(PipelineStage::computeShader) |
                static_cast<std::uint32_t>(PipelineStage::copy) | static_cast<std::uint32_t>(PipelineStage::resolve) |
                static_cast<std::uint32_t>(PipelineStage::renderTarget) |
                static_cast<std::uint32_t>(PipelineStage::depthStencil) |
                static_cast<std::uint32_t>(PipelineStage::indirect) |
                static_cast<std::uint32_t>(PipelineStage::geometryShader) |
                static_cast<std::uint32_t>(PipelineStage::hullShader) |
                static_cast<std::uint32_t>(PipelineStage::domainShader) |
                static_cast<std::uint32_t>(PipelineStage::taskShader) |
                static_cast<std::uint32_t>(PipelineStage::meshShader) |
                static_cast<std::uint32_t>(PipelineStage::conditionalExecution) |
                static_cast<std::uint32_t>(PipelineStage::accelerationStructureBuild) |
                static_cast<std::uint32_t>(PipelineStage::rayTracingShader) |
                static_cast<std::uint32_t>(PipelineStage::accelerationStructureCopy) |
                static_cast<std::uint32_t>(PipelineStage::streamOutput);
            return (static_cast<std::uint32_t>(stages) & ~allowed) == 0;
        }

        [[nodiscard]] constexpr bool validMetalAccess(Access access) noexcept
        {
            constexpr auto allowed =
                static_cast<std::uint32_t>(Access::common) | static_cast<std::uint32_t>(Access::vertexBuffer) |
                static_cast<std::uint32_t>(Access::constantBuffer) | static_cast<std::uint32_t>(Access::indexBuffer) |
                static_cast<std::uint32_t>(Access::shaderResource) |
                static_cast<std::uint32_t>(Access::unorderedAccess) | static_cast<std::uint32_t>(Access::copySource) |
                static_cast<std::uint32_t>(Access::copyDestination) |
                static_cast<std::uint32_t>(Access::indirectArgument) |
                static_cast<std::uint32_t>(Access::renderTarget) |
                static_cast<std::uint32_t>(Access::depthStencilRead) |
                static_cast<std::uint32_t>(Access::depthStencilWrite) |
                static_cast<std::uint32_t>(Access::resolveSource) |
                static_cast<std::uint32_t>(Access::resolveDestination) | static_cast<std::uint32_t>(Access::predicate) |
                static_cast<std::uint32_t>(Access::shadingRateSource) |
                static_cast<std::uint32_t>(Access::accelerationStructureBuildInput) |
                static_cast<std::uint32_t>(Access::accelerationStructureScratch) |
                static_cast<std::uint32_t>(Access::streamOutputWrite);
            const auto raw = static_cast<std::uint32_t>(access);
            return (raw & ~allowed) == 0 && (!hasAny(access, Access::common) || access == Access::common);
        }

        [[nodiscard]] constexpr bool hasAtMostOneMetalWriteAccess(Access access) noexcept
        {
            constexpr auto writeAccess = Access::unorderedAccess | Access::copyDestination | Access::renderTarget |
                                         Access::depthStencilWrite | Access::resolveDestination |
                                         Access::accelerationStructureScratch | Access::streamOutputWrite;
            const auto writes = static_cast<std::uint32_t>(access & writeAccess);
            return writes == 0 || (writes & (writes - 1U)) == 0;
        }

        [[nodiscard]] constexpr bool hasMetalWriteAccess(Access access) noexcept
        {
            constexpr auto writeAccess = Access::unorderedAccess | Access::copyDestination | Access::renderTarget |
                                         Access::depthStencilWrite | Access::resolveDestination |
                                         Access::accelerationStructureScratch | Access::streamOutputWrite;
            return hasAny(access, writeAccess);
        }

        [[nodiscard]] constexpr bool metalAccessMatchesStages(PipelineStage stages, Access access) noexcept
        {
            const auto matches = [&](Access requiredAccess, PipelineStage compatibleStages) {
                return !hasAny(access, requiredAccess) || hasAny(stages, PipelineStage::all | compatibleStages);
            };
            return matches(Access::vertexBuffer, PipelineStage::draw | PipelineStage::vertexShader) &&
                   matches(Access::constantBuffer,
                           PipelineStage::draw | PipelineStage::vertexShader | PipelineStage::pixelShader |
                               PipelineStage::computeShader | PipelineStage::geometryShader |
                               PipelineStage::hullShader | PipelineStage::domainShader | PipelineStage::taskShader |
                               PipelineStage::meshShader | PipelineStage::rayTracingShader) &&
                   matches(Access::indexBuffer, PipelineStage::draw | PipelineStage::vertexInput) &&
                   matches(Access::shaderResource,
                           PipelineStage::draw | PipelineStage::vertexShader | PipelineStage::pixelShader |
                               PipelineStage::computeShader | PipelineStage::geometryShader |
                               PipelineStage::hullShader | PipelineStage::domainShader | PipelineStage::taskShader |
                               PipelineStage::meshShader | PipelineStage::rayTracingShader |
                               PipelineStage::accelerationStructureCopy) &&
                   matches(Access::unorderedAccess,
                           PipelineStage::draw | PipelineStage::vertexShader | PipelineStage::pixelShader |
                               PipelineStage::computeShader | PipelineStage::geometryShader |
                               PipelineStage::hullShader | PipelineStage::domainShader | PipelineStage::taskShader |
                               PipelineStage::meshShader | PipelineStage::rayTracingShader |
                               PipelineStage::accelerationStructureCopy) &&
                   matches(Access::copySource | Access::copyDestination, PipelineStage::copy) &&
                   matches(Access::indirectArgument, PipelineStage::indirect) &&
                   matches(Access::predicate, PipelineStage::conditionalExecution) &&
                   matches(Access::shadingRateSource, PipelineStage::draw | PipelineStage::pixelShader) &&
                   matches(Access::accelerationStructureBuildInput | Access::accelerationStructureScratch,
                           PipelineStage::accelerationStructureBuild) &&
                   matches(Access::streamOutputWrite, PipelineStage::streamOutput) &&
                   matches(Access::renderTarget, PipelineStage::draw | PipelineStage::renderTarget) &&
                   matches(Access::depthStencilRead | Access::depthStencilWrite,
                           PipelineStage::draw | PipelineStage::depthStencil) &&
                   matches(Access::resolveSource | Access::resolveDestination, PipelineStage::resolve);
        }

        [[nodiscard]] constexpr bool supportsMetalBarrierScopeOnQueue(PipelineStage stages,
                                                                      Access access,
                                                                      QueueType queueType) noexcept
        {
            if (stages == PipelineStage::none && access != Access::none)
            {
                return false;
            }
            if (queueType == QueueType::graphics)
            {
                return true;
            }
            const auto rawStages = static_cast<std::uint32_t>(stages);
            const auto rawAccess = static_cast<std::uint32_t>(access);
            const auto stagesFit = [&](PipelineStage allowed) {
                return (rawStages & ~static_cast<std::uint32_t>(allowed)) == 0;
            };
            const auto accessFits = [&](Access allowed) {
                return (rawAccess & ~static_cast<std::uint32_t>(allowed)) == 0;
            };
            if (queueType == QueueType::compute)
            {
                return stagesFit(PipelineStage::all | PipelineStage::computeShader | PipelineStage::copy |
                                 PipelineStage::indirect | PipelineStage::conditionalExecution |
                                 PipelineStage::accelerationStructureBuild | PipelineStage::rayTracingShader |
                                 PipelineStage::accelerationStructureCopy) &&
                       accessFits(Access::common | Access::constantBuffer | Access::shaderResource |
                                  Access::unorderedAccess | Access::copySource | Access::copyDestination |
                                  Access::indirectArgument | Access::predicate |
                                  Access::accelerationStructureBuildInput | Access::accelerationStructureScratch);
            }
            return stagesFit(PipelineStage::all | PipelineStage::copy) &&
                   accessFits(Access::common | Access::copySource | Access::copyDestination);
        }

        [[nodiscard]] constexpr bool validMetalBarrierScope(PipelineStage stages,
                                                            Access access,
                                                            QueueType queueType) noexcept
        {
            return validMetalPipelineStages(stages) && validMetalAccess(access) &&
                   metalAccessMatchesStages(stages, access) &&
                   supportsMetalBarrierScopeOnQueue(stages, access, queueType);
        }

        [[nodiscard]] constexpr bool validMetalResourceBarrierScope(PipelineStage stages,
                                                                    Access access,
                                                                    QueueType queueType) noexcept
        {
            return validMetalBarrierScope(stages, access, queueType) && hasAtMostOneMetalWriteAccess(access);
        }

        [[nodiscard]] constexpr bool supportsMetalBufferAccess(BufferUsage usage, Access access) noexcept
        {
            if (access == Access::none || access == Access::common)
            {
                return true;
            }
            Access allowed = Access::none;
            if (hasAny(usage, BufferUsage::copySource))
            {
                allowed = allowed | Access::copySource;
            }
            if (hasAny(usage, BufferUsage::copyDestination))
            {
                allowed = allowed | Access::copyDestination;
            }
            if (hasAny(usage, BufferUsage::vertex))
            {
                allowed = allowed | Access::vertexBuffer;
            }
            if (hasAny(usage, BufferUsage::index))
            {
                allowed = allowed | Access::indexBuffer;
            }
            if (hasAny(usage, BufferUsage::constant))
            {
                allowed = allowed | Access::constantBuffer;
            }
            if (hasAny(usage, BufferUsage::storage))
            {
                allowed = allowed | Access::shaderResource | Access::unorderedAccess;
            }
            if (hasAny(usage, BufferUsage::indirect))
            {
                allowed = allowed | Access::indirectArgument;
            }
            if (hasAny(usage, BufferUsage::predicate))
            {
                allowed = allowed | Access::predicate;
            }
            return (static_cast<std::uint32_t>(access) & ~static_cast<std::uint32_t>(allowed)) == 0;
        }

        [[nodiscard]] constexpr bool supportsMetalTextureAccess(TextureUsage usage, Access access) noexcept
        {
            if (access == Access::none || access == Access::common)
            {
                return true;
            }
            Access allowed = Access::none;
            if (hasAny(usage, TextureUsage::sampled))
            {
                allowed = allowed | Access::shaderResource;
            }
            if (hasAny(usage, TextureUsage::storage))
            {
                allowed = allowed | Access::unorderedAccess;
            }
            if (hasAny(usage, TextureUsage::copySource))
            {
                allowed = allowed | Access::copySource | Access::resolveSource;
            }
            if (hasAny(usage, TextureUsage::copyDestination))
            {
                allowed = allowed | Access::copyDestination | Access::resolveDestination;
            }
            if (hasAny(usage, TextureUsage::renderTarget))
            {
                allowed = allowed | Access::renderTarget | Access::resolveSource | Access::resolveDestination;
            }
            if (hasAny(usage, TextureUsage::depthStencil))
            {
                allowed = allowed | Access::depthStencilRead | Access::depthStencilWrite | Access::resolveSource |
                          Access::resolveDestination;
            }
            if (hasAny(usage, TextureUsage::shadingRateSource))
            {
                allowed = allowed | Access::shadingRateSource;
            }
            return (static_cast<std::uint32_t>(access) & ~static_cast<std::uint32_t>(allowed)) == 0;
        }

        [[nodiscard]] constexpr bool metalAccessMatchesTextureLayout(TextureLayout layout, Access access) noexcept
        {
            if (layout == TextureLayout::undefined || layout == TextureLayout::present)
            {
                return access == Access::none;
            }
            if (access == Access::none || access == Access::common)
            {
                return true;
            }
            Access allowed = Access::none;
            switch (layout)
            {
            case TextureLayout::common:
                allowed = Access::shaderResource | Access::copySource | Access::copyDestination;
                break;
            case TextureLayout::shaderResource:
                allowed = Access::shaderResource;
                break;
            case TextureLayout::unorderedAccess:
                allowed = Access::unorderedAccess;
                break;
            case TextureLayout::renderTarget:
                allowed = Access::renderTarget;
                break;
            case TextureLayout::depthStencilWrite:
                allowed = Access::depthStencilRead | Access::depthStencilWrite;
                break;
            case TextureLayout::depthStencilRead:
                allowed = Access::depthStencilRead;
                break;
            case TextureLayout::copySource:
                allowed = Access::copySource;
                break;
            case TextureLayout::copyDestination:
                allowed = Access::copyDestination;
                break;
            case TextureLayout::resolveSource:
                allowed = Access::resolveSource;
                break;
            case TextureLayout::resolveDestination:
                allowed = Access::resolveDestination;
                break;
            case TextureLayout::shadingRateSource:
                allowed = Access::shadingRateSource;
                break;
            case TextureLayout::undefined:
            case TextureLayout::present:
                break;
            }
            return (static_cast<std::uint32_t>(access) & ~static_cast<std::uint32_t>(allowed)) == 0;
        }

        [[nodiscard]] constexpr bool validMetalTextureLayout(TextureLayout layout) noexcept
        {
            switch (layout)
            {
            case TextureLayout::undefined:
            case TextureLayout::common:
            case TextureLayout::present:
            case TextureLayout::shaderResource:
            case TextureLayout::unorderedAccess:
            case TextureLayout::renderTarget:
            case TextureLayout::depthStencilWrite:
            case TextureLayout::depthStencilRead:
            case TextureLayout::copySource:
            case TextureLayout::copyDestination:
            case TextureLayout::resolveSource:
            case TextureLayout::resolveDestination:
            case TextureLayout::shadingRateSource:
                return true;
            }
            return false;
        }

        [[nodiscard]] constexpr bool supportsMetalLayout(TextureUsage usage, TextureLayout layout) noexcept
        {
            switch (layout)
            {
            case TextureLayout::undefined:
            case TextureLayout::common:
                return true;
            case TextureLayout::shaderResource:
                return hasAny(usage, TextureUsage::sampled);
            case TextureLayout::unorderedAccess:
                return hasAny(usage, TextureUsage::storage);
            case TextureLayout::renderTarget:
                return hasAny(usage, TextureUsage::renderTarget);
            case TextureLayout::depthStencilWrite:
            case TextureLayout::depthStencilRead:
                return hasAny(usage, TextureUsage::depthStencil);
            case TextureLayout::copySource:
                return hasAny(usage, TextureUsage::copySource);
            case TextureLayout::copyDestination:
                return hasAny(usage, TextureUsage::copyDestination);
            case TextureLayout::resolveSource:
                return hasAny(usage,
                              TextureUsage::copySource | TextureUsage::renderTarget | TextureUsage::depthStencil);
            case TextureLayout::resolveDestination:
                return hasAny(usage,
                              TextureUsage::copyDestination | TextureUsage::renderTarget | TextureUsage::depthStencil);
            case TextureLayout::shadingRateSource:
                return hasAny(usage, TextureUsage::shadingRateSource);
            case TextureLayout::present:
                return hasAny(usage, TextureUsage::renderTarget);
            }
            return false;
        }

        [[nodiscard]] constexpr bool supportsMetalTextureLayoutOnQueue(TextureLayout layout,
                                                                       QueueType queueType) noexcept
        {
            if (queueType == QueueType::graphics)
            {
                return true;
            }
            if (queueType == QueueType::compute)
            {
                return layout == TextureLayout::undefined || layout == TextureLayout::common ||
                       layout == TextureLayout::shaderResource || layout == TextureLayout::unorderedAccess ||
                       layout == TextureLayout::copySource || layout == TextureLayout::copyDestination;
            }
            return layout == TextureLayout::undefined || layout == TextureLayout::common ||
                   layout == TextureLayout::copySource || layout == TextureLayout::copyDestination;
        }

        [[nodiscard]] constexpr std::uint32_t metalTextureArraySize(const TextureDesc& desc) noexcept
        {
            return desc.dimension == TextureDimension::texture3D ? 1U : desc.depthOrArrayLayers;
        }

        [[nodiscard]] std::optional<std::uint32_t> metalAspectMask(Format format, TextureAspect aspect) noexcept
        {
            switch (aspect)
            {
            case TextureAspect::color:
                return formatIsDepth(format) ? std::nullopt : std::optional<std::uint32_t>{1U};
            case TextureAspect::depth:
                return formatIsDepth(format) ? std::optional<std::uint32_t>{2U} : std::nullopt;
            case TextureAspect::stencil:
                return format == Format::d24UnormS8Uint ? std::optional<std::uint32_t>{4U} : std::nullopt;
            }
            return std::nullopt;
        }

        [[nodiscard]] Result<MetalTextureRange> resolveMetalTextureRange(const TextureDesc& desc,
                                                                         const TextureSubresourceRange& range)
        {
            const auto aspect = metalAspectMask(desc.format, range.aspect);
            const auto arraySize = metalTextureArraySize(desc);
            const std::uint32_t mipCount =
                range.mipLevelCount == remainingSubresources && range.firstMipLevel < desc.mipLevels
                    ? desc.mipLevels - range.firstMipLevel
                    : range.mipLevelCount;
            const std::uint32_t arrayCount =
                range.arrayLayerCount == remainingSubresources && range.firstArrayLayer < arraySize
                    ? arraySize - range.firstArrayLayer
                    : range.arrayLayerCount;
            if (!aspect.has_value() || range.firstMipLevel >= desc.mipLevels || mipCount == 0 ||
                mipCount > desc.mipLevels - range.firstMipLevel || range.firstArrayLayer >= arraySize ||
                arrayCount == 0 || arrayCount > arraySize - range.firstArrayLayer)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "The texture barrier range has an invalid mip, layer, count, or aspect."));
            }
            return MetalTextureRange{range.firstMipLevel, mipCount, range.firstArrayLayer, arrayCount, *aspect};
        }

        [[nodiscard]] constexpr bool metalTextureRangesOverlap(const MetalTextureRange& left,
                                                               const MetalTextureRange& right) noexcept
        {
            const auto mipOverlaps =
                left.firstMip < right.firstMip + right.mipCount && right.firstMip < left.firstMip + left.mipCount;
            const auto arrayOverlaps = left.firstArray < right.firstArray + right.arrayCount &&
                                       right.firstArray < left.firstArray + left.arrayCount;
            return mipOverlaps && arrayOverlaps && (left.aspect & right.aspect) != 0;
        }

        [[nodiscard]] MTLStages metalStages(PipelineStage stages) noexcept
        {
            if (stages == PipelineStage::none)
            {
                return static_cast<MTLStages>(0);
            }
            if (hasAny(stages, PipelineStage::all))
            {
                return MTLStageAll;
            }
            MTLStages flags = static_cast<MTLStages>(0);
            if (hasAny(stages,
                       PipelineStage::draw | PipelineStage::vertexInput | PipelineStage::vertexShader |
                           PipelineStage::geometryShader | PipelineStage::hullShader | PipelineStage::domainShader |
                           PipelineStage::streamOutput))
            {
                flags |= MTLStageVertex;
            }
            if (hasAny(stages, PipelineStage::taskShader))
            {
                flags |= MTLStageObject;
            }
            if (hasAny(stages, PipelineStage::meshShader))
            {
                flags |= MTLStageMesh;
            }
            if (hasAny(stages, PipelineStage::pixelShader | PipelineStage::renderTarget | PipelineStage::depthStencil))
            {
                flags |= MTLStageFragment;
            }
            if (hasAny(stages,
                       PipelineStage::computeShader | PipelineStage::indirect | PipelineStage::conditionalExecution))
            {
                flags |= MTLStageDispatch;
            }
            if (hasAny(stages, PipelineStage::indirect))
            {
                flags |= MTLStageVertex | MTLStageObject | MTLStageMesh;
            }
            if (hasAny(stages, PipelineStage::copy))
            {
                flags |= MTLStageBlit | MTLStageDispatch;
            }
            if (hasAny(stages, PipelineStage::resolve))
            {
                flags |= MTLStageFragment;
            }
            if (hasAny(stages,
                       PipelineStage::accelerationStructureBuild | PipelineStage::accelerationStructureCopy |
                           PipelineStage::rayTracingShader))
            {
                flags |= MTLStageAccelerationStructure | MTLStageDispatch;
            }
            return flags == static_cast<MTLStages>(0) ? MTLStageAll : flags;
        }

        struct MetalBarrierResources final
        {
            MTLStages afterStages = static_cast<MTLStages>(0);
            MTLStages beforeStages = static_cast<MTLStages>(0);
            std::vector<std::shared_ptr<void>> retained{};
            std::vector<id<MTLBuffer>> buffers;
            std::vector<id<MTLTexture>> textures;
            std::vector<MetalBufferRecord*> bufferRecords;
            std::vector<MetalTextureRecord*> textureRecords;
            std::vector<MetalTextureRange> textureRanges;
            std::vector<MetalLayoutTransition> layoutTransitions;
        };

        [[nodiscard]] Status validateMetalBarrierBatchShape(const ResourceBarrierBatch& barriers)
        {
            if ((barriers.globalBarriers.empty() && barriers.bufferBarriers.empty() &&
                 barriers.textureBarriers.empty() && barriers.samplerFeedbackBarriers.empty()))
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, "A resource-barrier batch must be non-empty."));
            }
            if (!barriers.samplerFeedbackBarriers.empty())
            {
                return std::unexpected(metalSelectionError(ErrorCode::unsupported,
                                                           "Sampler-feedback barriers are not supported on Metal."));
            }
            return {};
        }

        [[nodiscard]] Status appendMetalGlobalBarriers(std::span<const GlobalBarrier> globalBarriers,
                                                       QueueType queueType,
                                                       MetalBarrierResources& resources)
        {
            for (const auto& barrier : globalBarriers)
            {
                if (!validMetalBarrierScope(barrier.stagesBefore, barrier.accessBefore, queueType) ||
                    !validMetalBarrierScope(barrier.stagesAfter, barrier.accessAfter, queueType) ||
                    (barrier.accessBefore == Access::common && barrier.accessAfter != Access::common))
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument,
                        "A global barrier contains an invalid or queue-incompatible stage/access scope."));
                }
                resources.afterStages |= metalStages(barrier.stagesBefore);
                resources.beforeStages |= metalStages(barrier.stagesAfter);
            }
            return {};
        }

        [[nodiscard]] bool metalBufferAlreadyBarriered(const std::vector<MetalBufferRecord*>& bufferRecords,
                                                       const MetalBufferRecord* record)
        {
            for (const auto* previous : bufferRecords)
            {
                if (previous == record)
                {
                    return true;
                }
            }
            return false;
        }

        [[nodiscard]] Status appendMetalBufferBarrier(MetalDeviceNative& device,
                                                      const BufferBarrier& barrier,
                                                      QueueType queueType,
                                                      MetalBarrierResources& resources)
        {
            if (!validMetalResourceBarrierScope(barrier.stagesBefore, barrier.accessBefore, queueType) ||
                !validMetalResourceBarrierScope(barrier.stagesAfter, barrier.accessAfter, queueType))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "A buffer barrier contains an invalid or queue-incompatible stage/access scope."));
            }
            auto* record = device.buffers.find(barrier.buffer);
            if (record == nullptr || record->buffer == nil)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "A buffer barrier references an invalid or stale buffer handle."));
            }
            if (barrier.ownership != OwnershipTransfer::none)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "Buffer ownership transfers require a buffer created or imported through the shared-buffer API."));
            }
            if (!supportsMetalBufferAccess(record->desc.usage, barrier.accessBefore) ||
                !supportsMetalBufferAccess(record->desc.usage, barrier.accessAfter))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "A buffer barrier access is incompatible with the buffer usage."));
            }
            if (!validMetalBarrierRange(record->desc.size, barrier.offset, barrier.size))
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "A buffer barrier range is empty or out of bounds."));
            }
            if (metalBufferAlreadyBarriered(resources.bufferRecords, record))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "A resource-barrier batch cannot contain multiple transitions for one buffer."));
            }
            resources.afterStages |= metalStages(barrier.stagesBefore);
            resources.beforeStages |= metalStages(barrier.stagesAfter);
            resources.bufferRecords.push_back(record);
            resources.buffers.push_back(record->buffer);
            if (record->owner != nullptr)
            {
                resources.retained.push_back(record->owner);
            }
            return {};
        }

        [[nodiscard]] bool metalPresentEndpointHasAccess(TextureLayout layout, PipelineStage stages, Access access)
        {
            return layout == TextureLayout::present && (stages != PipelineStage::none || access != Access::none);
        }

        [[nodiscard]] bool validMetalTextureBarrierScope(const TextureBarrier& barrier, QueueType queueType)
        {
            return validMetalResourceBarrierScope(barrier.stagesBefore, barrier.accessBefore, queueType) &&
                   validMetalResourceBarrierScope(barrier.stagesAfter, barrier.accessAfter, queueType) &&
                   validMetalTextureLayout(barrier.layoutBefore) && validMetalTextureLayout(barrier.layoutAfter) &&
                   barrier.layoutAfter != TextureLayout::undefined &&
                   !metalPresentEndpointHasAccess(barrier.layoutBefore, barrier.stagesBefore, barrier.accessBefore) &&
                   !metalPresentEndpointHasAccess(barrier.layoutAfter, barrier.stagesAfter, barrier.accessAfter);
        }

        [[nodiscard]] bool supportsMetalTextureBarrierLayouts(TextureUsage usage,
                                                              const TextureBarrier& barrier,
                                                              QueueType queueType)
        {
            return supportsMetalLayout(usage, barrier.layoutBefore) &&
                   supportsMetalLayout(usage, barrier.layoutAfter) &&
                   supportsMetalTextureLayoutOnQueue(barrier.layoutBefore, queueType) &&
                   supportsMetalTextureLayoutOnQueue(barrier.layoutAfter, queueType);
        }

        [[nodiscard]] bool supportsMetalTextureBarrierAccess(TextureUsage usage, const TextureBarrier& barrier)
        {
            return supportsMetalTextureAccess(usage, barrier.accessBefore) &&
                   supportsMetalTextureAccess(usage, barrier.accessAfter) &&
                   metalAccessMatchesTextureLayout(barrier.layoutBefore, barrier.accessBefore) &&
                   metalAccessMatchesTextureLayout(barrier.layoutAfter, barrier.accessAfter);
        }

        [[nodiscard]] bool metalTextureRangeAlreadyBarriered(const MetalBarrierResources& resources,
                                                             const MetalTextureRecord* record,
                                                             const MetalTextureRange& range)
        {
            for (std::size_t index = 0; index < resources.textureRecords.size(); ++index)
            {
                if (resources.textureRecords[index] == record &&
                    metalTextureRangesOverlap(resources.textureRanges[index], range))
                {
                    return true;
                }
            }
            return false;
        }

        [[nodiscard]] Status appendMetalTextureBarrier(MetalDeviceNative& device,
                                                       const TextureBarrier& barrier,
                                                       QueueType queueType,
                                                       MetalBarrierResources& resources)
        {
            if (!validMetalTextureBarrierScope(barrier, queueType))
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "A texture barrier contains an unknown stage, access, or layout value, ends in "
                                        "the undefined layout, or uses present outside its no-access endpoint scope."));
            }
            auto* record = device.textures.find(barrier.texture);
            if (record == nullptr || record->texture == nil)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "A texture barrier references an invalid or stale texture handle."));
            }
            if (barrier.ownership != OwnershipTransfer::none)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "Texture ownership transfers require a texture created or "
                                                           "imported through the shared-texture API."));
            }
            auto range = resolveMetalTextureRange(record->desc, barrier.subresources);
            if (!range)
            {
                return std::unexpected(std::move(range.error()));
            }
            if (!supportsMetalTextureBarrierLayouts(record->desc.usage, barrier, queueType))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "A texture barrier layout is incompatible with the texture usage or command queue."));
            }
            if (!supportsMetalTextureBarrierAccess(record->desc.usage, barrier))
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "A texture barrier access is incompatible with the texture usage or layout."));
            }
            if (metalTextureRangeAlreadyBarriered(resources, record, *range))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "A resource-barrier batch cannot contain overlapping transitions for one texture."));
            }
            resources.afterStages |= metalStages(barrier.stagesBefore);
            resources.beforeStages |= metalStages(barrier.stagesAfter);
            resources.textureRecords.push_back(record);
            resources.textureRanges.push_back(*range);
            resources.layoutTransitions.push_back(MetalLayoutTransition{
                .table = metalTextureLayoutTable(*record),
                .firstMip = range->firstMip,
                .mipCount = range->mipCount,
                .firstLayer = range->firstArray,
                .layerCount = range->arrayCount,
                .aspects = range->aspect,
                .before = barrier.layoutBefore,
                .after = barrier.layoutAfter,
                .operation = "A texture barrier's layoutBefore",
            });
            resources.textures.push_back(record->texture);
            if (record->owner != nullptr)
            {
                resources.retained.push_back(record->owner);
            }
            return {};
        }

        [[nodiscard]] Status collectMetalBarrierResources(MetalDeviceNative& device,
                                                          const ResourceBarrierBatch& barriers,
                                                          QueueType queueType,
                                                          MetalBarrierResources& resources)
        {
            auto globals = appendMetalGlobalBarriers(barriers.globalBarriers, queueType, resources);
            if (!globals)
            {
                return globals;
            }
            for (const auto& barrier : barriers.bufferBarriers)
            {
                auto appended = appendMetalBufferBarrier(device, barrier, queueType, resources);
                if (!appended)
                {
                    return appended;
                }
            }
            for (const auto& barrier : barriers.textureBarriers)
            {
                auto appended = appendMetalTextureBarrier(device, barrier, queueType, resources);
                if (!appended)
                {
                    return appended;
                }
            }
            return {};
        }

        [[nodiscard]] Status retainMetalBarrierOwners(MetalCommandBuffer& native,
                                                      std::vector<std::shared_ptr<void>>& retained)
        {
            for (auto& resource : retained)
            {
                auto kept = retainMetalApplicationResource(native, std::move(resource));
                if (!kept)
                {
                    return kept;
                }
            }
            return {};
        }

        [[nodiscard]] Status encodeMetalBarrierResources(MetalCommandBuffer& native,
                                                         const MetalBarrierResources& resources)
        {
            const std::scoped_lock lock(native.owner()->mutex);
            auto encoded = native.encodeStageBarrier(resources.afterStages, resources.beforeStages);
            if (!encoded)
            {
                return encoded;
            }
            auto* entry = native.entry();
            if (entry != nullptr)
            {
                for (const id<MTLBuffer>& buffer : resources.buffers)
                {
                    metalRetainResidency(*entry, buffer);
                }
                for (const id<MTLTexture>& texture : resources.textures)
                {
                    metalRetainResidency(*entry, texture);
                }
            }
            return {};
        }

    } // namespace

    namespace
    {

        [[nodiscard]] constexpr std::string_view metalLayoutName(TextureLayout layout) noexcept
        {
            switch (layout)
            {
            case TextureLayout::undefined:
                return "undefined";
            case TextureLayout::common:
                return "common";
            case TextureLayout::present:
                return "present";
            case TextureLayout::shaderResource:
                return "shaderResource";
            case TextureLayout::unorderedAccess:
                return "unorderedAccess";
            case TextureLayout::renderTarget:
                return "renderTarget";
            case TextureLayout::depthStencilWrite:
                return "depthStencilWrite";
            case TextureLayout::depthStencilRead:
                return "depthStencilRead";
            case TextureLayout::copySource:
                return "copySource";
            case TextureLayout::copyDestination:
                return "copyDestination";
            case TextureLayout::resolveSource:
                return "resolveSource";
            case TextureLayout::resolveDestination:
                return "resolveDestination";
            case TextureLayout::shadingRateSource:
                return "shadingRateSource";
            }
            return "an unknown layout";
        }

        [[nodiscard]] constexpr std::string_view metalAspectName(TextureAspect aspect) noexcept
        {
            switch (aspect)
            {
            case TextureAspect::depth:
                return "depth";
            case TextureAspect::stencil:
                return "stencil";
            case TextureAspect::color:
                break;
            }
            return "color";
        }

        [[nodiscard]] Error metalLayoutMismatch(std::string_view operation,
                                                TextureLayout before,
                                                TextureLayout current,
                                                const TextureSubresource& subresource,
                                                std::string_view where)
        {
            std::string message(operation);
            message += " (";
            message += metalLayoutName(before);
            message += ") is not the current layout of ";
            message += metalAspectName(subresource.aspect);
            message += " mip ";
            message += std::to_string(subresource.mipLevel);
            message += ", array layer ";
            message += std::to_string(subresource.arrayLayer);
            message += ": ";
            message += where;
            message += " ";
            message += metalLayoutName(current);
            message += ". Name the current layout, or undefined to discard the contents.";
            return metalSelectionError(ErrorCode::invalidState, std::move(message));
        }

        template<typename Visit>
        void forEachMetalLayoutCell(const MetalLayoutTransition& transition, Visit&& visit)
        {
            const auto& table = *transition.table;
            for (const auto [bit, aspect] : {std::pair{1U, TextureAspect::color},
                                             std::pair{2U, TextureAspect::depth},
                                             std::pair{4U, TextureAspect::stencil}})
            {
                if ((transition.aspects & bit) == 0)
                {
                    continue;
                }
                const std::size_t plane = aspect == TextureAspect::stencil ? 1U : 0U;
                for (std::uint32_t layer = transition.firstLayer; layer < transition.firstLayer + transition.layerCount;
                     ++layer)
                {
                    for (std::uint32_t mip = transition.firstMip; mip < transition.firstMip + transition.mipCount;
                         ++mip)
                    {
                        const auto index = (plane * table.arrayLayers + layer) * table.mipLevels + mip;
                        if (layer < table.arrayLayers && mip < table.mipLevels && index < table.layouts.size())
                        {
                            visit(index, TextureSubresource{.mipLevel = mip, .arrayLayer = layer, .aspect = aspect});
                        }
                    }
                }
            }
        }

    } // namespace

    std::shared_ptr<MetalTextureLayoutTable> metalTextureLayoutTable(MetalTextureRecord& record)
    {
        if (record.layouts == nullptr)
        {
            auto table = std::make_shared<MetalTextureLayoutTable>();
            table->mipLevels = std::max(record.desc.mipLevels, 1U);
            table->arrayLayers = std::max(metalTextureArraySize(record.desc), 1U);
            table->layouts.assign(static_cast<std::size_t>(table->mipLevels) * table->arrayLayers * 2U,
                                  record.swapchainImage ? TextureLayout::present : TextureLayout::undefined);
            record.layouts = std::move(table);
        }
        return record.layouts;
    }

    Status metalCheckLayoutTransitions(const MetalCommandLayoutState& state,
                                       std::span<const MetalLayoutTransition> transitions)
    {
        if (state.writeIndex.empty())
        {
            return {};
        }
        for (const auto& transition : transitions)
        {
            if (transition.table == nullptr || transition.before == TextureLayout::undefined)
            {
                continue;
            }
            std::optional<Error> mismatch;
            forEachMetalLayoutCell(transition, [&](std::size_t index, const TextureSubresource& subresource) {
                if (mismatch.has_value())
                {
                    return;
                }
                const auto found = state.writeIndex.find(MetalLayoutCell{transition.table.get(), index});
                if (found != state.writeIndex.end() && state.writes[found->second].layout != transition.before)
                {
                    mismatch = metalLayoutMismatch(transition.operation,
                                                   transition.before,
                                                   state.writes[found->second].layout,
                                                   subresource,
                                                   "this command buffer leaves it in");
                }
            });
            if (mismatch.has_value())
            {
                return std::unexpected(std::move(*mismatch));
            }
        }
        return {};
    }

    void metalStageLayoutTransitions(MetalCommandLayoutState& state, std::span<const MetalLayoutTransition> transitions)
    {
        for (const auto& transition : transitions)
        {
            if (transition.table == nullptr)
            {
                continue;
            }
            forEachMetalLayoutCell(transition, [&](std::size_t index, const TextureSubresource& subresource) {
                const MetalLayoutCell cell{transition.table.get(), index};
                if (const auto found = state.writeIndex.find(cell); found != state.writeIndex.end())
                {
                    state.writes[found->second].layout = transition.after;
                    return;
                }
                if (transition.before != TextureLayout::undefined)
                {
                    state.entries.push_back(MetalLayoutEntryRequirement{
                        .table = transition.table,
                        .index = index,
                        .layout = transition.before,
                        .subresource = subresource,
                        .operation = transition.operation,
                    });
                }
                state.writeIndex.emplace(cell, state.writes.size());
                state.writes.push_back(
                    MetalLayoutWrite{.table = transition.table, .index = index, .layout = transition.after});
            });
        }
    }

    Result<MetalBatchLayouts> metalPlanBatchLayouts(std::span<MetalCommandBuffer* const> commands)
    {
        MetalBatchLayouts batch;
        for (const auto* command : commands)
        {
            const auto& state = command->layoutState();
            for (const auto& entry : state.entries)
            {
                const MetalLayoutCell cell{entry.table.get(), entry.index};
                const auto planned = batch.writeIndex.find(cell);
                const auto current = planned != batch.writeIndex.end() ? batch.writes[planned->second].layout
                                                                       : entry.table->layouts[entry.index];
                if (current != entry.layout)
                {
                    return std::unexpected(metalLayoutMismatch(entry.operation,
                                                               entry.layout,
                                                               current,
                                                               entry.subresource,
                                                               "the work submitted before it leaves it in"));
                }
            }
            for (const auto& write : state.writes)
            {
                const MetalLayoutCell cell{write.table.get(), write.index};
                if (const auto planned = batch.writeIndex.find(cell); planned != batch.writeIndex.end())
                {
                    batch.writes[planned->second].layout = write.layout;
                    continue;
                }
                batch.writeIndex.emplace(cell, batch.writes.size());
                batch.writes.push_back(write);
            }
        }
        return batch;
    }

    void metalPublishBatchLayouts(MetalBatchLayouts& layouts) noexcept
    {
        for (const auto& write : layouts.writes)
        {
            write.table->layouts[write.index] = write.layout;
        }
        layouts.writes.clear();
        layouts.writeIndex.clear();
    }

    Status MetalDevice::bufferBarrier(detail::BackendCommandBuffer& commandBuffer, const BufferBarrier& barrier)
    {
        const std::array barriers{barrier};
        return resourceBarriers(commandBuffer, ResourceBarrierBatch{.bufferBarriers = barriers});
    }

    Status MetalDevice::resourceBarriers(detail::BackendCommandBuffer& commandBuffer,
                                         const ResourceBarrierBatch& barriers)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        auto shape = validateMetalBarrierBatchShape(barriers);
        if (!shape)
        {
            return shape;
        }

        std::uint64_t deviceToken = 0;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal resource barriers require an active device."));
            }
            deviceToken = native_->deviceToken;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        auto ready = metalRecordingReady(*native);
        if (!ready)
        {
            return ready;
        }

        const auto queueType = native->queueType();
        MetalBarrierResources resources{};

        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal resource barriers require an active device."));
            }
            auto collected = collectMetalBarrierResources(*native_, barriers, queueType, resources);
            if (!collected)
            {
                return collected;
            }
        }
        if (auto checked = metalCheckLayoutTransitions(native->layoutState(), resources.layoutTransitions); !checked)
        {
            return checked;
        }

        auto kept = retainMetalBarrierOwners(*native, resources.retained);
        if (!kept)
        {
            return kept;
        }
        if (auto encoded = encodeMetalBarrierResources(*native, resources); !encoded)
        {
            return encoded;
        }
        metalStageLayoutTransitions(native->layoutState(), resources.layoutTransitions);
        return {};
    }

    Status MetalDevice::activateBufferAlias(detail::BackendCommandBuffer& commandBuffer,
                                            const BufferAliasActivation& activation)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        std::uint64_t deviceToken = 0;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                           "Metal buffer alias activation requires an active device."));
            }
            deviceToken = native_->deviceToken;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        auto ready = metalRecordingReady(*native);
        if (!ready)
        {
            return ready;
        }
        const auto queueType = native->queueType();
        if (activation.before == activation.after ||
            !validMetalBarrierScope(activation.stagesBefore, activation.accessBefore, queueType) ||
            !validMetalBarrierScope(activation.stagesAfter, activation.accessAfter, queueType) ||
            !hasMetalWriteAccess(activation.accessAfter))
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument,
                "Buffer alias activation requires distinct aliases plus compatible scopes and a destination write."));
        }

        std::shared_ptr<MetalBufferOwner> beforeOwner;
        std::shared_ptr<MetalBufferOwner> afterOwner;
        id<MTLBuffer> beforeBuffer = nil;
        id<MTLBuffer> afterBuffer = nil;
        MTLStages afterStages = static_cast<MTLStages>(0);
        MTLStages beforeStages = static_cast<MTLStages>(0);
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                           "Metal buffer alias activation requires an active device."));
            }
            auto* before = native_->buffers.find(activation.before);
            auto* after = native_->buffers.find(activation.after);
            if (before == nullptr || before->buffer == nil || after == nullptr || after->buffer == nil)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "Buffer alias activation references an invalid or stale buffer handle."));
            }
            if (!metalPlacementsAlias(before->owner != nullptr ? before->owner->placement : nullptr,
                                      after->owner != nullptr ? after->owner->placement : nullptr))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "Buffer alias activation requires overlapping placed resources from the same aliasable heap."));
            }
            if (!supportsMetalBufferAccess(before->desc.usage, activation.accessBefore) ||
                !supportsMetalBufferAccess(after->desc.usage, activation.accessAfter))
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "A buffer alias activation access is incompatible with its buffer usage."));
            }
            beforeOwner = before->owner;
            afterOwner = after->owner;
            beforeBuffer = before->buffer;
            afterBuffer = after->buffer;
            afterStages = metalStages(activation.stagesBefore);
            beforeStages = metalStages(activation.stagesAfter);
        }

        auto keptBefore = retainMetalApplicationResource(*native, std::move(beforeOwner));
        if (!keptBefore)
        {
            return keptBefore;
        }
        auto keptAfter = retainMetalApplicationResource(*native, std::move(afterOwner));
        if (!keptAfter)
        {
            return keptAfter;
        }

        const std::scoped_lock lock(native->owner()->mutex);
        auto encoded = native->encodeStageBarrier(afterStages, beforeStages);
        if (!encoded)
        {
            return encoded;
        }
        auto* entry = native->entry();
        if (entry != nullptr)
        {
            metalRetainResidency(*entry, beforeBuffer);
            metalRetainResidency(*entry, afterBuffer);
        }
        return {};
    }

    Status MetalDevice::textureBarrier(detail::BackendCommandBuffer& commandBuffer, const TextureBarrier& barrier)
    {
        const std::array barriers{barrier};
        return resourceBarriers(commandBuffer, ResourceBarrierBatch{.textureBarriers = barriers});
    }

    namespace
    {

        [[nodiscard]] bool malformedMetalTextureAliasActivation(const TextureAliasActivation& activation,
                                                                QueueType queueType)
        {
            return activation.before == activation.after || activation.layoutBefore == TextureLayout::undefined ||
                   activation.layoutAfter == TextureLayout::undefined ||
                   !validMetalTextureLayout(activation.layoutBefore) ||
                   !validMetalTextureLayout(activation.layoutAfter) ||
                   !validMetalBarrierScope(activation.stagesBefore, activation.accessBefore, queueType) ||
                   !validMetalBarrierScope(activation.stagesAfter, activation.accessAfter, queueType) ||
                   !hasMetalWriteAccess(activation.accessAfter);
        }

        struct MetalTextureAliasResources final
        {
            std::shared_ptr<MetalTextureOwner> beforeOwner;
            std::shared_ptr<MetalTextureOwner> afterOwner;
            id<MTLTexture> beforeTexture = nil;
            id<MTLTexture> afterTexture = nil;
            MTLStages afterStages = static_cast<MTLStages>(0);
            MTLStages beforeStages = static_cast<MTLStages>(0);
            MetalLayoutTransition afterLayout{};
        };

        [[nodiscard]] bool supportsMetalTextureAliasScopes(const MetalTextureRecord& before,
                                                           const MetalTextureRecord& after,
                                                           const TextureAliasActivation& activation,
                                                           QueueType queueType)
        {
            const bool beforeLayoutSupported = supportsMetalLayout(before.desc.usage, activation.layoutBefore) &&
                                               supportsMetalTextureLayoutOnQueue(activation.layoutBefore, queueType);
            const bool afterLayoutSupported = supportsMetalLayout(after.desc.usage, activation.layoutAfter) &&
                                              supportsMetalTextureLayoutOnQueue(activation.layoutAfter, queueType);
            return beforeLayoutSupported && afterLayoutSupported &&
                   supportsMetalTextureAccess(before.desc.usage, activation.accessBefore) &&
                   supportsMetalTextureAccess(after.desc.usage, activation.accessAfter) &&
                   metalAccessMatchesTextureLayout(activation.layoutBefore, activation.accessBefore) &&
                   metalAccessMatchesTextureLayout(activation.layoutAfter, activation.accessAfter);
        }

        [[nodiscard]] Status collectMetalTextureAliasResources(MetalDeviceNative& device,
                                                               const TextureAliasActivation& activation,
                                                               QueueType queueType,
                                                               MetalTextureAliasResources& resources)
        {
            auto* before = device.textures.find(activation.before);
            auto* after = device.textures.find(activation.after);
            if (before == nullptr || before->texture == nil || after == nullptr || after->texture == nil)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "Texture alias activation references an invalid or stale texture handle."));
            }
            if (!metalPlacementsAlias(before->owner != nullptr ? before->owner->placement : nullptr,
                                      after->owner != nullptr ? after->owner->placement : nullptr))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "Texture alias activation requires overlapping placed resources from the same aliasable heap."));
            }
            if (!supportsMetalTextureAliasScopes(*before, *after, activation, queueType))
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "A texture alias activation scope is incompatible with its "
                                                           "texture usage, layout, or command queue."));
            }
            if (hasAny(after->desc.usage, TextureUsage::renderTarget | TextureUsage::depthStencil) &&
                queueType != QueueType::graphics)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::unsupported, "Aliased attachment metadata must be initialized on a graphics queue."));
            }
            resources.beforeOwner = before->owner;
            resources.afterOwner = after->owner;
            resources.beforeTexture = before->texture;
            resources.afterTexture = after->texture;
            resources.afterStages = metalStages(activation.stagesBefore);
            resources.beforeStages = metalStages(activation.stagesAfter);
            std::uint32_t aspects = 0;
            for (const auto aspect : {TextureAspect::color, TextureAspect::depth, TextureAspect::stencil})
            {
                aspects |= metalAspectMask(after->desc.format, aspect).value_or(0U);
            }
            const auto afterTable = metalTextureLayoutTable(*after);
            resources.afterLayout = MetalLayoutTransition{
                .table = afterTable,
                .firstMip = 0,
                .mipCount = afterTable->mipLevels,
                .firstLayer = 0,
                .layerCount = afterTable->arrayLayers,
                .aspects = aspects,
                .before = TextureLayout::undefined,
                .after = activation.layoutAfter,
                .operation = "A texture alias activation",
            };
            return {};
        }

        [[nodiscard]] Status retainMetalTextureAliasOwners(MetalCommandBuffer& native,
                                                           MetalTextureAliasResources& resources)
        {
            auto keptBefore = retainMetalApplicationResource(native, std::move(resources.beforeOwner));
            if (!keptBefore)
            {
                return keptBefore;
            }
            auto keptAfter = retainMetalApplicationResource(native, std::move(resources.afterOwner));
            if (!keptAfter)
            {
                return keptAfter;
            }
            return {};
        }

        [[nodiscard]] Status encodeMetalTextureAliasBarrier(MetalCommandBuffer& native,
                                                            const MetalTextureAliasResources& resources)
        {
            const std::scoped_lock lock(native.owner()->mutex);
            auto encoded = native.encodeStageBarrier(resources.afterStages, resources.beforeStages);
            if (!encoded)
            {
                return encoded;
            }
            auto* entry = native.entry();
            if (entry != nullptr)
            {
                metalRetainResidency(*entry, resources.beforeTexture);
                metalRetainResidency(*entry, resources.afterTexture);
            }
            return {};
        }

    } // namespace

    Status MetalDevice::activateTextureAlias(detail::BackendCommandBuffer& commandBuffer,
                                             const TextureAliasActivation& activation)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        std::uint64_t deviceToken = 0;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidState, "Metal texture alias activation requires an active device."));
            }
            deviceToken = native_->deviceToken;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        auto ready = metalRecordingReady(*native);
        if (!ready)
        {
            return ready;
        }
        const auto queueType = native->queueType();
        if (malformedMetalTextureAliasActivation(activation, queueType))
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "Texture alias activation requires distinct aliases, defined "
                                                       "layouts, compatible scopes, and a destination write."));
        }

        MetalTextureAliasResources resources{};
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidState, "Metal texture alias activation requires an active device."));
            }
            auto collected = collectMetalTextureAliasResources(*native_, activation, queueType, resources);
            if (!collected)
            {
                return collected;
            }
        }

        auto kept = retainMetalTextureAliasOwners(*native, resources);
        if (!kept)
        {
            return kept;
        }
        if (auto encoded = encodeMetalTextureAliasBarrier(*native, resources); !encoded)
        {
            return encoded;
        }
        metalStageLayoutTransitions(native->layoutState(), std::span(&resources.afterLayout, 1));
        return {};
    }

} // namespace rachis::metal
