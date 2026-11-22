#include "Metal/MetalInternal.h"

#include "RachisGenerateMipmaps2D.pnms.h"
#include "RachisGenerateMipmaps3D.pnms.h"

namespace rachis::metal
{
    namespace
    {

        [[nodiscard]] constexpr std::size_t metalMipmapDimensionIndex(TextureDimension dimension) noexcept
        {
            return dimension == TextureDimension::texture3D ? 1U : 0U;
        }

        [[nodiscard]] constexpr std::array<std::uint32_t, 3> metalMipmapThreadGroupSize(
            TextureDimension dimension) noexcept
        {
            if (dimension == TextureDimension::texture3D)
            {
                return {4, 4, 4};
            }
            return {8, 8, 1};
        }

        [[nodiscard]] constexpr TextureViewDimension metalMipmapViewDimension(TextureDimension dimension) noexcept
        {
            return dimension == TextureDimension::texture3D ? TextureViewDimension::texture3D
                                                            : TextureViewDimension::texture2DArray;
        }

        [[nodiscard]] constexpr std::uint32_t metalMipDimension(std::uint32_t size, std::uint32_t mipLevel) noexcept
        {
            return std::max(1U, size >> mipLevel);
        }

        [[nodiscard]] std::span<const std::byte> metalEmbeddedPnms(const std::uint8_t* data, std::size_t size) noexcept
        {
            return {reinterpret_cast<const std::byte*>(data), size};
        }

        [[nodiscard]] Result<std::uint32_t> metalSelectedCount(std::uint32_t first,
                                                               std::uint32_t requested,
                                                               std::uint32_t available,
                                                               std::string_view subject)
        {
            if (first >= available)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, std::string(subject) + " starts out of range."));
            }
            const auto count = requested == remainingSubresources ? available - first : requested;
            if (count == 0 || count > available - first)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, std::string(subject) + " has an invalid count."));
            }
            return count;
        }

        [[nodiscard]] Status validateMetalImageCopyRegion(const TextureDesc& desc,
                                                          const TextureSubresource& subresource,
                                                          const Origin3D& origin,
                                                          const Extent3D& extent)
        {
            if (!metalAspectMatchesFormat(desc.format, subresource.aspect))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "The texture copy aspect does not match the texture format."));
            }
            const auto arraySize = desc.dimension == TextureDimension::texture3D ? 1U : desc.depthOrArrayLayers;
            if (subresource.mipLevel >= desc.mipLevels || subresource.arrayLayer >= arraySize ||
                (desc.dimension == TextureDimension::texture3D && subresource.arrayLayer != 0))
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "The texture copy subresource mip or layer is invalid."));
            }
            const auto mip = metalSubresourceExtent(desc, subresource.mipLevel);
            if (extent.width == 0 || extent.height == 0 || extent.depth == 0 || origin.x > mip.width ||
                extent.width > mip.width - origin.x || origin.y > mip.height || extent.height > mip.height - origin.y ||
                origin.z > mip.depth || extent.depth > mip.depth - origin.z ||
                (desc.dimension != TextureDimension::texture3D && origin.z != 0) ||
                (desc.dimension == TextureDimension::texture1D && (origin.y != 0 || extent.height != 1)))
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "The texture copy region is empty or outside the selected subresource."));
            }
            const auto footprint = metalFormatFootprint(desc.format, subresource.aspect);
            if (footprint.bytesPerBlock == 0 ||
                !metalBlockAligned(origin.x, extent.width, mip.width, footprint.blockWidth) ||
                !metalBlockAligned(origin.y, extent.height, mip.height, footprint.blockHeight))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "Compressed texture copies must be aligned to the format block size."));
            }
            const bool wholeSubresource = desc.sampleCount != 1 || formatIsDepth(desc.format);
            if (wholeSubresource && (origin.x != 0 || origin.y != 0 || origin.z != 0 || extent.width != mip.width ||
                                     extent.height != mip.height || extent.depth != mip.depth))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "Depth-stencil and multisampled texture copies require whole matching subresources."));
            }
            return {};
        }

        [[nodiscard]] Result<MetalFormatFootprint> validateMetalTransferSubresource(
            const TextureDesc& desc, const BufferTextureCopyRegion& region)
        {
            if (!metalAspectMatchesFormat(desc.format, region.textureSubresource.aspect))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "The texture copy aspect does not match the texture format."));
            }
            if (desc.sampleCount != 1)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "Buffer-texture copies require a single-sampled texture."));
            }

            const auto footprint = metalFormatFootprint(desc.format, region.textureSubresource.aspect);
            if (footprint.bytesPerBlock == 0)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "The texture format has no portable transfer footprint."));
            }

            const auto arraySize = desc.dimension == TextureDimension::texture3D ? 1U : desc.depthOrArrayLayers;
            if (region.textureSubresource.mipLevel >= desc.mipLevels ||
                region.textureSubresource.arrayLayer >= arraySize ||
                (desc.dimension == TextureDimension::texture3D && region.textureSubresource.arrayLayer != 0))
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "The texture copy subresource mip or layer is invalid."));
            }
            return footprint;
        }

        [[nodiscard]] Extent3D metalTransferSubresourceExtent(const TextureDesc& desc,
                                                              const BufferTextureCopyRegion& region)
        {
            return Extent3D{
                .width = metalMipExtent(desc.width, region.textureSubresource.mipLevel),
                .height = desc.dimension == TextureDimension::texture1D
                              ? 1U
                              : metalMipExtent(desc.height, region.textureSubresource.mipLevel),
                .depth = desc.dimension == TextureDimension::texture3D
                             ? metalMipExtent(desc.depthOrArrayLayers, region.textureSubresource.mipLevel)
                             : 1U,
            };
        }

        [[nodiscard]] Status validateMetalTransferRegion(const TextureDesc& desc,
                                                         const BufferTextureCopyRegion& region,
                                                         const Extent3D& mip,
                                                         const MetalFormatFootprint& footprint)
        {
            if (region.extent.width == 0 || region.extent.height == 0 || region.extent.depth == 0 ||
                region.textureOrigin.x > mip.width || region.extent.width > mip.width - region.textureOrigin.x ||
                region.textureOrigin.y > mip.height || region.extent.height > mip.height - region.textureOrigin.y ||
                region.textureOrigin.z > mip.depth || region.extent.depth > mip.depth - region.textureOrigin.z ||
                (desc.dimension != TextureDimension::texture3D && region.textureOrigin.z != 0) ||
                (desc.dimension == TextureDimension::texture1D &&
                 (region.textureOrigin.y != 0 || region.extent.height != 1)))
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "The texture copy region is empty or outside the selected subresource."));
            }
            if (!metalBlockAligned(region.textureOrigin.x, region.extent.width, mip.width, footprint.blockWidth) ||
                !metalBlockAligned(region.textureOrigin.y, region.extent.height, mip.height, footprint.blockHeight))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "Compressed texture copies must be aligned to the format block size."));
            }
            return {};
        }

        [[nodiscard]] Result<TextureTransferLayout> metalTransferLayout(const BufferTextureCopyRegion& region,
                                                                        const MetalFormatFootprint& footprint,
                                                                        std::uint32_t bufferOffsetAlignment,
                                                                        std::uint32_t rowPitchAlignment)
        {
            const auto blocksWide = (region.extent.width + footprint.blockWidth - 1U) / footprint.blockWidth;
            const auto copiedRows = (region.extent.height + footprint.blockHeight - 1U) / footprint.blockHeight;
            const auto rowsPerImage = region.rowsPerImage == 0 ? region.extent.height : region.rowsPerImage;
            const auto sliceRows =
                (static_cast<std::uint64_t>(rowsPerImage) + footprint.blockHeight - 1U) / footprint.blockHeight;
            std::uint64_t minimumRowBytes = 0;
            if (!metalCheckedMultiply(blocksWide, footprint.bytesPerBlock, minimumRowBytes))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "The buffer-texture copy footprint exceeds portable size limits."));
            }

            const auto resolvedRowPitch = region.bytesPerRow == 0
                                              ? static_cast<std::uint64_t>(metalAlignUp(
                                                    static_cast<std::uint32_t>(minimumRowBytes), rowPitchAlignment))
                                              : static_cast<std::uint64_t>(region.bytesPerRow);
            if (rowsPerImage < region.extent.height || copiedRows == 0 || resolvedRowPitch < minimumRowBytes ||
                resolvedRowPitch > std::numeric_limits<std::uint32_t>::max() ||
                resolvedRowPitch % rowPitchAlignment != 0)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "Buffer-texture row pitch or rows-per-image is too small or incorrectly aligned."));
            }

            std::uint64_t bytesPerImage = 0;
            std::uint64_t priorSliceBytes = 0;
            std::uint64_t copiedRowBytes = 0;
            std::uint64_t requiredBytes = 0;
            const bool validSize = metalCheckedMultiply(resolvedRowPitch, sliceRows, bytesPerImage) &&
                                   metalCheckedMultiply(bytesPerImage, region.extent.depth - 1U, priorSliceBytes) &&
                                   metalCheckedMultiply(resolvedRowPitch, copiedRows - 1U, copiedRowBytes) &&
                                   metalCheckedAdd(priorSliceBytes, copiedRowBytes, requiredBytes) &&
                                   metalCheckedAdd(requiredBytes, minimumRowBytes, requiredBytes) &&
                                   metalCheckedAdd(region.bufferOffset, requiredBytes, requiredBytes);
            if (!validSize)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "The buffer-texture copy footprint exceeds portable size limits."));
            }

            return TextureTransferLayout{
                .bytesPerRow = static_cast<std::uint32_t>(resolvedRowPitch),
                .rowsPerImage = rowsPerImage,
                .rowCount = static_cast<std::uint32_t>(copiedRows),
                .rowDataSize = minimumRowBytes,
                .bytesPerImage = bytesPerImage,
                .requiredBufferSize = requiredBytes,
                .bufferOffsetAlignment = bufferOffsetAlignment,
                .rowPitchAlignment = rowPitchAlignment,
            };
        }

        [[nodiscard]] MTLBlitOption metalTransferBlitOptions(const TextureDesc& desc,
                                                             const BufferTextureCopyRegion& region)
        {
            MTLBlitOption options = MTLBlitOptionNone;
            if (desc.format == Format::d24UnormS8Uint)
            {
                options = region.textureSubresource.aspect == TextureAspect::stencil
                              ? MTLBlitOptionStencilFromDepthStencil
                              : MTLBlitOptionDepthFromDepthStencil;
            }
            return options;
        }

    } // namespace

    [[nodiscard]] Result<MetalResolvedTransfer> resolveMetalTransfer(const TextureDesc& desc,
                                                                     const BufferTextureCopyRegion& region,
                                                                     std::uint32_t bufferOffsetAlignment,
                                                                     std::uint32_t rowPitchAlignment)
    {
        auto footprint = validateMetalTransferSubresource(desc, region);
        if (!footprint)
        {
            return std::unexpected(std::move(footprint.error()));
        }
        const auto mip = metalTransferSubresourceExtent(desc, region);
        if (auto valid = validateMetalTransferRegion(desc, region, mip, *footprint); !valid)
        {
            return std::unexpected(std::move(valid.error()));
        }
        if (bufferOffsetAlignment == 0 || rowPitchAlignment == 0 || region.bufferOffset % bufferOffsetAlignment != 0)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument, "A buffer-texture copy offset must satisfy Metal transfer alignment."));
        }
        auto layout = metalTransferLayout(region, *footprint, bufferOffsetAlignment, rowPitchAlignment);
        if (!layout)
        {
            return std::unexpected(std::move(layout.error()));
        }
        return MetalResolvedTransfer{
            .layout = *layout,
            .size = MTLSizeMake(region.extent.width, region.extent.height, region.extent.depth),
            .origin = MTLOriginMake(region.textureOrigin.x, region.textureOrigin.y, region.textureOrigin.z),
            .slice = desc.dimension == TextureDimension::texture3D ? 0 : region.textureSubresource.arrayLayer,
            .level = region.textureSubresource.mipLevel,
            .options = metalTransferBlitOptions(desc, region),
        };
    }

    [[nodiscard]] std::pair<std::uint32_t, std::uint32_t> metalTransferAlignments(id<MTLDevice> device,
                                                                                  Format format,
                                                                                  TextureAspect aspect) noexcept
    {
        const auto footprint = metalFormatFootprint(format, aspect);
        const auto pixel = metalPixelFormat(format);
        const bool queryLinear =
            device != nil && pixel != MTLPixelFormatInvalid && !footprint.compressed && !formatIsDepth(format);
        const auto linear =
            queryLinear ? static_cast<std::uint32_t>([device minimumLinearTextureAlignmentForPixelFormat:pixel]) : 1U;
        const auto offset = std::max(1U, footprint.bytesPerBlock);
        const auto row = std::max({offset, 1U, linear});
        return {offset, row};
    }

    namespace
    {

        struct MetalImmediateBufferCopy final
        {
            __strong id<MTLDevice> device = nil;
            __strong id<MTL4CommandQueue> queue = nil;
            __strong id<MTLBuffer> sourceBuffer = nil;
            __strong id<MTLBuffer> destinationBuffer = nil;
            MetalNativeQueueGate* gate = nullptr;
            std::shared_ptr<MetalDeviceNative> backend;
        };

        [[nodiscard]] Result<MetalImmediateBufferCopy> captureMetalImmediateBufferCopy(
            rachis::detail::TimedMutex& deviceMutex,
            const std::shared_ptr<MetalDeviceNative>& native,
            BufferHandle source,
            BufferHandle destination,
            const BufferCopyRegion& range)
        {
            std::scoped_lock const lock(deviceMutex);
            if (native == nullptr || native->device == nil || native->commandQueue == nil ||
                native->queueGate == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal buffer copy requires an active device."));
            }
            auto* src = native->buffers.find(source);
            auto* dst = native->buffers.find(destination);
            if (src == nullptr || dst == nullptr || src->buffer == nil || dst->buffer == nil)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, "Buffer copy requires two live Metal buffers."));
            }
            if (range.size == 0 || range.sourceOffset > src->desc.size ||
                range.size > src->desc.size - range.sourceOffset || range.destinationOffset > dst->desc.size ||
                range.size > dst->desc.size - range.destinationOffset)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, "The Metal copy range is outside a buffer."));
            }
            MetalImmediateBufferCopy copy;
            copy.device = native->device;
            copy.queue = native->commandQueue;
            copy.sourceBuffer = src->buffer;
            copy.destinationBuffer = dst->buffer;
            copy.gate = native->queueGate.get();
            copy.backend = native;
            return copy;
        }

        [[nodiscard]] Result<id<MTLResidencySet>> makeMetalBufferCopyResidency(const MetalImmediateBufferCopy& copy)
        {
            MTLResidencySetDescriptor* const residencyDesc = [MTLResidencySetDescriptor new];
            residencyDesc.initialCapacity = 2;
            NSError* residencyError = nil;
            id<MTLResidencySet> const residency = [copy.device newResidencySetWithDescriptor:residencyDesc
                                                                                       error:&residencyError];
            if (residency == nil)
            {
                return std::unexpected(
                    metalNsError(ErrorCode::nativeFailure, residencyError, "newResidencySetWithDescriptor"));
            }
            [residency addAllocation:copy.sourceBuffer];
            if (copy.destinationBuffer != copy.sourceBuffer)
            {
                [residency addAllocation:copy.destinationBuffer];
            }
            [residency commit];
            return residency;
        }

        struct MetalRecordedBufferCopy final
        {
            __strong id<MTL4CommandAllocator> allocator = nil;
            __strong id<MTL4CommandBuffer> command = nil;
        };

        [[nodiscard]] Result<MetalRecordedBufferCopy> recordMetalBufferCopy(const MetalImmediateBufferCopy& copy,
                                                                            id<MTLResidencySet> residency,
                                                                            const BufferCopyRegion& range)
        {
            id<MTL4CommandAllocator> const allocator = [copy.device newCommandAllocator];
            id<MTL4CommandBuffer> const command = [copy.device newCommandBuffer];
            if (allocator == nil || command == nil)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::nativeFailure,
                    "Metal buffer copy could not create command objects. Partial Metal objects were released."));
            }
            [command beginCommandBufferWithAllocator:allocator];
            [command useResidencySet:residency];
            id<MTL4ComputeCommandEncoder> const encoder = [command computeCommandEncoder];
            if (encoder == nil)
            {
                [command endCommandBuffer];
                return std::unexpected(metalSelectionError(ErrorCode::nativeFailure,
                                                           "Metal buffer copy could not create a compute encoder."));
            }
            encoder.label = @"RHI.Metal.BufferCopy";
            [encoder copyFromBuffer:copy.sourceBuffer
                       sourceOffset:static_cast<NSUInteger>(range.sourceOffset)
                           toBuffer:copy.destinationBuffer
                  destinationOffset:static_cast<NSUInteger>(range.destinationOffset)
                               size:static_cast<NSUInteger>(range.size)];
            [encoder endEncoding];
            [command endCommandBuffer];
            MetalRecordedBufferCopy recorded;
            recorded.allocator = allocator;
            recorded.command = command;
            return recorded;
        }

        [[nodiscard]] Status submitMetalBufferCopyAndWait(const std::shared_ptr<MetalDeviceNative> backend,
                                                          const MetalImmediateBufferCopy& copy,
                                                          id<MTLResidencySet> residency,
                                                          const MetalRecordedBufferCopy& recorded,
                                                          id<MTLSharedEvent> event)
        {
            id<MTL4CommandAllocator> const allocator = recorded.allocator;
            id<MTL4CommandBuffer> const command = recorded.command;
            id<MTLBuffer> const sourceBuffer = copy.sourceBuffer;
            id<MTLBuffer> const destinationBuffer = copy.destinationBuffer;
            MTL4CommitOptions* const commitOptions = [MTL4CommitOptions new];
            const auto completion = std::make_shared<MetalQueueTimeline>();
            completion->event = event;
            completion->feedback.begin(1);
            [commitOptions addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
              static_cast<void>(allocator);
              static_cast<void>(command);
              static_cast<void>(residency);
              static_cast<void>(sourceBuffer);
              static_cast<void>(destinationBuffer);
              publishMetalSubmissionFailure(*backend, feedback.error, "RHI.Metal.BufferCopy", "RHI.Metal.Graphics");
              completion->feedback.complete(1);
            }];
            id<MTL4CommandBuffer> buffers[] = {command};
            {
                const std::scoped_lock gateLock(copy.gate->mutex);
                [copy.queue commit:buffers count:1 options:commitOptions];
                [copy.queue signalEvent:event value:1];
            }
            auto waited = waitForMetalValue(*backend, *completion, 1, 5000);
            if (!waited)
            {
                return std::unexpected(std::move(waited.error()));
            }
            if (*waited != TimelinePointStatus::complete)
            {
                return std::unexpected(metalSelectionError(ErrorCode::nativeFailure,
                                                           "Metal buffer copy timed out waiting for GPU completion."));
            }
            return {};
        }

    } // namespace

    Status MetalDevice::copyBuffersImmediate(BufferHandle source,
                                             BufferHandle destination,
                                             std::uint64_t sourceOffset,
                                             std::uint64_t destinationOffset,
                                             std::uint64_t size)
    {
        const BufferCopyRegion range{
            .sourceOffset = sourceOffset,
            .destinationOffset = destinationOffset,
            .size = size,
        };
        auto copy = captureMetalImmediateBufferCopy(mutex_, native_, source, destination, range);
        if (!copy)
        {
            return std::unexpected(std::move(copy.error()));
        }

        @autoreleasepool
        {
            auto residency = makeMetalBufferCopyResidency(*copy);
            if (!residency)
            {
                return std::unexpected(std::move(residency.error()));
            }
            auto recorded = recordMetalBufferCopy(*copy, *residency, range);
            if (!recorded)
            {
                return std::unexpected(std::move(recorded.error()));
            }
            id<MTLSharedEvent> const event = [copy->device newSharedEvent];
            if (event == nil)
            {
                return std::unexpected(metalSelectionError(ErrorCode::nativeFailure,
                                                           "Metal buffer copy could not create a shared event."));
            }
            return submitMetalBufferCopyAndWait(copy->backend, *copy, *residency, *recorded, event);
        }
    }

#ifdef RACHIS_TESTING
    Status MetalDevice::copyCapturedBindingBuffer(detail::BackendCommandBuffer& commandBuffer,
                                                  std::uint32_t setIndex,
                                                  BufferHandle destination,
                                                  std::uint64_t size)
    {
        const MetalAutoreleasePool autoreleasePool;
        std::uint64_t deviceToken = 0;
        id<MTLBuffer> destinationBuffer = nil;
        std::uint64_t destinationSize = 0;
        std::shared_ptr<MetalBufferOwner> destinationOwner;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidState, "Metal copyCapturedBindingBuffer requires an active device."));
            }
            deviceToken = native_->deviceToken;
            auto* dst = native_->buffers.find(destination);
            if (dst == nullptr || dst->buffer == nil)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "Captured binding copy requires a live destination buffer."));
            }
            destinationBuffer = dst->buffer;
            destinationSize = dst->desc.size;
            destinationOwner = dst->owner;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        if (size == 0 || size > destinationSize)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument, "The captured binding copy range is empty or outside the destination."));
        }
        auto retainedDestination = retainMetalApplicationResource(*native, std::move(destinationOwner));
        if (!retainedDestination)
        {
            return retainedDestination;
        }
        const std::scoped_lock lock(native->owner()->mutex);
        auto* entry = native->entry();
        if (entry == nullptr || setIndex >= entry->boundComputeVersions.size() ||
            entry->boundComputeVersions[setIndex] == nullptr)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidState, "The command buffer has no captured Metal binding set at that index."));
        }
        id<MTLBuffer> sourceBuffer = nil;
        std::uint64_t sourceOffset = 0;
        std::uint64_t sourceRange = 0;
        for (const auto& slot : entry->boundComputeVersions[setIndex]->slots)
        {
            if (slot.buffer != nil)
            {
                sourceBuffer = slot.buffer;
                sourceOffset = slot.offset;
                sourceRange = slot.range;
                break;
            }
        }
        if (sourceBuffer == nil || size > sourceRange)
        {
            return std::unexpected(metalSelectionError(
                ErrorCode::invalidArgument, "The captured binding set has no buffer range large enough to copy."));
        }
        auto prepared = native->ensureCopyEncoder();
        if (!prepared)
        {
            return prepared;
        }
        metalRetainResidency(*entry, sourceBuffer);
        metalRetainResidency(*entry, destinationBuffer);
        [entry->computeEncoder copyFromBuffer:sourceBuffer
                                 sourceOffset:static_cast<NSUInteger>(sourceOffset)
                                     toBuffer:destinationBuffer
                            destinationOffset:0
                                         size:static_cast<NSUInteger>(size)];
        return {};
    }
#endif

    namespace
    {

        struct MetalBufferInterval final
        {
            std::uint64_t begin = 0;
            std::uint64_t end = 0;
        };

        [[nodiscard]] bool metalSortedIntervalsOverlap(std::span<const MetalBufferInterval> intervals) noexcept
        {
            for (std::size_t index = 1; index < intervals.size(); ++index)
            {
                if (intervals[index].begin < intervals[index - 1].end)
                {
                    return true;
                }
            }
            return false;
        }

        [[nodiscard]] bool metalSortedIntervalSetsOverlap(std::span<const MetalBufferInterval> sources,
                                                          std::span<const MetalBufferInterval> destinations) noexcept
        {
            std::size_t sourceIndex = 0;
            std::size_t destinationIndex = 0;
            while (sourceIndex < sources.size() && destinationIndex < destinations.size())
            {
                const auto& sourceInterval = sources[sourceIndex];
                const auto& destinationInterval = destinations[destinationIndex];
                if (sourceInterval.begin < destinationInterval.end && destinationInterval.begin < sourceInterval.end)
                {
                    return true;
                }
                if (sourceInterval.end <= destinationInterval.begin)
                {
                    ++sourceIndex;
                }
                else
                {
                    ++destinationIndex;
                }
            }
            return false;
        }

        [[nodiscard]] Status validateMetalBufferCopyRegions(std::span<const BufferCopyRegion> regions,
                                                            std::uint64_t sourceSize,
                                                            std::uint64_t destinationSize,
                                                            bool sameBuffer)
        {
            std::vector<MetalBufferInterval> sourceIntervals;
            std::vector<MetalBufferInterval> destinationIntervals;
            try
            {
                sourceIntervals.reserve(regions.size());
                destinationIntervals.reserve(regions.size());
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::outOfMemory, "Failed to allocate Metal buffer-copy validation."));
            }
            for (const auto& region : regions)
            {
                if (region.size == 0 || region.sourceOffset > sourceSize ||
                    region.size > sourceSize - region.sourceOffset || region.destinationOffset > destinationSize ||
                    region.size > destinationSize - region.destinationOffset)
                {
                    return std::unexpected(
                        metalSelectionError(ErrorCode::invalidArgument, "The Metal copy range is outside a buffer."));
                }
                sourceIntervals.push_back(MetalBufferInterval{
                    .begin = region.sourceOffset,
                    .end = region.sourceOffset + region.size,
                });
                destinationIntervals.push_back(MetalBufferInterval{
                    .begin = region.destinationOffset,
                    .end = region.destinationOffset + region.size,
                });
            }
            std::ranges::sort(destinationIntervals, {}, &MetalBufferInterval::begin);
            if (metalSortedIntervalsOverlap(destinationIntervals))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "A buffer-copy batch cannot write overlapping destination regions."));
            }
            if (!sameBuffer)
            {
                return {};
            }
            std::ranges::sort(sourceIntervals, {}, &MetalBufferInterval::begin);
            if (metalSortedIntervalSetsOverlap(sourceIntervals, destinationIntervals))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "A same-buffer copy batch cannot access one byte as both source and destination."));
            }
            return {};
        }

    } // namespace

    Status MetalDevice::copyBufferRegions(detail::BackendCommandBuffer& commandBuffer,
                                          BufferHandle source,
                                          BufferHandle destination,
                                          std::span<const BufferCopyRegion> regions)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (auto blocked = requireActive(); !blocked)
        {
            return blocked;
        }
        std::uint64_t deviceToken = 0;
        id<MTLBuffer> sourceBuffer = nil;
        id<MTLBuffer> destinationBuffer = nil;
        std::uint64_t sourceSize = 0;
        std::uint64_t destinationSize = 0;
        std::shared_ptr<MetalBufferOwner> sourceOwner;
        std::shared_ptr<MetalBufferOwner> destinationOwner;
        {
            std::scoped_lock const lock(mutex_);
            if (native_ == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal copyBufferRegions requires an active device."));
            }
            deviceToken = native_->deviceToken;
            auto* src = native_->buffers.find(source);
            auto* dst = native_->buffers.find(destination);
            if (src == nullptr || dst == nullptr || src->buffer == nil || dst->buffer == nil)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument, "Buffer copy requires two live Metal buffers."));
            }
            if (!hasAny(src->desc.usage, BufferUsage::copySource) ||
                !hasAny(dst->desc.usage, BufferUsage::copyDestination))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "Buffer copy usages do not include copy source and copy destination."));
            }
            sourceBuffer = src->buffer;
            destinationBuffer = dst->buffer;
            sourceSize = src->desc.size;
            destinationSize = dst->desc.size;
            sourceOwner = src->owner;
            destinationOwner = dst->owner;
        }
        auto* native = asMetalCommand(commandBuffer, deviceToken);
        if (native == nullptr)
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "The command buffer belongs to another device or backend."));
        }
        if (regions.empty())
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidArgument, "copyBufferRegions requires at least one region."));
        }
        if (auto valid = validateMetalBufferCopyRegions(regions, sourceSize, destinationSize, source == destination);
            !valid)
        {
            return valid;
        }
        auto owner = metalRecordingPool(*native);
        if (!owner)
        {
            return std::unexpected(std::move(owner.error()));
        }
        const std::scoped_lock lock((*owner)->mutex);
        if (auto ready = metalRecordingReady(*native); !ready)
        {
            return ready;
        }
        auto retainedSource = retainMetalApplicationResource(*native, std::move(sourceOwner));
        if (!retainedSource)
        {
            return retainedSource;
        }
        auto retainedDestination = retainMetalApplicationResource(*native, std::move(destinationOwner));
        if (!retainedDestination)
        {
            return retainedDestination;
        }
        auto prepared = native->ensureCopyEncoder();
        if (!prepared)
        {
            return prepared;
        }
        auto* entry = native->entry();
        metalRetainResidency(*entry, sourceBuffer);
        metalRetainResidency(*entry, destinationBuffer);
        for (const auto& region : regions)
        {
            [entry->computeEncoder copyFromBuffer:sourceBuffer
                                     sourceOffset:static_cast<NSUInteger>(region.sourceOffset)
                                         toBuffer:destinationBuffer
                                destinationOffset:static_cast<NSUInteger>(region.destinationOffset)
                                             size:static_cast<NSUInteger>(region.size)];
        }
        return {};
    }

    namespace
    {

        struct MetalBufferTextureCopyResources final
        {
            std::uint64_t deviceToken = 0;
            __strong id<MTLBuffer> buffer = nil;
            __strong id<MTLTexture> texture = nil;
            std::uint64_t bufferSize = 0;
            TextureDesc textureDesc{};
            std::shared_ptr<MetalBufferOwner> bufferOwner;
            std::shared_ptr<MetalTextureOwner> textureOwner;
            std::array<MetalWaitedDrawable, 1> drawables{};
        };

        [[nodiscard]] MetalBufferTextureCopyResources metalBufferTextureCopyResources(const MetalDeviceNative& native,
                                                                                      const MetalBufferRecord& buffer,
                                                                                      const MetalTextureRecord& texture)
        {
            MetalBufferTextureCopyResources resources;
            resources.deviceToken = native.deviceToken;
            resources.buffer = buffer.buffer;
            resources.texture = texture.texture;
            resources.bufferSize = buffer.desc.size;
            resources.textureDesc = texture.desc;
            resources.bufferOwner = buffer.owner;
            resources.textureOwner = texture.owner;
            resources.drawables[0] = metalSwapchainImageReference(native, texture);
            return resources;
        }

        [[nodiscard]] Result<MetalBufferTextureCopyResources> captureMetalBufferToTextureResources(
            rachis::detail::TimedMutex& deviceMutex,
            const std::shared_ptr<MetalDeviceNative>& native,
            BufferHandle source,
            TextureHandle destination)
        {
            std::scoped_lock const lock(deviceMutex);
            if (native == nullptr)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidState, "Metal copyBufferToTextureRegions requires an active device."));
            }
            auto* buffer = native->buffers.find(source);
            auto* texture = native->textures.find(destination);
            if (buffer == nullptr || buffer->buffer == nil)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "A buffer-to-texture copy references an invalid or stale buffer."));
            }
            if (texture == nullptr || texture->texture == nil)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "A buffer-to-texture copy references an invalid or stale texture."));
            }
            if (!hasAny(buffer->desc.usage, BufferUsage::copySource) ||
                !hasAny(texture->desc.usage, TextureUsage::copyDestination))
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "A buffer-to-texture copy requires copy-source and copy-destination usages."));
            }
            return metalBufferTextureCopyResources(*native, *buffer, *texture);
        }

        [[nodiscard]] Result<MetalCommandBuffer*> metalRecordingCommand(detail::BackendCommandBuffer& commandBuffer,
                                                                        std::uint64_t deviceToken)
        {
            auto* native = asMetalCommand(commandBuffer, deviceToken);
            if (native == nullptr)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "The command buffer belongs to another device or backend."));
            }
            auto ready = metalRecordingReady(*native);
            if (!ready)
            {
                return std::unexpected(std::move(ready.error()));
            }
            return native;
        }

        [[nodiscard]] Result<MetalResolvedTransfer> resolveMetalBufferTextureRegion(
            const MetalCommandBuffer& native,
            const TextureDesc& textureDesc,
            const BufferTextureCopyRegion& region,
            std::uint64_t bufferSize)
        {
            const auto alignments = metalTransferAlignments(native.owner() != nullptr ? native.owner()->device : nil,
                                                            textureDesc.format,
                                                            region.textureSubresource.aspect);
            auto resolved = resolveMetalTransfer(textureDesc, region, alignments.first, alignments.second);
            if (!resolved)
            {
                return std::unexpected(std::move(resolved.error()));
            }
            if (resolved->layout.requiredBufferSize > bufferSize)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                           "The buffer-texture copy footprint exceeds the buffer."));
            }
            return resolved;
        }

        [[nodiscard]] MetalTextureBox metalTextureBox(const TextureSubresource& subresource,
                                                      const Origin3D& origin,
                                                      const Extent3D& extent)
        {
            return MetalTextureBox{
                .mipLevel = subresource.mipLevel,
                .arrayLayer = subresource.arrayLayer,
                .aspect = subresource.aspect,
                .origin = origin,
                .extent = extent,
            };
        }

        [[nodiscard]] bool metalTextureBoxOverlapsAny(const MetalTextureBox& box,
                                                      const std::vector<MetalTextureBox>& boxes)
        {
            for (const auto& previous : boxes)
            {
                if (metalTextureBoxesOverlap(box, previous))
                {
                    return true;
                }
            }
            return false;
        }

        [[nodiscard]] Result<std::vector<MetalResolvedTransfer>> resolveMetalBufferToTextureCopies(
            const MetalCommandBuffer& native,
            const MetalBufferTextureCopyResources& resources,
            std::span<const BufferTextureCopyRegion> regions)
        {
            std::vector<MetalResolvedTransfer> copies;
            std::vector<MetalTextureBox> destinations;
            copies.reserve(regions.size());
            destinations.reserve(regions.size());
            for (const auto& region : regions)
            {
                auto resolved =
                    resolveMetalBufferTextureRegion(native, resources.textureDesc, region, resources.bufferSize);
                if (!resolved)
                {
                    return std::unexpected(std::move(resolved.error()));
                }
                const auto box = metalTextureBox(region.textureSubresource, region.textureOrigin, region.extent);
                if (metalTextureBoxOverlapsAny(box, destinations))
                {
                    return std::unexpected(
                        metalSelectionError(ErrorCode::invalidArgument,
                                            "A buffer-to-texture batch cannot write overlapping destination texels."));
                }
                copies.push_back(*resolved);
                destinations.push_back(box);
            }
            return copies;
        }

        [[nodiscard]] Status retainMetalTransferOwners(MetalCommandBuffer& native,
                                                       std::shared_ptr<void> sourceOwner,
                                                       std::shared_ptr<void> destinationOwner)
        {
            auto retainedSource = retainMetalApplicationResource(native, std::move(sourceOwner));
            if (!retainedSource)
            {
                return retainedSource;
            }
            auto retainedDestination = retainMetalApplicationResource(native, std::move(destinationOwner));
            if (!retainedDestination)
            {
                return retainedDestination;
            }
            return {};
        }

        [[nodiscard]] Status encodeMetalBufferToTextureCopies(MetalCommandBuffer& native,
                                                              const MetalBufferTextureCopyResources& resources,
                                                              std::span<const BufferTextureCopyRegion> regions,
                                                              const std::vector<MetalResolvedTransfer>& copies)
        {
            const std::scoped_lock lock(native.owner()->mutex);
            if (auto reserved = metalReserveWaitedDrawableTracking(native.entry(), resources.drawables); !reserved)
            {
                return reserved;
            }
            auto prepared = native.ensureCopyEncoder();
            if (!prepared)
            {
                return prepared;
            }
            auto* entry = native.entry();
            metalRetainResidency(*entry, resources.buffer);
            metalRetainResidency(*entry, resources.texture);
            for (std::size_t index = 0; index < copies.size(); ++index)
            {
                const auto& copy = copies[index];
                const auto bytesPerImage =
                    copy.size.depth == 1 ? 0 : static_cast<NSUInteger>(copy.layout.bytesPerImage);
                [entry->computeEncoder copyFromBuffer:resources.buffer
                                         sourceOffset:static_cast<NSUInteger>(regions[index].bufferOffset)
                                    sourceBytesPerRow:copy.layout.bytesPerRow
                                  sourceBytesPerImage:bytesPerImage
                                           sourceSize:copy.size
                                            toTexture:resources.texture
                                     destinationSlice:copy.slice
                                     destinationLevel:copy.level
                                    destinationOrigin:copy.origin
                                              options:copy.options];
            }
            metalMergeWaitedDrawables(entry, resources.drawables);
            return {};
        }

    } // namespace

    Status MetalDevice::copyBufferToTextureRegions(detail::BackendCommandBuffer& commandBuffer,
                                                   BufferHandle source,
                                                   TextureHandle destination,
                                                   std::span<const BufferTextureCopyRegion> regions)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (regions.empty())
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "A buffer-to-texture batch must contain at least one region."));
        }
        auto resources = captureMetalBufferToTextureResources(mutex_, native_, source, destination);
        if (!resources)
        {
            return std::unexpected(std::move(resources.error()));
        }
        auto recording = metalRecordingCommand(commandBuffer, resources->deviceToken);
        if (!recording)
        {
            return std::unexpected(std::move(recording.error()));
        }
        auto& native = **recording;
        auto copies = resolveMetalBufferToTextureCopies(native, *resources, regions);
        if (!copies)
        {
            return std::unexpected(std::move(copies.error()));
        }
        if (auto retained = retainMetalTransferOwners(
                native, std::move(resources->bufferOwner), std::move(resources->textureOwner));
            !retained)
        {
            return retained;
        }
        return encodeMetalBufferToTextureCopies(native, *resources, regions, *copies);
    }

    namespace
    {

        [[nodiscard]] Result<MetalBufferTextureCopyResources> captureMetalTextureToBufferResources(
            rachis::detail::TimedMutex& deviceMutex,
            const std::shared_ptr<MetalDeviceNative>& native,
            TextureHandle source,
            BufferHandle destination)
        {
            std::scoped_lock const lock(deviceMutex);
            if (native == nullptr)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidState, "Metal copyTextureToBufferRegions requires an active device."));
            }
            auto* texture = native->textures.find(source);
            auto* buffer = native->buffers.find(destination);
            if (texture == nullptr || texture->texture == nil)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "A texture-to-buffer copy references an invalid or stale texture."));
            }
            if (buffer == nullptr || buffer->buffer == nil)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "A texture-to-buffer copy references an invalid or stale buffer."));
            }
            if (!hasAny(texture->desc.usage, TextureUsage::copySource) ||
                !hasAny(buffer->desc.usage, BufferUsage::copyDestination))
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "A texture-to-buffer copy requires copy-source and copy-destination usages."));
            }
            return metalBufferTextureCopyResources(*native, *buffer, *texture);
        }

        [[nodiscard]] Result<std::vector<MetalResolvedTransfer>> resolveMetalTextureToBufferCopies(
            const MetalCommandBuffer& native,
            const MetalBufferTextureCopyResources& resources,
            std::span<const BufferTextureCopyRegion> regions)
        {
            std::vector<MetalResolvedTransfer> copies;
            copies.reserve(regions.size());
            for (const auto& region : regions)
            {
                auto resolved =
                    resolveMetalBufferTextureRegion(native, resources.textureDesc, region, resources.bufferSize);
                if (!resolved)
                {
                    return std::unexpected(std::move(resolved.error()));
                }
                copies.push_back(*resolved);
            }
            return copies;
        }

        [[nodiscard]] Status encodeMetalTextureToBufferCopies(MetalCommandBuffer& native,
                                                              const MetalBufferTextureCopyResources& resources,
                                                              std::span<const BufferTextureCopyRegion> regions,
                                                              const std::vector<MetalResolvedTransfer>& copies)
        {
            const std::scoped_lock lock(native.owner()->mutex);
            if (auto reserved = metalReserveWaitedDrawableTracking(native.entry(), resources.drawables); !reserved)
            {
                return reserved;
            }
            auto prepared = native.ensureCopyEncoder();
            if (!prepared)
            {
                return prepared;
            }
            auto* entry = native.entry();
            metalRetainResidency(*entry, resources.texture);
            metalRetainResidency(*entry, resources.buffer);
            for (std::size_t index = 0; index < copies.size(); ++index)
            {
                const auto& copy = copies[index];
                const auto bytesPerImage =
                    copy.size.depth == 1 ? 0 : static_cast<NSUInteger>(copy.layout.bytesPerImage);
                [entry->computeEncoder copyFromTexture:resources.texture
                                           sourceSlice:copy.slice
                                           sourceLevel:copy.level
                                          sourceOrigin:copy.origin
                                            sourceSize:copy.size
                                              toBuffer:resources.buffer
                                     destinationOffset:static_cast<NSUInteger>(regions[index].bufferOffset)
                                destinationBytesPerRow:copy.layout.bytesPerRow
                              destinationBytesPerImage:bytesPerImage
                                               options:copy.options];
            }
            metalMergeWaitedDrawables(entry, resources.drawables);
            return {};
        }

    } // namespace

    Status MetalDevice::copyTextureToBufferRegions(detail::BackendCommandBuffer& commandBuffer,
                                                   TextureHandle source,
                                                   BufferHandle destination,
                                                   std::span<const BufferTextureCopyRegion> regions)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (regions.empty())
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "A texture-to-buffer batch must contain at least one region."));
        }
        auto resources = captureMetalTextureToBufferResources(mutex_, native_, source, destination);
        if (!resources)
        {
            return std::unexpected(std::move(resources.error()));
        }
        auto recording = metalRecordingCommand(commandBuffer, resources->deviceToken);
        if (!recording)
        {
            return std::unexpected(std::move(recording.error()));
        }
        auto& native = **recording;
        auto copies = resolveMetalTextureToBufferCopies(native, *resources, regions);
        if (!copies)
        {
            return std::unexpected(std::move(copies.error()));
        }
        if (auto retained = retainMetalTransferOwners(
                native, std::move(resources->textureOwner), std::move(resources->bufferOwner));
            !retained)
        {
            return retained;
        }
        return encodeMetalTextureToBufferCopies(native, *resources, regions, *copies);
    }

    namespace
    {

        struct MetalTexturePair final
        {
            std::uint64_t deviceToken = 0;
            __strong id<MTLTexture> sourceTexture = nil;
            __strong id<MTLTexture> destinationTexture = nil;
            TextureDesc sourceDesc{};
            TextureDesc destinationDesc{};
            std::shared_ptr<MetalTextureOwner> sourceOwner;
            std::shared_ptr<MetalTextureOwner> destinationOwner;
            std::array<MetalWaitedDrawable, 2> drawables{};
        };

        [[nodiscard]] MetalTexturePair metalTexturePair(const MetalDeviceNative& native,
                                                        const MetalTextureRecord& sourceRecord,
                                                        const MetalTextureRecord& destinationRecord)
        {
            MetalTexturePair textures;
            textures.deviceToken = native.deviceToken;
            textures.sourceTexture = sourceRecord.texture;
            textures.destinationTexture = destinationRecord.texture;
            textures.sourceDesc = sourceRecord.desc;
            textures.destinationDesc = destinationRecord.desc;
            textures.sourceOwner = sourceRecord.owner;
            textures.destinationOwner = destinationRecord.owner;
            textures.drawables[0] = metalSwapchainImageReference(native, sourceRecord);
            textures.drawables[1] = metalSwapchainImageReference(native, destinationRecord);
            return textures;
        }

        [[nodiscard]] Result<MetalTexturePair> captureMetalTextureCopyPair(
            rachis::detail::TimedMutex& deviceMutex,
            const std::shared_ptr<MetalDeviceNative>& native,
            TextureHandle source,
            TextureHandle destination)
        {
            std::scoped_lock const lock(deviceMutex);
            if (native == nullptr)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                           "Metal copyTextureRegions requires an active device."));
            }
            auto* sourceRecord = native->textures.find(source);
            auto* destinationRecord = native->textures.find(destination);
            if (sourceRecord == nullptr || sourceRecord->texture == nil)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "A texture copy references an invalid or stale source texture."));
            }
            if (destinationRecord == nullptr || destinationRecord->texture == nil)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "A texture copy references an invalid or stale destination texture."));
            }
            if (!hasAny(sourceRecord->desc.usage, TextureUsage::copySource) ||
                !hasAny(destinationRecord->desc.usage, TextureUsage::copyDestination))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "A texture copy requires copy-source and copy-destination usages."));
            }
            if (sourceRecord->desc.format != destinationRecord->desc.format ||
                sourceRecord->desc.sampleCount != destinationRecord->desc.sampleCount)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "Texture copies require identical formats and matching sample counts."));
            }
            return metalTexturePair(*native, *sourceRecord, *destinationRecord);
        }

        [[nodiscard]] Status validateMetalTextureCopyRegion(const MetalTexturePair& textures,
                                                            bool sameTexture,
                                                            const TextureCopyRegion& region)
        {
            if (region.sourceSubresource.aspect != region.destinationSubresource.aspect)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "Texture copies require matching source and destination aspects."));
            }
            if (auto valid = validateMetalImageCopyRegion(
                    textures.sourceDesc, region.sourceSubresource, region.sourceOrigin, region.extent);
                !valid)
            {
                return valid;
            }
            if (auto valid = validateMetalImageCopyRegion(
                    textures.destinationDesc, region.destinationSubresource, region.destinationOrigin, region.extent);
                !valid)
            {
                return valid;
            }
            if (sameTexture && region.sourceSubresource.mipLevel == region.destinationSubresource.mipLevel &&
                region.sourceSubresource.arrayLayer == region.destinationSubresource.arrayLayer &&
                region.sourceSubresource.aspect == region.destinationSubresource.aspect)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "A texture copy cannot use the same subresource as both source and destination."));
            }
            return {};
        }

        [[nodiscard]] Status validateMetalSameTextureCopyAccess(std::span<const TextureCopyRegion> regions,
                                                                const std::vector<MetalTextureBox>& destinations)
        {
            for (const auto& region : regions)
            {
                const auto sourceBox = metalTextureBox(region.sourceSubresource, region.sourceOrigin, region.extent);
                if (metalTextureBoxOverlapsAny(sourceBox, destinations))
                {
                    return std::unexpected(metalSelectionError(
                        ErrorCode::invalidArgument,
                        "A same-texture copy batch cannot access one texel as both source and destination."));
                }
            }
            return {};
        }

        [[nodiscard]] Status validateMetalTextureCopyRegions(const MetalTexturePair& textures,
                                                             bool sameTexture,
                                                             std::span<const TextureCopyRegion> regions)
        {
            std::vector<MetalTextureBox> destinations;
            destinations.reserve(regions.size());
            for (const auto& region : regions)
            {
                if (auto valid = validateMetalTextureCopyRegion(textures, sameTexture, region); !valid)
                {
                    return valid;
                }
                const auto destinationBox =
                    metalTextureBox(region.destinationSubresource, region.destinationOrigin, region.extent);
                if (metalTextureBoxOverlapsAny(destinationBox, destinations))
                {
                    return std::unexpected(
                        metalSelectionError(ErrorCode::invalidArgument,
                                            "A texture-copy batch cannot write overlapping destination texels."));
                }
                destinations.push_back(destinationBox);
            }
            if (!sameTexture)
            {
                return {};
            }
            return validateMetalSameTextureCopyAccess(regions, destinations);
        }

        [[nodiscard]] Status encodeMetalTextureCopies(MetalCommandBuffer& native,
                                                      const MetalTexturePair& textures,
                                                      std::span<const TextureCopyRegion> regions)
        {
            const std::scoped_lock lock(native.owner()->mutex);
            if (auto reserved = metalReserveWaitedDrawableTracking(native.entry(), textures.drawables); !reserved)
            {
                return reserved;
            }
            auto prepared = native.ensureCopyEncoder();
            if (!prepared)
            {
                return prepared;
            }
            auto* entry = native.entry();
            metalRetainResidency(*entry, textures.sourceTexture);
            metalRetainResidency(*entry, textures.destinationTexture);
            for (const auto& region : regions)
            {
                const auto sourceSlice = textures.sourceDesc.dimension == TextureDimension::texture3D
                                             ? 0
                                             : region.sourceSubresource.arrayLayer;
                const auto destinationSlice = textures.destinationDesc.dimension == TextureDimension::texture3D
                                                  ? 0
                                                  : region.destinationSubresource.arrayLayer;
                [entry->computeEncoder
                      copyFromTexture:textures.sourceTexture
                          sourceSlice:sourceSlice
                          sourceLevel:region.sourceSubresource.mipLevel
                         sourceOrigin:MTLOriginMake(region.sourceOrigin.x, region.sourceOrigin.y, region.sourceOrigin.z)
                           sourceSize:MTLSizeMake(region.extent.width, region.extent.height, region.extent.depth)
                            toTexture:textures.destinationTexture
                     destinationSlice:destinationSlice
                     destinationLevel:region.destinationSubresource.mipLevel
                    destinationOrigin:MTLOriginMake(region.destinationOrigin.x,
                                                    region.destinationOrigin.y,
                                                    region.destinationOrigin.z)];
            }
            metalMergeWaitedDrawables(entry, textures.drawables);
            return {};
        }

    } // namespace

    Status MetalDevice::copyTextureRegions(detail::BackendCommandBuffer& commandBuffer,
                                           TextureHandle source,
                                           TextureHandle destination,
                                           std::span<const TextureCopyRegion> regions)
    {
        const MetalAutoreleasePool autoreleasePool;
        if (regions.empty())
        {
            return std::unexpected(metalSelectionError(ErrorCode::invalidArgument,
                                                       "A texture-copy batch must contain at least one region."));
        }
        auto textures = captureMetalTextureCopyPair(mutex_, native_, source, destination);
        if (!textures)
        {
            return std::unexpected(std::move(textures.error()));
        }
        auto recording = metalRecordingCommand(commandBuffer, textures->deviceToken);
        if (!recording)
        {
            return std::unexpected(std::move(recording.error()));
        }
        auto& native = **recording;
        if (auto valid = validateMetalTextureCopyRegions(*textures, source == destination, regions); !valid)
        {
            return valid;
        }
        if (auto retained = retainMetalTransferOwners(
                native, std::move(textures->sourceOwner), std::move(textures->destinationOwner));
            !retained)
        {
            return retained;
        }
        return encodeMetalTextureCopies(native, *textures, regions);
    }

    namespace
    {

        struct MetalResolveTextures final
        {
            MetalTexturePair pair;
            FormatProperties formatProperties{};
        };

        [[nodiscard]] Result<MetalResolveTextures> captureMetalResolveTextures(
            rachis::detail::TimedMutex& deviceMutex,
            const std::shared_ptr<MetalDeviceNative>& native,
            const TextureResolve& resolve)
        {
            std::scoped_lock const lock(deviceMutex);
            if (native == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal resolveTexture requires an active device."));
            }
            auto* sourceRecord = native->textures.find(resolve.source);
            auto* destinationRecord = native->textures.find(resolve.destination);
            if (sourceRecord == nullptr || sourceRecord->texture == nil)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "A texture resolve references an invalid or stale source texture."));
            }
            if (destinationRecord == nullptr || destinationRecord->texture == nil)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidArgument,
                                        "A texture resolve references an invalid or stale destination texture."));
            }
            MetalResolveTextures captured;
            captured.pair = metalTexturePair(*native, *sourceRecord, *destinationRecord);
            captured.formatProperties = probeFormat(native->device, captured.pair.sourceDesc.format);
            return captured;
        }

        [[nodiscard]] Status validateMetalResolveTextures(const MetalTexturePair& textures,
                                                          const TextureResolve& resolve,
                                                          bool depthStencilFormat)
        {
            if (textures.sourceDesc.format != textures.destinationDesc.format ||
                resolve.sourceSubresource.aspect != resolve.destinationSubresource.aspect)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "A texture resolve requires identical formats and matching aspects."));
            }
            if (textures.sourceDesc.sampleCount == 1 || textures.destinationDesc.sampleCount != 1 ||
                textures.sourceDesc.dimension != TextureDimension::texture2D ||
                textures.destinationDesc.dimension != TextureDimension::texture2D)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "A texture resolve requires a multisampled 2D source and a single-sampled 2D destination."));
            }
            const auto requiredUsage = depthStencilFormat ? TextureUsage::depthStencil : TextureUsage::renderTarget;
            if (!hasAny(textures.sourceDesc.usage, requiredUsage) ||
                !hasAny(textures.destinationDesc.usage, requiredUsage))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "Texture resolve resources must both have matching attachment usage."));
            }
            return {};
        }

        [[nodiscard]] Status validateMetalResolveMode(const TextureResolve& resolve,
                                                      bool depthStencilFormat,
                                                      const FormatProperties& formatProperties)
        {
            const bool depthResolve = resolve.sourceSubresource.aspect == TextureAspect::depth;
            const bool stencilResolve = resolve.sourceSubresource.aspect == TextureAspect::stencil;
            if ((depthStencilFormat && ((!depthResolve && !stencilResolve) || resolve.mode == ResolveMode::average)) ||
                (!depthStencilFormat &&
                 (resolve.sourceSubresource.aspect != TextureAspect::color || resolve.mode != ResolveMode::average)))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "Color resolves use average mode; depth and stencil resolves use minimum or maximum mode."));
            }
            if (stencilResolve)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::unsupported, "Metal stencil-plane min/max resolve is unsupported."));
            }
            if (!hasAny(formatProperties.features, FormatFeature::multisampleResolve))
            {
                return std::unexpected(metalSelectionError(ErrorCode::unsupported,
                                                           "The texture format does not support multisample resolve."));
            }
            return {};
        }

        [[nodiscard]] Result<Extent3D> validateMetalResolveExtents(const MetalTexturePair& textures,
                                                                   const TextureResolve& resolve)
        {
            const auto sourceMip = metalSubresourceExtent(textures.sourceDesc, resolve.sourceSubresource.mipLevel);
            const auto destinationMip =
                metalSubresourceExtent(textures.destinationDesc, resolve.destinationSubresource.mipLevel);
            const auto sourceLayers = textures.sourceDesc.depthOrArrayLayers;
            const auto destinationLayers = textures.destinationDesc.depthOrArrayLayers;
            if (resolve.sourceSubresource.mipLevel >= textures.sourceDesc.mipLevels ||
                resolve.destinationSubresource.mipLevel >= textures.destinationDesc.mipLevels ||
                resolve.sourceSubresource.arrayLayer >= sourceLayers ||
                resolve.destinationSubresource.arrayLayer >= destinationLayers ||
                sourceMip.width != destinationMip.width || sourceMip.height != destinationMip.height)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "A texture resolve requires matching source and destination subresource extents."));
            }
            return sourceMip;
        }

        void configureMetalResolveAttachment(MTL4RenderPassDescriptor* pass,
                                             const MetalTexturePair& textures,
                                             const TextureResolve& resolve,
                                             bool depthStencilFormat)
        {
            if (!depthStencilFormat)
            {
                auto* const color = pass.colorAttachments[0];
                color.texture = textures.sourceTexture;
                color.level = resolve.sourceSubresource.mipLevel;
                color.slice = resolve.sourceSubresource.arrayLayer;
                color.resolveTexture = textures.destinationTexture;
                color.resolveLevel = resolve.destinationSubresource.mipLevel;
                color.resolveSlice = resolve.destinationSubresource.arrayLayer;
                color.loadAction = MTLLoadActionLoad;
                color.storeAction = MTLStoreActionStoreAndMultisampleResolve;
                return;
            }
            auto* const depth = pass.depthAttachment;
            depth.texture = textures.sourceTexture;
            depth.level = resolve.sourceSubresource.mipLevel;
            depth.slice = resolve.sourceSubresource.arrayLayer;
            depth.resolveTexture = textures.destinationTexture;
            depth.resolveLevel = resolve.destinationSubresource.mipLevel;
            depth.resolveSlice = resolve.destinationSubresource.arrayLayer;
            depth.loadAction = MTLLoadActionLoad;
            depth.storeAction = MTLStoreActionStoreAndMultisampleResolve;
            depth.depthResolveFilter = resolve.mode == ResolveMode::maximum ? MTLMultisampleDepthResolveFilterMax
                                                                            : MTLMultisampleDepthResolveFilterMin;
        }

        [[nodiscard]] Status encodeMetalResolvePass(MetalCommandBuffer& native,
                                                    const MetalTexturePair& textures,
                                                    const TextureResolve& resolve,
                                                    bool depthStencilFormat,
                                                    const Extent3D& sourceMip)
        {
            const std::scoped_lock lock(native.owner()->mutex);
            auto* entry = native.entry();
            if (auto reserved = metalReserveWaitedDrawableTracking(entry, textures.drawables); !reserved)
            {
                return reserved;
            }
            native.endEncoders();
            MTL4RenderPassDescriptor* const pass = [MTL4RenderPassDescriptor new];
            pass.renderTargetWidth = sourceMip.width;
            pass.renderTargetHeight = sourceMip.height;
            configureMetalResolveAttachment(pass, textures, resolve, depthStencilFormat);
            metalRetainResidency(*entry, textures.sourceTexture);
            metalRetainResidency(*entry, textures.destinationTexture);
            entry->renderEncoder = [entry->command renderCommandEncoderWithDescriptor:pass];
            if (entry->renderEncoder == nil)
            {
                return std::unexpected(metalSelectionError(ErrorCode::nativeFailure,
                                                           "Metal could not create a resolve render command encoder."));
            }
            entry->encoder = MetalEncoderKind::render;
            flushMetalRenderPendingBarriers(*entry);
            native.endEncoders();
            metalMergeWaitedDrawables(entry, textures.drawables);
            return {};
        }

    } // namespace

    Status MetalDevice::resolveTexture(detail::BackendCommandBuffer& commandBuffer, const TextureResolve& resolve)
    {
        const MetalAutoreleasePool autoreleasePool;
        auto captured = captureMetalResolveTextures(mutex_, native_, resolve);
        if (!captured)
        {
            return std::unexpected(std::move(captured.error()));
        }
        auto& textures = captured->pair;
        auto recording = metalRecordingCommand(commandBuffer, textures.deviceToken);
        if (!recording)
        {
            return std::unexpected(std::move(recording.error()));
        }
        auto& native = **recording;
        if (native.queueType() != QueueType::graphics || native.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::invalidState,
                                    "A texture resolve requires a graphics command buffer outside a rendering scope."));
        }
        const bool depthStencilFormat = formatIsDepth(textures.sourceDesc.format);
        if (auto valid = validateMetalResolveTextures(textures, resolve, depthStencilFormat); !valid)
        {
            return valid;
        }
        if (auto valid = validateMetalResolveMode(resolve, depthStencilFormat, captured->formatProperties); !valid)
        {
            return valid;
        }
        auto sourceMip = validateMetalResolveExtents(textures, resolve);
        if (!sourceMip)
        {
            return std::unexpected(std::move(sourceMip.error()));
        }
        if (auto retained = retainMetalTransferOwners(
                native, std::move(textures.sourceOwner), std::move(textures.destinationOwner));
            !retained)
        {
            return retained;
        }
        return encodeMetalResolvePass(native, textures, resolve, depthStencilFormat, *sourceMip);
    }

    void MetalDevice::destroyMipmapPipelines() noexcept
    {
        auto destroyHandle = [this](auto handle, auto destroy) noexcept {
            if (handle.isValid())
            {
                static_cast<void>((this->*destroy)(handle));
            }
        };
        for (auto pipeline : mipmapPipelines_)
        {
            destroyHandle(pipeline, &MetalDevice::destroyPipeline);
        }
        for (auto shader : mipmapShaders_)
        {
            destroyHandle(shader, &MetalDevice::destroyShaderModule);
        }
        destroyHandle(mipmapPipelineLayout_, &MetalDevice::destroyPipelineLayout);
        for (auto layout : mipmapBindingLayouts_)
        {
            destroyHandle(layout, &MetalDevice::destroyBindingLayout);
        }
        destroyHandle(mipmapSampler_, &MetalDevice::destroySampler);
        {
            std::scoped_lock const lock(mutex_);
            if (native_ != nullptr)
            {
                native_->mipmapEncodeOn = nil;
                native_->mipmapEncodeOff = nil;
            }
        }
        mipmapPipelines_ = {};
        mipmapShaders_ = {};
        mipmapPipelineLayout_ = {};
        mipmapBindingLayouts_ = {};
        mipmapSampler_ = {};
        mipmapsReady_ = false;
    }

    namespace
    {

        struct MetalMipmapConstantBuffers final
        {
            __strong id<MTLBuffer> off = nil;
            __strong id<MTLBuffer> on = nil;
        };

        [[nodiscard]] id<MTLBuffer> newMetalMipmapConstantBuffer(id<MTLDevice> device,
                                                                 std::uint32_t encodeSrgb,
                                                                 NSString* label)
        {
            constexpr std::uint64_t uniformBytes = 256;
            std::array<std::uint32_t, uniformBytes / sizeof(std::uint32_t)> words{};
            words[0] = encodeSrgb;
            id<MTLBuffer> const buffer =
                [device newBufferWithBytes:words.data()
                                    length:sizeof(words)
                                   options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache |
                                           MTLResourceHazardTrackingModeTracked];
            if (buffer != nil)
            {
                buffer.label = label;
            }
            return buffer;
        }

        [[nodiscard]] Result<MetalMipmapConstantBuffers> createMetalMipmapConstantBuffers(id<MTLDevice> device)
        {
            MetalMipmapConstantBuffers constants;
            constants.off = newMetalMipmapConstantBuffer(device, 0, @"Metal mipmap encode-off constants");
            constants.on = newMetalMipmapConstantBuffer(device, 1, @"Metal mipmap encode-on constants");
            if (constants.off == nil || constants.on == nil)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::unsupported, "Mipmap generation constant buffers are unavailable."));
            }
            return constants;
        }

        void destroyMetalMipmapBindingLayouts(MetalDevice& device,
                                              const std::array<BindingLayoutHandle, 3>& bindingLayouts)
        {
            for (auto layout : bindingLayouts)
            {
                if (layout)
                {
                    static_cast<void>(device.destroyBindingLayout(layout));
                }
            }
        }

        [[nodiscard]] Result<std::array<BindingLayoutHandle, 3>> createMetalMipmapBindingLayouts(MetalDevice& device)
        {
            const std::array bufferEntries{
                BindingLayoutEntry{0, BindingType::uniformBuffer, ShaderStage::compute},
            };
            const std::array textureEntries{
                BindingLayoutEntry{0, BindingType::sampledTexture, ShaderStage::compute},
                BindingLayoutEntry{1, BindingType::storageTexture, ShaderStage::compute},
            };
            const std::array samplerEntries{
                BindingLayoutEntry{0, BindingType::sampler, ShaderStage::compute},
            };
            const std::array<std::span<const BindingLayoutEntry>, 3> layoutEntries{
                bufferEntries,
                textureEntries,
                samplerEntries,
            };
            std::array<BindingLayoutHandle, 3> bindingLayouts{};
            for (std::size_t index = 0; index < bindingLayouts.size(); ++index)
            {
                auto layout = device.createBindingLayout(BindingLayoutDesc{
                    .entries = layoutEntries[index],
                    .label = "Metal mipmap generation layout",
                });
                if (!layout)
                {
                    destroyMetalMipmapBindingLayouts(device, bindingLayouts);
                    return std::unexpected(std::move(layout.error()));
                }
                bindingLayouts[index] = *layout;
            }
            return bindingLayouts;
        }

        struct MetalMipmapPrograms final
        {
            std::array<ShaderModuleHandle, 2> modules{};
            std::array<PipelineHandle, 2> compute{};
        };

        void destroyFirstMetalMipmapPrograms(MetalDevice& device,
                                             const MetalMipmapPrograms& programs,
                                             std::size_t count)
        {
            for (std::size_t created = 0; created < count; ++created)
            {
                static_cast<void>(device.destroyPipeline(programs.compute[created]));
                static_cast<void>(device.destroyShaderModule(programs.modules[created]));
            }
        }

        [[nodiscard]] Result<MetalMipmapPrograms> createMetalMipmapPrograms(MetalDevice& device,
                                                                            PipelineLayoutHandle pipelineLayout)
        {
            static constexpr std::array bindings{
                ShaderBindingRequirement{0, 0, ShaderResourceType::uniformBuffer, 1},
                ShaderBindingRequirement{1, 0, ShaderResourceType::sampledTexture, 1},
                ShaderBindingRequirement{1, 1, ShaderResourceType::storageTexture, 1},
                ShaderBindingRequirement{2, 0, ShaderResourceType::sampler, 1},
            };
            const std::array programs{
                std::pair{metalEmbeddedPnms(g_rachisGenerateMipmaps2DPnms, g_rachisGenerateMipmaps2DPnmsSize),
                          ShaderWorkgroupSize{8, 8, 1}},
                std::pair{metalEmbeddedPnms(g_rachisGenerateMipmaps3DPnms, g_rachisGenerateMipmaps3DPnmsSize),
                          ShaderWorkgroupSize{4, 4, 4}},
            };
            MetalMipmapPrograms compiled;
            for (std::size_t index = 0; index < programs.size(); ++index)
            {
                auto shader = device.createShaderModule(ShaderModuleDesc{
                    .stage = ShaderStage::compute,
                    .bytecode = programs[index].first,
                    .entryPoint = "main",
                    .label = "Metal mipmap generation",
                    .shaderInterface =
                        ShaderInterfaceDesc{
                            .bindings = bindings,
                            .workgroupSize = programs[index].second,
                        },
                });
                if (!shader)
                {
                    destroyFirstMetalMipmapPrograms(device, compiled, index);
                    return std::unexpected(std::move(shader.error()));
                }
                auto pipeline = device.createComputePipeline(ComputePipelineDesc{
                    .layout = pipelineLayout,
                    .computeShader = *shader,
                    .label = "Metal mipmap generation pipeline",
                });
                if (!pipeline)
                {
                    static_cast<void>(device.destroyShaderModule(*shader));
                    destroyFirstMetalMipmapPrograms(device, compiled, index);
                    return std::unexpected(std::move(pipeline.error()));
                }
                compiled.modules[index] = *shader;
                compiled.compute[index] = *pipeline;
            }
            return compiled;
        }

        struct MetalMipmapPipelineObjects final
        {
            std::array<BindingLayoutHandle, 3> bindingLayouts{};
            PipelineLayoutHandle pipelineLayout{};
            MetalMipmapPrograms programs;
        };

        [[nodiscard]] Result<MetalMipmapPipelineObjects> createMetalMipmapPipelineObjects(MetalDevice& device)
        {
            auto bindingLayouts = createMetalMipmapBindingLayouts(device);
            if (!bindingLayouts)
            {
                return std::unexpected(std::move(bindingLayouts.error()));
            }
            auto pipelineLayout = device.createPipelineLayout(PipelineLayoutDesc{
                .bindingLayouts = *bindingLayouts,
                .label = "Metal mipmap generation pipeline layout",
            });
            if (!pipelineLayout)
            {
                destroyMetalMipmapBindingLayouts(device, *bindingLayouts);
                return std::unexpected(std::move(pipelineLayout.error()));
            }
            auto programs = createMetalMipmapPrograms(device, *pipelineLayout);
            if (!programs)
            {
                static_cast<void>(device.destroyPipelineLayout(*pipelineLayout));
                destroyMetalMipmapBindingLayouts(device, *bindingLayouts);
                return std::unexpected(std::move(programs.error()));
            }
            return MetalMipmapPipelineObjects{
                .bindingLayouts = *bindingLayouts,
                .pipelineLayout = *pipelineLayout,
                .programs = *programs,
            };
        }

        void destroyMetalMipmapPipelineObjects(MetalDevice& device, const MetalMipmapPipelineObjects& objects)
        {
            for (auto pipeline : objects.programs.compute)
            {
                static_cast<void>(device.destroyPipeline(pipeline));
            }
            for (auto shader : objects.programs.modules)
            {
                static_cast<void>(device.destroyShaderModule(shader));
            }
            static_cast<void>(device.destroyPipelineLayout(objects.pipelineLayout));
            destroyMetalMipmapBindingLayouts(device, objects.bindingLayouts);
        }

    } // namespace

    Status MetalDevice::ensureMipmapPipelines()
    {
        id<MTLDevice> device = nil;
        {
            std::scoped_lock const lock(mutex_);
            if (mipmapsReady_)
            {
                return {};
            }
            if (native_ == nullptr || native_->device == nil)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal generateMipmaps requires an active device."));
            }
            device = native_->device;
        }

        auto constants = createMetalMipmapConstantBuffers(device);
        if (!constants)
        {
            return std::unexpected(std::move(constants.error()));
        }
        auto sampler = createSampler(SamplerDesc{
            .minFilter = FilterMode::linear,
            .magFilter = FilterMode::linear,
            .mipFilter = MipmapFilterMode::nearest,
            .label = "Metal mipmap linear clamp",
        });
        if (!sampler)
        {
            return std::unexpected(std::move(sampler.error()));
        }
        auto objects = createMetalMipmapPipelineObjects(*this);
        if (!objects)
        {
            static_cast<void>(destroySampler(*sampler));
            return std::unexpected(std::move(objects.error()));
        }

        {
            std::scoped_lock const lock(mutex_);
            if (!mipmapsReady_)
            {
                native_->mipmapEncodeOff = constants->off;
                native_->mipmapEncodeOn = constants->on;
                mipmapSampler_ = *sampler;
                mipmapBindingLayouts_ = objects->bindingLayouts;
                mipmapPipelineLayout_ = objects->pipelineLayout;
                mipmapShaders_ = objects->programs.modules;
                mipmapPipelines_ = objects->programs.compute;
                mipmapsReady_ = true;
                return {};
            }
        }
        destroyMetalMipmapPipelineObjects(*this, *objects);
        static_cast<void>(destroySampler(*sampler));
        return {};
    }

    namespace
    {

        struct MetalMipmapTexture final
        {
            std::uint64_t deviceToken = 0;
            TextureDesc desc{};
            std::vector<Format> viewFormats;
            std::shared_ptr<MetalTextureOwner> owner;
            std::shared_ptr<MetalTextureLayoutTable> layouts;
        };

        [[nodiscard]] Result<MetalMipmapTexture> captureMetalMipmapTexture(
            rachis::detail::TimedMutex& deviceMutex,
            const std::shared_ptr<MetalDeviceNative>& native,
            TextureHandle texture)
        {
            std::scoped_lock const lock(deviceMutex);
            if (native == nullptr)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal generateMipmaps requires an active device."));
            }
            auto* record = native->textures.find(texture);
            if (record == nullptr || record->texture == nil)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "Mipmap generation references an invalid or stale texture."));
            }
            MetalMipmapTexture captured;
            captured.deviceToken = native->deviceToken;
            captured.desc = record->desc;
            captured.viewFormats = record->viewFormats;
            captured.owner = record->owner;
            captured.layouts = metalTextureLayoutTable(*record);
            return captured;
        }

        struct MetalMipmapRange final
        {
            std::uint32_t mipCount = 0;
            std::uint32_t layerCount = 0;
        };

        [[nodiscard]] Result<MetalMipmapRange> validateMetalMipmapGeneration(const MetalCommandBuffer& native,
                                                                             const TextureDesc& desc,
                                                                             const TextureMipmapGeneration& generation)
        {
            if ((native.queueType() != QueueType::graphics && native.queueType() != QueueType::compute) ||
                native.graphicsState().renderingOwner != nullptr)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidState,
                    "Mipmap generation requires a graphics or compute command buffer outside a rendering scope."));
            }
            if (desc.sampleCount != 1 || !hasAny(desc.usage, TextureUsage::sampled) ||
                !hasAny(desc.usage, TextureUsage::storage) || generation.subresources.aspect != TextureAspect::color)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "Mipmap generation requires a single-sampled color texture with sampled and storage usage."));
            }
            if (desc.dimension == TextureDimension::texture1D)
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::unsupported,
                    "Metal 1D textures have a single mip level, so mipmap generation is unsupported for them."));
            }
            auto mipCount = metalSelectedCount(generation.subresources.firstMipLevel,
                                               generation.subresources.mipLevelCount,
                                               desc.mipLevels,
                                               "The mipmap range");
            const auto availableLayerCount =
                desc.dimension == TextureDimension::texture3D ? 1U : desc.depthOrArrayLayers;
            auto layerCount = metalSelectedCount(generation.subresources.firstArrayLayer,
                                                 generation.subresources.arrayLayerCount,
                                                 availableLayerCount,
                                                 "The mipmap layer range");
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
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument, "Mipmap generation requires at least two selected mip levels."));
            }
            if (!metalLayoutCompatible(desc.usage, generation.baseLayout) ||
                !metalLayoutCompatible(desc.usage, generation.finalLayout))
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::invalidArgument,
                    "The mipmap seed or final layout is undefined or incompatible with the texture usage."));
            }
            return MetalMipmapRange{.mipCount = *mipCount, .layerCount = *layerCount};
        }

        struct MetalMipmapStorage final
        {
            bool encodeSrgb = false;
            Format storageFormat = Format::unknown;
        };

        [[nodiscard]] Result<MetalMipmapStorage> selectMetalMipmapStorage(const TextureDesc& desc,
                                                                          std::span<const Format> viewFormats)
        {
            const auto linearFormat = metalLinearViewFormat(desc.format);
            if (linearFormat == Format::unknown)
            {
                return MetalMipmapStorage{.encodeSrgb = false, .storageFormat = desc.format};
            }
            if (std::ranges::find(viewFormats, linearFormat) == viewFormats.end())
            {
                return std::unexpected(metalSelectionError(
                    ErrorCode::unsupported,
                    "sRGB mipmap generation requires the compatible linear format in TextureDesc::viewFormats."));
            }
            return MetalMipmapStorage{.encodeSrgb = true, .storageFormat = linearFormat};
        }

        struct MetalMipmapResources final
        {
            __strong id<MTLDevice> device = nil;
            __strong id<MTLBuffer> constants = nil;
            __strong id<MTLSamplerState> sampler = nil;
            std::shared_ptr<MetalPipelineRecord> pipeline;
            ShaderWorkgroupSize workgroup{1, 1, 1};
        };

        [[nodiscard]] Result<MetalMipmapResources> captureMetalMipmapResources(
            rachis::detail::TimedMutex& deviceMutex,
            const std::shared_ptr<MetalDeviceNative>& native,
            bool encodeSrgb,
            const SamplerHandle& samplerHandle,
            PipelineHandle pipelineHandle)
        {
            std::scoped_lock const lock(deviceMutex);
            if (native == nullptr || native->device == nil)
            {
                return std::unexpected(
                    metalSelectionError(ErrorCode::invalidState, "Metal generateMipmaps requires an active device."));
            }
            id<MTLBuffer> const constants = encodeSrgb ? native->mipmapEncodeOn : native->mipmapEncodeOff;
            auto* samplerRecord = native->samplers.find(samplerHandle);
            auto* pipeline = native->pipelines.find(pipelineHandle);
            if (constants == nil || samplerRecord == nullptr || samplerRecord->sampler == nil || pipeline == nullptr ||
                *pipeline == nullptr || (*pipeline)->compute == nil)
            {
                return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                           "Metal mipmap generation lost a helper pipeline resource."));
            }
            MetalMipmapResources resources;
            resources.device = native->device;
            resources.constants = constants;
            resources.sampler = samplerRecord->sampler;
            resources.pipeline = *pipeline;
            resources.workgroup = (*pipeline)->workgroup;
            return resources;
        }

        struct MetalMipmapChain final
        {
            TextureHandle texture{};
            TextureDesc desc{};
            Format storageFormat = Format::unknown;
            TextureViewDimension viewDimension = TextureViewDimension::texture2D;
            std::array<std::uint32_t, 3> group{};
            std::uint32_t firstMipLevel = 0;
            std::uint32_t firstArrayLayer = 0;
            MetalMipmapRange range{};
            TextureLayout baseLayout = TextureLayout::undefined;
            TextureLayout finalLayout = TextureLayout::undefined;
        };

        [[nodiscard]] constexpr MTLStages metalMipmapLayoutStages(TextureLayout layout) noexcept
        {
            switch (layout)
            {
            case TextureLayout::renderTarget:
            case TextureLayout::resolveSource:
            case TextureLayout::resolveDestination:
                return MTLStageFragment;
            case TextureLayout::shaderResource:
            case TextureLayout::unorderedAccess:
                return static_cast<MTLStages>(MTLStageVertex | MTLStageFragment | MTLStageObject | MTLStageMesh |
                                              MTLStageDispatch);
            case TextureLayout::copySource:
            case TextureLayout::copyDestination:
                return static_cast<MTLStages>(MTLStageBlit | MTLStageDispatch);
            default:
                return MTLStageAll;
            }
        }

        struct MetalMipmapViews final
        {
            TextureViewHandle sampled{};
            TextureViewHandle storage{};
        };

        [[nodiscard]] Result<MetalMipmapViews> createMetalMipmapViews(MetalDevice& device,
                                                                      const MetalMipmapChain& chain,
                                                                      std::uint32_t sourceMip,
                                                                      std::uint32_t destinationMip)
        {
            auto sampled = device.createTextureView(TextureViewDesc{
                .texture = chain.texture,
                .usage = TextureViewUsage::sampled,
                .dimension = chain.viewDimension,
                .format = chain.desc.format,
                .firstMipLevel = sourceMip,
                .mipLevelCount = 1,
                .firstArrayLayer = chain.firstArrayLayer,
                .arrayLayerCount = chain.range.layerCount,
            });
            auto storage = device.createTextureView(TextureViewDesc{
                .texture = chain.texture,
                .usage = TextureViewUsage::storage,
                .dimension = chain.viewDimension,
                .format = chain.storageFormat,
                .firstMipLevel = destinationMip,
                .mipLevelCount = 1,
                .firstArrayLayer = chain.firstArrayLayer,
                .arrayLayerCount = chain.range.layerCount,
            });
            if (!sampled || !storage)
            {
                if (sampled)
                {
                    static_cast<void>(device.destroyTextureView(*sampled));
                }
                if (storage)
                {
                    static_cast<void>(device.destroyTextureView(*storage));
                }
                return std::unexpected(sampled ? std::move(storage.error()) : std::move(sampled.error()));
            }
            return MetalMipmapViews{.sampled = *sampled, .storage = *storage};
        }

        struct MetalMipmapViewTextures final
        {
            __strong id<MTLTexture> sampledTexture = nil;
            __strong id<MTLTexture> storageTexture = nil;
            std::shared_ptr<MetalTextureOwner> sampledOwner;
            std::shared_ptr<MetalTextureOwner> storageOwner;
        };

        [[nodiscard]] std::optional<MetalMipmapViewTextures> findMetalMipmapViewTextures(
            rachis::detail::TimedMutex& deviceMutex,
            const std::shared_ptr<MetalDeviceNative>& native,
            const MetalMipmapViews& views)
        {
            std::scoped_lock const lock(deviceMutex);
            auto* sampledRecord = native->views.find(views.sampled);
            auto* storageRecord = native->views.find(views.storage);
            if (sampledRecord == nullptr || sampledRecord->view == nil || storageRecord == nullptr ||
                storageRecord->view == nil)
            {
                return std::nullopt;
            }
            MetalMipmapViewTextures textures;
            textures.sampledTexture = sampledRecord->view;
            textures.storageTexture = storageRecord->view;
            textures.sampledOwner = sampledRecord->owner;
            textures.storageOwner = storageRecord->owner;
            return textures;
        }

        [[nodiscard]] Result<MetalMipmapViewTextures> takeMetalMipmapViewTextures(
            MetalDevice& device,
            rachis::detail::TimedMutex& deviceMutex,
            const std::shared_ptr<MetalDeviceNative>& native,
            const MetalMipmapViews& views)
        {
            auto textures = findMetalMipmapViewTextures(deviceMutex, native, views);
            if (!textures)
            {
                static_cast<void>(device.destroyTextureView(views.sampled));
                static_cast<void>(device.destroyTextureView(views.storage));
                return std::unexpected(metalSelectionError(ErrorCode::invalidState,
                                                           "Metal mipmap generation lost a temporary texture view."));
            }
            return std::move(*textures);
        }

        [[nodiscard]] Status retainMetalMipmapViewOwnersAndReleaseViews(MetalDevice& device,
                                                                        MetalCommandBuffer& native,
                                                                        const MetalMipmapViews& views,
                                                                        const MetalMipmapViewTextures& textures)
        {
            auto keptSampled =
                retainMetalUtilityResource(native, std::static_pointer_cast<void>(textures.sampledOwner));
            auto keptStorage =
                retainMetalUtilityResource(native, std::static_pointer_cast<void>(textures.storageOwner));
            static_cast<void>(device.destroyTextureView(views.sampled));
            static_cast<void>(device.destroyTextureView(views.storage));
            if (!keptSampled)
            {
                return keptSampled;
            }
            if (!keptStorage)
            {
                return keptStorage;
            }
            return {};
        }

        [[nodiscard]] MTLSize metalMipmapThreadgroupCount(const MetalMipmapChain& chain, std::uint32_t destinationMip)
        {
            const auto width = metalMipDimension(chain.desc.width, destinationMip);
            const auto height = metalMipDimension(chain.desc.height, destinationMip);
            const auto depth = chain.desc.dimension == TextureDimension::texture3D
                                   ? metalMipDimension(chain.desc.depthOrArrayLayers, destinationMip)
                                   : chain.range.layerCount;
            const auto groupsX = (width + chain.group[0] - 1U) / chain.group[0];
            const auto groupsY = (height + chain.group[1] - 1U) / chain.group[1];
            const auto groupsZ = (depth + chain.group[2] - 1U) / chain.group[2];
            return MTLSizeMake(groupsX, groupsY, groupsZ);
        }

        [[nodiscard]] Status dispatchMetalMipmapLevel(MetalCommandBuffer& native,
                                                      const MetalMipmapResources& resources,
                                                      const MetalMipmapViewTextures& textures,
                                                      MTLSize threadgroups)
        {
            const std::scoped_lock lock(native.owner()->mutex);
            auto* entry = native.entry();
            auto prepared = native.ensureComputeEncoder();
            if (!prepared)
            {
                return prepared;
            }
            auto table = acquireMetalArgumentTable(resources.device, *entry, MetalArgumentTableStage::compute, 1, 2, 1);
            if (!table)
            {
                return std::unexpected(std::move(table.error()));
            }
            [*table setAddress:resources.constants.gpuAddress atIndex:0];
            [*table setTexture:textures.sampledTexture.gpuResourceID atIndex:0];
            [*table setTexture:textures.storageTexture.gpuResourceID atIndex:1];
            [*table setSamplerState:resources.sampler.gpuResourceID atIndex:0];
            metalRetainResidency(*entry, resources.constants);
            metalRetainResidency(*entry, textures.sampledTexture);
            metalRetainResidency(*entry, textures.storageTexture);
            [entry->computeEncoder setComputePipelineState:resources.pipeline->compute];
            [entry->computeEncoder setArgumentTable:*table];
            entry->activeComputeTable = *table;
            entry->computeTableDirty = true;
            entry->activeComputePipeline = nullptr;
            [entry->computeEncoder
                 dispatchThreadgroups:threadgroups
                threadsPerThreadgroup:MTLSizeMake(resources.workgroup.x, resources.workgroup.y, resources.workgroup.z)];
            return native.encodeStageBarrier(MTLStageDispatch, MTLStageDispatch);
        }

        void resetMetalComputeBinding(MetalCommandBuffer& native)
        {
            auto* entry = native.entry();
            entry->boundComputePipeline.reset();
            entry->boundComputeVersions.clear();
            entry->computePushConstants = {};
            entry->computePushConstantBytes = 0;
            entry->computePushConstantsDirty = true;
            entry->activeComputePushSlice = {};
            entry->computeTableDirty = true;
            entry->activeComputeTable = nil;
            entry->activeComputePipeline = nullptr;
            native.computeState() = {};
        }

        [[nodiscard]] Status finishMetalMipmapChain(MetalCommandBuffer& native, MTLStages consumerStages)
        {
            const std::scoped_lock lock(native.owner()->mutex);
            auto published = native.encodeStageBarrier(MTLStageDispatch, consumerStages);
            if (!published)
            {
                return published;
            }
            resetMetalComputeBinding(native);
            return {};
        }

        [[nodiscard]] Status encodeMetalMipmapChain(MetalDevice& device,
                                                    rachis::detail::TimedMutex& deviceMutex,
                                                    const std::shared_ptr<MetalDeviceNative>& deviceNative,
                                                    MetalCommandBuffer& command,
                                                    const MetalMipmapChain& chain,
                                                    const MetalMipmapResources& resources)
        {
            const auto blitAndDispatch = static_cast<MTLStages>(MTLStageBlit | MTLStageDispatch);
            const auto producerStages =
                static_cast<MTLStages>(metalMipmapLayoutStages(chain.baseLayout) | blitAndDispatch);
            const auto consumerStages =
                static_cast<MTLStages>(metalMipmapLayoutStages(chain.finalLayout) | blitAndDispatch);
            {
                const std::scoped_lock lock(command.owner()->mutex);
                resetMetalComputeBinding(command);
                auto prepared = command.ensureComputeEncoder();
                if (!prepared)
                {
                    return prepared;
                }
                auto hazard = command.encodeStageBarrier(producerStages, MTLStageDispatch);
                if (!hazard)
                {
                    return hazard;
                }
            }

            for (std::uint32_t mipOffset = 1; mipOffset < chain.range.mipCount; ++mipOffset)
            {
                const auto sourceMip = chain.firstMipLevel + mipOffset - 1U;
                const auto destinationMip = sourceMip + 1U;
                auto views = createMetalMipmapViews(device, chain, sourceMip, destinationMip);
                if (!views)
                {
                    return std::unexpected(std::move(views.error()));
                }
                auto textures = takeMetalMipmapViewTextures(device, deviceMutex, deviceNative, *views);
                if (!textures)
                {
                    return std::unexpected(std::move(textures.error()));
                }
                if (auto kept = retainMetalMipmapViewOwnersAndReleaseViews(device, command, *views, *textures); !kept)
                {
                    return kept;
                }
                if (auto dispatched = dispatchMetalMipmapLevel(
                        command, resources, *textures, metalMipmapThreadgroupCount(chain, destinationMip));
                    !dispatched)
                {
                    return dispatched;
                }
            }
            return finishMetalMipmapChain(command, consumerStages);
        }

    } // namespace

    Status MetalDevice::generateMipmaps(detail::BackendCommandBuffer& commandBuffer,
                                        const TextureMipmapGeneration& generation)
    {
        const MetalAutoreleasePool autoreleasePool;
        auto texture = captureMetalMipmapTexture(mutex_, native_, generation.texture);
        if (!texture)
        {
            return std::unexpected(std::move(texture.error()));
        }
        auto recording = metalRecordingCommand(commandBuffer, texture->deviceToken);
        if (!recording)
        {
            return std::unexpected(std::move(recording.error()));
        }
        auto& native = **recording;
        const auto& desc = texture->desc;
        auto range = validateMetalMipmapGeneration(native, desc, generation);
        if (!range)
        {
            return std::unexpected(std::move(range.error()));
        }
        if (!hasAny(queryFormatProperties(desc.format).features, FormatFeature::filterable))
        {
            return std::unexpected(
                metalSelectionError(ErrorCode::unsupported, "Mipmap generation requires a filterable texture format."));
        }
        const std::array layoutTransitions{
            MetalLayoutTransition{
                .table = texture->layouts,
                .firstMip = generation.subresources.firstMipLevel,
                .mipCount = 1,
                .firstLayer = generation.subresources.firstArrayLayer,
                .layerCount = range->layerCount,
                .before = generation.baseLayout,
                .after = generation.finalLayout,
                .operation = "Mipmap generation's baseLayout",
            },
            MetalLayoutTransition{
                .table = texture->layouts,
                .firstMip = generation.subresources.firstMipLevel + 1,
                .mipCount = range->mipCount - 1,
                .firstLayer = generation.subresources.firstArrayLayer,
                .layerCount = range->layerCount,
                .before = TextureLayout::undefined,
                .after = generation.finalLayout,
                .operation = "Mipmap generation",
            },
        };
        if (auto checked = metalCheckLayoutTransitions(native.layoutState(), layoutTransitions); !checked)
        {
            return checked;
        }
        if (auto pipelines = ensureMipmapPipelines(); !pipelines)
        {
            return pipelines;
        }
        auto storage = selectMetalMipmapStorage(desc, texture->viewFormats);
        if (!storage)
        {
            return std::unexpected(std::move(storage.error()));
        }
        auto retainedTexture = retainMetalApplicationResource(native, std::move(texture->owner));
        if (!retainedTexture)
        {
            return retainedTexture;
        }

        if (auto blocked = requireActive(); !blocked)
        {
            return std::unexpected(std::move(blocked.error()));
        }
        const auto pipelineHandle = mipmapPipelines_[metalMipmapDimensionIndex(desc.dimension)];
        auto resources =
            captureMetalMipmapResources(mutex_, native_, storage->encodeSrgb, mipmapSampler_, pipelineHandle);
        if (!resources)
        {
            return std::unexpected(std::move(resources.error()));
        }
        auto retainedPipeline = retainMetalUtilityResource(native, std::static_pointer_cast<void>(resources->pipeline));
        if (!retainedPipeline)
        {
            return retainedPipeline;
        }
        const MetalMipmapChain chain{
            .texture = generation.texture,
            .desc = desc,
            .storageFormat = storage->storageFormat,
            .viewDimension = metalMipmapViewDimension(desc.dimension),
            .group = metalMipmapThreadGroupSize(desc.dimension),
            .firstMipLevel = generation.subresources.firstMipLevel,
            .firstArrayLayer = generation.subresources.firstArrayLayer,
            .range = *range,
            .baseLayout = generation.baseLayout,
            .finalLayout = generation.finalLayout,
        };
        if (auto encoded = encodeMetalMipmapChain(*this, mutex_, native_, native, chain, *resources); !encoded)
        {
            return encoded;
        }
        metalStageLayoutTransitions(native.layoutState(), layoutTransitions);
        return {};
    }

} // namespace rachis::metal
