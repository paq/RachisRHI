#include "D3D12/D3D12ResourceInternal.h"

#include "D3D12/D3D12Binding.h"
#include "D3D12/D3D12Pipeline.h"
#include "Rachis/Rendering.h"

namespace rachis::d3d12
{

    using namespace resource_detail;

    namespace
    {

        [[nodiscard]] constexpr std::uint32_t blockAlignedDimension(std::uint32_t size,
                                                                    std::uint32_t blockSize) noexcept
        {
            return (size + blockSize - 1U) / blockSize * blockSize;
        }

        [[nodiscard]] constexpr Extent3D blockAlignedExtent(const format::Info& info, Extent3D extent) noexcept
        {
            return Extent3D{
                blockAlignedDimension(extent.width, info.blockWidth),
                blockAlignedDimension(extent.height, info.blockHeight),
                extent.depth,
            };
        }

        [[nodiscard]] bool sparseTileRangeIsMapped(const std::shared_ptr<const SparseMappingState>& mappings,
                                                   std::uint32_t firstTile,
                                                   std::uint32_t tileCount) noexcept
        {
            if (mappings == nullptr || tileCount == 0)
            {
                return false;
            }
            const auto tileLimit = static_cast<std::uint64_t>(firstTile) + tileCount;
            auto coveredUntil = static_cast<std::uint64_t>(firstTile);
            for (const auto& region : mappings->regions)
            {
                const auto regionBegin = static_cast<std::uint64_t>(region.firstTile);
                const auto regionEnd = regionBegin + region.tileCount;
                if (regionEnd <= coveredUntil)
                {
                    continue;
                }
                if (regionBegin > coveredUntil)
                {
                    return false;
                }
                coveredUntil = regionEnd;
                if (coveredUntil >= tileLimit)
                {
                    return true;
                }
            }
            return false;
        }

        [[nodiscard]] bool sparseBufferRangeIsMapped(const BufferRecord& record,
                                                     std::uint64_t offset,
                                                     std::uint64_t size) noexcept
        {
            if (!record.sparse)
            {
                return true;
            }
            if (size == 0)
            {
                return false;
            }

            const auto firstTile = offset / sparseBufferTileSize;
            const auto tileLimit = ((offset + size - 1) / sparseBufferTileSize) + 1;
            return tileLimit <= std::numeric_limits<std::uint32_t>::max() &&
                   sparseTileRangeIsMapped(record.sparseMappings,
                                           static_cast<std::uint32_t>(firstTile),
                                           static_cast<std::uint32_t>(tileLimit - firstTile));
        }

    } // namespace

    namespace resource_detail
    {

        [[nodiscard]] bool sparseTextureRegionIsMapped(const TextureRecord& record,
                                                       std::uint32_t subresource,
                                                       Origin3D origin,
                                                       Extent3D extent) noexcept
        {
            if (!record.sparse)
            {
                return true;
            }
            if (record.sparseMappings == nullptr || subresource >= record.desc.mipLevels || extent.width == 0 ||
                extent.height == 0 || extent.depth == 0)
            {
                return false;
            }
            if (subresource >= record.sparseProperties.standardMipCount)
            {
                return record.sparseProperties.mipTailTileCount != 0 &&
                       sparseTileRangeIsMapped(record.sparseMappings,
                                               record.sparseMipTailFirstTile,
                                               record.sparseProperties.mipTailTileCount);
            }
            if (subresource >= record.sparseSubresources.size())
            {
                return false;
            }
            const auto& tiling = record.sparseSubresources[subresource];
            const auto& tile = record.sparseProperties.standardTileExtent;
            if (tile.width == 0 || tile.height == 0 || tile.depth == 0 || tiling.WidthInTiles == 0 ||
                tiling.HeightInTiles == 0 || tiling.DepthInTiles == 0)
            {
                return false;
            }
            const auto firstX = origin.x / tile.width;
            const auto firstY = origin.y / tile.height;
            const auto firstZ = origin.z / tile.depth;
            const auto lastX = (static_cast<std::uint64_t>(origin.x) + extent.width - 1U) / tile.width;
            const auto lastY = (static_cast<std::uint64_t>(origin.y) + extent.height - 1U) / tile.height;
            const auto lastZ = (static_cast<std::uint64_t>(origin.z) + extent.depth - 1U) / tile.depth;
            const auto rowTileCount = static_cast<std::uint32_t>(lastX - firstX + 1U);
            for (auto z = static_cast<std::uint64_t>(firstZ); z <= lastZ; ++z)
            {
                for (auto y = static_cast<std::uint64_t>(firstY); y <= lastY; ++y)
                {
                    const auto rowStart = static_cast<std::uint64_t>(tiling.StartTileIndexInOverallResource) + firstX +
                                          y * tiling.WidthInTiles + z * tiling.WidthInTiles * tiling.HeightInTiles;
                    if (rowStart > std::numeric_limits<std::uint32_t>::max() ||
                        !sparseTileRangeIsMapped(
                            record.sparseMappings, static_cast<std::uint32_t>(rowStart), rowTileCount))
                    {
                        return false;
                    }
                }
            }
            return true;
        }

    } // namespace resource_detail

    namespace
    {

        struct NativeSubresource final
        {
            std::uint32_t index = 0;
            std::uint32_t plane = 0;
            Extent3D mipExtent;
        };

        [[nodiscard]] Result<NativeSubresource> nativeSubresource(const TextureRecord& texture,
                                                                  const TextureSubresource& subresource)
        {
            const auto info = format::info(texture.desc.format);
            const auto plane = info.has_value() ? aspectPlane(*info, subresource.aspect) : std::nullopt;
            const std::uint32_t arraySize =
                texture.desc.dimension == TextureDimension::texture3D ? 1U : texture.desc.depthOrArrayLayers;
            if (!plane.has_value() || subresource.mipLevel >= texture.desc.mipLevels ||
                subresource.arrayLayer >= arraySize)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The texture subresource mip, layer, or aspect is invalid.",
                });
            }

            const auto index = subresource.mipLevel + subresource.arrayLayer * texture.desc.mipLevels +
                               *plane * texture.desc.mipLevels * arraySize;
            return NativeSubresource{
                index,
                *plane,
                Extent3D{
                    mipDimension(texture.desc.width, subresource.mipLevel),
                    texture.desc.dimension == TextureDimension::texture1D
                        ? 1U
                        : mipDimension(texture.desc.height, subresource.mipLevel),
                    texture.desc.dimension == TextureDimension::texture3D
                        ? mipDimension(texture.desc.depthOrArrayLayers, subresource.mipLevel)
                        : 1U,
                },
            };
        }

        [[nodiscard]] Status validateTextureRegion(const TextureRecord& texture,
                                                   const NativeSubresource& subresource,
                                                   Origin3D origin,
                                                   Extent3D extent)
        {
            if (extent.width == 0 || extent.height == 0 || extent.depth == 0 ||
                origin.x > subresource.mipExtent.width || extent.width > subresource.mipExtent.width - origin.x ||
                origin.y > subresource.mipExtent.height || extent.height > subresource.mipExtent.height - origin.y ||
                origin.z > subresource.mipExtent.depth || extent.depth > subresource.mipExtent.depth - origin.z)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The texture copy region is empty or outside the selected subresource.",
                });
            }

            const auto info = *format::info(texture.desc.format);
            if (info.compressed)
            {
                const bool alignedOrigin = origin.x % info.blockWidth == 0 && origin.y % info.blockHeight == 0;
                const bool alignedExtent =
                    (extent.width % info.blockWidth == 0 || origin.x + extent.width == subresource.mipExtent.width) &&
                    (extent.height % info.blockHeight == 0 || origin.y + extent.height == subresource.mipExtent.height);
                if (!alignedOrigin || !alignedExtent)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Block-compressed texture copies must align to blocks except at mip edges.",
                    });
                }
            }
            if (info.depth &&
                (origin.x != 0 || origin.y != 0 || origin.z != 0 || extent.width != subresource.mipExtent.width ||
                 extent.height != subresource.mipExtent.height || extent.depth != subresource.mipExtent.depth))
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "Depth-stencil copies currently require a whole subresource.",
                });
            }
            return {};
        }

        [[nodiscard]] bool checkedMultiply(std::uint64_t left, std::uint64_t right, std::uint64_t& result) noexcept
        {
            if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)
            {
                return false;
            }
            result = left * right;
            return true;
        }

        [[nodiscard]] bool checkedAdd(std::uint64_t left, std::uint64_t right, std::uint64_t& result) noexcept
        {
            if (right > std::numeric_limits<std::uint64_t>::max() - left)
            {
                return false;
            }
            result = left + right;
            return true;
        }

        struct LinearTextureLayout final
        {
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
            NativeSubresource subresource;
            TextureTransferLayout transfer;
            Extent3D blockExtent;
            bool wholeSubresource = false;
        };

        [[nodiscard]] Result<NativeSubresource> linearCopySubresource(const TextureRecord& texture,
                                                                      const BufferTextureCopyRegion& copy)
        {
            auto subresource = nativeSubresource(texture, copy.textureSubresource);
            if (!subresource)
            {
                return std::unexpected(std::move(subresource.error()));
            }
            auto validRegion = validateTextureRegion(texture, *subresource, copy.textureOrigin, copy.extent);
            if (!validRegion)
            {
                return std::unexpected(std::move(validRegion.error()));
            }
            if (texture.desc.sampleCount != 1)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Buffer-texture copies require a single-sampled texture.",
                });
            }
            if (copy.bufferOffset % D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT != 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A buffer-texture copy offset must satisfy D3D12 texture placement alignment.",
                });
            }
            return subresource;
        }

        struct LinearRowMetrics final
        {
            std::uint64_t minimumRowBytes = 0;
            std::uint64_t copiedRows = 0;
            std::uint64_t sliceRows = 0;
            DXGI_FORMAT footprintFormat = DXGI_FORMAT_UNKNOWN;
        };

        [[nodiscard]] Result<LinearRowMetrics> linearRowMetrics(ID3D12Device& device,
                                                                const TextureRecord& texture,
                                                                const NativeSubresource& subresource,
                                                                const BufferTextureCopyRegion& copy,
                                                                std::uint32_t rowsPerImage)
        {
            const auto info = *format::info(texture.desc.format);
            if (!info.depth)
            {
                const std::uint64_t blocksWide = (copy.extent.width + info.blockWidth - 1U) / info.blockWidth;
                return LinearRowMetrics{
                    .minimumRowBytes = blocksWide * info.bytesPerBlock,
                    .copiedRows = (copy.extent.height + info.blockHeight - 1U) / info.blockHeight,
                    .sliceRows = (static_cast<std::uint64_t>(rowsPerImage) + info.blockHeight - 1U) / info.blockHeight,
                    .footprintFormat = info.view,
                };
            }

            D3D12_PLACED_SUBRESOURCE_FOOTPRINT canonical{};
            UINT canonicalRows = 0;
            UINT64 canonicalRowBytes = 0;
            const auto nativeDesc = texture.resource->GetDesc();
            device.GetCopyableFootprints(
                &nativeDesc, subresource.index, 1, 0, &canonical, &canonicalRows, &canonicalRowBytes, nullptr);
            if (canonical.Footprint.Format == DXGI_FORMAT_UNKNOWN || canonicalRows == 0 || canonicalRowBytes == 0 ||
                canonical.Footprint.Width != copy.extent.width || canonical.Footprint.Height != copy.extent.height ||
                canonical.Footprint.Depth != copy.extent.depth)
            {
                return std::unexpected(Error{
                    ErrorCode::nativeFailure,
                    0,
                    "D3D12 returned an invalid copyable footprint for a depth-stencil plane.",
                });
            }
            if (rowsPerImage != copy.extent.height)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Depth-stencil buffer copies require rows-per-image to match the complete plane height.",
                });
            }
            return LinearRowMetrics{
                .minimumRowBytes = canonicalRowBytes,
                .copiedRows = canonicalRows,
                .sliceRows = canonicalRows,
                .footprintFormat = canonical.Footprint.Format,
            };
        }

        [[nodiscard]] Result<std::uint64_t> linearRowPitch(const BufferTextureCopyRegion& copy,
                                                           std::uint32_t rowsPerImage,
                                                           std::uint64_t minimumRowBytes)
        {
            const std::uint64_t resolvedRowPitch =
                copy.bytesPerRow == 0 ? (minimumRowBytes + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1U) &
                                            ~(static_cast<std::uint64_t>(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT) - 1U)
                                      : copy.bytesPerRow;
            if (rowsPerImage < copy.extent.height || rowsPerImage > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
                resolvedRowPitch < minimumRowBytes || resolvedRowPitch > std::numeric_limits<std::uint32_t>::max() ||
                resolvedRowPitch % D3D12_TEXTURE_DATA_PITCH_ALIGNMENT != 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Buffer-texture row pitch or rows-per-image is too small or incorrectly aligned.",
                });
            }
            return resolvedRowPitch;
        }

        struct LinearCopySize final
        {
            std::uint64_t bytesPerImage = 0;
            std::uint64_t requiredBufferSize = 0;
        };

        [[nodiscard]] Result<LinearCopySize> linearCopySize(const BufferTextureCopyRegion& copy,
                                                            const LinearRowMetrics& rows,
                                                            std::uint64_t rowPitch)
        {
            std::uint64_t bytesPerImage = 0;
            std::uint64_t priorSliceBytes = 0;
            std::uint64_t copiedRowBytes = 0;
            std::uint64_t requiredBytes = 0;
            const bool validSize = checkedMultiply(rowPitch, rows.sliceRows, bytesPerImage) &&
                                   checkedMultiply(bytesPerImage, copy.extent.depth - 1U, priorSliceBytes) &&
                                   checkedMultiply(rowPitch, rows.copiedRows - 1U, copiedRowBytes) &&
                                   checkedAdd(priorSliceBytes, copiedRowBytes, requiredBytes) &&
                                   checkedAdd(requiredBytes, rows.minimumRowBytes, requiredBytes) &&
                                   checkedAdd(copy.bufferOffset, requiredBytes, requiredBytes);
            if (!validSize)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The buffer-texture copy footprint exceeds portable size limits.",
                });
            }
            return LinearCopySize{
                .bytesPerImage = bytesPerImage,
                .requiredBufferSize = requiredBytes,
            };
        }

        [[nodiscard]] Result<LinearTextureLayout> linearTextureLayout(ID3D12Device& device,
                                                                      const TextureRecord& texture,
                                                                      const BufferTextureCopyRegion& copy)
        {
            auto subresource = linearCopySubresource(texture, copy);
            if (!subresource)
            {
                return std::unexpected(std::move(subresource.error()));
            }
            const std::uint32_t rowsPerImage = copy.rowsPerImage == 0 ? copy.extent.height : copy.rowsPerImage;
            auto rows = linearRowMetrics(device, texture, *subresource, copy, rowsPerImage);
            if (!rows)
            {
                return std::unexpected(std::move(rows.error()));
            }
            auto rowPitch = linearRowPitch(copy, rowsPerImage, rows->minimumRowBytes);
            if (!rowPitch)
            {
                return std::unexpected(std::move(rowPitch.error()));
            }
            auto size = linearCopySize(copy, *rows, *rowPitch);
            if (!size)
            {
                return std::unexpected(std::move(size.error()));
            }

            const auto info = *format::info(texture.desc.format);
            const auto blockExtent = blockAlignedExtent(info, copy.extent);
            return LinearTextureLayout{
                .footprint =
                    D3D12_PLACED_SUBRESOURCE_FOOTPRINT{
                        .Offset = copy.bufferOffset,
                        .Footprint =
                            {
                                .Format = rows->footprintFormat,
                                .Width = blockExtent.width,
                                .Height = blockAlignedDimension(rowsPerImage, info.blockHeight),
                                .Depth = copy.extent.depth,
                                .RowPitch = static_cast<std::uint32_t>(*rowPitch),
                            },
                    },
                .subresource = *subresource,
                .transfer =
                    TextureTransferLayout{
                        .bytesPerRow = static_cast<std::uint32_t>(*rowPitch),
                        .rowsPerImage = rowsPerImage,
                        .rowCount = static_cast<std::uint32_t>(rows->copiedRows),
                        .rowDataSize = rows->minimumRowBytes,
                        .bytesPerImage = size->bytesPerImage,
                        .requiredBufferSize = size->requiredBufferSize,
                        .bufferOffsetAlignment = D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT,
                        .rowPitchAlignment = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT,
                    },
                .blockExtent = blockExtent,
                .wholeSubresource = info.depth,
            };
        }

        template<typename Region, std::size_t InlineCapacity>
        void appendRegions(SmallVector<Region, InlineCapacity>& destination, std::span<const Region> regions)
        {
            destination.reserve(regions.size());
            for (const auto& region : regions)
            {
                destination.push_back(region);
            }
        }

        struct PreparedBufferCopyBatch final
        {
            std::shared_ptr<BufferRecord> source;
            std::shared_ptr<BufferRecord> destination;
            SmallVector<BufferCopyRegion, 2> regions;
            std::array<std::shared_ptr<ResidencyRecord>, 2> residencies;
            SparseMappingCommandLeases sparseMappings;
        };

        struct PreparedBufferTextureCopyBatch final
        {
            std::shared_ptr<BufferRecord> buffer;
            std::shared_ptr<TextureRecord> texture;
            SmallVector<BufferTextureCopyRegion, 1> regions;
            SmallVector<LinearTextureLayout, 1> layouts;
            std::array<std::shared_ptr<ResidencyRecord>, 2> residencies;
            SparseMappingCommandLeases sparseMappings;
        };

        struct PreparedTextureCopyRegion final
        {
            TextureCopyRegion region;
            NativeSubresource sourceSubresource;
            NativeSubresource destinationSubresource;
            Extent3D blockExtent;
            bool wholeSubresource = false;
        };

        struct PreparedTextureCopyBatch final
        {
            std::shared_ptr<TextureRecord> source;
            std::shared_ptr<TextureRecord> destination;
            SmallVector<PreparedTextureCopyRegion, 1> regions;
            std::array<std::shared_ptr<ResidencyRecord>, 2> residencies;
            SparseMappingCommandLeases sparseMappings;
        };

        struct BufferInterval final
        {
            std::uint64_t begin = 0;
            std::uint64_t end = 0;
        };

        struct NativeTextureBox final
        {
            std::uint32_t subresource = 0;
            Origin3D origin;
            Extent3D extent;
        };

        [[nodiscard]] constexpr bool intervalsOverlap(std::uint64_t leftOrigin,
                                                      std::uint64_t leftExtent,
                                                      std::uint64_t rightOrigin,
                                                      std::uint64_t rightExtent) noexcept
        {
            return leftOrigin < rightOrigin + rightExtent && rightOrigin < leftOrigin + leftExtent;
        }

        [[nodiscard]] constexpr bool textureBoxesOverlap(const NativeTextureBox& left,
                                                         const NativeTextureBox& right) noexcept
        {
            return left.subresource == right.subresource &&
                   intervalsOverlap(left.origin.x, left.extent.width, right.origin.x, right.extent.width) &&
                   intervalsOverlap(left.origin.y, left.extent.height, right.origin.y, right.extent.height) &&
                   intervalsOverlap(left.origin.z, left.extent.depth, right.origin.z, right.extent.depth);
        }

        [[nodiscard]] constexpr bool isWholeSubresource(Origin3D origin,
                                                        Extent3D extent,
                                                        const NativeSubresource& subresource) noexcept
        {
            return origin.x == 0 && origin.y == 0 && origin.z == 0 && extent.width == subresource.mipExtent.width &&
                   extent.height == subresource.mipExtent.height && extent.depth == subresource.mipExtent.depth;
        }

        [[nodiscard]] constexpr std::uint64_t linearRowCount(const LinearTextureLayout& layout) noexcept
        {
            return static_cast<std::uint64_t>(layout.transfer.rowCount) * layout.footprint.Footprint.Depth;
        }

        [[nodiscard]] constexpr BufferInterval linearRowInterval(const LinearTextureLayout& layout,
                                                                 std::uint64_t row) noexcept
        {
            const std::uint64_t slice = row / layout.transfer.rowCount;
            const std::uint64_t rowInSlice = row % layout.transfer.rowCount;
            const std::uint64_t rowPitch = layout.footprint.Footprint.RowPitch;
            const std::uint64_t begin =
                layout.footprint.Offset + slice * layout.transfer.bytesPerImage + rowInSlice * rowPitch;
            return BufferInterval{
                .begin = begin,
                .end = begin + layout.transfer.rowDataSize,
            };
        }

        [[nodiscard]] constexpr bool linearLayoutsOverlap(const LinearTextureLayout& left,
                                                          const LinearTextureLayout& right) noexcept
        {
            std::uint64_t leftRow = 0;
            std::uint64_t rightRow = 0;
            while (leftRow < linearRowCount(left) && rightRow < linearRowCount(right))
            {
                const auto leftInterval = linearRowInterval(left, leftRow);
                const auto rightInterval = linearRowInterval(right, rightRow);
                if (leftInterval.begin < rightInterval.end && rightInterval.begin < leftInterval.end)
                {
                    return true;
                }
                if (leftInterval.end <= rightInterval.begin)
                {
                    ++leftRow;
                }
                else
                {
                    ++rightRow;
                }
            }
            return false;
        }

        [[nodiscard]] Status validateBufferCopyRegion(const PreparedBufferCopyBatch& batch,
                                                      const BufferCopyRegion& region)
        {
            if (region.size == 0 || region.sourceOffset > batch.source->desc.size ||
                region.size > batch.source->desc.size - region.sourceOffset ||
                region.destinationOffset > batch.destination->desc.size ||
                region.size > batch.destination->desc.size - region.destinationOffset)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A buffer-copy batch contains an empty or out-of-bounds region.",
                });
            }
            if (!sparseBufferRangeIsMapped(*batch.source, region.sourceOffset, region.size) ||
                !sparseBufferRangeIsMapped(*batch.destination, region.destinationOffset, region.size))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A buffer-copy batch cannot access an unmapped sparse-buffer tile.",
                });
            }
            return {};
        }

        [[nodiscard]] bool sortedIntervalsContainOverlap(std::span<const BufferInterval> intervals) noexcept
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

        [[nodiscard]] bool sortedIntervalSetsOverlap(std::span<const BufferInterval> sources,
                                                     std::span<const BufferInterval> destinations) noexcept
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

        [[nodiscard]] Status validateBufferCopyRegions(const PreparedBufferCopyBatch& batch, bool sameBuffer)
        {
            SmallVector<BufferInterval, 8> sourceIntervals;
            SmallVector<BufferInterval, 8> destinationIntervals;
            sourceIntervals.reserve(batch.regions.size());
            destinationIntervals.reserve(batch.regions.size());
            for (const auto& region : batch.regions)
            {
                auto validRegion = validateBufferCopyRegion(batch, region);
                if (!validRegion)
                {
                    return validRegion;
                }
                sourceIntervals.push_back(BufferInterval{
                    .begin = region.sourceOffset,
                    .end = region.sourceOffset + region.size,
                });
                destinationIntervals.push_back(BufferInterval{
                    .begin = region.destinationOffset,
                    .end = region.destinationOffset + region.size,
                });
            }

            const auto byBeginning = [](const BufferInterval& left, const BufferInterval& right) {
                return left.begin < right.begin;
            };
            std::ranges::sort(destinationIntervals, byBeginning);
            if (sortedIntervalsContainOverlap(destinationIntervals))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A buffer-copy batch cannot write overlapping destination regions.",
                });
            }
            if (!sameBuffer)
            {
                return {};
            }

            std::ranges::sort(sourceIntervals, byBeginning);
            if (sortedIntervalSetsOverlap(sourceIntervals, destinationIntervals))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A same-buffer copy batch cannot access one byte as both source and destination.",
                });
            }
            return {};
        }

        [[nodiscard]] Result<std::shared_ptr<PreparedBufferCopyBatch>> prepareBufferCopyBatch(
            std::shared_ptr<BufferRecord> source,
            std::shared_ptr<BufferRecord> destination,
            std::span<const BufferCopyRegion> regions,
            bool sameBuffer)
        {
            if (!hasAny(source->desc.usage, BufferUsage::copySource) ||
                !hasAny(destination->desc.usage, BufferUsage::copyDestination))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Buffer copy usages do not include copy source and copy destination.",
                });
            }

            auto prepared = std::make_shared<PreparedBufferCopyBatch>();
            prepared->source = std::move(source);
            prepared->destination = std::move(destination);
            appendRegions(prepared->regions, regions);
            prepared->residencies = {prepared->source->residency, prepared->destination->residency};
            auto validRegions = validateBufferCopyRegions(*prepared, sameBuffer);
            if (!validRegions)
            {
                return std::unexpected(std::move(validRegions.error()));
            }
            return prepared;
        }

        [[nodiscard]] Result<ID3D12GraphicsCommandList*> retainCopyBatchForRecording(
            QueueManager& queues,
            detail::BackendCommandBuffer& commandBuffer,
            const std::shared_ptr<PreparedBufferCopyBatch>& prepared)
        {
            SparseMappingDependencies sparseDependencies;
            auto sourceDependency =
                captureSparseBufferMapping(prepared->source, prepared->sparseMappings, sparseDependencies);
            if (!sourceDependency)
            {
                return std::unexpected(std::move(sourceDependency.error()));
            }
            auto destinationDependency =
                captureSparseBufferMapping(prepared->destination, prepared->sparseMappings, sparseDependencies);
            if (!destinationDependency)
            {
                return std::unexpected(std::move(destinationDependency.error()));
            }
            auto commandList = queues.commandList(commandBuffer);
            if (!commandList)
            {
                return std::unexpected(std::move(commandList.error()));
            }
            auto retained = retainResourceCommandDependencies(queues,
                                                              commandBuffer,
                                                              prepared,
                                                              prepared->residencies,
                                                              prepared->sparseMappings,
                                                              sparseDependencies,
                                                              {});
            if (!retained)
            {
                return std::unexpected(std::move(retained.error()));
            }
            return commandList;
        }

        void recordBufferCopies(ID3D12GraphicsCommandList& commandList, const PreparedBufferCopyBatch& batch)
        {
            for (const auto& region : batch.regions)
            {
                commandList.CopyBufferRegion(batch.destination->resource.Get(),
                                             region.destinationOffset,
                                             batch.source->resource.Get(),
                                             region.sourceOffset,
                                             region.size);
            }
        }

        [[nodiscard]] Result<LinearTextureLayout> linearTextureLayoutWithinBuffer(
            ID3D12Device& device, const PreparedBufferTextureCopyBatch& batch, const BufferTextureCopyRegion& region)
        {
            auto layout = linearTextureLayout(device, *batch.texture, region);
            if (!layout)
            {
                return std::unexpected(std::move(layout.error()));
            }
            if (layout->transfer.requiredBufferSize > batch.buffer->desc.size)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The buffer-texture copy footprint exceeds the buffer.",
                });
            }
            return layout;
        }

        [[nodiscard]] constexpr NativeTextureBox copiedTextureBox(const LinearTextureLayout& layout,
                                                                  const BufferTextureCopyRegion& region) noexcept
        {
            return NativeTextureBox{
                .subresource = layout.subresource.index,
                .origin = region.textureOrigin,
                .extent = region.extent,
            };
        }

        [[nodiscard]] constexpr BatchSweepKey boxSweepKey(const NativeTextureBox& box) noexcept
        {
            return textureBoxSweepKey(box.subresource, box.origin, box.extent);
        }

        [[nodiscard]] Status prepareBufferToTextureLayout(ID3D12Device& device,
                                                          PreparedBufferTextureCopyBatch& batch,
                                                          const BufferTextureCopyRegion& region)
        {
            auto layout = linearTextureLayoutWithinBuffer(device, batch, region);
            if (!layout)
            {
                return std::unexpected(std::move(layout.error()));
            }
            if (!sparseBufferRangeIsMapped(
                    *batch.buffer, region.bufferOffset, layout->transfer.requiredBufferSize - region.bufferOffset))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A buffer-to-texture copy cannot access an unmapped sparse-buffer tile.",
                });
            }
            if (!sparseTextureRegionIsMapped(
                    *batch.texture, layout->subresource.index, region.textureOrigin, region.extent))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A buffer-to-texture copy cannot access an unmapped sparse-texture tile.",
                });
            }
            batch.layouts.push_back(*layout);
            return {};
        }

        [[nodiscard]] Status prepareBufferToTextureLayouts(ID3D12Device& device, PreparedBufferTextureCopyBatch& batch)
        {
            Status firstRejected = {};
            for (const auto& region : batch.regions)
            {
                firstRejected = prepareBufferToTextureLayout(device, batch, region);
                if (!firstRejected)
                {
                    break;
                }
            }
            const auto destinationBox = [&](std::size_t index) {
                return copiedTextureBox(batch.layouts[index], batch.regions[index]);
            };
            const auto overlapping = anyOverlappingPair(
                batch.layouts.size(),
                [&](std::size_t index) { return boxSweepKey(destinationBox(index)); },
                [&](std::size_t earlier, std::size_t later) {
                    return textureBoxesOverlap(destinationBox(earlier), destinationBox(later));
                });
            if (overlapping)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A buffer-to-texture batch cannot write overlapping destination texels.",
                });
            }
            return firstRejected;
        }

        [[nodiscard]] Result<std::shared_ptr<PreparedBufferTextureCopyBatch>> prepareBufferToTextureCopyBatch(
            ID3D12Device& device,
            std::shared_ptr<BufferRecord> buffer,
            std::shared_ptr<TextureRecord> texture,
            std::span<const BufferTextureCopyRegion> regions)
        {
            if (!hasAny(buffer->desc.usage, BufferUsage::copySource) ||
                !hasAny(texture->desc.usage, TextureUsage::copyDestination))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A buffer-to-texture copy requires copy-source and copy-destination usages.",
                });
            }

            auto prepared = std::make_shared<PreparedBufferTextureCopyBatch>();
            prepared->buffer = std::move(buffer);
            prepared->texture = std::move(texture);
            appendRegions(prepared->regions, regions);
            prepared->layouts.reserve(regions.size());
            prepared->residencies = {prepared->buffer->residency, prepared->texture->residency};
            auto layouts = prepareBufferToTextureLayouts(device, *prepared);
            if (!layouts)
            {
                return std::unexpected(std::move(layouts.error()));
            }
            return prepared;
        }

        [[nodiscard]] Result<ID3D12GraphicsCommandList*> retainCopyBatchForRecording(
            QueueManager& queues,
            detail::BackendCommandBuffer& commandBuffer,
            const std::shared_ptr<PreparedBufferTextureCopyBatch>& prepared)
        {
            SparseMappingDependencies sparseDependencies;
            auto sparseDependency =
                captureSparseBufferMapping(prepared->buffer, prepared->sparseMappings, sparseDependencies);
            if (!sparseDependency)
            {
                return std::unexpected(std::move(sparseDependency.error()));
            }
            sparseDependency =
                captureSparseTextureMapping(prepared->texture, prepared->sparseMappings, sparseDependencies);
            if (!sparseDependency)
            {
                return std::unexpected(std::move(sparseDependency.error()));
            }
            auto commandList = queues.commandList(commandBuffer);
            if (!commandList)
            {
                return std::unexpected(std::move(commandList.error()));
            }
            const std::array swapchainLeases{prepared->texture->swapchainLease};
            auto retained = retainResourceCommandDependencies(queues,
                                                              commandBuffer,
                                                              prepared,
                                                              prepared->residencies,
                                                              prepared->sparseMappings,
                                                              sparseDependencies,
                                                              swapchainLeases);
            if (!retained)
            {
                return std::unexpected(std::move(retained.error()));
            }
            return commandList;
        }

        [[nodiscard]] D3D12_TEXTURE_COPY_LOCATION placedFootprintCopyLocation(
            const BufferRecord& buffer, const LinearTextureLayout& layout) noexcept
        {
            return D3D12_TEXTURE_COPY_LOCATION{
                .pResource = buffer.resource.Get(),
                .Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                .PlacedFootprint = layout.footprint,
            };
        }

        [[nodiscard]] D3D12_TEXTURE_COPY_LOCATION subresourceCopyLocation(const TextureRecord& texture,
                                                                          const NativeSubresource& subresource) noexcept
        {
            return D3D12_TEXTURE_COPY_LOCATION{
                .pResource = texture.resource.Get(),
                .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                .SubresourceIndex = subresource.index,
            };
        }

        void recordBufferToTextureCopies(ID3D12GraphicsCommandList& commandList,
                                         const PreparedBufferTextureCopyBatch& batch)
        {
            for (std::size_t index = 0; index < batch.regions.size(); ++index)
            {
                const auto& region = batch.regions[index];
                const auto& layout = batch.layouts[index];
                const auto source = placedFootprintCopyLocation(*batch.buffer, layout);
                const auto destination = subresourceCopyLocation(*batch.texture, layout.subresource);
                const auto& extent = layout.blockExtent;
                const D3D12_BOX sourceBox{0, 0, 0, extent.width, extent.height, extent.depth};
                commandList.CopyTextureRegion(&destination,
                                              region.textureOrigin.x,
                                              region.textureOrigin.y,
                                              region.textureOrigin.z,
                                              &source,
                                              layout.wholeSubresource ? nullptr : &sourceBox);
            }
        }

        [[nodiscard]] Status prepareTextureToBufferLayout(ID3D12Device& device,
                                                          PreparedBufferTextureCopyBatch& batch,
                                                          const BufferTextureCopyRegion& region)
        {
            auto layout = linearTextureLayoutWithinBuffer(device, batch, region);
            if (!layout)
            {
                return std::unexpected(std::move(layout.error()));
            }
            if (!sparseBufferRangeIsMapped(
                    *batch.buffer, region.bufferOffset, layout->transfer.requiredBufferSize - region.bufferOffset))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A texture-to-buffer copy cannot access an unmapped sparse-buffer tile.",
                });
            }
            if (!sparseTextureRegionIsMapped(
                    *batch.texture, layout->subresource.index, region.textureOrigin, region.extent))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A texture-to-buffer copy cannot access an unmapped sparse-texture tile.",
                });
            }
            batch.layouts.push_back(*layout);
            return {};
        }

        [[nodiscard]] Status prepareTextureToBufferLayouts(ID3D12Device& device, PreparedBufferTextureCopyBatch& batch)
        {
            Status firstRejected = {};
            for (const auto& region : batch.regions)
            {
                firstRejected = prepareTextureToBufferLayout(device, batch, region);
                if (!firstRejected)
                {
                    break;
                }
            }
            const std::uint64_t pitch = batch.layouts.empty() ? 1U : batch.layouts[0].footprint.Footprint.RowPitch;
            const auto overlapping = anyOverlappingPair(
                batch.layouts.size(),
                [&](std::size_t index) {
                    const auto& layout = batch.layouts[index];
                    return linearFootprintSweepKey(layout.footprint.Offset, layout.transfer, pitch);
                },
                [&](std::size_t earlier, std::size_t later) {
                    return linearLayoutsOverlap(batch.layouts[later], batch.layouts[earlier]);
                });
            if (overlapping)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A texture-to-buffer batch cannot write overlapping destination bytes.",
                });
            }
            return firstRejected;
        }

        [[nodiscard]] Result<std::shared_ptr<PreparedBufferTextureCopyBatch>> prepareTextureToBufferCopyBatch(
            ID3D12Device& device,
            std::shared_ptr<TextureRecord> texture,
            std::shared_ptr<BufferRecord> buffer,
            std::span<const BufferTextureCopyRegion> regions)
        {
            if (!hasAny(texture->desc.usage, TextureUsage::copySource) ||
                !hasAny(buffer->desc.usage, BufferUsage::copyDestination))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A texture-to-buffer copy requires copy-source and copy-destination usages.",
                });
            }

            auto prepared = std::make_shared<PreparedBufferTextureCopyBatch>();
            prepared->buffer = std::move(buffer);
            prepared->texture = std::move(texture);
            appendRegions(prepared->regions, regions);
            prepared->layouts.reserve(regions.size());
            prepared->residencies = {prepared->texture->residency, prepared->buffer->residency};
            auto layouts = prepareTextureToBufferLayouts(device, *prepared);
            if (!layouts)
            {
                return std::unexpected(std::move(layouts.error()));
            }
            return prepared;
        }

        [[nodiscard]] constexpr D3D12_BOX nativeBox(Origin3D origin, Extent3D extent) noexcept
        {
            return D3D12_BOX{
                origin.x,
                origin.y,
                origin.z,
                origin.x + extent.width,
                origin.y + extent.height,
                origin.z + extent.depth,
            };
        }

        void recordTextureToBufferCopies(ID3D12GraphicsCommandList& commandList,
                                         const PreparedBufferTextureCopyBatch& batch)
        {
            for (std::size_t index = 0; index < batch.regions.size(); ++index)
            {
                const auto& region = batch.regions[index];
                const auto& layout = batch.layouts[index];
                const auto source = subresourceCopyLocation(*batch.texture, layout.subresource);
                const auto destination = placedFootprintCopyLocation(*batch.buffer, layout);
                const auto sourceBox = nativeBox(region.textureOrigin, layout.blockExtent);
                commandList.CopyTextureRegion(
                    &destination, 0, 0, 0, &source, layout.wholeSubresource ? nullptr : &sourceBox);
            }
        }

        [[nodiscard]] Result<PreparedTextureCopyRegion> prepareTextureCopyRegion(const PreparedTextureCopyBatch& batch,
                                                                                 const TextureCopyRegion& region,
                                                                                 bool requiresWholeSubresource)
        {
            if (region.sourceSubresource.aspect != region.destinationSubresource.aspect)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Texture copies require matching source and destination aspects.",
                });
            }
            auto sourceSubresource = nativeSubresource(*batch.source, region.sourceSubresource);
            auto destinationSubresource = nativeSubresource(*batch.destination, region.destinationSubresource);
            if (!sourceSubresource)
            {
                return std::unexpected(std::move(sourceSubresource.error()));
            }
            if (!destinationSubresource)
            {
                return std::unexpected(std::move(destinationSubresource.error()));
            }
            auto validSource =
                validateTextureRegion(*batch.source, *sourceSubresource, region.sourceOrigin, region.extent);
            auto validDestination = validateTextureRegion(
                *batch.destination, *destinationSubresource, region.destinationOrigin, region.extent);
            if (!validSource)
            {
                return std::unexpected(std::move(validSource.error()));
            }
            if (!validDestination)
            {
                return std::unexpected(std::move(validDestination.error()));
            }
            if (!sparseTextureRegionIsMapped(
                    *batch.source, sourceSubresource->index, region.sourceOrigin, region.extent) ||
                !sparseTextureRegionIsMapped(
                    *batch.destination, destinationSubresource->index, region.destinationOrigin, region.extent))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A texture copy cannot access an unmapped sparse-texture tile.",
                });
            }
            if (requiresWholeSubresource &&
                (!isWholeSubresource(region.sourceOrigin, region.extent, *sourceSubresource) ||
                 !isWholeSubresource(region.destinationOrigin, region.extent, *destinationSubresource)))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Depth-stencil and multisampled texture copies require whole matching subresources.",
                });
            }
            return PreparedTextureCopyRegion{
                .region = region,
                .sourceSubresource = *sourceSubresource,
                .destinationSubresource = *destinationSubresource,
                .blockExtent = blockAlignedExtent(*format::info(batch.source->desc.format), region.extent),
                .wholeSubresource = requiresWholeSubresource,
            };
        }

        [[nodiscard]] constexpr NativeTextureBox sourceTextureBox(const PreparedTextureCopyRegion& prepared) noexcept
        {
            return NativeTextureBox{
                .subresource = prepared.sourceSubresource.index,
                .origin = prepared.region.sourceOrigin,
                .extent = prepared.region.extent,
            };
        }

        [[nodiscard]] constexpr NativeTextureBox destinationTextureBox(
            const PreparedTextureCopyRegion& prepared) noexcept
        {
            return NativeTextureBox{
                .subresource = prepared.destinationSubresource.index,
                .origin = prepared.region.destinationOrigin,
                .extent = prepared.region.extent,
            };
        }

        [[nodiscard]] bool anySourceOverlapsAnyDestination(std::span<const PreparedTextureCopyRegion> regions)
        {
            return anyCrossOverlap(
                regions.size(),
                [&](std::size_t index) { return boxSweepKey(sourceTextureBox(regions[index])); },
                [&](std::size_t index) { return boxSweepKey(destinationTextureBox(regions[index])); },
                [&](std::size_t source, std::size_t destination) {
                    return textureBoxesOverlap(sourceTextureBox(regions[source]),
                                               destinationTextureBox(regions[destination]));
                });
        }

        [[nodiscard]] Status appendTextureCopyRegion(PreparedTextureCopyBatch& batch,
                                                     const TextureCopyRegion& region,
                                                     bool requiresWholeSubresource,
                                                     bool sameTexture)
        {
            auto prepared = prepareTextureCopyRegion(batch, region, requiresWholeSubresource);
            if (!prepared)
            {
                return std::unexpected(std::move(prepared.error()));
            }
            if (sameTexture && prepared->sourceSubresource.index == prepared->destinationSubresource.index)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A texture copy cannot use the same subresource as both source and destination.",
                });
            }
            batch.regions.push_back(*prepared);
            return {};
        }

        [[nodiscard]] Status prepareTextureCopyRegions(PreparedTextureCopyBatch& batch,
                                                       std::span<const TextureCopyRegion> regions,
                                                       bool requiresWholeSubresource,
                                                       bool sameTexture)
        {
            Status firstRejected = {};
            for (const auto& region : regions)
            {
                firstRejected = appendTextureCopyRegion(batch, region, requiresWholeSubresource, sameTexture);
                if (!firstRejected)
                {
                    break;
                }
            }
            const auto overlapping = anyOverlappingPair(
                batch.regions.size(),
                [&](std::size_t index) { return boxSweepKey(destinationTextureBox(batch.regions[index])); },
                [&](std::size_t earlier, std::size_t later) {
                    return textureBoxesOverlap(destinationTextureBox(batch.regions[later]),
                                               destinationTextureBox(batch.regions[earlier]));
                });
            if (overlapping)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A texture-copy batch cannot write overlapping destination texels.",
                });
            }
            if (!firstRejected)
            {
                return firstRejected;
            }

            if (sameTexture && anySourceOverlapsAnyDestination(batch.regions))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A same-texture copy batch cannot access one texel as both source and destination.",
                });
            }
            return {};
        }

        [[nodiscard]] Result<std::shared_ptr<PreparedTextureCopyBatch>> prepareTextureCopyBatch(
            std::shared_ptr<TextureRecord> source,
            std::shared_ptr<TextureRecord> destination,
            std::span<const TextureCopyRegion> regions,
            bool sameTexture)
        {
            if (!hasAny(source->desc.usage, TextureUsage::copySource) ||
                !hasAny(destination->desc.usage, TextureUsage::copyDestination))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A texture copy requires copy-source and copy-destination usages.",
                });
            }
            const auto sourceInfo = format::info(source->desc.format);
            if (!sourceInfo.has_value() ||
                !isTextureViewFormatCompatible(source->desc.format, destination->desc.format) ||
                source->desc.sampleCount != destination->desc.sampleCount)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Texture copies require copy-compatible formats and matching sample counts.",
                });
            }

            auto prepared = std::make_shared<PreparedTextureCopyBatch>();
            prepared->source = std::move(source);
            prepared->destination = std::move(destination);
            prepared->regions.reserve(regions.size());
            prepared->residencies = {prepared->source->residency, prepared->destination->residency};
            const bool requiresWholeSubresource = sourceInfo->depth || prepared->source->desc.sampleCount != 1;
            auto preparedRegions = prepareTextureCopyRegions(*prepared, regions, requiresWholeSubresource, sameTexture);
            if (!preparedRegions)
            {
                return std::unexpected(std::move(preparedRegions.error()));
            }
            return prepared;
        }

        [[nodiscard]] Result<ID3D12GraphicsCommandList*> retainCopyBatchForRecording(
            QueueManager& queues,
            detail::BackendCommandBuffer& commandBuffer,
            const std::shared_ptr<PreparedTextureCopyBatch>& prepared)
        {
            SparseMappingDependencies sparseDependencies;
            auto sparseDependency =
                captureSparseTextureMapping(prepared->source, prepared->sparseMappings, sparseDependencies);
            if (!sparseDependency)
            {
                return std::unexpected(std::move(sparseDependency.error()));
            }
            sparseDependency =
                captureSparseTextureMapping(prepared->destination, prepared->sparseMappings, sparseDependencies);
            if (!sparseDependency)
            {
                return std::unexpected(std::move(sparseDependency.error()));
            }
            auto commandList = queues.commandList(commandBuffer);
            if (!commandList)
            {
                return std::unexpected(std::move(commandList.error()));
            }
            const std::array swapchainLeases{
                prepared->source->swapchainLease,
                prepared->destination->swapchainLease,
            };
            auto retained = retainResourceCommandDependencies(queues,
                                                              commandBuffer,
                                                              prepared,
                                                              prepared->residencies,
                                                              prepared->sparseMappings,
                                                              sparseDependencies,
                                                              swapchainLeases);
            if (!retained)
            {
                return std::unexpected(std::move(retained.error()));
            }
            return commandList;
        }

        void recordTextureCopies(ID3D12GraphicsCommandList& commandList, const PreparedTextureCopyBatch& batch)
        {
            for (const auto& item : batch.regions)
            {
                const auto sourceLocation = subresourceCopyLocation(*batch.source, item.sourceSubresource);
                const auto destinationLocation =
                    subresourceCopyLocation(*batch.destination, item.destinationSubresource);
                const auto& region = item.region;
                const auto sourceBox = nativeBox(region.sourceOrigin, item.blockExtent);
                commandList.CopyTextureRegion(&destinationLocation,
                                              region.destinationOrigin.x,
                                              region.destinationOrigin.y,
                                              region.destinationOrigin.z,
                                              &sourceLocation,
                                              item.wholeSubresource ? nullptr : &sourceBox);
            }
        }

        [[nodiscard]] Status validateTextureResolve(ID3D12Device& device,
                                                    const TextureRecord& source,
                                                    const TextureRecord& destination,
                                                    const TextureResolve& resolve)
        {
            const auto sourceInfo = format::info(source.desc.format);
            if (!sourceInfo.has_value() || source.desc.format != destination.desc.format ||
                resolve.sourceSubresource.aspect != resolve.destinationSubresource.aspect)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A texture resolve requires identical formats and matching aspects.",
                });
            }
            if (source.desc.sampleCount == 1 || destination.desc.sampleCount != 1 ||
                source.desc.dimension != TextureDimension::texture2D ||
                destination.desc.dimension != TextureDimension::texture2D)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A texture resolve requires a multisampled 2D source and a single-sampled 2D destination.",
                });
            }
            const bool depthStencilFormat = isDepthFormat(*sourceInfo);
            const TextureUsage requiredUsage =
                depthStencilFormat ? TextureUsage::depthStencil : TextureUsage::renderTarget;
            if (!hasAny(source.desc.usage, requiredUsage) || !hasAny(destination.desc.usage, requiredUsage))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Texture resolve resources must both have matching attachment usage.",
                });
            }
            const bool depthResolve = resolve.sourceSubresource.aspect == TextureAspect::depth;
            const bool stencilResolve = resolve.sourceSubresource.aspect == TextureAspect::stencil;
            const bool colorResolve = resolve.sourceSubresource.aspect == TextureAspect::color;
            const bool averaging = resolve.mode == ResolveMode::average;
            const bool validAspectAndMode =
                depthStencilFormat ? (depthResolve || stencilResolve) && !averaging : colorResolve && averaging;
            if (!validAspectAndMode)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Color resolves use average mode; depth and stencil resolves use minimum or maximum mode.",
                });
            }
            const bool resolvableDepthStencil = (source.desc.format == Format::d32Float && depthResolve) ||
                                                source.desc.format == Format::d24UnormS8Uint;
            if (depthStencilFormat && !resolvableDepthStencil)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The requested depth/stencil format and aspect do not support multisample resolve.",
                });
            }
            if (colorResolve)
            {
                return requireFormatSupport(device,
                                            sourceInfo->view,
                                            D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RESOLVE,
                                            D3D12_FORMAT_SUPPORT2_NONE,
                                            "average multisample resolves");
            }
            return {};
        }

        struct PreparedTextureResolve final
        {
            std::shared_ptr<TextureRecord> source;
            std::shared_ptr<TextureRecord> destination;
            TextureResolve request;
            format::Info sourceInfo;
            NativeSubresource sourceSubresource;
            NativeSubresource destinationSubresource;
        };

        [[nodiscard]] Result<PreparedTextureResolve> prepareTextureResolve(ID3D12Device& device,
                                                                           std::shared_ptr<TextureRecord> source,
                                                                           std::shared_ptr<TextureRecord> destination,
                                                                           const TextureResolve& resolve)
        {
            auto validResolve = validateTextureResolve(device, *source, *destination, resolve);
            if (!validResolve)
            {
                return std::unexpected(std::move(validResolve.error()));
            }
            auto sourceSubresource = nativeSubresource(*source, resolve.sourceSubresource);
            auto destinationSubresource = nativeSubresource(*destination, resolve.destinationSubresource);
            if (!sourceSubresource)
            {
                return std::unexpected(std::move(sourceSubresource.error()));
            }
            if (!destinationSubresource)
            {
                return std::unexpected(std::move(destinationSubresource.error()));
            }
            if (sourceSubresource->mipExtent.width != destinationSubresource->mipExtent.width ||
                sourceSubresource->mipExtent.height != destinationSubresource->mipExtent.height ||
                sourceSubresource->mipExtent.depth != destinationSubresource->mipExtent.depth)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A texture resolve requires matching source and destination subresource extents.",
                });
            }

            const auto sourceInfo = *format::info(source->desc.format);
            return PreparedTextureResolve{
                .source = std::move(source),
                .destination = std::move(destination),
                .request = resolve,
                .sourceInfo = sourceInfo,
                .sourceSubresource = *sourceSubresource,
                .destinationSubresource = *destinationSubresource,
            };
        }

        [[nodiscard]] QueueManager::ResourceRetention textureRetention(const std::shared_ptr<TextureRecord>& texture)
        {
            return QueueManager::ResourceRetention{
                .owner = texture,
                .residency = texture->residency,
            };
        }

        [[nodiscard]] Status retainResolveTextures(QueueManager& queues,
                                                   detail::BackendCommandBuffer& commandBuffer,
                                                   const PreparedTextureResolve& prepared)
        {
            const std::array dependencies{
                textureRetention(prepared.source),
                textureRetention(prepared.destination),
            };
            const std::array swapchainLeases{prepared.source->swapchainLease, prepared.destination->swapchainLease};
            return queues.retainCommandDependencies(commandBuffer, dependencies, swapchainLeases);
        }

        [[nodiscard]] Status recordColorResolve(QueueManager& queues,
                                                detail::BackendCommandBuffer& commandBuffer,
                                                ID3D12GraphicsCommandList& commandList,
                                                const PreparedTextureResolve& prepared)
        {
            auto retained = retainResolveTextures(queues, commandBuffer, prepared);
            if (!retained)
            {
                return retained;
            }
            commandList.ResolveSubresource(prepared.destination->resource.Get(),
                                           prepared.destinationSubresource.index,
                                           prepared.source->resource.Get(),
                                           prepared.sourceSubresource.index,
                                           prepared.sourceInfo.view);
            return {};
        }

        [[nodiscard]] Status recordPackedDepthStencilResolve(QueueManager& queues,
                                                             detail::BackendCommandBuffer& commandBuffer,
                                                             const PreparedTextureResolve& prepared)
        {
            auto commandList = queues.commandList1(commandBuffer);
            if (!commandList)
            {
                return std::unexpected(std::move(commandList.error()));
            }
            auto retained = retainResolveTextures(queues, commandBuffer, prepared);
            if (!retained)
            {
                return retained;
            }
            const auto nativeMode =
                prepared.request.mode == ResolveMode::maximum ? D3D12_RESOLVE_MODE_MAX : D3D12_RESOLVE_MODE_MIN;
            (*commandList)
                ->ResolveSubresourceRegion(prepared.destination->resource.Get(),
                                           prepared.destinationSubresource.index,
                                           0,
                                           0,
                                           prepared.source->resource.Get(),
                                           prepared.sourceSubresource.index,
                                           nullptr,
                                           DXGI_FORMAT_D24_UNORM_S8_UINT,
                                           nativeMode);
            return {};
        }

        struct DepthResolvePipeline final
        {
            ID3D12Device& device;
            CpuDescriptorPool& destinationDescriptorPool;
            ID3D12RootSignature* rootSignature = nullptr;
            ID3D12PipelineState* pipelineState = nullptr;
            bool enhancedBarriers = false;
        };

        struct DepthResolveDraw final
        {
            NativePipelineContinuation pipelineContinuation;
            std::shared_ptr<GpuDescriptorLease> sourceDescriptor;
            std::shared_ptr<CpuDescriptorLease> destinationDescriptor;
            ID3D12GraphicsCommandList7* enhancedCommandList = nullptr;
        };

        [[nodiscard]] Result<DepthResolveDraw> prepareDepthResolveDraw(const DepthResolvePipeline& resolvePipeline,
                                                                       QueueManager& queues,
                                                                       BindingManager& bindings,
                                                                       const PipelineManager& pipelines,
                                                                       detail::BackendCommandBuffer& commandBuffer)
        {
            auto pipelineContinuation = pipelines.prepareNativePipelineContinuation(queues, commandBuffer);
            if (!pipelineContinuation)
            {
                return std::unexpected(std::move(pipelineContinuation.error()));
            }
            auto sourceDescriptor = bindings.allocateResourceDescriptors(1);
            if (!sourceDescriptor)
            {
                return std::unexpected(std::move(sourceDescriptor.error()));
            }
            auto destinationDescriptor = resolvePipeline.destinationDescriptorPool.allocate(resolvePipeline.device);
            if (!destinationDescriptor)
            {
                return std::unexpected(std::move(destinationDescriptor.error()));
            }
            ID3D12GraphicsCommandList7* enhancedCommandList = nullptr;
            if (resolvePipeline.enhancedBarriers)
            {
                auto preparedCommandList = queues.commandList7(commandBuffer);
                if (!preparedCommandList)
                {
                    return std::unexpected(std::move(preparedCommandList.error()));
                }
                enhancedCommandList = *preparedCommandList;
            }
            return DepthResolveDraw{
                .pipelineContinuation = *pipelineContinuation,
                .sourceDescriptor = std::move(*sourceDescriptor),
                .destinationDescriptor = std::move(*destinationDescriptor),
                .enhancedCommandList = enhancedCommandList,
            };
        }

        void createDepthResolveViews(ID3D12Device& device,
                                     const PreparedTextureResolve& prepared,
                                     const DepthResolveDraw& draw)
        {
            const D3D12_SHADER_RESOURCE_VIEW_DESC sourceView{
                .Format = DXGI_FORMAT_R32_FLOAT,
                .ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY,
                .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                .Texture2DMSArray =
                    {
                        .FirstArraySlice = prepared.request.sourceSubresource.arrayLayer,
                        .ArraySize = 1,
                    },
            };
            const D3D12_DEPTH_STENCIL_VIEW_DESC destinationView{
                .Format = DXGI_FORMAT_D32_FLOAT,
                .ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY,
                .Flags = D3D12_DSV_FLAG_NONE,
                .Texture2DArray =
                    {
                        .MipSlice = prepared.request.destinationSubresource.mipLevel,
                        .FirstArraySlice = prepared.request.destinationSubresource.arrayLayer,
                        .ArraySize = 1,
                    },
            };
            device.CreateShaderResourceView(
                prepared.source->resource.Get(), &sourceView, draw.sourceDescriptor->cpuHandle());
            device.CreateDepthStencilView(
                prepared.destination->resource.Get(), &destinationView, draw.destinationDescriptor->handle());
        }

        [[nodiscard]] Status retainDepthResolveDraw(QueueManager& queues,
                                                    detail::BackendCommandBuffer& commandBuffer,
                                                    const PreparedTextureResolve& prepared,
                                                    const DepthResolveDraw& draw)
        {
            const std::array dependencies{
                textureRetention(prepared.source),
                textureRetention(prepared.destination),
                QueueManager::ResourceRetention{
                    .owner = draw.sourceDescriptor,
                    .internal = true,
                },
                QueueManager::ResourceRetention{
                    .owner = draw.destinationDescriptor,
                    .internal = true,
                },
            };
            const std::array swapchainLeases{prepared.source->swapchainLease, prepared.destination->swapchainLease};
            return queues.retainCommandDependencies(commandBuffer, dependencies, swapchainLeases);
        }

        [[nodiscard]] constexpr D3D12_BARRIER_SUBRESOURCE_RANGE barrierSubresourceRange(
            const TextureSubresource& subresource, const NativeSubresource& native) noexcept
        {
            return D3D12_BARRIER_SUBRESOURCE_RANGE{
                .IndexOrFirstMipLevel = subresource.mipLevel,
                .NumMipLevels = 1,
                .FirstArraySlice = subresource.arrayLayer,
                .NumArraySlices = 1,
                .FirstPlane = native.plane,
                .NumPlanes = 1,
            };
        }

        void recordTextureBarriers(ID3D12GraphicsCommandList7& commandList,
                                   std::span<const D3D12_TEXTURE_BARRIER> barriers)
        {
            const D3D12_BARRIER_GROUP group{
                .Type = D3D12_BARRIER_TYPE_TEXTURE,
                .NumBarriers = static_cast<UINT>(barriers.size()),
                .pTextureBarriers = barriers.data(),
            };
            commandList.Barrier(1, &group);
        }

        void beginDepthResolveDrawWithEnhancedBarriers(ID3D12GraphicsCommandList7& commandList,
                                                       const PreparedTextureResolve& prepared)
        {
            const std::array barriers{
                D3D12_TEXTURE_BARRIER{
                    .SyncBefore = D3D12_BARRIER_SYNC_RESOLVE,
                    .SyncAfter = D3D12_BARRIER_SYNC_PIXEL_SHADING,
                    .AccessBefore = D3D12_BARRIER_ACCESS_RESOLVE_SOURCE,
                    .AccessAfter = D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                    .LayoutBefore = D3D12_BARRIER_LAYOUT_RESOLVE_SOURCE,
                    .LayoutAfter = D3D12_BARRIER_LAYOUT_SHADER_RESOURCE,
                    .pResource = prepared.source->resource.Get(),
                    .Subresources =
                        barrierSubresourceRange(prepared.request.sourceSubresource, prepared.sourceSubresource),
                    .Flags = {},
                },
                D3D12_TEXTURE_BARRIER{
                    .SyncBefore = D3D12_BARRIER_SYNC_RESOLVE,
                    .SyncAfter = D3D12_BARRIER_SYNC_DEPTH_STENCIL,
                    .AccessBefore = D3D12_BARRIER_ACCESS_RESOLVE_DEST,
                    .AccessAfter = D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE,
                    .LayoutBefore = D3D12_BARRIER_LAYOUT_RESOLVE_DEST,
                    .LayoutAfter = D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE,
                    .pResource = prepared.destination->resource.Get(),
                    .Subresources = barrierSubresourceRange(prepared.request.destinationSubresource,
                                                            prepared.destinationSubresource),
                    .Flags = {},
                },
            };
            recordTextureBarriers(commandList, barriers);
        }

        void endDepthResolveDrawWithEnhancedBarriers(ID3D12GraphicsCommandList7& commandList,
                                                     const PreparedTextureResolve& prepared)
        {
            const std::array barriers{
                D3D12_TEXTURE_BARRIER{
                    .SyncBefore = D3D12_BARRIER_SYNC_PIXEL_SHADING,
                    .SyncAfter = D3D12_BARRIER_SYNC_RESOLVE,
                    .AccessBefore = D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                    .AccessAfter = D3D12_BARRIER_ACCESS_RESOLVE_SOURCE,
                    .LayoutBefore = D3D12_BARRIER_LAYOUT_SHADER_RESOURCE,
                    .LayoutAfter = D3D12_BARRIER_LAYOUT_RESOLVE_SOURCE,
                    .pResource = prepared.source->resource.Get(),
                    .Subresources =
                        barrierSubresourceRange(prepared.request.sourceSubresource, prepared.sourceSubresource),
                    .Flags = {},
                },
                D3D12_TEXTURE_BARRIER{
                    .SyncBefore = D3D12_BARRIER_SYNC_DEPTH_STENCIL,
                    .SyncAfter = D3D12_BARRIER_SYNC_RESOLVE,
                    .AccessBefore = D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE,
                    .AccessAfter = D3D12_BARRIER_ACCESS_RESOLVE_DEST,
                    .LayoutBefore = D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE,
                    .LayoutAfter = D3D12_BARRIER_LAYOUT_RESOLVE_DEST,
                    .pResource = prepared.destination->resource.Get(),
                    .Subresources = barrierSubresourceRange(prepared.request.destinationSubresource,
                                                            prepared.destinationSubresource),
                    .Flags = {},
                },
            };
            recordTextureBarriers(commandList, barriers);
        }

        [[nodiscard]] D3D12_RESOURCE_BARRIER legacyTransition(const TextureRecord& texture,
                                                              const NativeSubresource& subresource,
                                                              D3D12_RESOURCE_STATES before,
                                                              D3D12_RESOURCE_STATES after) noexcept
        {
            return D3D12_RESOURCE_BARRIER{
                .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
                .Transition =
                    {
                        .pResource = texture.resource.Get(),
                        .Subresource = subresource.index,
                        .StateBefore = before,
                        .StateAfter = after,
                    },
            };
        }

        void beginDepthResolveDrawWithLegacyBarriers(ID3D12GraphicsCommandList& commandList,
                                                     const PreparedTextureResolve& prepared)
        {
            const std::array barriers{
                legacyTransition(*prepared.source,
                                 prepared.sourceSubresource,
                                 D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
                                 D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
                legacyTransition(*prepared.destination,
                                 prepared.destinationSubresource,
                                 D3D12_RESOURCE_STATE_RESOLVE_DEST,
                                 D3D12_RESOURCE_STATE_DEPTH_WRITE),
            };
            commandList.ResourceBarrier(static_cast<UINT>(barriers.size()), barriers.data());
        }

        void endDepthResolveDrawWithLegacyBarriers(ID3D12GraphicsCommandList& commandList,
                                                   const PreparedTextureResolve& prepared)
        {
            const std::array barriers{
                legacyTransition(*prepared.source,
                                 prepared.sourceSubresource,
                                 D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                                 D3D12_RESOURCE_STATE_RESOLVE_SOURCE),
                legacyTransition(*prepared.destination,
                                 prepared.destinationSubresource,
                                 D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                 D3D12_RESOURCE_STATE_RESOLVE_DEST),
            };
            commandList.ResourceBarrier(static_cast<UINT>(barriers.size()), barriers.data());
        }

        void drawFullScreenDepthResolve(ID3D12GraphicsCommandList& commandList,
                                        const BindingManager& bindings,
                                        const DepthResolvePipeline& resolvePipeline,
                                        const PreparedTextureResolve& prepared,
                                        const DepthResolveDraw& draw)
        {
            const D3D12_DISCARD_REGION discard{
                .NumRects = 0,
                .pRects = nullptr,
                .FirstSubresource = prepared.destinationSubresource.index,
                .NumSubresources = 1,
            };
            commandList.DiscardResource(prepared.destination->resource.Get(), &discard);

            if (!draw.pipelineContinuation.hasPipeline())
            {
                const auto nativeHeaps = bindings.nativeDescriptorHeaps();
                ID3D12DescriptorHeap* const heaps[]{nativeHeaps.resources, nativeHeaps.samplers};
                commandList.SetDescriptorHeaps(static_cast<UINT>(std::size(heaps)), heaps);
            }
            commandList.SetGraphicsRootSignature(resolvePipeline.rootSignature);
            commandList.SetPipelineState(resolvePipeline.pipelineState);
            commandList.SetGraphicsRootDescriptorTable(0, draw.sourceDescriptor->gpuHandle());
            const std::array constants{
                prepared.source->desc.sampleCount,
                prepared.request.mode == ResolveMode::maximum ? 1U : 0U,
            };
            commandList.SetGraphicsRoot32BitConstants(1, static_cast<UINT>(constants.size()), constants.data(), 0);
            const D3D12_VIEWPORT viewport{
                .TopLeftX = 0.0F,
                .TopLeftY = 0.0F,
                .Width = static_cast<float>(prepared.sourceSubresource.mipExtent.width),
                .Height = static_cast<float>(prepared.sourceSubresource.mipExtent.height),
                .MinDepth = 0.0F,
                .MaxDepth = 1.0F,
            };
            const D3D12_RECT scissor{
                .left = 0,
                .top = 0,
                .right = static_cast<LONG>(prepared.sourceSubresource.mipExtent.width),
                .bottom = static_cast<LONG>(prepared.sourceSubresource.mipExtent.height),
            };
            const auto depthStencil = draw.destinationDescriptor->handle();
            commandList.RSSetViewports(1, &viewport);
            commandList.RSSetScissorRects(1, &scissor);
            commandList.OMSetRenderTargets(0, nullptr, FALSE, &depthStencil);
            commandList.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            commandList.DrawInstanced(3, 1, 0, 0);
        }

        void recordDepthResolveBetweenBarriers(ID3D12GraphicsCommandList& commandList,
                                               const BindingManager& bindings,
                                               const DepthResolvePipeline& resolvePipeline,
                                               const PreparedTextureResolve& prepared,
                                               const DepthResolveDraw& draw)
        {
            if (resolvePipeline.enhancedBarriers)
            {
                beginDepthResolveDrawWithEnhancedBarriers(*draw.enhancedCommandList, prepared);
                drawFullScreenDepthResolve(commandList, bindings, resolvePipeline, prepared, draw);
                endDepthResolveDrawWithEnhancedBarriers(*draw.enhancedCommandList, prepared);
            }
            else
            {
                beginDepthResolveDrawWithLegacyBarriers(commandList, prepared);
                drawFullScreenDepthResolve(commandList, bindings, resolvePipeline, prepared, draw);
                endDepthResolveDrawWithLegacyBarriers(commandList, prepared);
            }
            draw.pipelineContinuation.restore(commandList);
        }

        using CopiedSubresources = SmallVector<std::uint32_t, 2>;

        [[nodiscard]] CopiedSubresources copiedSubresources(std::span<const LinearTextureLayout> layouts)
        {
            CopiedSubresources subresources;
            subresources.reserve(layouts.size());
            for (const auto& layout : layouts)
            {
                subresources.push_back(layout.subresource.index);
            }
            return subresources;
        }

        struct CopiedTextureSubresources final
        {
            CopiedSubresources sources;
            CopiedSubresources destinations;
            CopiedSubresources shared;
        };

        void removeSubresources(CopiedSubresources& subresources, std::span<const std::uint32_t> removed) noexcept
        {
            const auto kept = std::ranges::remove_if(subresources, [&](std::uint32_t subresource) {
                return std::ranges::binary_search(removed, subresource);
            });
            subresources.truncate(static_cast<std::size_t>(kept.begin() - subresources.begin()));
        }

        void sortUniqueSubresources(CopiedSubresources& subresources) noexcept
        {
            std::ranges::sort(subresources);
            const auto duplicates = std::ranges::unique(subresources);
            subresources.truncate(static_cast<std::size_t>(duplicates.begin() - subresources.begin()));
        }

        [[nodiscard]] CopiedTextureSubresources copiedSubresources(const PreparedTextureCopyBatch& batch)
        {
            CopiedTextureSubresources copied;
            copied.sources.reserve(batch.regions.size());
            copied.destinations.reserve(batch.regions.size());
            for (const auto& region : batch.regions)
            {
                copied.sources.push_back(region.sourceSubresource.index);
                copied.destinations.push_back(region.destinationSubresource.index);
            }
            if (batch.source != batch.destination)
            {
                return copied;
            }
            sortUniqueSubresources(copied.sources);
            sortUniqueSubresources(copied.destinations);
            for (const auto subresource : copied.sources)
            {
                if (std::ranges::binary_search(copied.destinations, subresource))
                {
                    copied.shared.push_back(subresource);
                }
            }
            removeSubresources(copied.sources, copied.shared.span());
            removeSubresources(copied.destinations, copied.shared.span());
            return copied;
        }

    } // namespace

    Status ResourceManager::copyBufferRegions(QueueManager& queues,
                                              detail::BackendCommandBuffer& commandBuffer,
                                              BufferHandle sourceHandle,
                                              BufferHandle destinationHandle,
                                              std::span<const BufferCopyRegion> regions)
    {
        if (regions.empty() || regions.size() > std::numeric_limits<UINT>::max())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A buffer-copy batch must contain at least one region and fit within portable count limits.",
            });
        }
        if (commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Buffer copies cannot be recorded inside a rendering scope.",
            });
        }

        std::unique_lock<std::mutex> mappingLock(impl_->sparseMappingsMutex, std::defer_lock);
        if (impl_->activeSparseResourceCount.load(std::memory_order_acquire) != 0)
        {
            mappingLock.lock();
        }
        auto source = impl_->buffer(sourceHandle);
        auto destination = impl_->buffer(destinationHandle);
        if (!mappingLock.owns_lock() && ((source && (*source)->sparse) || (destination && (*destination)->sparse)))
        {
            mappingLock.lock();
            source = impl_->buffer(sourceHandle);
            destination = impl_->buffer(destinationHandle);
        }
        if (!source)
        {
            return std::unexpected(std::move(source.error()));
        }
        if (!destination)
        {
            return std::unexpected(std::move(destination.error()));
        }
        auto prepared = prepareBufferCopyBatch(
            std::move(*source), std::move(*destination), regions, sourceHandle == destinationHandle);
        if (!prepared)
        {
            return std::unexpected(std::move(prepared.error()));
        }
        if (sourceHandle == destinationHandle && !impl_->enhancedBarriers)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Legacy D3D12 cannot copy within one buffer, which would need the copy-source and copy-destination "
                "states at once.",
            });
        }

        auto commandList = retainCopyBatchForRecording(queues, commandBuffer, *prepared);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        recordBufferCopies(**commandList, **prepared);
        test::noteResourceCommandRecorded();
        return {};
    }

    Result<TextureTransferLayout> ResourceManager::queryTextureTransferLayout(
        TextureHandle textureHandle, const BufferTextureCopyRegion& region) const
    {
        auto texture = impl_->texture(textureHandle);
        if (!texture)
        {
            return std::unexpected(std::move(texture.error()));
        }
        auto layout = linearTextureLayout(*impl_->device, **texture, region);
        if (!layout)
        {
            return std::unexpected(std::move(layout.error()));
        }
        return layout->transfer;
    }

    Status ResourceManager::copyBufferToTextureRegions(QueueManager& queues,
                                                       detail::BackendCommandBuffer& commandBuffer,
                                                       BufferHandle sourceHandle,
                                                       TextureHandle destinationHandle,
                                                       std::span<const BufferTextureCopyRegion> regions)
    {
        if (regions.empty() || regions.size() > std::numeric_limits<UINT>::max())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A buffer-to-texture batch must contain at least one region and fit within portable count limits.",
            });
        }
        if (commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Buffer-to-texture copies cannot be recorded inside a rendering scope.",
            });
        }

        std::unique_lock<std::mutex> mappingLock(impl_->sparseMappingsMutex, std::defer_lock);
        if (impl_->activeSparseResourceCount.load(std::memory_order_acquire) != 0)
        {
            mappingLock.lock();
        }
        auto buffer = impl_->buffer(sourceHandle);
        auto texture = impl_->texture(destinationHandle);
        if (!mappingLock.owns_lock() && ((buffer && (*buffer)->sparse) || (texture && (*texture)->sparse)))
        {
            mappingLock.lock();
            buffer = impl_->buffer(sourceHandle);
            texture = impl_->texture(destinationHandle);
        }
        if (!buffer)
        {
            return std::unexpected(std::move(buffer.error()));
        }
        if (!texture)
        {
            return std::unexpected(std::move(texture.error()));
        }
        auto prepared =
            prepareBufferToTextureCopyBatch(*impl_->device, std::move(*buffer), std::move(*texture), regions);
        if (!prepared)
        {
            return std::unexpected(std::move(prepared.error()));
        }
        auto layoutRecording = queues.legacyTextureStates(commandBuffer);
        if (!layoutRecording)
        {
            return std::unexpected(std::move(layoutRecording.error()));
        }
        LegacyTextureStateInsertionGuard insertions(*layoutRecording);
        const auto copied = copiedSubresources((*prepared)->layouts.span());
        auto layouts = prepareTextureLayoutRequirement(**layoutRecording,
                                                       *(*prepared)->texture,
                                                       copied.span(),
                                                       TextureLayout::copyDestination,
                                                       TextureLayout::common,
                                                       "copyBufferToTexture");
        if (!layouts)
        {
            return std::unexpected(std::move(layouts.error()));
        }

        auto commandList = retainCopyBatchForRecording(queues, commandBuffer, *prepared);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        recordBufferToTextureCopies(**commandList, **prepared);
        commitTextureLayoutRequirement(*layouts);
        insertions.commit();
        test::noteResourceCommandRecorded();
        return {};
    }

    Status ResourceManager::copyTextureToBufferRegions(QueueManager& queues,
                                                       detail::BackendCommandBuffer& commandBuffer,
                                                       TextureHandle sourceHandle,
                                                       BufferHandle destinationHandle,
                                                       std::span<const BufferTextureCopyRegion> regions)
    {
        if (regions.empty() || regions.size() > std::numeric_limits<UINT>::max())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A texture-to-buffer batch must contain at least one region and fit within portable count limits.",
            });
        }
        if (commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Texture-to-buffer copies cannot be recorded inside a rendering scope.",
            });
        }

        std::unique_lock<std::mutex> mappingLock(impl_->sparseMappingsMutex, std::defer_lock);
        if (impl_->activeSparseResourceCount.load(std::memory_order_acquire) != 0)
        {
            mappingLock.lock();
        }
        auto texture = impl_->texture(sourceHandle);
        auto buffer = impl_->buffer(destinationHandle);
        if (!mappingLock.owns_lock() && ((texture && (*texture)->sparse) || (buffer && (*buffer)->sparse)))
        {
            mappingLock.lock();
            texture = impl_->texture(sourceHandle);
            buffer = impl_->buffer(destinationHandle);
        }
        if (!texture)
        {
            return std::unexpected(std::move(texture.error()));
        }
        if (!buffer)
        {
            return std::unexpected(std::move(buffer.error()));
        }
        auto prepared =
            prepareTextureToBufferCopyBatch(*impl_->device, std::move(*texture), std::move(*buffer), regions);
        if (!prepared)
        {
            return std::unexpected(std::move(prepared.error()));
        }
        auto layoutRecording = queues.legacyTextureStates(commandBuffer);
        if (!layoutRecording)
        {
            return std::unexpected(std::move(layoutRecording.error()));
        }
        LegacyTextureStateInsertionGuard insertions(*layoutRecording);
        const auto copied = copiedSubresources((*prepared)->layouts.span());
        auto layouts = prepareTextureLayoutRequirement(**layoutRecording,
                                                       *(*prepared)->texture,
                                                       copied.span(),
                                                       TextureLayout::copySource,
                                                       TextureLayout::common,
                                                       "copyTextureToBuffer");
        if (!layouts)
        {
            return std::unexpected(std::move(layouts.error()));
        }

        auto commandList = retainCopyBatchForRecording(queues, commandBuffer, *prepared);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        recordTextureToBufferCopies(**commandList, **prepared);
        commitTextureLayoutRequirement(*layouts);
        insertions.commit();
        test::noteResourceCommandRecorded();
        return {};
    }

    Status ResourceManager::copyTextureRegions(QueueManager& queues,
                                               detail::BackendCommandBuffer& commandBuffer,
                                               TextureHandle sourceHandle,
                                               TextureHandle destinationHandle,
                                               std::span<const TextureCopyRegion> regions)
    {
        if (regions.empty() || regions.size() > std::numeric_limits<UINT>::max())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A texture-copy batch must contain at least one region and fit within portable count limits.",
            });
        }
        if (commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Texture copies cannot be recorded inside a rendering scope.",
            });
        }

        std::unique_lock<std::mutex> mappingLock(impl_->sparseMappingsMutex, std::defer_lock);
        if (impl_->activeSparseResourceCount.load(std::memory_order_acquire) != 0)
        {
            mappingLock.lock();
        }
        auto source = impl_->texture(sourceHandle);
        auto destination = impl_->texture(destinationHandle);
        if (!mappingLock.owns_lock() && ((source && (*source)->sparse) || (destination && (*destination)->sparse)))
        {
            mappingLock.lock();
            source = impl_->texture(sourceHandle);
            destination = impl_->texture(destinationHandle);
        }
        if (!source)
        {
            return std::unexpected(std::move(source.error()));
        }
        if (!destination)
        {
            return std::unexpected(std::move(destination.error()));
        }
        auto prepared = prepareTextureCopyBatch(
            std::move(*source), std::move(*destination), regions, sourceHandle == destinationHandle);
        if (!prepared)
        {
            return std::unexpected(std::move(prepared.error()));
        }
        auto layoutRecording = queues.legacyTextureStates(commandBuffer);
        if (!layoutRecording)
        {
            return std::unexpected(std::move(layoutRecording.error()));
        }
        LegacyTextureStateInsertionGuard insertions(*layoutRecording);
        const auto copied = copiedSubresources(**prepared);
        auto sourceLayouts = prepareTextureLayoutRequirement(**layoutRecording,
                                                             *(*prepared)->source,
                                                             copied.sources.span(),
                                                             TextureLayout::copySource,
                                                             TextureLayout::common,
                                                             "copyTexture");
        if (!sourceLayouts)
        {
            return std::unexpected(std::move(sourceLayouts.error()));
        }
        auto destinationLayouts = prepareTextureLayoutRequirement(**layoutRecording,
                                                                  *(*prepared)->destination,
                                                                  copied.destinations.span(),
                                                                  TextureLayout::copyDestination,
                                                                  TextureLayout::common,
                                                                  "copyTexture");
        if (!destinationLayouts)
        {
            return std::unexpected(std::move(destinationLayouts.error()));
        }
        auto sharedLayouts = prepareTextureLayoutRequirement(**layoutRecording,
                                                             *(*prepared)->source,
                                                             copied.shared.span(),
                                                             TextureLayout::common,
                                                             TextureLayout::undefined,
                                                             "copyTexture");
        if (!sharedLayouts)
        {
            return std::unexpected(std::move(sharedLayouts.error()));
        }

        auto commandList = retainCopyBatchForRecording(queues, commandBuffer, *prepared);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }
        recordTextureCopies(**commandList, **prepared);
        commitTextureLayoutRequirement(*sourceLayouts);
        commitTextureLayoutRequirement(*destinationLayouts);
        commitTextureLayoutRequirement(*sharedLayouts);
        insertions.commit();
        test::noteResourceCommandRecorded();
        return {};
    }

    Status ResourceManager::resolveTexture(QueueManager& queues,
                                           BindingManager& bindings,
                                           PipelineManager& pipelines,
                                           detail::BackendCommandBuffer& commandBuffer,
                                           const TextureResolve& resolve)
    {
        auto source = impl_->texture(resolve.source);
        auto destination = impl_->texture(resolve.destination);
        if (!source)
        {
            return std::unexpected(std::move(source.error()));
        }
        if (!destination)
        {
            return std::unexpected(std::move(destination.error()));
        }
        auto prepared = prepareTextureResolve(*impl_->device, std::move(*source), std::move(*destination), resolve);
        if (!prepared)
        {
            return std::unexpected(std::move(prepared.error()));
        }
        if (commandBuffer.queueType() != QueueType::graphics || commandBuffer.graphicsState().renderingOwner != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A texture resolve requires a graphics command buffer outside a rendering scope.",
            });
        }
        auto commandList = queues.commandList(commandBuffer);
        if (!commandList)
        {
            return std::unexpected(std::move(commandList.error()));
        }

        if (!isDepthFormat(prepared->sourceInfo))
        {
            return recordColorResolve(queues, commandBuffer, **commandList, *prepared);
        }
        if (prepared->source->desc.format == Format::d24UnormS8Uint)
        {
            return recordPackedDepthStencilResolve(queues, commandBuffer, *prepared);
        }

        const DepthResolvePipeline resolvePipeline{
            .device = *impl_->device,
            .destinationDescriptorPool = impl_->depthResolveDescriptors,
            .rootSignature = impl_->depthResolveRootSignature.Get(),
            .pipelineState = impl_->depthResolvePipeline.Get(),
            .enhancedBarriers = impl_->enhancedBarriers,
        };
        auto draw = prepareDepthResolveDraw(resolvePipeline, queues, bindings, pipelines, commandBuffer);
        if (!draw)
        {
            return std::unexpected(std::move(draw.error()));
        }
        createDepthResolveViews(resolvePipeline.device, *prepared, *draw);
        auto retained = retainDepthResolveDraw(queues, commandBuffer, *prepared, *draw);
        if (!retained)
        {
            return retained;
        }
        recordDepthResolveBetweenBarriers(**commandList, bindings, resolvePipeline, *prepared, *draw);
        return {};
    }

} // namespace rachis::d3d12
