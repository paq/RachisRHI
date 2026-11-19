#include "Common/VectorReserve.h"
#include "Internal/FillAndClearValidation.h"
#include "Internal/VulkanBufferBarriers.h"
#include "Internal/VulkanDiagnostics.h"
#include "Internal/VulkanResources.h"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace rachis::vulkan
{
    namespace
    {

        struct FormatFootprint final
        {
            std::uint32_t blockWidth = 1;
            std::uint32_t blockHeight = 1;
            std::uint32_t bytesPerBlock = 0;
            bool depth = false;
            bool stencil = false;
            bool compressed = false;
        };

        struct BufferInterval final
        {
            std::uint64_t begin = 0;
            std::uint64_t end = 0;
        };

        struct TextureBox final
        {
            std::uint32_t mipLevel = 0;
            std::uint32_t arrayLayer = 0;
            TextureAspect aspect = TextureAspect::color;
            Origin3D origin{};
            Extent3D extent{};
        };

        struct ResolvedTransfer final
        {
            TextureTransferLayout layout{};
            VkBufferImageCopy copy{};
        };

        struct TransferRowLayout final
        {
            std::uint32_t rowsPerImage = 0;
            std::uint32_t copiedRows = 0;
            std::uint64_t sliceRows = 0;
            std::uint32_t nativeImageHeight = 0;
            std::uint64_t minimumRowBytes = 0;
            std::uint64_t rowPitch = 0;
        };

        struct TransferByteSizes final
        {
            std::uint64_t bytesPerImage = 0;
            std::uint64_t requiredBufferSize = 0;
        };

        struct BufferImageCopyBatch final
        {
            std::vector<VkBufferImageCopy> copies;
            std::vector<VkImageLayout> layouts;
        };

        struct ImageCopyLayouts final
        {
            VkImageLayout source = VK_IMAGE_LAYOUT_UNDEFINED;
            VkImageLayout destination = VK_IMAGE_LAYOUT_UNDEFINED;

            [[nodiscard]] constexpr bool operator==(const ImageCopyLayouts&) const noexcept = default;
        };

        struct ImageCopyBatch final
        {
            std::vector<VkImageCopy> copies;
            std::vector<ImageCopyLayouts> layouts;
        };

        [[nodiscard]] constexpr bool checkedAdd(std::uint64_t left, std::uint64_t right, std::uint64_t& result) noexcept
        {
            if (right > std::numeric_limits<std::uint64_t>::max() - left)
            {
                return false;
            }
            result = left + right;
            return true;
        }

        [[nodiscard]] constexpr bool checkedMultiply(std::uint64_t left,
                                                     std::uint64_t right,
                                                     std::uint64_t& result) noexcept
        {
            if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)
            {
                return false;
            }
            result = left * right;
            return true;
        }

        [[nodiscard]] constexpr std::uint32_t mipExtent(std::uint32_t value, std::uint32_t mipLevel) noexcept
        {
            return std::max(1U, value >> mipLevel);
        }

        [[nodiscard]] constexpr std::uint32_t alignUp(std::uint32_t value, std::uint32_t alignment) noexcept
        {
            if (alignment <= 1)
            {
                return value;
            }
            return (value + alignment - 1U) / alignment * alignment;
        }

        [[nodiscard]] constexpr bool isDepthFormat(Format format) noexcept
        {
            return format == Format::d32Float || format == Format::d24UnormS8Uint;
        }

        [[nodiscard]] constexpr FormatFootprint formatFootprint(Format format, TextureAspect aspect) noexcept
        {
            FormatFootprint footprint{};
            switch (format)
            {
            case Format::r8Unorm:
            case Format::r8Uint:
            case Format::r8Sint:
                footprint.bytesPerBlock = 1;
                break;
            case Format::rg8Unorm:
            case Format::rg8Uint:
            case Format::rg8Sint:
            case Format::r16Float:
            case Format::r16Uint:
            case Format::r16Sint:
                footprint.bytesPerBlock = 2;
                break;
            case Format::rgba8Unorm:
            case Format::rgba8UnormSrgb:
            case Format::bgra8Unorm:
            case Format::bgra8UnormSrgb:
            case Format::rgba8Uint:
            case Format::rgba8Sint:
            case Format::r32Float:
            case Format::r32Uint:
            case Format::r32Sint:
            case Format::rg16Float:
            case Format::rg16Uint:
            case Format::rg16Sint:
                footprint.bytesPerBlock = 4;
                break;
            case Format::rgba16Float:
            case Format::rgba16Uint:
            case Format::rgba16Sint:
            case Format::rg32Float:
            case Format::rg32Uint:
            case Format::rg32Sint:
            case Format::r64Uint:
                footprint.bytesPerBlock = 8;
                break;
            case Format::rgba32Float:
            case Format::rgba32Uint:
            case Format::rgba32Sint:
                footprint.bytesPerBlock = 16;
                break;
            case Format::d32Float:
                footprint.bytesPerBlock = 4;
                footprint.depth = true;
                break;
            case Format::d24UnormS8Uint:
                footprint.depth = true;
                footprint.stencil = true;
                footprint.bytesPerBlock = aspect == TextureAspect::stencil ? 1 : 4;
                break;
            case Format::bc1RgbaUnorm:
            case Format::bc1RgbaUnormSrgb:
            case Format::bc4RUnorm:
            case Format::bc4RSnorm:
                footprint.blockWidth = 4;
                footprint.blockHeight = 4;
                footprint.bytesPerBlock = 8;
                footprint.compressed = true;
                break;
            case Format::bc2RgbaUnorm:
            case Format::bc2RgbaUnormSrgb:
            case Format::bc3RgbaUnorm:
            case Format::bc3RgbaUnormSrgb:
            case Format::bc5RgUnorm:
            case Format::bc5RgSnorm:
            case Format::bc6hRgbUfloat:
            case Format::bc6hRgbSfloat:
            case Format::bc7RgbaUnorm:
            case Format::bc7RgbaUnormSrgb:
                footprint.blockWidth = 4;
                footprint.blockHeight = 4;
                footprint.bytesPerBlock = 16;
                footprint.compressed = true;
                break;
            case Format::unknown:
                break;
            }
            return footprint;
        }

        [[nodiscard]] constexpr bool aspectMatchesFormat(Format format, TextureAspect aspect) noexcept
        {
            switch (aspect)
            {
            case TextureAspect::depth:
                return format == Format::d32Float || format == Format::d24UnormS8Uint;
            case TextureAspect::stencil:
                return format == Format::d24UnormS8Uint;
            case TextureAspect::color:
                return !isDepthFormat(format);
            }
            return false;
        }

        [[nodiscard]] constexpr VkImageAspectFlags nativeAspect(TextureAspect aspect) noexcept
        {
            switch (aspect)
            {
            case TextureAspect::depth:
                return VK_IMAGE_ASPECT_DEPTH_BIT;
            case TextureAspect::stencil:
                return VK_IMAGE_ASPECT_STENCIL_BIT;
            case TextureAspect::color:
                break;
            }
            return VK_IMAGE_ASPECT_COLOR_BIT;
        }

        [[nodiscard]] constexpr bool intervalsOverlap(const BufferInterval& left, const BufferInterval& right) noexcept
        {
            return left.begin < right.end && right.begin < left.end;
        }

        [[nodiscard]] constexpr bool textureBoxesOverlap(const TextureBox& left, const TextureBox& right) noexcept
        {
            if (left.mipLevel != right.mipLevel || left.arrayLayer != right.arrayLayer || left.aspect != right.aspect)
            {
                return false;
            }
            return left.origin.x < right.origin.x + right.extent.width &&
                   right.origin.x < left.origin.x + left.extent.width &&
                   left.origin.y < right.origin.y + right.extent.height &&
                   right.origin.y < left.origin.y + left.extent.height &&
                   left.origin.z < right.origin.z + right.extent.depth &&
                   right.origin.z < left.origin.z + left.extent.depth;
        }

        [[nodiscard]] constexpr bool blockAligned(std::uint32_t origin,
                                                  std::uint32_t extent,
                                                  std::uint32_t mipSize,
                                                  std::uint32_t block) noexcept
        {
            if (block <= 1)
            {
                return true;
            }
            if (origin % block != 0)
            {
                return false;
            }
            return extent % block == 0 || origin + extent == mipSize;
        }

        [[nodiscard]] Error invalidCopy(const char* message) noexcept
        {
            return Error{
                ErrorCode::invalidArgument,
                0,
                message,
            };
        }

        [[nodiscard]] Extent3D mipExtent3D(const TextureDesc& desc, std::uint32_t mipLevel) noexcept
        {
            return Extent3D{
                .width = mipExtent(desc.width, mipLevel),
                .height = desc.dimension == TextureDimension::texture1D ? 1U : mipExtent(desc.height, mipLevel),
                .depth =
                    desc.dimension == TextureDimension::texture3D ? mipExtent(desc.depthOrArrayLayers, mipLevel) : 1U,
            };
        }

        [[nodiscard]] Status validateRegionInsideSubresource(const TextureDesc& desc,
                                                             const TextureSubresource& subresource,
                                                             const Origin3D& origin,
                                                             const Extent3D& extent)
        {
            const auto arraySize = desc.dimension == TextureDimension::texture3D ? 1U : desc.depthOrArrayLayers;
            if (subresource.mipLevel >= desc.mipLevels || subresource.arrayLayer >= arraySize ||
                (desc.dimension == TextureDimension::texture3D && subresource.arrayLayer != 0))
            {
                return std::unexpected(invalidCopy("The texture copy subresource mip or layer is invalid."));
            }

            const auto mip = mipExtent3D(desc, subresource.mipLevel);
            const bool empty = extent.width == 0 || extent.height == 0 || extent.depth == 0;
            const bool outsideMip = origin.x > mip.width || extent.width > mip.width - origin.x ||
                                    origin.y > mip.height || extent.height > mip.height - origin.y ||
                                    origin.z > mip.depth || extent.depth > mip.depth - origin.z;
            const bool usesAbsentAxis =
                (desc.dimension != TextureDimension::texture3D && origin.z != 0) ||
                (desc.dimension == TextureDimension::texture1D && (origin.y != 0 || extent.height != 1));
            if (empty || outsideMip || usesAbsentAxis)
            {
                return std::unexpected(
                    invalidCopy("The texture copy region is empty or outside the selected subresource."));
            }
            return {};
        }

        [[nodiscard]] Result<TransferRowLayout> resolveTransferRowLayout(const BufferTextureCopyRegion& region,
                                                                         const FormatFootprint& footprint)
        {
            const auto blocksWide = (region.extent.width + footprint.blockWidth - 1U) / footprint.blockWidth;
            const auto copiedRows = (region.extent.height + footprint.blockHeight - 1U) / footprint.blockHeight;
            const auto rowsPerImage = region.rowsPerImage == 0 ? region.extent.height : region.rowsPerImage;
            const auto sliceRows =
                (static_cast<std::uint64_t>(rowsPerImage) + footprint.blockHeight - 1U) / footprint.blockHeight;

            const auto nativeImageHeight = sliceRows * footprint.blockHeight;
            if (nativeImageHeight > std::numeric_limits<std::uint32_t>::max())
            {
                return std::unexpected(invalidCopy("The buffer-texture copy footprint exceeds portable size limits."));
            }
            std::uint64_t minimumRowBytes = 0;
            if (!checkedMultiply(blocksWide, footprint.bytesPerBlock, minimumRowBytes))
            {
                return std::unexpected(invalidCopy("The buffer-texture copy footprint exceeds portable size limits."));
            }

            const std::uint32_t rowPitchAlignment = footprint.bytesPerBlock;
            const auto rowPitch = region.bytesPerRow == 0
                                      ? static_cast<std::uint64_t>(
                                            alignUp(static_cast<std::uint32_t>(minimumRowBytes), rowPitchAlignment))
                                      : static_cast<std::uint64_t>(region.bytesPerRow);
            if (rowsPerImage < region.extent.height || copiedRows == 0 || rowPitch < minimumRowBytes ||
                rowPitch > std::numeric_limits<std::uint32_t>::max() || rowPitch % rowPitchAlignment != 0)
            {
                return std::unexpected(
                    invalidCopy("Buffer-texture row pitch or rows-per-image is too small or incorrectly aligned."));
            }
            return TransferRowLayout{
                .rowsPerImage = rowsPerImage,
                .copiedRows = copiedRows,
                .sliceRows = sliceRows,
                .nativeImageHeight = static_cast<std::uint32_t>(nativeImageHeight),
                .minimumRowBytes = minimumRowBytes,
                .rowPitch = rowPitch,
            };
        }

        [[nodiscard]] Result<TransferByteSizes> resolveTransferByteSizes(const BufferTextureCopyRegion& region,
                                                                         const TransferRowLayout& rows)
        {
            std::uint64_t bytesPerImage = 0;
            std::uint64_t priorSliceBytes = 0;
            std::uint64_t copiedRowBytes = 0;
            std::uint64_t requiredBytes = 0;
            const bool validSize = checkedMultiply(rows.rowPitch, rows.sliceRows, bytesPerImage) &&
                                   checkedMultiply(bytesPerImage, region.extent.depth - 1U, priorSliceBytes) &&
                                   checkedMultiply(rows.rowPitch, rows.copiedRows - 1U, copiedRowBytes) &&
                                   checkedAdd(priorSliceBytes, copiedRowBytes, requiredBytes) &&
                                   checkedAdd(requiredBytes, rows.minimumRowBytes, requiredBytes) &&
                                   checkedAdd(region.bufferOffset, requiredBytes, requiredBytes);
            if (!validSize)
            {
                return std::unexpected(invalidCopy("The buffer-texture copy footprint exceeds portable size limits."));
            }
            return TransferByteSizes{
                .bytesPerImage = bytesPerImage,
                .requiredBufferSize = requiredBytes,
            };
        }

        [[nodiscard]] VkImageSubresourceLayers nativeSubresourceLayers(const TextureDesc& desc,
                                                                       const TextureSubresource& subresource) noexcept
        {
            return VkImageSubresourceLayers{
                .aspectMask = nativeAspect(subresource.aspect),
                .mipLevel = subresource.mipLevel,
                .baseArrayLayer = desc.dimension == TextureDimension::texture3D ? 0U : subresource.arrayLayer,
                .layerCount = 1,
            };
        }

        [[nodiscard]] constexpr VkOffset3D nativeOffset(const Origin3D& origin) noexcept
        {
            return VkOffset3D{
                static_cast<std::int32_t>(origin.x),
                static_cast<std::int32_t>(origin.y),
                static_cast<std::int32_t>(origin.z),
            };
        }

        [[nodiscard]] constexpr VkExtent3D nativeExtent(const Extent3D& extent) noexcept
        {
            return VkExtent3D{
                extent.width,
                extent.height,
                extent.depth,
            };
        }

        [[nodiscard]] Result<ResolvedTransfer> resolveTransfer(const TextureDesc& desc,
                                                               const BufferTextureCopyRegion& region)
        {
            if (!aspectMatchesFormat(desc.format, region.textureSubresource.aspect))
            {
                return std::unexpected(invalidCopy("The texture copy aspect does not match the texture format."));
            }
            if (desc.sampleCount != 1)
            {
                return std::unexpected(invalidCopy("Buffer-texture copies require a single-sampled texture."));
            }

            const auto footprint = formatFootprint(desc.format, region.textureSubresource.aspect);
            if (footprint.bytesPerBlock == 0)
            {
                return std::unexpected(invalidCopy("The texture format has no portable transfer footprint."));
            }
            if (auto inside = validateRegionInsideSubresource(
                    desc, region.textureSubresource, region.textureOrigin, region.extent);
                !inside)
            {
                return std::unexpected(std::move(inside.error()));
            }
            const auto mip = mipExtent3D(desc, region.textureSubresource.mipLevel);
            if (!blockAligned(region.textureOrigin.x, region.extent.width, mip.width, footprint.blockWidth) ||
                !blockAligned(region.textureOrigin.y, region.extent.height, mip.height, footprint.blockHeight))
            {
                return std::unexpected(
                    invalidCopy("Compressed texture copies must be aligned to the format block size."));
            }

            const std::uint32_t bufferOffsetAlignment = std::max(4U, footprint.bytesPerBlock);
            if (region.bufferOffset % bufferOffsetAlignment != 0)
            {
                return std::unexpected(
                    invalidCopy("A buffer-texture copy offset must satisfy Vulkan transfer alignment."));
            }

            auto rows = resolveTransferRowLayout(region, footprint);
            if (!rows)
            {
                return std::unexpected(std::move(rows.error()));
            }
            auto sizes = resolveTransferByteSizes(region, *rows);
            if (!sizes)
            {
                return std::unexpected(std::move(sizes.error()));
            }

            const auto texelsPerRow =
                static_cast<std::uint32_t>((rows->rowPitch / footprint.bytesPerBlock) * footprint.blockWidth);
            return ResolvedTransfer{
                .layout =
                    TextureTransferLayout{
                        .bytesPerRow = static_cast<std::uint32_t>(rows->rowPitch),
                        .rowsPerImage = rows->rowsPerImage,
                        .rowCount = rows->copiedRows,
                        .rowDataSize = rows->minimumRowBytes,
                        .bytesPerImage = sizes->bytesPerImage,
                        .requiredBufferSize = sizes->requiredBufferSize,
                        .bufferOffsetAlignment = bufferOffsetAlignment,
                        .rowPitchAlignment = footprint.bytesPerBlock,
                    },
                .copy =
                    VkBufferImageCopy{
                        .bufferOffset = region.bufferOffset,
                        .bufferRowLength = region.bytesPerRow == 0 ? 0U : texelsPerRow,
                        .bufferImageHeight = region.rowsPerImage == 0 ? 0U : rows->nativeImageHeight,
                        .imageSubresource = nativeSubresourceLayers(desc, region.textureSubresource),
                        .imageOffset = nativeOffset(region.textureOrigin),
                        .imageExtent = nativeExtent(region.extent),
                    },
            };
        }

        [[nodiscard]] bool copyCompatibleLayout(VkImageLayout layout, bool destination) noexcept
        {
            if (layout == VK_IMAGE_LAYOUT_GENERAL)
            {
                return true;
            }
            return destination ? layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                               : layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        }

        [[nodiscard]] constexpr bool copyCompatibleFormats(Format left, Format right) noexcept
        {
            if (left == right)
            {
                return true;
            }
            switch (left)
            {
            case Format::rgba8Unorm:
            case Format::rgba8UnormSrgb:
                return right == Format::rgba8Unorm || right == Format::rgba8UnormSrgb;
            case Format::bgra8Unorm:
            case Format::bgra8UnormSrgb:
                return right == Format::bgra8Unorm || right == Format::bgra8UnormSrgb;
            case Format::bc1RgbaUnorm:
            case Format::bc1RgbaUnormSrgb:
                return right == Format::bc1RgbaUnorm || right == Format::bc1RgbaUnormSrgb;
            case Format::bc3RgbaUnorm:
            case Format::bc3RgbaUnormSrgb:
                return right == Format::bc3RgbaUnorm || right == Format::bc3RgbaUnormSrgb;
            case Format::bc7RgbaUnorm:
            case Format::bc7RgbaUnormSrgb:
                return right == Format::bc7RgbaUnorm || right == Format::bc7RgbaUnormSrgb;
            default:
                return false;
            }
        }

        [[nodiscard]] bool anyAdjacentIntervalsOverlap(std::span<const BufferInterval> sortedByBeginning) noexcept
        {
            for (std::size_t index = 1; index < sortedByBeginning.size(); ++index)
            {
                if (sortedByBeginning[index].begin < sortedByBeginning[index - 1].end)
                {
                    return true;
                }
            }
            return false;
        }

        [[nodiscard]] bool anySourceOverlapsDestination(std::span<const BufferInterval> sortedSources,
                                                        std::span<const BufferInterval> sortedDestinations) noexcept
        {
            std::size_t sourceIndex = 0;
            std::size_t destinationIndex = 0;
            while (sourceIndex < sortedSources.size() && destinationIndex < sortedDestinations.size())
            {
                if (intervalsOverlap(sortedSources[sourceIndex], sortedDestinations[destinationIndex]))
                {
                    return true;
                }
                if (sortedSources[sourceIndex].end <= sortedDestinations[destinationIndex].begin)
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

        [[nodiscard]] Status validateDisjointBufferCopyIntervals(std::vector<BufferInterval> sourceIntervals,
                                                                 std::vector<BufferInterval> destinationIntervals,
                                                                 bool sameBuffer)
        {
            std::ranges::sort(destinationIntervals, {}, &BufferInterval::begin);
            if (anyAdjacentIntervalsOverlap(destinationIntervals))
            {
                return std::unexpected(
                    invalidCopy("A buffer-copy batch cannot write overlapping destination regions."));
            }
            if (!sameBuffer)
            {
                return {};
            }

            std::ranges::sort(sourceIntervals, {}, &BufferInterval::begin);
            if (anySourceOverlapsDestination(sourceIntervals, destinationIntervals))
            {
                return std::unexpected(
                    invalidCopy("A same-buffer copy batch cannot access one byte as both source and destination."));
            }
            return {};
        }

        [[nodiscard]] Result<std::vector<VkBufferCopy>> resolveBufferCopies(const BufferDesc& source,
                                                                            const BufferDesc& destination,
                                                                            bool sameBuffer,
                                                                            std::span<const BufferCopyRegion> regions)
        {
            std::vector<VkBufferCopy> copies;
            std::vector<BufferInterval> sourceIntervals;
            std::vector<BufferInterval> destinationIntervals;
            copies.reserve(regions.size());
            sourceIntervals.reserve(regions.size());
            destinationIntervals.reserve(regions.size());
            for (const auto& region : regions)
            {
                if (region.size == 0 || region.sourceOffset > source.size ||
                    region.size > source.size - region.sourceOffset || region.destinationOffset > destination.size ||
                    region.size > destination.size - region.destinationOffset)
                {
                    return std::unexpected(
                        invalidCopy("A buffer-copy batch contains an empty or out-of-bounds region."));
                }
                copies.push_back(VkBufferCopy{
                    .srcOffset = region.sourceOffset,
                    .dstOffset = region.destinationOffset,
                    .size = region.size,
                });
                sourceIntervals.push_back(BufferInterval{
                    .begin = region.sourceOffset,
                    .end = region.sourceOffset + region.size,
                });
                destinationIntervals.push_back(BufferInterval{
                    .begin = region.destinationOffset,
                    .end = region.destinationOffset + region.size,
                });
            }

            if (auto disjoint = validateDisjointBufferCopyIntervals(
                    std::move(sourceIntervals), std::move(destinationIntervals), sameBuffer);
                !disjoint)
            {
                return std::unexpected(std::move(disjoint.error()));
            }
            return copies;
        }

        void retainKeepAlive(std::vector<std::shared_ptr<void>>* retained,
                             const std::shared_ptr<VulkanNativeKeepAlive>& keepAlive)
        {
            if (retained != nullptr && keepAlive != nullptr)
            {
                retained->push_back(keepAlive);
            }
        }

        [[nodiscard]] constexpr TextureBox textureBox(const TextureSubresource& subresource,
                                                      const Origin3D& origin,
                                                      const Extent3D& extent) noexcept
        {
            return TextureBox{
                .mipLevel = subresource.mipLevel,
                .arrayLayer = subresource.arrayLayer,
                .aspect = subresource.aspect,
                .origin = origin,
                .extent = extent,
            };
        }

        [[nodiscard]] bool overlapsAny(const TextureBox& box, std::span<const TextureBox> others) noexcept
        {
            return std::ranges::any_of(others,
                                       [&box](const TextureBox& other) { return textureBoxesOverlap(box, other); });
        }

        template<std::invocable<const TextureSubresource&> RecordedLayoutLookup>
        [[nodiscard]] Result<BufferImageCopyBatch> resolveBufferToTextureBatch(
            const BufferDesc& buffer,
            const TextureDesc& texture,
            std::span<const BufferTextureCopyRegion> regions,
            const RecordedLayoutLookup& recordedLayoutOf)
        {
            BufferImageCopyBatch batch;
            std::vector<TextureBox> destinations;
            batch.copies.reserve(regions.size());
            batch.layouts.reserve(regions.size());
            destinations.reserve(regions.size());
            for (const auto& region : regions)
            {
                auto resolved = resolveTransfer(texture, region);
                if (!resolved)
                {
                    return std::unexpected(std::move(resolved.error()));
                }
                const VkImageLayout layout = recordedLayoutOf(region.textureSubresource);
                if (!copyCompatibleLayout(layout, true))
                {
                    return std::unexpected(
                        Error{ErrorCode::invalidState,
                              0,
                              "A buffer-to-texture copy requires a copy-destination or general subresource layout."});
                }
                if (resolved->layout.requiredBufferSize > buffer.size)
                {
                    return std::unexpected(invalidCopy("The buffer-texture copy footprint exceeds the buffer."));
                }
                const auto box = textureBox(region.textureSubresource, region.textureOrigin, region.extent);
                if (overlapsAny(box, destinations))
                {
                    return std::unexpected(
                        invalidCopy("A buffer-to-texture batch cannot write overlapping destination texels."));
                }
                batch.copies.push_back(resolved->copy);
                batch.layouts.push_back(layout);
                destinations.push_back(box);
            }
            return batch;
        }

        template<typename Layout>
        [[nodiscard]] std::size_t endOfLayoutRun(const std::vector<Layout>& layouts, std::size_t first) noexcept
        {
            auto end = first + 1;
            while (end < layouts.size() && layouts[end] == layouts[first])
            {
                ++end;
            }
            return end;
        }

        void recordBufferToImageCopies(const VolkDeviceTable& table,
                                       VkCommandBuffer command,
                                       VkBuffer source,
                                       VkImage destination,
                                       const BufferImageCopyBatch& batch)
        {
            for (std::size_t first = 0; first < batch.copies.size();)
            {
                const auto end = endOfLayoutRun(batch.layouts, first);
                table.vkCmdCopyBufferToImage(command,
                                             source,
                                             destination,
                                             batch.layouts[first],
                                             static_cast<std::uint32_t>(end - first),
                                             batch.copies.data() + first);
                first = end;
            }
        }

        [[nodiscard]] bool overlapsAny(const BufferInterval& interval, std::span<const BufferInterval> others) noexcept
        {
            return std::ranges::any_of(
                others, [&interval](const BufferInterval& other) { return intervalsOverlap(interval, other); });
        }

        template<std::invocable<const TextureSubresource&> RecordedLayoutLookup>
        [[nodiscard]] Result<BufferImageCopyBatch> resolveTextureToBufferBatch(
            const TextureDesc& texture,
            const BufferDesc& buffer,
            std::span<const BufferTextureCopyRegion> regions,
            const RecordedLayoutLookup& recordedLayoutOf)
        {
            BufferImageCopyBatch batch;
            std::vector<BufferInterval> destinations;
            batch.copies.reserve(regions.size());
            batch.layouts.reserve(regions.size());
            destinations.reserve(regions.size());
            for (const auto& region : regions)
            {
                auto resolved = resolveTransfer(texture, region);
                if (!resolved)
                {
                    return std::unexpected(std::move(resolved.error()));
                }
                const VkImageLayout layout = recordedLayoutOf(region.textureSubresource);
                if (!copyCompatibleLayout(layout, false))
                {
                    return std::unexpected(
                        Error{ErrorCode::invalidState,
                              0,
                              "A texture-to-buffer copy requires a copy-source or general subresource layout."});
                }
                if (resolved->layout.requiredBufferSize > buffer.size)
                {
                    return std::unexpected(invalidCopy("The buffer-texture copy footprint exceeds the buffer."));
                }
                const BufferInterval interval{
                    .begin = region.bufferOffset,
                    .end = resolved->layout.requiredBufferSize,
                };
                if (overlapsAny(interval, destinations))
                {
                    return std::unexpected(
                        invalidCopy("A texture-to-buffer batch cannot write overlapping destination ranges."));
                }
                batch.copies.push_back(resolved->copy);
                batch.layouts.push_back(layout);
                destinations.push_back(interval);
            }
            return batch;
        }

        void recordImageToBufferCopies(const VolkDeviceTable& table,
                                       VkCommandBuffer command,
                                       VkImage source,
                                       VkBuffer destination,
                                       const BufferImageCopyBatch& batch)
        {
            for (std::size_t first = 0; first < batch.copies.size();)
            {
                const auto end = endOfLayoutRun(batch.layouts, first);
                table.vkCmdCopyImageToBuffer(command,
                                             source,
                                             batch.layouts[first],
                                             destination,
                                             static_cast<std::uint32_t>(end - first),
                                             batch.copies.data() + first);
                first = end;
            }
        }

        [[nodiscard]] Status validateImageCopyRegion(const TextureDesc& desc,
                                                     const TextureSubresource& subresource,
                                                     const Origin3D& origin,
                                                     const Extent3D& extent)
        {
            if (!aspectMatchesFormat(desc.format, subresource.aspect))
            {
                return std::unexpected(invalidCopy("The texture copy aspect does not match the texture format."));
            }
            if (auto inside = validateRegionInsideSubresource(desc, subresource, origin, extent); !inside)
            {
                return inside;
            }
            const auto mip = mipExtent3D(desc, subresource.mipLevel);
            const auto footprint = formatFootprint(desc.format, subresource.aspect);
            if (footprint.bytesPerBlock == 0 ||
                !blockAligned(origin.x, extent.width, mip.width, footprint.blockWidth) ||
                !blockAligned(origin.y, extent.height, mip.height, footprint.blockHeight))
            {
                return std::unexpected(
                    invalidCopy("Compressed texture copies must be aligned to the format block size."));
            }
            return {};
        }

        [[nodiscard]] Status validateTextureCopyRegion(const TextureDesc& source,
                                                       const TextureDesc& destination,
                                                       const TextureCopyRegion& region)
        {
            if (region.sourceSubresource.aspect != region.destinationSubresource.aspect)
            {
                return std::unexpected(invalidCopy("Texture copies require matching source and destination aspects."));
            }
            if (auto valid =
                    validateImageCopyRegion(source, region.sourceSubresource, region.sourceOrigin, region.extent);
                !valid)
            {
                return valid;
            }
            return validateImageCopyRegion(
                destination, region.destinationSubresource, region.destinationOrigin, region.extent);
        }

        [[nodiscard]] constexpr bool isWholeSubresource(const Origin3D& origin,
                                                        const Extent3D& extent,
                                                        const Extent3D& mip) noexcept
        {
            return origin.x == 0 && origin.y == 0 && origin.z == 0 && extent.width == mip.width &&
                   extent.height == mip.height && extent.depth == mip.depth;
        }

        [[nodiscard]] bool coversWholeSubresources(const TextureDesc& source,
                                                   const TextureDesc& destination,
                                                   const TextureCopyRegion& region) noexcept
        {
            const auto sourceMip = mipExtent3D(source, region.sourceSubresource.mipLevel);
            const auto destinationMip = mipExtent3D(destination, region.destinationSubresource.mipLevel);
            return isWholeSubresource(region.sourceOrigin, region.extent, sourceMip) &&
                   isWholeSubresource(region.destinationOrigin, region.extent, destinationMip);
        }

        [[nodiscard]] constexpr bool copiesWithinOneSubresource(const TextureCopyRegion& region) noexcept
        {
            return region.sourceSubresource.mipLevel == region.destinationSubresource.mipLevel &&
                   region.sourceSubresource.arrayLayer == region.destinationSubresource.arrayLayer &&
                   region.sourceSubresource.aspect == region.destinationSubresource.aspect;
        }

        [[nodiscard]] bool anySourceBoxOverlapsDestinations(std::span<const TextureCopyRegion> regions,
                                                            std::span<const TextureBox> destinations) noexcept
        {
            return std::ranges::any_of(regions, [destinations](const TextureCopyRegion& region) {
                return overlapsAny(textureBox(region.sourceSubresource, region.sourceOrigin, region.extent),
                                   destinations);
            });
        }

        template<std::invocable<const TextureCopyRegion&> RecordedLayoutsLookup>
        [[nodiscard]] Result<ImageCopyBatch> resolveTextureCopyBatch(const TextureDesc& source,
                                                                     const TextureDesc& destination,
                                                                     bool sameTexture,
                                                                     std::span<const TextureCopyRegion> regions,
                                                                     const RecordedLayoutsLookup& recordedLayoutsOf)
        {
            const auto sourceFootprint = formatFootprint(source.format, TextureAspect::color);
            const bool requiresWholeSubresources = sourceFootprint.depth || source.sampleCount != 1;
            ImageCopyBatch batch;
            std::vector<TextureBox> destinations;
            batch.copies.reserve(regions.size());
            batch.layouts.reserve(regions.size());
            destinations.reserve(regions.size());
            for (const auto& region : regions)
            {
                if (auto valid = validateTextureCopyRegion(source, destination, region); !valid)
                {
                    return std::unexpected(std::move(valid.error()));
                }
                const ImageCopyLayouts layouts = recordedLayoutsOf(region);
                if (!copyCompatibleLayout(layouts.source, false) || !copyCompatibleLayout(layouts.destination, true))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidState,
                        0,
                        "A texture copy requires copy-source and copy-destination or general subresource layouts."});
                }
                if (requiresWholeSubresources && !coversWholeSubresources(source, destination, region))
                {
                    return std::unexpected(invalidCopy(
                        "Depth-stencil and multisampled texture copies require whole matching subresources."));
                }
                if (sameTexture && copiesWithinOneSubresource(region))
                {
                    return std::unexpected(
                        invalidCopy("A texture copy cannot use the same subresource as both source and destination."));
                }
                const auto destinationBox =
                    textureBox(region.destinationSubresource, region.destinationOrigin, region.extent);
                if (overlapsAny(destinationBox, destinations))
                {
                    return std::unexpected(
                        invalidCopy("A texture-copy batch cannot write overlapping destination texels."));
                }
                batch.copies.push_back(VkImageCopy{
                    .srcSubresource = nativeSubresourceLayers(source, region.sourceSubresource),
                    .srcOffset = nativeOffset(region.sourceOrigin),
                    .dstSubresource = nativeSubresourceLayers(destination, region.destinationSubresource),
                    .dstOffset = nativeOffset(region.destinationOrigin),
                    .extent = nativeExtent(region.extent),
                });
                batch.layouts.push_back(layouts);
                destinations.push_back(destinationBox);
            }

            if (sameTexture && anySourceBoxOverlapsDestinations(regions, destinations))
            {
                return std::unexpected(
                    invalidCopy("A same-texture copy batch cannot access one texel as both source and destination."));
            }
            return batch;
        }

        void recordImageCopies(const VolkDeviceTable& table,
                               VkCommandBuffer command,
                               VkImage source,
                               VkImage destination,
                               const ImageCopyBatch& batch)
        {
            for (std::size_t first = 0; first < batch.copies.size();)
            {
                const auto end = endOfLayoutRun(batch.layouts, first);
                table.vkCmdCopyImage(command,
                                     source,
                                     batch.layouts[first].source,
                                     destination,
                                     batch.layouts[first].destination,
                                     static_cast<std::uint32_t>(end - first),
                                     batch.copies.data() + first);
                first = end;
            }
        }

    } // namespace

    Status VulkanResourceAllocator::copyBufferRegions(VkCommandBuffer command,
                                                      BufferHandle source,
                                                      BufferHandle destination,
                                                      std::span<const BufferCopyRegion> regions,
                                                      std::vector<std::shared_ptr<void>>* retained)
    {
        if (regions.empty() || regions.size() > std::numeric_limits<std::uint32_t>::max())
        {
            return std::unexpected(invalidCopy(
                "A buffer-copy batch must contain at least one region and fit within portable count limits."));
        }

        const std::scoped_lock lock(mutex_);
        if (table_ == nullptr || table_->vkCmdCopyBuffer == nullptr || table_->vkCmdPipelineBarrier2 == nullptr ||
            command == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Vulkan buffer copies are unavailable because the device table is incomplete.",
            });
        }

        auto* sourceRecord = findBuffer(source);
        auto* destinationRecord = findBuffer(destination);
        if (sourceRecord == nullptr || !sourceRecord->alive || sourceRecord->buffer == VK_NULL_HANDLE)
        {
            return std::unexpected(invalidCopy("A buffer-copy batch references an invalid or stale source buffer."));
        }
        if (destinationRecord == nullptr || !destinationRecord->alive || destinationRecord->buffer == VK_NULL_HANDLE)
        {
            return std::unexpected(
                invalidCopy("A buffer-copy batch references an invalid or stale destination buffer."));
        }
        if (!hasAny(sourceRecord->desc.usage, BufferUsage::copySource) ||
            !hasAny(destinationRecord->desc.usage, BufferUsage::copyDestination))
        {
            return std::unexpected(invalidCopy("Buffer copy usages do not include copy source and copy destination."));
        }

        auto copies = resolveBufferCopies(sourceRecord->desc, destinationRecord->desc, source == destination, regions);
        if (!copies)
        {
            return std::unexpected(std::move(copies.error()));
        }

        retainKeepAlive(retained, sourceRecord->keepAlive);
        retainKeepAlive(retained, destinationRecord->keepAlive);
        table_->vkCmdCopyBuffer(command,
                                sourceRecord->buffer,
                                destinationRecord->buffer,
                                static_cast<std::uint32_t>(copies->size()),
                                copies->data());
        if (destinationRecord->desc.memoryType == MemoryType::readback)
        {
            makeCopyWritesHostVisible(*table_, command, destinationRecord->buffer, destinationRecord->desc.size);
        }
        return {};
    }

    Result<TextureTransferLayout> VulkanResourceAllocator::queryTextureTransferLayout(
        TextureHandle texture, const BufferTextureCopyRegion& region) const
    {
        const std::scoped_lock lock(mutex_);
        const auto* record = textures_.find(texture);
        if (record == nullptr || !record->alive || record->image == VK_NULL_HANDLE)
        {
            return std::unexpected(
                invalidCopy("A texture transfer layout query references an invalid or stale texture."));
        }
        auto resolved = resolveTransfer(record->desc, region);
        if (!resolved)
        {
            return std::unexpected(std::move(resolved.error()));
        }
        return resolved->layout;
    }

    Status VulkanResourceAllocator::copyBufferToTextureRegions(VkCommandBuffer command,
                                                               QueueType queueType,
                                                               BufferHandle source,
                                                               TextureHandle destination,
                                                               std::span<const BufferTextureCopyRegion> regions,
                                                               std::vector<std::shared_ptr<void>>* retained)
    {
        if (regions.empty() || regions.size() > std::numeric_limits<std::uint32_t>::max())
        {
            return std::unexpected(invalidCopy(
                "A buffer-to-texture batch must contain at least one region and fit within portable count limits."));
        }

        const std::scoped_lock lock(mutex_);
        if (table_ == nullptr || table_->vkCmdCopyBufferToImage == nullptr || command == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Vulkan buffer-to-texture copies are unavailable because the device table is incomplete.",
            });
        }

        auto* buffer = findBuffer(source);
        auto* texture = textures_.find(destination);
        if (buffer == nullptr || !buffer->alive || buffer->buffer == VK_NULL_HANDLE)
        {
            return std::unexpected(invalidCopy("A buffer-to-texture copy references an invalid or stale buffer."));
        }
        if (texture == nullptr || !texture->alive || texture->image == VK_NULL_HANDLE)
        {
            return std::unexpected(invalidCopy("A buffer-to-texture copy references an invalid or stale texture."));
        }
        if (!hasAny(buffer->desc.usage, BufferUsage::copySource) ||
            !hasAny(texture->desc.usage, TextureUsage::copyDestination))
        {
            return std::unexpected(
                invalidCopy("A buffer-to-texture copy requires copy-source and copy-destination usages."));
        }

        const auto recordedLayoutOf = [this, command, texture](const TextureSubresource& subresource) {
            return recordedLayout(command, *texture, subresource);
        };
        auto batch = resolveBufferToTextureBatch(buffer->desc, texture->desc, regions, recordedLayoutOf);
        if (!batch)
        {
            return std::unexpected(std::move(batch.error()));
        }
        if (auto supported = validation_detail::validateBufferTextureCopyQueueFamily(
                queueFamilySupportsGraphics(queueType), regions);
            !supported)
        {
            return supported;
        }
        const std::array swapchainLeases{texture->swapchainLease};
        if (auto referenced = retainSwapchainImagesLocked(command, swapchainLeases); !referenced)
        {
            return referenced;
        }

        for (std::size_t index = 0; index < regions.size(); ++index)
        {
            recordTransferRequirementLocked(command,
                                            *texture,
                                            regions[index].textureSubresource,
                                            batch->layouts[index],
                                            validation_detail::SubresourceTransferUse::copyDestination);
        }
        retainKeepAlive(retained, buffer->keepAlive);
        retainKeepAlive(retained, texture->keepAlive);
        recordBufferToImageCopies(*table_, command, buffer->buffer, texture->image, *batch);
        return {};
    }

    Status VulkanResourceAllocator::copyTextureToBufferRegions(VkCommandBuffer command,
                                                               QueueType queueType,
                                                               TextureHandle source,
                                                               BufferHandle destination,
                                                               std::span<const BufferTextureCopyRegion> regions,
                                                               std::vector<std::shared_ptr<void>>* retained)
    {
        if (regions.empty() || regions.size() > std::numeric_limits<std::uint32_t>::max())
        {
            return std::unexpected(invalidCopy(
                "A texture-to-buffer batch must contain at least one region and fit within portable count limits."));
        }

        const std::scoped_lock lock(mutex_);
        if (table_ == nullptr || table_->vkCmdCopyImageToBuffer == nullptr ||
            table_->vkCmdPipelineBarrier2 == nullptr || command == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Vulkan texture-to-buffer copies are unavailable because the device table is incomplete.",
            });
        }

        auto* texture = textures_.find(source);
        auto* buffer = findBuffer(destination);
        if (texture == nullptr || !texture->alive || texture->image == VK_NULL_HANDLE)
        {
            return std::unexpected(invalidCopy("A texture-to-buffer copy references an invalid or stale texture."));
        }
        if (buffer == nullptr || !buffer->alive || buffer->buffer == VK_NULL_HANDLE)
        {
            return std::unexpected(invalidCopy("A texture-to-buffer copy references an invalid or stale buffer."));
        }
        if (!hasAny(texture->desc.usage, TextureUsage::copySource) ||
            !hasAny(buffer->desc.usage, BufferUsage::copyDestination))
        {
            return std::unexpected(
                invalidCopy("A texture-to-buffer copy requires copy-source and copy-destination usages."));
        }

        const auto recordedLayoutOf = [this, command, texture](const TextureSubresource& subresource) {
            return recordedLayout(command, *texture, subresource);
        };
        auto batch = resolveTextureToBufferBatch(texture->desc, buffer->desc, regions, recordedLayoutOf);
        if (!batch)
        {
            return std::unexpected(std::move(batch.error()));
        }
        if (auto supported = validation_detail::validateBufferTextureCopyQueueFamily(
                queueFamilySupportsGraphics(queueType), regions);
            !supported)
        {
            return supported;
        }
        const std::array swapchainLeases{texture->swapchainLease};
        if (auto referenced = retainSwapchainImagesLocked(command, swapchainLeases); !referenced)
        {
            return referenced;
        }

        for (std::size_t index = 0; index < regions.size(); ++index)
        {
            recordTransferRequirementLocked(command,
                                            *texture,
                                            regions[index].textureSubresource,
                                            batch->layouts[index],
                                            validation_detail::SubresourceTransferUse::copySource);
        }
        retainKeepAlive(retained, texture->keepAlive);
        retainKeepAlive(retained, buffer->keepAlive);
        recordImageToBufferCopies(*table_, command, texture->image, buffer->buffer, *batch);
        if (buffer->desc.memoryType == MemoryType::readback)
        {
            makeCopyWritesHostVisible(*table_, command, buffer->buffer, buffer->desc.size);
        }
        return {};
    }

    Status VulkanResourceAllocator::copyTextureRegions(VkCommandBuffer command,
                                                       TextureHandle source,
                                                       TextureHandle destination,
                                                       std::span<const TextureCopyRegion> regions,
                                                       std::vector<std::shared_ptr<void>>* retained)
    {
        if (regions.empty() || regions.size() > std::numeric_limits<std::uint32_t>::max())
        {
            return std::unexpected(invalidCopy(
                "A texture-copy batch must contain at least one region and fit within portable count limits."));
        }

        const std::scoped_lock lock(mutex_);
        if (table_ == nullptr || table_->vkCmdCopyImage == nullptr || command == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Vulkan texture copies are unavailable because the device table is incomplete.",
            });
        }

        auto* sourceRecord = textures_.find(source);
        auto* destinationRecord = textures_.find(destination);
        if (sourceRecord == nullptr || !sourceRecord->alive || sourceRecord->image == VK_NULL_HANDLE)
        {
            return std::unexpected(invalidCopy("A texture copy references an invalid or stale source texture."));
        }
        if (destinationRecord == nullptr || !destinationRecord->alive || destinationRecord->image == VK_NULL_HANDLE)
        {
            return std::unexpected(invalidCopy("A texture copy references an invalid or stale destination texture."));
        }
        if (!hasAny(sourceRecord->desc.usage, TextureUsage::copySource) ||
            !hasAny(destinationRecord->desc.usage, TextureUsage::copyDestination))
        {
            return std::unexpected(invalidCopy("A texture copy requires copy-source and copy-destination usages."));
        }
        if (!copyCompatibleFormats(sourceRecord->desc.format, destinationRecord->desc.format) ||
            sourceRecord->desc.sampleCount != destinationRecord->desc.sampleCount)
        {
            return std::unexpected(
                invalidCopy("Texture copies require copy-compatible formats and matching sample counts."));
        }

        const auto recordedLayoutsOf =
            [this, command, sourceRecord, destinationRecord](const TextureCopyRegion& region) {
                return ImageCopyLayouts{
                    .source = recordedLayout(command, *sourceRecord, region.sourceSubresource),
                    .destination = recordedLayout(command, *destinationRecord, region.destinationSubresource),
                };
            };
        auto batch = resolveTextureCopyBatch(
            sourceRecord->desc, destinationRecord->desc, source == destination, regions, recordedLayoutsOf);
        if (!batch)
        {
            return std::unexpected(std::move(batch.error()));
        }
        const std::array swapchainLeases{sourceRecord->swapchainLease, destinationRecord->swapchainLease};
        if (auto referenced = retainSwapchainImagesLocked(command, swapchainLeases); !referenced)
        {
            return referenced;
        }

        for (std::size_t index = 0; index < regions.size(); ++index)
        {
            recordTransferRequirementLocked(command,
                                            *sourceRecord,
                                            regions[index].sourceSubresource,
                                            batch->layouts[index].source,
                                            validation_detail::SubresourceTransferUse::copySource);
            recordTransferRequirementLocked(command,
                                            *destinationRecord,
                                            regions[index].destinationSubresource,
                                            batch->layouts[index].destination,
                                            validation_detail::SubresourceTransferUse::copyDestination);
        }
        retainKeepAlive(retained, sourceRecord->keepAlive);
        retainKeepAlive(retained, destinationRecord->keepAlive);
        recordImageCopies(*table_, command, sourceRecord->image, destinationRecord->image, *batch);
        return {};
    }

    Status VulkanResourceAllocator::fillBuffer(VkCommandBuffer command,
                                               BufferHandle buffer,
                                               std::uint64_t offset,
                                               std::uint64_t size,
                                               std::uint32_t value,
                                               std::vector<std::shared_ptr<void>>* retained)
    {
        const std::scoped_lock lock(mutex_);
        if (table_ == nullptr || table_->vkCmdFillBuffer == nullptr || command == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Vulkan buffer fills are unavailable because the device table is incomplete.",
            });
        }
        const auto* record = findBuffer(buffer);
        if (record == nullptr || !record->alive || record->buffer == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The fillBuffer buffer handle is invalid or stale.",
            });
        }
        auto range = detail::resolveBufferFillRange(record->desc, offset, size);
        if (!range)
        {
            return std::unexpected(std::move(range.error()));
        }
        recordFillBufferLocked(command, *record, offset, *range, value, retained);
        return {};
    }

    Status VulkanResourceAllocator::clearStorageTexture(VkCommandBuffer command,
                                                        TextureViewHandle viewHandle,
                                                        const StorageTextureClearValue& value,
                                                        std::vector<std::shared_ptr<void>>* retained)
    {
        const std::scoped_lock lock(mutex_);
        if (table_ == nullptr || table_->vkCmdClearColorImage == nullptr || command == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Vulkan storage-texture clears are unavailable because the device table is incomplete.",
            });
        }
        const auto* view = views_.find(viewHandle);
        if (view == nullptr || !view->alive || view->view == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The storage-texture clear view handle is invalid or stale.",
            });
        }
        auto* texture = textures_.find(view->texture);
        if (texture == nullptr || !texture->alive || texture->image == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The storage-texture clear view does not reference a live texture.",
            });
        }
        auto format =
            detail::validateStorageTextureClear(texture->desc, view->desc.usage, view->resolved.format, value);
        if (!format)
        {
            return std::unexpected(std::move(format.error()));
        }

        const bool volume = texture->desc.dimension == TextureDimension::texture3D;
        const VkImageSubresourceRange range{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = view->desc.firstMipLevel,
            .levelCount = 1,
            .baseArrayLayer = volume ? 0U : view->desc.firstArrayLayer,
            .layerCount = volume ? 1U : view->resolved.layerCount,
        };
        const auto* state = commandLayoutState(command);
        for (std::uint32_t layer = range.baseArrayLayer; layer < range.baseArrayLayer + range.layerCount; ++layer)
        {
            const TextureSubresource subresource{.mipLevel = range.baseMipLevel, .arrayLayer = layer};
            if (texture->submittedLayouts != nullptr)
            {
                const auto cell = static_cast<std::size_t>(layer) * texture->desc.mipLevels + range.baseMipLevel;
                const auto known = commandPublicLayout(state, LayoutCell{texture->submittedLayouts.get(), cell});
                if (known.has_value() && *known != TextureLayout::unorderedAccess)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidState,
                        0,
                        "clearStorageTexture requires the unorderedAccess layout, but this command buffer leaves a "
                        "cleared subresource in another layout.",
                    });
                }
            }
            if (recordedLayout(command, *texture, subresource) != VK_IMAGE_LAYOUT_GENERAL)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "clearStorageTexture requires the cleared subresources to be in the unorderedAccess layout.",
                });
            }
        }
        if (retained != nullptr)
        {
            detail::reserveAdditional(*retained, 2U);
        }
        ensureTrackedLayouts(*texture);
        const auto mark = markCommandLayoutsLocked(command);
        try
        {
            auto& commandState = commandLayouts_[command];
            for (std::uint32_t layer = range.baseArrayLayer; layer < range.baseArrayLayer + range.layerCount; ++layer)
            {
                const TextureSubresource subresource{.mipLevel = range.baseMipLevel, .arrayLayer = layer};
                const auto cell = static_cast<std::size_t>(layer) * texture->desc.mipLevels + range.baseMipLevel;
                if (cell < texture->submittedLayouts->size() &&
                    !commandPublicLayout(&commandState, LayoutCell{texture->submittedLayouts.get(), cell}).has_value())
                {
                    commandState.entries.push_back(LayoutEntryRequirement{
                        .submittedPublic = texture->submittedPublicLayouts,
                        .index = cell,
                        .layout = TextureLayout::unorderedAccess,
                        .subresource = subresource,
                        .entryLayoutUnknown = texture->shared,
                    });
                }
                recordTransferRequirementLocked(command,
                                                *texture,
                                                subresource,
                                                VK_IMAGE_LAYOUT_GENERAL,
                                                validation_detail::SubresourceTransferUse::copyDestination);
            }
        }
        catch (...)
        {
            rollBackCommandLayoutsLocked(command, mark);
            throw;
        }
        const std::array swapchainLeases{texture->swapchainLease};
        if (auto referenced = retainSwapchainImagesLocked(command, swapchainLeases); !referenced)
        {
            rollBackCommandLayoutsLocked(command, mark);
            return referenced;
        }
        retainKeepAlive(retained, view->keepAlive);
        retainKeepAlive(retained, texture->keepAlive);

        VkClearColorValue clear{};
        switch (format->type)
        {
        case StorageTextureClearType::float32:
            std::ranges::copy(value.float32, std::begin(clear.float32));
            break;
        case StorageTextureClearType::uint32:
            std::ranges::copy(value.uint32, std::begin(clear.uint32));
            break;
        case StorageTextureClearType::sint32:
            std::ranges::copy(value.sint32, std::begin(clear.int32));
            break;
        }
        table_->vkCmdClearColorImage(command, texture->image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
        return {};
    }

    Result<VulkanResourceAllocator::NativeTexture> VulkanResourceAllocator::nativeTexture(
        TextureHandle texture, VkCommandBuffer command, TextureSubresource subresource) const
    {
        const std::scoped_lock lock(mutex_);
        const auto* record = textures_.find(texture);
        if (record == nullptr || !record->alive || record->image == VK_NULL_HANDLE)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The texture handle is invalid or stale.",
            });
        }
        return NativeTexture{
            .desc = record->desc,
            .viewFormats = record->viewFormats,
            .image = record->image,
            .layout = recordedLayout(command, *record, subresource),
            .keepAlive = record->keepAlive,
            .swapchainLease = record->swapchainLease,
        };
    }

} // namespace rachis::vulkan
