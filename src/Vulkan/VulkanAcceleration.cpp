#include "Internal/VulkanAcceleration.h"

#include "Common/VectorReserve.h"
#include "Internal/VulkanDiagnostics.h"

#ifdef RACHIS_TESTING
#include "Vulkan/VulkanTestHooks.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <limits>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace rachis::vulkan
{
    namespace
    {

        constexpr std::uint64_t accelerationStructureAlignment = 256;
        constexpr std::string_view immutableDestinationMessage =
            "Acceleration structures are immutable after their first recorded build; create a new destination.";

        [[nodiscard]] constexpr bool validAccelerationStructureBuildFlags(AccelerationStructureBuildFlag flags) noexcept
        {
            constexpr auto allowed = static_cast<std::uint8_t>(AccelerationStructureBuildFlag::preferFastTrace) |
                                     static_cast<std::uint8_t>(AccelerationStructureBuildFlag::preferFastBuild) |
                                     static_cast<std::uint8_t>(AccelerationStructureBuildFlag::minimizeMemory) |
                                     static_cast<std::uint8_t>(AccelerationStructureBuildFlag::allowUpdate) |
                                     static_cast<std::uint8_t>(AccelerationStructureBuildFlag::allowCompaction);
            const auto raw = static_cast<std::uint8_t>(flags);
            return (raw & ~allowed) == 0 && !(hasAny(flags, AccelerationStructureBuildFlag::preferFastTrace) &&
                                              hasAny(flags, AccelerationStructureBuildFlag::preferFastBuild));
        }

        [[nodiscard]] constexpr VkBuildAccelerationStructureFlagsKHR nativeBuildFlags(
            AccelerationStructureBuildFlag flags) noexcept
        {
            VkBuildAccelerationStructureFlagsKHR result = 0;
            if (hasAny(flags, AccelerationStructureBuildFlag::preferFastTrace))
            {
                result |= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
            }
            if (hasAny(flags, AccelerationStructureBuildFlag::preferFastBuild))
            {
                result |= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
            }
            if (hasAny(flags, AccelerationStructureBuildFlag::minimizeMemory))
            {
                result |= VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR;
            }
            if (hasAny(flags, AccelerationStructureBuildFlag::allowUpdate))
            {
                result |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
            }
            if (hasAny(flags, AccelerationStructureBuildFlag::allowCompaction))
            {
                result |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
            }
            return result;
        }

        [[nodiscard]] constexpr VkGeometryFlagsKHR nativeGeometryFlags(AccelerationStructureGeometryFlag flags) noexcept
        {
            VkGeometryFlagsKHR result = 0;
            if (hasAny(flags, AccelerationStructureGeometryFlag::opaque))
            {
                result |= VK_GEOMETRY_OPAQUE_BIT_KHR;
            }
            if (hasAny(flags, AccelerationStructureGeometryFlag::noDuplicateAnyHitInvocation))
            {
                result |= VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
            }
            return result;
        }

        [[nodiscard]] constexpr VkGeometryInstanceFlagsKHR nativeInstanceFlags(
            AccelerationStructureInstanceFlag flags) noexcept
        {
            VkGeometryInstanceFlagsKHR result = 0;
            if (hasAny(flags, AccelerationStructureInstanceFlag::triangleCullDisable))
            {
                result |= VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            }
            if (hasAny(flags, AccelerationStructureInstanceFlag::triangleFrontCounterClockwise))
            {
                result |= VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_KHR;
            }
            if (hasAny(flags, AccelerationStructureInstanceFlag::forceOpaque))
            {
                result |= VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
            }
            if (hasAny(flags, AccelerationStructureInstanceFlag::forceNonOpaque))
            {
                result |= VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;
            }
            return result;
        }

        struct VertexFormatInfo final
        {
            VkFormat format = VK_FORMAT_UNDEFINED;
            std::uint32_t byteSize = 0;
            std::uint32_t alignment = 0;
            std::uint32_t componentSize = 0;
        };

        [[nodiscard]] constexpr std::optional<VertexFormatInfo> nativeVertexFormat(
            AccelerationStructureVertexFormat format) noexcept
        {
            switch (format)
            {
            case AccelerationStructureVertexFormat::float32x2:
                return VertexFormatInfo{VK_FORMAT_R32G32_SFLOAT, 8, 8, 4};
            case AccelerationStructureVertexFormat::float32x3:
                return VertexFormatInfo{VK_FORMAT_R32G32B32_SFLOAT, 12, 4, 4};
            case AccelerationStructureVertexFormat::float16x2:
                return VertexFormatInfo{VK_FORMAT_R16G16_SFLOAT, 4, 4, 2};
            case AccelerationStructureVertexFormat::float16x4:
                return VertexFormatInfo{VK_FORMAT_R16G16B16A16_SFLOAT, 8, 8, 2};
            case AccelerationStructureVertexFormat::snorm16x2:
                return VertexFormatInfo{VK_FORMAT_R16G16_SNORM, 4, 4, 2};
            case AccelerationStructureVertexFormat::snorm16x4:
                return VertexFormatInfo{VK_FORMAT_R16G16B16A16_SNORM, 8, 8, 2};
            }
            return std::nullopt;
        }

        [[nodiscard]] constexpr bool validGeometryFlags(AccelerationStructureGeometryFlag flags) noexcept
        {
            constexpr auto allowed =
                static_cast<std::uint8_t>(AccelerationStructureGeometryFlag::opaque) |
                static_cast<std::uint8_t>(AccelerationStructureGeometryFlag::noDuplicateAnyHitInvocation);
            return (static_cast<std::uint8_t>(flags) & ~allowed) == 0;
        }

        [[nodiscard]] constexpr bool validRange(std::uint64_t resourceSize,
                                                std::uint64_t offset,
                                                std::uint64_t size) noexcept
        {
            return offset < resourceSize && size != 0 && (size == wholeResource || size <= resourceSize - offset);
        }

        [[nodiscard]] constexpr std::uint64_t resolveSize(std::uint64_t resourceSize,
                                                          std::uint64_t offset,
                                                          std::uint64_t size) noexcept
        {
            return size == wholeResource ? resourceSize - offset : size;
        }

        [[nodiscard]] constexpr bool validStridedBufferRange(std::uint64_t resourceSize,
                                                             std::uint64_t offset,
                                                             std::uint32_t count,
                                                             std::uint32_t stride,
                                                             std::uint32_t elementSize) noexcept
        {
            return count != 0 && elementSize != 0 && stride >= elementSize && offset <= resourceSize &&
                   elementSize <= resourceSize - offset &&
                   static_cast<std::uint64_t>(count - 1U) <= (resourceSize - offset - elementSize) / stride;
        }

        [[nodiscard]] constexpr bool validInstanceFlags(AccelerationStructureInstanceFlag flags) noexcept
        {
            constexpr auto allowed = AccelerationStructureInstanceFlag::triangleCullDisable |
                                     AccelerationStructureInstanceFlag::triangleFrontCounterClockwise |
                                     AccelerationStructureInstanceFlag::forceOpaque |
                                     AccelerationStructureInstanceFlag::forceNonOpaque;
            return (static_cast<std::uint8_t>(flags) & ~static_cast<std::uint8_t>(allowed)) == 0 &&
                   !(hasAny(flags, AccelerationStructureInstanceFlag::forceOpaque) &&
                     hasAny(flags, AccelerationStructureInstanceFlag::forceNonOpaque));
        }

        [[nodiscard]] bool invertibleLinearPart(const std::array<float, 12>& transform) noexcept
        {
            const auto at = [&](std::size_t row, std::size_t column) {
                return static_cast<double>(transform[(row * 4U) + column]);
            };
            const double determinant = at(0, 0) * ((at(1, 1) * at(2, 2)) - (at(1, 2) * at(2, 1))) -
                                       at(0, 1) * ((at(1, 0) * at(2, 2)) - (at(1, 2) * at(2, 0))) +
                                       at(0, 2) * ((at(1, 0) * at(2, 1)) - (at(1, 1) * at(2, 0)));
            return std::isfinite(determinant) && determinant != 0.0;
        }

        [[nodiscard]] bool validTopLevelInstance(const AccelerationStructureInstance& instance) noexcept
        {
            return instance.instanceId <= 0x00ff'ffffU && instance.instanceMask <= 0xffU &&
                   instance.hitGroupOffset <= 0x00ff'ffffU && validInstanceFlags(instance.flags) &&
                   std::ranges::all_of(instance.transform, [](float value) { return std::isfinite(value); }) &&
                   invertibleLinearPart(instance.transform);
        }

        [[nodiscard]] VkAccelerationStructureInstanceKHR nativeInstance(const AccelerationStructureInstance& instance,
                                                                        VkDeviceAddress bottomLevelAddress) noexcept
        {
            VkAccelerationStructureInstanceKHR native{};
            std::memcpy(&native.transform, instance.transform.data(), sizeof(instance.transform));
            native.instanceCustomIndex = instance.instanceId;
            native.mask = instance.instanceMask;
            native.instanceShaderBindingTableRecordOffset = instance.hitGroupOffset;
            native.flags = static_cast<std::uint8_t>(nativeInstanceFlags(instance.flags));
            native.accelerationStructureReference = bottomLevelAddress;
            return native;
        }

        [[nodiscard]] bool liveBuildInputBuffer(
            const Result<VulkanResourceAllocator::ResolvedBindingBuffer>& buffer) noexcept
        {
            return buffer && hasAny(buffer->desc.usage, BufferUsage::accelerationStructureBuildInput) &&
                   buffer->deviceAddress != 0;
        }

        [[nodiscard]] constexpr VkIndexType nativeIndexType(IndexFormat format) noexcept
        {
            return format == IndexFormat::uint16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
        }

        [[nodiscard]] VkAccelerationStructureGeometryTrianglesDataKHR nativeTrianglesFromVertices(
            const AccelerationStructureTriangleGeometry& triangles,
            const VertexFormatInfo& vertexFormat,
            VkDeviceAddress vertexBufferAddress) noexcept
        {
            return VkAccelerationStructureGeometryTrianglesDataKHR{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                .pNext = {},
                .vertexFormat = vertexFormat.format,
                .vertexData = {vertexBufferAddress + triangles.vertexOffset},
                .vertexStride = triangles.vertexStride,
                .maxVertex = triangles.vertexCount == 0 ? 0 : triangles.vertexCount - 1,
                .indexType = VK_INDEX_TYPE_NONE_KHR,
                .indexData = {},
                .transformData = {},
            };
        }

        [[nodiscard]] VkAccelerationStructureGeometryKHR nativeTriangleGeometry(
            const VkAccelerationStructureGeometryTrianglesDataKHR& triangles,
            AccelerationStructureGeometryFlag flags) noexcept
        {
            VkAccelerationStructureGeometryKHR native{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                .pNext = {},
                .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
                .geometry = {},
                .flags = nativeGeometryFlags(flags),
            };
            native.geometry.triangles = triangles;
            return native;
        }

        [[nodiscard]] VkAccelerationStructureGeometryKHR nativeAabbGeometry(
            const AccelerationStructureAabbGeometry& aabbs,
            VkDeviceAddress boundsBufferAddress,
            AccelerationStructureGeometryFlag flags) noexcept
        {
            VkAccelerationStructureGeometryKHR native{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                .pNext = {},
                .geometryType = VK_GEOMETRY_TYPE_AABBS_KHR,
                .geometry = {},
                .flags = nativeGeometryFlags(flags),
            };
            native.geometry.aabbs = VkAccelerationStructureGeometryAabbsDataKHR{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR,
                .pNext = {},
                .data = {boundsBufferAddress + aabbs.offset},
                .stride = aabbs.stride,
            };
            return native;
        }

        [[nodiscard]] Result<VkAccelerationStructureGeometryTrianglesDataKHR> validatedTrianglesFromVertices(
            const VulkanResourceAllocator& resources, const AccelerationStructureTriangleGeometry& triangles)
        {
            const auto vertexFormat = nativeVertexFormat(triangles.vertexFormat);
            const auto vertices = resources.resolveBindingBuffer(triangles.vertexBuffer);
            if (!vertexFormat || !liveBuildInputBuffer(vertices) ||
                triangles.vertexOffset % vertexFormat->alignment != 0 ||
                triangles.vertexStride % vertexFormat->componentSize != 0 ||
                !validStridedBufferRange(vertices->desc.size,
                                         triangles.vertexOffset,
                                         triangles.vertexCount,
                                         triangles.vertexStride,
                                         vertexFormat->byteSize))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Triangle vertices require a live build-input buffer and an aligned in-bounds format/stride.",
                });
            }
            return nativeTrianglesFromVertices(triangles, *vertexFormat, vertices->deviceAddress);
        }

        [[nodiscard]] Result<VkDeviceAddress> validatedTriangleIndexAddress(
            const VulkanResourceAllocator& resources, const AccelerationStructureTriangleGeometry& triangles)
        {
            const auto indices = resources.resolveBindingBuffer(triangles.indexBuffer);
            const std::uint32_t indexSize = triangles.indexFormat == IndexFormat::uint16 ? 2U : 4U;
            const bool validIndexFormat =
                triangles.indexFormat == IndexFormat::uint16 || triangles.indexFormat == IndexFormat::uint32;
            const bool wholeTriangles = triangles.indexCount != 0 && triangles.indexCount % 3U == 0;
            if (!liveBuildInputBuffer(indices) || !validIndexFormat || !wholeTriangles ||
                triangles.indexOffset % indexSize != 0 ||
                !validStridedBufferRange(
                    indices->desc.size, triangles.indexOffset, triangles.indexCount, indexSize, indexSize))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Triangle indices require a live build-input buffer and an aligned in-bounds triangle list.",
                });
            }
            return indices->deviceAddress + triangles.indexOffset;
        }

        [[nodiscard]] Result<VkDeviceAddress> validatedTriangleTransformAddress(
            const VulkanResourceAllocator& resources, const AccelerationStructureTriangleGeometry& triangles)
        {
            const auto transform = resources.resolveBindingBuffer(triangles.transformBuffer);
            constexpr std::uint64_t transformSize = 12U * sizeof(float);
            if (!liveBuildInputBuffer(transform) || triangles.transformOffset % 16U != 0 ||
                !validRange(transform->desc.size, triangles.transformOffset, transformSize))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A triangle transform requires one aligned 3x4 float matrix in a build-input buffer.",
                });
            }
            return transform->deviceAddress + triangles.transformOffset;
        }

        struct NativeBottomLevelGeometry final
        {
            VkAccelerationStructureGeometryKHR geometry{};
            std::uint32_t primitiveCount = 0;
        };

        [[nodiscard]] Result<NativeBottomLevelGeometry> validatedTriangleGeometry(
            const VulkanResourceAllocator& resources,
            const AccelerationStructureTriangleGeometry& triangles,
            AccelerationStructureGeometryFlag flags)
        {
            auto native = validatedTrianglesFromVertices(resources, triangles);
            if (!native)
            {
                return std::unexpected(std::move(native.error()));
            }
            std::uint32_t primitiveCount = 0;
            if (triangles.indexBuffer.isValid())
            {
                auto indexAddress = validatedTriangleIndexAddress(resources, triangles);
                if (!indexAddress)
                {
                    return std::unexpected(std::move(indexAddress.error()));
                }
                native->indexType = nativeIndexType(triangles.indexFormat);
                native->indexData.deviceAddress = *indexAddress;
                primitiveCount = triangles.indexCount / 3U;
            }
            else
            {
                if (triangles.indexCount != 0 || triangles.vertexCount % 3U != 0)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Non-indexed BLAS triangles require zero indices and a multiple-of-three vertex count.",
                    });
                }
                primitiveCount = triangles.vertexCount / 3U;
            }
            if (triangles.transformBuffer.isValid())
            {
                auto transformAddress = validatedTriangleTransformAddress(resources, triangles);
                if (!transformAddress)
                {
                    return std::unexpected(std::move(transformAddress.error()));
                }
                native->transformData.deviceAddress = *transformAddress;
            }
            return NativeBottomLevelGeometry{
                .geometry = nativeTriangleGeometry(*native, flags),
                .primitiveCount = primitiveCount,
            };
        }

        [[nodiscard]] Result<NativeBottomLevelGeometry> validatedAabbGeometry(
            const VulkanResourceAllocator& resources,
            const AccelerationStructureAabbGeometry& aabbs,
            AccelerationStructureGeometryFlag flags)
        {
            const auto source = resources.resolveBindingBuffer(aabbs.buffer);
            constexpr std::uint32_t aabbSize = 6U * sizeof(float);
            if (!liveBuildInputBuffer(source) || aabbs.offset % 8U != 0 || aabbs.stride % 8U != 0 ||
                !validStridedBufferRange(source->desc.size, aabbs.offset, aabbs.count, aabbs.stride, aabbSize))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "AABB geometry requires aligned, strided bounds in a live build-input buffer.",
                });
            }
            return NativeBottomLevelGeometry{
                .geometry = nativeAabbGeometry(aabbs, source->deviceAddress, flags),
                .primitiveCount = aabbs.count,
            };
        }

        [[nodiscard]] Result<NativeBottomLevelGeometry> validatedBottomLevelGeometry(
            const VulkanResourceAllocator& resources, const AccelerationStructureGeometry& geometry)
        {
            if (!validGeometryFlags(geometry.flags))
            {
                return std::unexpected(Error{ErrorCode::invalidArgument, 0, "A BLAS geometry contains unknown flags."});
            }
            if (const auto* triangles = std::get_if<AccelerationStructureTriangleGeometry>(&geometry.data))
            {
                return validatedTriangleGeometry(resources, *triangles, geometry.flags);
            }
            if (const auto* aabbs = std::get_if<AccelerationStructureAabbGeometry>(&geometry.data))
            {
                return validatedAabbGeometry(resources, *aabbs, geometry.flags);
            }
            return std::unexpected(Error{ErrorCode::invalidArgument, 0, "A BLAS geometry type is invalid."});
        }

        [[nodiscard]] Status validateStorageRange(const BufferDesc& storage, const AccelerationStructureDesc& desc)
        {
            if (storage.usage != BufferUsage::accelerationStructureStorage ||
                !validRange(storage.size, desc.offset, desc.size))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Acceleration-structure storage must be a live, aligned, in-bounds dedicated storage-buffer range.",
                });
            }
            if (desc.offset % accelerationStructureAlignment != 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Acceleration-structure storage must be a live, in-bounds dedicated range with an aligned start "
                    "address.",
                });
            }
            return {};
        }

        [[nodiscard]] bool canRecordAccelerationStructureBuild(const VulkanCommandBuffer& commandBuffer) noexcept
        {
            const bool buildCapableQueue =
                commandBuffer.queueType() == QueueType::graphics || commandBuffer.queueType() == QueueType::compute;
            return buildCapableQueue && commandBuffer.graphicsState().renderingOwner == nullptr;
        }

        [[nodiscard]] bool validBuildScratchRange(const VulkanResourceAllocator::ResolvedBindingBuffer& scratch,
                                                  std::uint64_t offset,
                                                  std::uint32_t alignment,
                                                  std::uint64_t buildScratchSize) noexcept
        {
            return scratch.desc.usage == BufferUsage::accelerationStructureScratch && scratch.deviceAddress != 0 &&
                   offset % alignment == 0 && (scratch.deviceAddress + offset) % alignment == 0 &&
                   (buildScratchSize != 0 ? validRange(scratch.desc.size, offset, buildScratchSize)
                                          : offset < scratch.desc.size);
        }

        [[nodiscard]] Result<NativeBottomLevelGeometry> nativeTriangleGeometryRetainingBuffers(
            const VulkanResourceAllocator& resources,
            const AccelerationStructureTriangleGeometry& triangles,
            AccelerationStructureGeometryFlag flags,
            std::vector<std::shared_ptr<void>>& retained)
        {
            const auto vertexFormat = nativeVertexFormat(triangles.vertexFormat);
            auto vertices = resources.resolveBindingBuffer(triangles.vertexBuffer);
            if (!vertexFormat || !vertices)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Triangle vertices require a live build-input buffer and a supported format.",
                });
            }
            auto native = nativeTrianglesFromVertices(triangles, *vertexFormat, vertices->deviceAddress);
            std::uint32_t primitiveCount = triangles.vertexCount / 3U;
            retained.push_back(vertices->keepAlive);
            if (triangles.indexBuffer.isValid())
            {
                auto indices = resources.resolveBindingBuffer(triangles.indexBuffer);
                if (!liveBuildInputBuffer(indices))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Triangle indices require a live build-input buffer and an aligned in-bounds triangle list.",
                    });
                }
                native.indexType = nativeIndexType(triangles.indexFormat);
                native.indexData.deviceAddress = indices->deviceAddress + triangles.indexOffset;
                primitiveCount = triangles.indexCount / 3U;
                retained.push_back(indices->keepAlive);
            }
            if (triangles.transformBuffer.isValid())
            {
                auto transform = resources.resolveBindingBuffer(triangles.transformBuffer);
                if (!liveBuildInputBuffer(transform))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A triangle transform requires one aligned 3x4 float matrix in a build-input buffer.",
                    });
                }
                native.transformData.deviceAddress = transform->deviceAddress + triangles.transformOffset;
                retained.push_back(transform->keepAlive);
            }
            return NativeBottomLevelGeometry{
                .geometry = nativeTriangleGeometry(native, flags),
                .primitiveCount = primitiveCount,
            };
        }

        [[nodiscard]] Result<NativeBottomLevelGeometry> nativeAabbGeometryRetainingBuffer(
            const VulkanResourceAllocator& resources,
            const AccelerationStructureAabbGeometry& aabbs,
            AccelerationStructureGeometryFlag flags,
            std::vector<std::shared_ptr<void>>& retained)
        {
            auto source = resources.resolveBindingBuffer(aabbs.buffer);
            if (!liveBuildInputBuffer(source))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "AABB geometry requires aligned, strided bounds in a live build-input buffer.",
                });
            }
            const NativeBottomLevelGeometry native{
                .geometry = nativeAabbGeometry(aabbs, source->deviceAddress, flags),
                .primitiveCount = aabbs.count,
            };
            retained.push_back(source->keepAlive);
            return native;
        }

        [[nodiscard]] Result<NativeBottomLevelGeometry> nativeBottomLevelGeometryRetainingBuffers(
            const VulkanResourceAllocator& resources,
            const AccelerationStructureGeometry& geometry,
            std::vector<std::shared_ptr<void>>& retained)
        {
            if (const auto* triangles = std::get_if<AccelerationStructureTriangleGeometry>(&geometry.data))
            {
                return nativeTriangleGeometryRetainingBuffers(resources, *triangles, geometry.flags, retained);
            }
            const auto* aabbs = std::get_if<AccelerationStructureAabbGeometry>(&geometry.data);
            return nativeAabbGeometryRetainingBuffer(resources, *aabbs, geometry.flags, retained);
        }

        [[nodiscard]] constexpr VkAccelerationStructureBuildRangeInfoKHR nativeBuildRange(
            std::uint32_t primitiveCount) noexcept
        {
            return VkAccelerationStructureBuildRangeInfoKHR{
                .primitiveCount = primitiveCount,
                .primitiveOffset = {},
                .firstVertex = {},
                .transformOffset = {},
            };
        }

        struct NativeBottomLevelBuildInput final
        {
            std::vector<VkAccelerationStructureGeometryKHR> geometries;
            std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges;
        };

        [[nodiscard]] Result<NativeBottomLevelBuildInput> nativeBottomLevelBuildInputRetainingBuffers(
            const VulkanResourceAllocator& resources,
            std::span<const AccelerationStructureGeometry> geometries,
            std::vector<std::shared_ptr<void>>& retained)
        {
            NativeBottomLevelBuildInput nativeInput;
            nativeInput.geometries.reserve(geometries.size());
            nativeInput.ranges.reserve(geometries.size());
            for (const auto& geometry : geometries)
            {
                auto native = nativeBottomLevelGeometryRetainingBuffers(resources, geometry, retained);
                if (!native)
                {
                    return std::unexpected(std::move(native.error()));
                }
                nativeInput.geometries.push_back(native->geometry);
                nativeInput.ranges.push_back(nativeBuildRange(native->primitiveCount));
            }
            return nativeInput;
        }

        [[nodiscard]] VkAccelerationStructureBuildGeometryInfoKHR nativeBuildGeometryInfo(
            VkAccelerationStructureTypeKHR type,
            AccelerationStructureBuildFlag flags,
            std::span<const VkAccelerationStructureGeometryKHR> geometries,
            VkAccelerationStructureKHR destination = VK_NULL_HANDLE,
            VkDeviceAddress scratchAddress = 0) noexcept
        {
            return VkAccelerationStructureBuildGeometryInfoKHR{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
                .pNext = {},
                .type = type,
                .flags = nativeBuildFlags(flags),
                .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
                .srcAccelerationStructure = {},
                .dstAccelerationStructure = destination,
                .geometryCount = static_cast<std::uint32_t>(geometries.size()),
                .pGeometries = geometries.data(),
                .ppGeometries = {},
                .scratchData = {scratchAddress},
            };
        }

        [[nodiscard]] Result<std::vector<VkAccelerationStructureInstanceKHR>> nativeInstancesRetainingBottomLevels(
            const VulkanAccelerationManager& accelerationStructures,
            std::span<const AccelerationStructureInstance> instances,
            VulkanTopLevelInstances& prepared)
        {
            std::vector<VkAccelerationStructureInstanceKHR> nativeInstances;
            nativeInstances.reserve(instances.size());
            prepared.dependencies.reserve(instances.size() * 2U);
            prepared.dependencyTrackers.reserve(instances.size());
            for (const auto& instance : instances)
            {
                if (!validTopLevelInstance(instance))
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A TLAS instance needs an instanceId and hitGroupOffset below 2^24, an instanceMask below "
                        "2^8, known flags without both forceOpaque and forceNonOpaque, and a finite transform whose "
                        "3x3 part is invertible.",
                    });
                }
                auto bottom = accelerationStructures.resolve(instance.bottomLevel);
                if (!bottom || bottom->type != AccelerationStructureType::bottomLevel || bottom->deviceAddress == 0 ||
                    bottom->version == nullptr)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "A TLAS instance requires a built bottom-level acceleration structure.",
                    });
                }
                nativeInstances.push_back(nativeInstance(instance, bottom->deviceAddress));
                prepared.dependencies.push_back(std::move(bottom->keepAlive));
                prepared.dependencies.push_back(std::move(bottom->version));
                prepared.dependencyTrackers.push_back(std::move(bottom->use.executionTracker));
            }
            detail::deduplicateSharedPointers(prepared.dependencies);
            detail::deduplicateSharedPointers(prepared.dependencyTrackers);
            return nativeInstances;
        }

        class TemporaryBuffer final
        {
        public:
            explicit TemporaryBuffer(VulkanResourceAllocator& resources) noexcept : resources_(resources)
            {
            }

            TemporaryBuffer(const TemporaryBuffer&) = delete;
            TemporaryBuffer& operator=(const TemporaryBuffer&) = delete;
            TemporaryBuffer(TemporaryBuffer&&) = delete;
            TemporaryBuffer& operator=(TemporaryBuffer&&) = delete;

            ~TemporaryBuffer()
            {
                if (handle_.isValid())
                {
                    static_cast<void>(resources_.destroyBuffer(handle_));
                }
            }

            void adopt(BufferHandle handle) noexcept
            {
                handle_ = handle;
            }

        private:
            VulkanResourceAllocator& resources_;
            BufferHandle handle_{};
        };

        struct UploadedInstances final
        {
            VkDeviceAddress deviceAddress = 0;
            std::shared_ptr<void> keepAlive;
        };

        [[nodiscard]] Result<UploadedInstances> uploadInstances(
            VulkanResourceAllocator& resources,
            std::span<const VkAccelerationStructureInstanceKHR> instances,
            TemporaryBuffer& owner)
        {
            auto buffer = resources.createBuffer({
                .size = std::max<std::uint64_t>(instances.size() * sizeof(VkAccelerationStructureInstanceKHR), 16),
                .usage = BufferUsage::accelerationStructureBuildInput,
                .memoryType = MemoryType::upload,
                .label = "Vulkan TLAS instances",
            });
            if (!buffer)
            {
                return std::unexpected(std::move(buffer.error()));
            }
            owner.adopt(*buffer);
            if (auto written = resources.writeBuffer(*buffer, 0, std::as_bytes(instances)); !written)
            {
                return std::unexpected(std::move(written.error()));
            }
            auto resolved = resources.resolveBindingBuffer(*buffer);
            if (!resolved || resolved->deviceAddress == 0 ||
                resolved->deviceAddress % accelerationStructureInstanceAddressAlignment != 0)
            {
                return std::unexpected(Error{
                    ErrorCode::nativeFailure,
                    0,
                    "Vulkan could not give the TLAS instance buffer a 16-byte aligned device address.",
                });
            }
            return UploadedInstances{
                .deviceAddress = resolved->deviceAddress,
                .keepAlive = std::move(resolved->keepAlive),
            };
        }

        [[nodiscard]] constexpr VkAccelerationStructureGeometryKHR nativeInstancesGeometry(
            VkDeviceAddress instancesAddress = 0) noexcept
        {
            return VkAccelerationStructureGeometryKHR{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                .pNext = {},
                .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
                .geometry =
                    {
                        .instances =
                            {
                                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                                .pNext = {},
                                .arrayOfPointers = VK_FALSE,
                                .data = {instancesAddress},
                            },
                    },
                .flags = {},
            };
        }

        constexpr std::uint32_t instanceRecordAlignment = 16;

        static_assert(instanceRecordAlignment == accelerationStructureInstanceAddressAlignment);

        static_assert(sizeof(VkAccelerationStructureInstanceKHR) == sizeof(AccelerationStructureInstanceRecord));
        static_assert(offsetof(VkAccelerationStructureInstanceKHR, accelerationStructureReference) ==
                      offsetof(AccelerationStructureInstanceRecord, bottomLevel));
        static_assert(instanceRecordAlignment == alignof(AccelerationStructureInstanceRecord));
        static_assert(static_cast<std::uint32_t>(AccelerationStructureInstanceFlag::triangleCullDisable) ==
                      static_cast<std::uint32_t>(VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR));
        static_assert(static_cast<std::uint32_t>(AccelerationStructureInstanceFlag::triangleFrontCounterClockwise) ==
                      static_cast<std::uint32_t>(VK_GEOMETRY_INSTANCE_TRIANGLE_FRONT_COUNTERCLOCKWISE_BIT_KHR));
        static_assert(static_cast<std::uint32_t>(AccelerationStructureInstanceFlag::forceOpaque) ==
                      static_cast<std::uint32_t>(VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR));
        static_assert(static_cast<std::uint32_t>(AccelerationStructureInstanceFlag::forceNonOpaque) ==
                      static_cast<std::uint32_t>(VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR));

        [[nodiscard]] Result<VulkanResourceAllocator::ResolvedBindingBuffer> validatedInstanceBuffer(
            const VulkanResourceAllocator& resources, BufferHandle handle, std::uint64_t offset, std::uint32_t count)
        {
            auto buffer = resources.resolveBindingBuffer(handle);
            if (!buffer)
            {
                return std::unexpected(std::move(buffer.error()));
            }
            const std::uint64_t recordBytes =
                static_cast<std::uint64_t>(count) * sizeof(VkAccelerationStructureInstanceKHR);
            if (!liveBuildInputBuffer(buffer) || buffer->desc.memoryType == MemoryType::readback ||
                offset % instanceRecordAlignment != 0 ||
                (buffer->deviceAddress + offset) % instanceRecordAlignment != 0 ||
                !validRange(buffer->desc.size, offset, recordBytes))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "TLAS instance records require a live, non-readback build-input buffer that holds every record "
                    "from an aligned offset.",
                });
            }
            return buffer;
        }

        [[nodiscard]] Result<VulkanTopLevelInstances> prepareSpanInstances(
            const VulkanAccelerationManager& accelerationStructures,
            VulkanResourceAllocator& resources,
            std::span<const AccelerationStructureInstance> instances,
            TemporaryBuffer& uploadOwner)
        {
            VulkanTopLevelInstances prepared;
            auto native = nativeInstancesRetainingBottomLevels(accelerationStructures, instances, prepared);
            if (!native)
            {
                return std::unexpected(std::move(native.error()));
            }
            auto uploaded = uploadInstances(resources, *native, uploadOwner);
            if (!uploaded)
            {
                return std::unexpected(std::move(uploaded.error()));
            }
            prepared.deviceAddress = uploaded->deviceAddress;
            prepared.count = static_cast<std::uint32_t>(native->size());
            prepared.owners.push_back(std::move(uploaded->keepAlive));
            return prepared;
        }

        [[nodiscard]] Result<VulkanTopLevelInstances> prepareBufferInstances(const VulkanResourceAllocator& resources,
                                                                             BufferHandle instanceBuffer,
                                                                             std::uint64_t instanceOffset,
                                                                             std::uint32_t instanceCount)
        {
            auto buffer = validatedInstanceBuffer(resources, instanceBuffer, instanceOffset, instanceCount);
            if (!buffer)
            {
                return std::unexpected(std::move(buffer.error()));
            }
            VulkanTopLevelInstances prepared;
            prepared.deviceAddress = buffer->deviceAddress + instanceOffset;
            prepared.count = instanceCount;
            prepared.owners.push_back(std::move(buffer->keepAlive));
            return prepared;
        }

        constexpr std::string_view inPlaceUpdateConsumerConflict =
            "An acceleration structure cannot be read while another command recording updates it in place.";

        struct AccelerationStructureConsumerHold final
        {
            detail::AccelerationStructureExecutionLease lease;
            std::shared_ptr<VulkanAccelerationStructureVersion> version;
        };
    } // namespace

    Status retainVulkanAccelerationStructureConsumers(VulkanCommandBuffer& commandBuffer,
                                                      std::span<const VulkanAccelerationStructureUse> uses)
    {
        if (uses.empty())
        {
            return {};
        }
        std::vector<std::shared_ptr<void>> holds;
        try
        {
            holds.reserve(uses.size());
            detail::reserveAdditional(commandBuffer.retained(), uses.size());
            const auto recordingId = commandBuffer.accelerationStructureRecordingId();
            for (const auto& use : uses)
            {
                if (use.version == nullptr || use.executionTracker == nullptr)
                {
                    continue;
                }
                const std::scoped_lock lock(use.version->mutex);
                const auto version = use.version->current;
                if (version == nullptr)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidState,
                        0,
                        "A bound acceleration-structure descriptor has no built version.",
                    });
                }
                std::vector<std::shared_ptr<detail::AccelerationStructureExecutionTracker>> trackers;
                trackers.reserve(version->dependencyTrackers.size() + 1U);
                trackers.push_back(use.executionTracker);
                trackers.insert(trackers.end(), version->dependencyTrackers.begin(), version->dependencyTrackers.end());
                auto lease = detail::AccelerationStructureExecutionLease::tryAcquireConsumers(trackers, recordingId);
                if (!lease)
                {
                    return std::unexpected(
                        Error{ErrorCode::invalidState, 0, std::string(inPlaceUpdateConsumerConflict)});
                }
                holds.push_back(std::make_shared<AccelerationStructureConsumerHold>(AccelerationStructureConsumerHold{
                    .lease = std::move(*lease),
                    .version = version,
                }));
            }
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Acceleration-structure binding leases could not be retained.",
            });
        }
        auto& retained = commandBuffer.retained();
        std::ranges::move(holds, std::back_inserter(retained));
        return {};
    }

    VulkanAccelerationManager::AccelerationStructureKeepAlive::~AccelerationStructureKeepAlive()
    {
        releaseNow();
    }

    void VulkanAccelerationManager::AccelerationStructureKeepAlive::releaseNow() noexcept
    {
        std::shared_ptr<void> releasedStorage;
        std::shared_ptr<AccelerationStructureStorageLease> releasedLease;
        {
            const std::scoped_lock lock(mutex);
            if (table != nullptr && table->vkDestroyAccelerationStructureKHR != nullptr && device != VK_NULL_HANDLE &&
                accelerationStructure != VK_NULL_HANDLE)
            {
                table->vkDestroyAccelerationStructureKHR(device, accelerationStructure, nullptr);
#ifdef RACHIS_TESTING
                test::liveAccelerationStructures().fetch_sub(1, std::memory_order_relaxed);
#endif
            }
            accelerationStructure = VK_NULL_HANDLE;
            device = VK_NULL_HANDLE;
            table = nullptr;
            releasedStorage = std::move(storage);
            releasedLease = std::move(storageLease);
        }
    }

    VulkanAccelerationManager::~VulkanAccelerationManager()
    {
        shutdown();
    }

    void VulkanAccelerationManager::initialize(VkDevice device,
                                               const VolkDeviceTable* table,
                                               VulkanResourceAllocator* resources,
                                               const DeviceCapabilities* capabilities,
                                               std::uint32_t scratchAlignment,
                                               const VulkanInstanceTable* instanceTable) noexcept
    {
        device_ = device;
        table_ = table;
        instanceTable_ = instanceTable;
        resources_ = resources;
        capabilities_ = capabilities;
        scratchAlignment_ = scratchAlignment == 0 ? 256 : scratchAlignment;
    }

    void VulkanAccelerationManager::shutdown() noexcept
    {
        std::vector<std::weak_ptr<AccelerationStructureKeepAlive>> keepAlives;
        {
            const std::scoped_lock lock(mutex_);
            structures_.clear();
            storageLeases_.clear();
            keepAlives = std::move(keepAlives_);
            nextKeepAlivePruneThreshold_ = 64;
            device_ = VK_NULL_HANDLE;
            table_ = nullptr;
            instanceTable_ = nullptr;
            resources_ = nullptr;
            capabilities_ = nullptr;
        }
        for (const auto& weakKeepAlive : keepAlives)
        {
            if (auto keepAlive = weakKeepAlive.lock(); keepAlive != nullptr)
            {
                keepAlive->releaseNow();
            }
        }
    }

    bool VulkanAccelerationManager::supported() const noexcept
    {
        return device_ != VK_NULL_HANDLE && table_ != nullptr && resources_ != nullptr && capabilities_ != nullptr &&
               (capabilities_->rayTracing || capabilities_->rayQuery) &&
               table_->vkCreateAccelerationStructureKHR != nullptr &&
               table_->vkGetAccelerationStructureBuildSizesKHR != nullptr &&
               table_->vkCmdBuildAccelerationStructuresKHR != nullptr;
    }

    Result<AccelerationStructureBuildSizes> VulkanAccelerationManager::queryBuildSizes(
        VkAccelerationStructureTypeKHR type,
        AccelerationStructureBuildFlag flags,
        std::uint32_t geometryCount,
        const VkAccelerationStructureGeometryKHR* geometries,
        const std::uint32_t* primitiveCounts) const
    {
        if (!supported())
        {
            return std::unexpected(Error{
                ErrorCode::unsupported, 0, "Acceleration-structure builds require acceleration-structure support."});
        }
        const auto info = nativeBuildGeometryInfo(type, flags, std::span(geometries, geometryCount));
        VkAccelerationStructureBuildSizesInfoKHR sizes{};
        sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

        table_->vkGetAccelerationStructureBuildSizesKHR(
            device_, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, primitiveCounts, &sizes);
        if (sizes.accelerationStructureSize == 0)
        {
            return std::unexpected(Error{
                ErrorCode::nativeFailure,
                0,
                "Vulkan returned an empty acceleration-structure size for otherwise valid input.",
            });
        }
        return AccelerationStructureBuildSizes{
            .accelerationStructureSize = sizes.accelerationStructureSize,
            .buildScratchSize = sizes.buildScratchSize,
            .updateScratchSize = sizes.updateScratchSize,
        };
    }

    VkPipelineStageFlags2 VulkanAccelerationManager::readerStages(QueueType queueType) const noexcept
    {
        VkPipelineStageFlags2 stages =
            VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        if (queueType == QueueType::graphics)
        {
            stages |= VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            if (capabilities_ != nullptr && capabilities_->geometryShaders)
            {
                stages |= VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT;
            }
            if (capabilities_ != nullptr && capabilities_->tessellationShaders)
            {
                stages |= VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT |
                          VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT;
            }
            if (capabilities_ != nullptr && capabilities_->taskShaders)
            {
                stages |= VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT;
            }
            if (capabilities_ != nullptr && capabilities_->meshShaders)
            {
                stages |= VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
            }
        }
        if (capabilities_ != nullptr && capabilities_->rayTracing)
        {
            stages |= VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        }
        return stages;
    }

    void VulkanAccelerationManager::recordInPlaceUpdateBarrier(VkCommandBuffer command,
                                                               QueueType queueType) const noexcept
    {
        const VkMemoryBarrier2 barrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .pNext = {},
            .srcStageMask = readerStages(queueType),
            .srcAccessMask = VK_ACCESS_2_NONE,
            .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            .dstAccessMask =
                VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        };
        const VkDependencyInfo dependency{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = {},
            .dependencyFlags = {},
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &barrier,
            .bufferMemoryBarrierCount = {},
            .pBufferMemoryBarriers = {},
            .imageMemoryBarrierCount = {},
            .pImageMemoryBarriers = {},
        };
        table_->vkCmdPipelineBarrier2(command, &dependency);
    }

    Status VulkanAccelerationManager::recordBuildBarrier(VkCommandBuffer command, QueueType queueType) const
    {
        const VkMemoryBarrier2 barrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .pNext = {},
            .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            .srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
            .dstStageMask = readerStages(queueType),
            .dstAccessMask =
                VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        };
        const VkDependencyInfo dependency{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pNext = {},
            .dependencyFlags = {},
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &barrier,
            .bufferMemoryBarrierCount = {},
            .pBufferMemoryBarriers = {},
            .imageMemoryBarrierCount = {},
            .pImageMemoryBarriers = {},
        };
        table_->vkCmdPipelineBarrier2(command, &dependency);
        return {};
    }

    Result<AccelerationStructureBuildSizes> VulkanAccelerationManager::queryBottomLevelBuildSizes(
        const BottomLevelAccelerationStructureBuildInput& input) const
    {
        if (!supported())
        {
            return std::unexpected(Error{
                ErrorCode::unsupported, 0, "Acceleration-structure builds require acceleration-structure support."});
        }
        if (input.geometries.empty() || !validAccelerationStructureBuildFlags(input.flags) ||
            (capabilities_->rayTracingLimits.maxGeometriesPerBottomLevel != 0 &&
             input.geometries.size() > capabilities_->rayTracingLimits.maxGeometriesPerBottomLevel))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A BLAS build requires geometry, valid build flags, and a supported geometry count.",
            });
        }

        std::vector<VkAccelerationStructureGeometryKHR> geometries;
        std::vector<std::uint32_t> primitiveCounts;
        geometries.reserve(input.geometries.size());
        primitiveCounts.reserve(input.geometries.size());
        std::uint64_t totalPrimitives = 0;
        for (const auto& geometry : input.geometries)
        {
            auto native = validatedBottomLevelGeometry(*resources_, geometry);
            if (!native)
            {
                return std::unexpected(std::move(native.error()));
            }
            if (!geometries.empty() && native->geometry.geometryType != geometries.front().geometryType)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A BLAS build cannot mix triangle and AABB geometries.",
                });
            }
            geometries.push_back(native->geometry);
            primitiveCounts.push_back(native->primitiveCount);
            totalPrimitives += native->primitiveCount;
        }
        if (const auto limit = capabilities_->rayTracingLimits.maxPrimitivesPerBottomLevel;
            limit != 0 && totalPrimitives > limit)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The geometries of a BLAS build hold more primitives in total than the device's "
                "maxPrimitivesPerBottomLevel limit.",
            });
        }
        return queryBuildSizes(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                               input.flags,
                               static_cast<std::uint32_t>(geometries.size()),
                               geometries.data(),
                               primitiveCounts.data());
    }

    Result<AccelerationStructureBuildSizes> VulkanAccelerationManager::queryTopLevelBuildSizes(
        const TopLevelAccelerationStructureBuildInput& input) const
    {
        if (!supported())
        {
            return std::unexpected(Error{
                ErrorCode::unsupported, 0, "Acceleration-structure builds require acceleration-structure support."});
        }
        if (input.instanceCount == 0 || !validAccelerationStructureBuildFlags(input.flags) ||
            (capabilities_->rayTracingLimits.maxInstancesPerTopLevel != 0 &&
             input.instanceCount > capabilities_->rayTracingLimits.maxInstancesPerTopLevel))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A TLAS build requires a supported non-zero instance count and valid build flags.",
            });
        }
        const auto geometry = nativeInstancesGeometry();
        const auto primitives = input.instanceCount;
        return queryBuildSizes(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, input.flags, 1, &geometry, &primitives);
    }

    Result<std::shared_ptr<VulkanAccelerationManager::AccelerationStructureStorageLease>> VulkanAccelerationManager::
        leaseStorageRange(BufferHandle storage, std::uint64_t offset, std::uint64_t size)
    {
        auto lease = std::make_shared<AccelerationStructureStorageLease>();
        lease->storage = storage;
        lease->offset = offset;
        lease->size = size;

        const std::scoped_lock lock(mutex_);
        std::erase_if(storageLeases_, [](const auto& existing) { return existing.expired(); });
        for (const auto& existingWeak : storageLeases_)
        {
            const auto existing = existingWeak.lock();
            if (existing != nullptr && existing->storage == lease->storage &&
                lease->offset < existing->offset + existing->size && existing->offset < lease->offset + lease->size)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Acceleration-structure storage ranges must not overlap retained acceleration structures.",
                });
            }
        }
        storageLeases_.push_back(lease);
        return lease;
    }

    Result<VkAccelerationStructureKHR> VulkanAccelerationManager::createNativeAccelerationStructure(
        const AccelerationStructureDesc& desc, VkBuffer storageBuffer, std::uint64_t size) const
    {
        const VkAccelerationStructureCreateInfoKHR createInfo{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
            .pNext = {},
            .createFlags = {},
            .buffer = storageBuffer,
            .offset = desc.offset,
            .size = size,
            .type = desc.type == AccelerationStructureType::topLevel ? VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR
                                                                     : VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            .deviceAddress = {},
        };
        VkAccelerationStructureKHR native = VK_NULL_HANDLE;
        const auto created = table_->vkCreateAccelerationStructureKHR(device_, &createInfo, nullptr, &native);
        if (created != VK_SUCCESS)
        {
            return std::unexpected(
                makeVulkanError(vulkanResultErrorCode(created), created, "vkCreateAccelerationStructureKHR"));
        }
#ifdef RACHIS_TESTING
        test::liveAccelerationStructures().fetch_add(1, std::memory_order_relaxed);
#endif
        return native;
    }

    std::shared_ptr<VulkanAccelerationManager::AccelerationStructureRecord> VulkanAccelerationManager::makeRecord(
        const AccelerationStructureDesc& desc,
        std::uint64_t size,
        std::shared_ptr<void> storageKeepAlive,
        std::shared_ptr<AccelerationStructureStorageLease> storageLease) const
    {
#ifdef RACHIS_TESTING
        throwInjectedVulkanBadAlloc(VulkanBadAllocPoint::accelerationStructureRecord);
#endif
        auto record = std::make_shared<AccelerationStructureRecord>();
        record->type = desc.type;
        record->storage = desc.storage;
        record->offset = desc.offset;
        record->size = size;
        record->label = std::string(desc.label);
        record->keepAlive = std::make_shared<AccelerationStructureKeepAlive>();
        record->keepAlive->device = device_;
        record->keepAlive->table = table_;
        record->keepAlive->storage = std::move(storageKeepAlive);
        record->keepAlive->storageLease = std::move(storageLease);
        return record;
    }

    void VulkanAccelerationManager::adoptNativeAccelerationStructure(AccelerationStructureRecord& record,
                                                                     VkAccelerationStructureKHR native) const noexcept
    {
        {
            const std::scoped_lock lock(record.keepAlive->mutex);
            record.keepAlive->accelerationStructure = native;
        }
        record.accelerationStructure = native;
        const VkAccelerationStructureDeviceAddressInfoKHR addressInfo{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
            .pNext = {},
            .accelerationStructure = native,
        };
        record.deviceAddress = table_->vkGetAccelerationStructureDeviceAddressKHR(device_, &addressInfo);
        if (!record.label.empty())
        {
            setVulkanObjectName(instanceTable_,
                                device_,
                                VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR,
                                reinterpret_cast<std::uint64_t>(native),
                                record.label.c_str());
        }
    }

    Result<AccelerationStructureHandle> VulkanAccelerationManager::registerRecord(
        std::shared_ptr<AccelerationStructureRecord> record)
    {
        const std::scoped_lock lock(mutex_);
        if (keepAlives_.size() >= nextKeepAlivePruneThreshold_)
        {
            std::erase_if(keepAlives_, [](const auto& existing) { return existing.expired(); });
            nextKeepAlivePruneThreshold_ = std::max<std::size_t>(64, keepAlives_.size() * 2);
        }
        keepAlives_.push_back(record->keepAlive);
        return structures_.emplace(std::move(record));
    }

    Result<AccelerationStructureHandle> VulkanAccelerationManager::createAccelerationStructure(
        const AccelerationStructureDesc& desc)
    {
        if (!supported())
        {
            return std::unexpected(
                Error{ErrorCode::unsupported, 0, "Acceleration structures require acceleration-structure support."});
        }
        if (desc.type != AccelerationStructureType::bottomLevel && desc.type != AccelerationStructureType::topLevel)
        {
            return std::unexpected(Error{ErrorCode::invalidArgument, 0, "The acceleration-structure type is invalid."});
        }
        auto storage = resources_->resolveBindingBuffer(desc.storage);
        if (!storage)
        {
            return std::unexpected(std::move(storage.error()));
        }
        if (auto validStorage = validateStorageRange(storage->desc, desc); !validStorage)
        {
            return std::unexpected(std::move(validStorage.error()));
        }
        const auto size = resolveSize(storage->desc.size, desc.offset, desc.size);

        auto lease = leaseStorageRange(desc.storage, desc.offset, size);
        if (!lease)
        {
            return std::unexpected(std::move(lease.error()));
        }
        auto record = makeRecord(desc, size, std::move(storage->keepAlive), std::move(*lease));
        auto native = createNativeAccelerationStructure(desc, storage->buffer, size);
        if (!native)
        {
            return std::unexpected(std::move(native.error()));
        }
        adoptNativeAccelerationStructure(*record, *native);
        return registerRecord(std::move(record));
    }

    Status VulkanAccelerationManager::destroyAccelerationStructure(AccelerationStructureHandle accelerationStructure)
    {
        const std::scoped_lock lock(mutex_);
        if (!structures_.erase(accelerationStructure))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The acceleration-structure handle is invalid or stale.",
            });
        }
        return {};
    }

    Result<VulkanAccelerationManager::ResolvedAccelerationStructure> VulkanAccelerationManager::resolve(
        AccelerationStructureHandle accelerationStructure) const
    {
        const std::scoped_lock lock(mutex_);
        const auto* found = structures_.find(accelerationStructure);
        if (found == nullptr || *found == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The acceleration-structure handle is invalid or stale.",
            });
        }
        std::shared_ptr<VulkanAccelerationStructureVersion> version;
        {
            const std::scoped_lock versionLock((*found)->version->mutex);
            version = (*found)->version->current;
        }
        return ResolvedAccelerationStructure{
            .accelerationStructure = (*found)->accelerationStructure,
            .deviceAddress = (*found)->deviceAddress,
            .type = (*found)->type,
            .size = (*found)->size,
            .keepAlive = (*found)->keepAlive,
            .version = std::move(version),
            .use =
                VulkanAccelerationStructureUse{
                    .version = (*found)->version,
                    .executionTracker = (*found)->executionTracker,
                },
        };
    }

    Status VulkanAccelerationManager::publishBuild(VulkanCommandBuffer& commandBuffer,
                                                   AccelerationStructureHandle destination,
                                                   PublishedBuild build)
    {
        const bool inPlace = build.write == detail::AccelerationStructureWrite::updateInPlace;
        std::shared_ptr<detail::AccelerationStructureExecutionLease> executionLease;
        try
        {
            executionLease = std::make_shared<detail::AccelerationStructureExecutionLease>();
            detail::reserveAdditional(commandBuffer.retained(), build.owners.size() + 4U);
            detail::reserveAdditional(commandBuffer.accelerationStructurePublications(), 1U);
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to allocate acceleration-structure build retention.",
            });
        }
        const auto recordingId = commandBuffer.accelerationStructureRecordingId();
        const std::scoped_lock lock(mutex_);
        auto* record = structures_.find(destination);
        if (record == nullptr || *record == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The acceleration-structure handle is invalid or stale.",
            });
        }
        std::shared_ptr<AccelerationStructureRecord> sourceRecord;
        if (build.sourceVersion != nullptr && build.source != destination)
        {
            auto* found = structures_.find(build.source);
            if (found == nullptr || *found == nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The TLAS update source handle is invalid or stale.",
                });
            }
            sourceRecord = *found;
        }
        std::unique_lock versionLock((*record)->version->mutex, std::defer_lock);
        std::unique_lock<std::mutex> sourceVersionLock;
        if (sourceRecord != nullptr)
        {
            sourceVersionLock = std::unique_lock(sourceRecord->version->mutex, std::defer_lock);
            std::lock(versionLock, sourceVersionLock);
        }
        else
        {
            versionLock.lock();
        }
        auto& slot = *(*record)->version;
        if (build.sourceVersion != nullptr &&
            (sourceRecord != nullptr ? sourceRecord->version->current : slot.current) != build.sourceVersion)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "The TLAS update source was rebuilt or rolled back while the update was being recorded.",
            });
        }
        if (!inPlace && slot.current != nullptr)
        {
            return std::unexpected(Error{ErrorCode::invalidState, 0, std::string(immutableDestinationMessage)});
        }
        if (inPlace && slot.current == nullptr)
        {
            return std::unexpected(
                Error{ErrorCode::invalidState, 0, "An in-place TLAS update requires a built target."});
        }
        if (inPlace && !(*record)->executionTracker->canUpdateInPlace(recordingId))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "An in-place TLAS update cannot overwrite an acceleration structure with active GPU execution.",
            });
        }
        auto lease = detail::AccelerationStructureExecutionLease::tryAcquireProducer(
            (*record)->executionTracker, recordingId, build.consumers, build.write);
        if (!lease)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                lease.error() == detail::AccelerationStructureLeaseConflict::consumer
                    ? std::string(inPlaceUpdateConsumerConflict)
                    : std::string("An acceleration-structure build cannot overwrite an acceleration structure with "
                                  "active GPU execution."),
            });
        }
        *executionLease = std::move(*lease);
        auto previous = std::exchange(slot.current, build.version);
        auto& retained = commandBuffer.retained();
        retained.push_back((*record)->keepAlive);
        retained.push_back(build.version);
        retained.push_back(std::move(executionLease));
        if (previous != nullptr)
        {
            retained.push_back(previous);
        }
        std::ranges::move(build.owners, std::back_inserter(retained));
        commandBuffer.accelerationStructurePublications().push_back(VulkanAccelerationStructurePublication{
            .slot = (*record)->version,
            .published = build.version,
            .previous = std::move(previous),
        });
        return {};
    }

    Status VulkanAccelerationManager::buildBottomLevel(VulkanCommandBuffer& commandBuffer,
                                                       const BottomLevelAccelerationStructureBuildDesc& desc)
    {
        if (!supported())
        {
            return std::unexpected(Error{
                ErrorCode::unsupported, 0, "Acceleration-structure builds require acceleration-structure support."});
        }
        if (!canRecordAccelerationStructureBuild(commandBuffer))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A BLAS build requires a graphics or compute command buffer outside a rendering scope.",
            });
        }
        auto sizes = queryBottomLevelBuildSizes(desc.input);
        if (!sizes)
        {
            return std::unexpected(std::move(sizes.error()));
        }
        auto destination = resolve(desc.destination);
        auto scratch = resources_->resolveBindingBuffer(desc.scratch);
        if (!destination)
        {
            return std::unexpected(std::move(destination.error()));
        }
        if (!scratch)
        {
            return std::unexpected(std::move(scratch.error()));
        }
        if (destination->type != AccelerationStructureType::bottomLevel ||
            destination->size < sizes->accelerationStructureSize ||
            !validBuildScratchRange(*scratch, desc.scratchOffset, scratchAlignment_, sizes->buildScratchSize))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A BLAS build requires a sufficiently large BLAS destination and aligned dedicated scratch range.",
            });
        }
        std::vector<std::shared_ptr<void>> buildOwners;
        auto nativeInput = nativeBottomLevelBuildInputRetainingBuffers(*resources_, desc.input.geometries, buildOwners);
        if (!nativeInput)
        {
            return std::unexpected(std::move(nativeInput.error()));
        }
        buildOwners.push_back(scratch->keepAlive);
        std::shared_ptr<VulkanAccelerationStructureVersion> version;
        try
        {
            version = std::make_shared<VulkanAccelerationStructureVersion>();
        }
        catch (const std::bad_alloc&)
        {
            return std::unexpected(Error{
                ErrorCode::outOfMemory,
                0,
                "Failed to allocate acceleration-structure build retention.",
            });
        }
        version->buildFlags = desc.input.flags;
        if (auto begun = publishBuild(commandBuffer,
                                      desc.destination,
                                      PublishedBuild{.version = std::move(version), .owners = std::move(buildOwners)});
            !begun)
        {
            return begun;
        }
        const auto info = nativeBuildGeometryInfo(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                                                  desc.input.flags,
                                                  nativeInput->geometries,
                                                  destination->accelerationStructure,
                                                  scratch->deviceAddress + desc.scratchOffset);
        const VkAccelerationStructureBuildRangeInfoKHR* rangePointer = nativeInput->ranges.data();
        table_->vkCmdBuildAccelerationStructuresKHR(commandBuffer.native(), 1, &info, &rangePointer);
        return recordBuildBarrier(commandBuffer.native(), commandBuffer.queueType());
    }

    Result<VulkanAccelerationManager::TopLevelBuildTarget> VulkanAccelerationManager::resolveTopLevelBuildTarget(
        const VulkanCommandBuffer& commandBuffer,
        AccelerationStructureHandle destination,
        std::size_t instanceCount,
        AccelerationStructureBuildFlag flags,
        BufferHandle scratch,
        std::uint64_t scratchOffset) const
    {
        if (!supported())
        {
            return std::unexpected(Error{
                ErrorCode::unsupported, 0, "Acceleration-structure builds require acceleration-structure support."});
        }
        if (!canRecordAccelerationStructureBuild(commandBuffer))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A TLAS build requires a graphics or compute command buffer outside a rendering scope.",
            });
        }
        if (instanceCount > std::numeric_limits<std::uint32_t>::max())
        {
            return std::unexpected(Error{ErrorCode::invalidArgument, 0, "The TLAS instance count is not portable."});
        }
        const TopLevelAccelerationStructureBuildInput input{
            .instanceCount = static_cast<std::uint32_t>(instanceCount),
            .flags = flags,
        };
        auto sizes = queryTopLevelBuildSizes(input);
        if (!sizes)
        {
            return std::unexpected(std::move(sizes.error()));
        }
        auto resolvedDestination = resolve(destination);
        auto resolvedScratch = resources_->resolveBindingBuffer(scratch);
        if (!resolvedDestination)
        {
            return std::unexpected(std::move(resolvedDestination.error()));
        }
        if (!resolvedScratch)
        {
            return std::unexpected(std::move(resolvedScratch.error()));
        }
        if (resolvedDestination->type != AccelerationStructureType::topLevel ||
            resolvedDestination->size < sizes->accelerationStructureSize ||
            !validBuildScratchRange(*resolvedScratch, scratchOffset, scratchAlignment_, sizes->buildScratchSize))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A TLAS build requires a sufficiently large TLAS destination and aligned dedicated scratch range.",
            });
        }
        if (resolvedDestination->version != nullptr)
        {
            return std::unexpected(Error{ErrorCode::invalidState, 0, std::string(immutableDestinationMessage)});
        }
        return TopLevelBuildTarget{
            .handle = destination,
            .destination = std::move(*resolvedDestination),
            .scratch = std::move(*resolvedScratch),
            .scratchOffset = scratchOffset,
            .flags = flags,
        };
    }

    namespace
    {

        [[nodiscard]] Result<std::shared_ptr<VulkanAccelerationStructureVersion>> makeTopLevelVersion(
            AccelerationStructureBuildFlag flags, VulkanTopLevelInstances& instances)
        {
            try
            {
                auto version = std::make_shared<VulkanAccelerationStructureVersion>();
                version->buildFlags = flags;
                version->instanceCount = instances.count;
                version->dependencies = std::move(instances.dependencies);
                version->dependencyTrackers = instances.dependencyTrackers;
                return version;
            }
            catch (const std::bad_alloc&)
            {
                return std::unexpected(Error{
                    ErrorCode::outOfMemory,
                    0,
                    "Failed to allocate acceleration-structure build retention.",
                });
            }
        }

    } // namespace

    Status VulkanAccelerationManager::recordTopLevelBuild(VulkanCommandBuffer& commandBuffer,
                                                          const TopLevelBuildTarget& target,
                                                          VulkanTopLevelInstances instances)
    {
        auto version = makeTopLevelVersion(target.flags, instances);
        if (!version)
        {
            return std::unexpected(std::move(version.error()));
        }
        instances.owners.push_back(target.scratch.keepAlive);
        if (auto begun = publishBuild(commandBuffer,
                                      target.handle,
                                      PublishedBuild{
                                          .version = std::move(*version),
                                          .owners = std::move(instances.owners),
                                          .consumers = std::move(instances.dependencyTrackers),
                                      });
            !begun)
        {
            return begun;
        }

        const auto geometry = nativeInstancesGeometry(instances.deviceAddress);
        const auto info = nativeBuildGeometryInfo(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                                                  target.flags,
                                                  std::span(&geometry, 1),
                                                  target.destination.accelerationStructure,
                                                  target.scratch.deviceAddress + target.scratchOffset);
        const auto range = nativeBuildRange(instances.count);
        const VkAccelerationStructureBuildRangeInfoKHR* rangePointer = &range;
        table_->vkCmdBuildAccelerationStructuresKHR(commandBuffer.native(), 1, &info, &rangePointer);
        return recordBuildBarrier(commandBuffer.native(), commandBuffer.queueType());
    }

    Status VulkanAccelerationManager::buildTopLevel(VulkanCommandBuffer& commandBuffer,
                                                    const TopLevelAccelerationStructureBuildDesc& desc)
    {
        auto target = resolveTopLevelBuildTarget(
            commandBuffer, desc.destination, desc.instances.size(), desc.flags, desc.scratch, desc.scratchOffset);
        if (!target)
        {
            return std::unexpected(std::move(target.error()));
        }
        TemporaryBuffer instanceBuffer(*resources_);
        auto instances = prepareSpanInstances(*this, *resources_, desc.instances, instanceBuffer);
        if (!instances)
        {
            return std::unexpected(std::move(instances.error()));
        }
        return recordTopLevelBuild(commandBuffer, *target, std::move(*instances));
    }

    Status VulkanAccelerationManager::buildTopLevelFromBuffer(VulkanCommandBuffer& commandBuffer,
                                                              const TopLevelAccelerationStructureBufferBuildDesc& desc)
    {
        auto target = resolveTopLevelBuildTarget(
            commandBuffer, desc.destination, desc.instanceCount, desc.flags, desc.scratch, desc.scratchOffset);
        if (!target)
        {
            return std::unexpected(std::move(target.error()));
        }
        auto instances =
            prepareBufferInstances(*resources_, desc.instanceBuffer, desc.instanceOffset, desc.instanceCount);
        if (!instances)
        {
            return std::unexpected(std::move(instances.error()));
        }
        return recordTopLevelBuild(commandBuffer, *target, std::move(*instances));
    }

    Result<VulkanAccelerationManager::TopLevelUpdateTarget> VulkanAccelerationManager::resolveTopLevelUpdateTarget(
        const VulkanCommandBuffer& commandBuffer, const TopLevelUpdateRequest& request) const
    {
        if (!supported())
        {
            return std::unexpected(Error{
                ErrorCode::unsupported, 0, "Acceleration-structure updates require acceleration-structure support."});
        }
        if (!canRecordAccelerationStructureBuild(commandBuffer))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A TLAS update requires a graphics or compute command buffer outside a rendering scope.",
            });
        }
        if (request.inPlace != (request.source == request.destination) ||
            request.instanceCount > std::numeric_limits<std::uint32_t>::max())
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                request.inPlace ? "An in-place TLAS update requires one target and a portable instance count."
                                : "A TLAS update requires distinct handles and a portable instance count.",
            });
        }
        auto sizes = queryTopLevelBuildSizes({
            .instanceCount = static_cast<std::uint32_t>(request.instanceCount),
            .flags = request.flags,
        });
        if (!sizes)
        {
            return std::unexpected(std::move(sizes.error()));
        }
        auto source = resolve(request.source);
        if (!source)
        {
            return std::unexpected(std::move(source.error()));
        }
        auto destination = request.inPlace ? source : resolve(request.destination);
        if (!destination)
        {
            return std::unexpected(std::move(destination.error()));
        }
        auto scratch = resources_->resolveBindingBuffer(request.scratch);
        if (!scratch)
        {
            return std::unexpected(std::move(scratch.error()));
        }
        if (source->type != AccelerationStructureType::topLevel ||
            destination->type != AccelerationStructureType::topLevel ||
            !validBuildScratchRange(*scratch, request.scratchOffset, scratchAlignment_, sizes->updateScratchSize))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A TLAS update requires compatible TLAS handles, destination storage, and aligned update scratch.",
            });
        }
        if (source->version == nullptr || (!request.inPlace && destination->version != nullptr))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                request.inPlace ? "An in-place TLAS update requires a built target."
                                : "A TLAS update requires a built source and an unbuilt immutable destination.",
            });
        }
        if (destination->size < sizes->accelerationStructureSize)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A TLAS update destination must fit the update input's prebuild size.",
            });
        }
        if (!hasAny(source->version->buildFlags, AccelerationStructureBuildFlag::allowUpdate) ||
            request.flags != source->version->buildFlags || request.instanceCount != source->version->instanceCount)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A TLAS update must preserve the update-enabled source flags and instance count.",
            });
        }
        return TopLevelUpdateTarget{
            .request = request,
            .source = std::move(*source),
            .destination = std::move(*destination),
            .scratch = std::move(*scratch),
        };
    }

    Status VulkanAccelerationManager::recordTopLevelUpdate(VulkanCommandBuffer& commandBuffer,
                                                           const TopLevelUpdateTarget& target,
                                                           VulkanTopLevelInstances instances)
    {
        auto version = makeTopLevelVersion(target.request.flags, instances);
        if (!version)
        {
            return std::unexpected(std::move(version.error()));
        }
        auto consumers = std::move(instances.dependencyTrackers);
        instances.owners.push_back(target.scratch.keepAlive);
#ifdef RACHIS_TESTING
        if (auto hook = std::exchange(test::beforeTopLevelUpdatePublication(), {}); hook)
        {
            hook();
        }
#endif
        if (!target.request.inPlace)
        {
            consumers.push_back(target.source.use.executionTracker);
            instances.owners.push_back(target.source.keepAlive);
            instances.owners.push_back(target.source.version);
        }
        if (auto begun =
                publishBuild(commandBuffer,
                             target.request.destination,
                             PublishedBuild{
                                 .version = std::move(*version),
                                 .owners = std::move(instances.owners),
                                 .consumers = std::move(consumers),
                                 .write = target.request.inPlace ? detail::AccelerationStructureWrite::updateInPlace
                                                                 : detail::AccelerationStructureWrite::publishVersion,
                                 .source = target.request.source,
                                 .sourceVersion = target.source.version,
                             });
            !begun)
        {
            return begun;
        }

        const auto geometry = nativeInstancesGeometry(instances.deviceAddress);
        auto info = nativeBuildGeometryInfo(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                                            target.request.flags,
                                            std::span(&geometry, 1),
                                            target.destination.accelerationStructure,
                                            target.scratch.deviceAddress + target.request.scratchOffset);
        info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        if (target.request.inPlace)
        {
            recordInPlaceUpdateBarrier(commandBuffer.native(), commandBuffer.queueType());
        }
        info.srcAccelerationStructure = target.source.accelerationStructure;
        const auto range = nativeBuildRange(instances.count);
        const VkAccelerationStructureBuildRangeInfoKHR* rangePointer = &range;
        table_->vkCmdBuildAccelerationStructuresKHR(commandBuffer.native(), 1, &info, &rangePointer);
        return recordBuildBarrier(commandBuffer.native(), commandBuffer.queueType());
    }

    Status VulkanAccelerationManager::updateTopLevelFromSpan(VulkanCommandBuffer& commandBuffer,
                                                             const TopLevelUpdateRequest& request,
                                                             std::span<const AccelerationStructureInstance> instances)
    {
        auto target = resolveTopLevelUpdateTarget(commandBuffer, request);
        if (!target)
        {
            return std::unexpected(std::move(target.error()));
        }
        TemporaryBuffer instanceBuffer(*resources_);
        auto prepared = prepareSpanInstances(*this, *resources_, instances, instanceBuffer);
        if (!prepared)
        {
            return std::unexpected(std::move(prepared.error()));
        }
        return recordTopLevelUpdate(commandBuffer, *target, std::move(*prepared));
    }

    Status VulkanAccelerationManager::updateTopLevelFromInstanceBuffer(VulkanCommandBuffer& commandBuffer,
                                                                       const TopLevelUpdateRequest& request,
                                                                       BufferHandle instanceBuffer,
                                                                       std::uint64_t instanceOffset)
    {
        auto target = resolveTopLevelUpdateTarget(commandBuffer, request);
        if (!target)
        {
            return std::unexpected(std::move(target.error()));
        }
        auto prepared = prepareBufferInstances(
            *resources_, instanceBuffer, instanceOffset, static_cast<std::uint32_t>(request.instanceCount));
        if (!prepared)
        {
            return std::unexpected(std::move(prepared.error()));
        }
        return recordTopLevelUpdate(commandBuffer, *target, std::move(*prepared));
    }

    Status VulkanAccelerationManager::updateTopLevel(VulkanCommandBuffer& commandBuffer,
                                                     const TopLevelAccelerationStructureUpdateDesc& desc)
    {
        return updateTopLevelFromSpan(commandBuffer,
                                      {
                                          .source = desc.source,
                                          .destination = desc.destination,
                                          .instanceCount = desc.instances.size(),
                                          .flags = desc.flags,
                                          .scratch = desc.scratch,
                                          .scratchOffset = desc.scratchOffset,
                                          .inPlace = false,
                                      },
                                      desc.instances);
    }

    Status VulkanAccelerationManager::updateTopLevelInPlace(VulkanCommandBuffer& commandBuffer,
                                                            const TopLevelAccelerationStructureInPlaceUpdateDesc& desc)
    {
        return updateTopLevelFromSpan(commandBuffer,
                                      {
                                          .source = desc.target,
                                          .destination = desc.target,
                                          .instanceCount = desc.instances.size(),
                                          .flags = desc.flags,
                                          .scratch = desc.scratch,
                                          .scratchOffset = desc.scratchOffset,
                                          .inPlace = true,
                                      },
                                      desc.instances);
    }

    Status VulkanAccelerationManager::updateTopLevelFromBuffer(
        VulkanCommandBuffer& commandBuffer, const TopLevelAccelerationStructureBufferUpdateDesc& desc)
    {
        return updateTopLevelFromInstanceBuffer(commandBuffer,
                                                {
                                                    .source = desc.source,
                                                    .destination = desc.destination,
                                                    .instanceCount = desc.instanceCount,
                                                    .flags = desc.flags,
                                                    .scratch = desc.scratch,
                                                    .scratchOffset = desc.scratchOffset,
                                                    .inPlace = false,
                                                },
                                                desc.instanceBuffer,
                                                desc.instanceOffset);
    }

    Status VulkanAccelerationManager::updateTopLevelInPlaceFromBuffer(
        VulkanCommandBuffer& commandBuffer, const TopLevelAccelerationStructureBufferInPlaceUpdateDesc& desc)
    {
        return updateTopLevelFromInstanceBuffer(commandBuffer,
                                                {
                                                    .source = desc.target,
                                                    .destination = desc.target,
                                                    .instanceCount = desc.instanceCount,
                                                    .flags = desc.flags,
                                                    .scratch = desc.scratch,
                                                    .scratchOffset = desc.scratchOffset,
                                                    .inPlace = true,
                                                },
                                                desc.instanceBuffer,
                                                desc.instanceOffset);
    }

    Result<AccelerationStructureInstanceLayout> VulkanAccelerationManager::instanceLayout() const
    {
        if (!supported())
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Acceleration-structure instance records require acceleration-structure support.",
            });
        }
        return AccelerationStructureInstanceLayout{
            .format = AccelerationStructureInstanceFormat::packed64,
            .size = sizeof(VkAccelerationStructureInstanceKHR),
            .alignment = instanceRecordAlignment,
        };
    }

    Result<std::uint64_t> VulkanAccelerationManager::reference(AccelerationStructureHandle bottomLevel) const
    {
        if (!supported())
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Acceleration-structure references require acceleration-structure support.",
            });
        }
        auto resolved = resolve(bottomLevel);
        if (!resolved)
        {
            return std::unexpected(std::move(resolved.error()));
        }
        if (resolved->type != AccelerationStructureType::bottomLevel)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Only a bottom-level acceleration structure has a reference that instance records can name.",
            });
        }
        if (resolved->deviceAddress == 0)
        {
            return std::unexpected(Error{
                ErrorCode::nativeFailure,
                0,
                "Vulkan returned no device address for the bottom-level acceleration structure.",
            });
        }
        return resolved->deviceAddress;
    }

    Status VulkanAccelerationManager::updateBottomLevel(VulkanCommandBuffer&,
                                                        const BottomLevelAccelerationStructureUpdateDesc&)
    {
        return std::unexpected(
            Error{ErrorCode::unsupported, 0, "Vulkan out-of-place BLAS updates are not implemented."});
    }

    Status VulkanAccelerationManager::updateBottomLevelInPlace(VulkanCommandBuffer&,
                                                               const BottomLevelAccelerationStructureInPlaceUpdateDesc&)
    {
        return std::unexpected(Error{ErrorCode::unsupported, 0, "Vulkan in-place BLAS updates are not implemented."});
    }

    Status VulkanAccelerationManager::compact(VulkanCommandBuffer&, const AccelerationStructureCompactionDesc&)
    {
        return std::unexpected(
            Error{ErrorCode::unsupported, 0, "Vulkan acceleration-structure compaction is not implemented."});
    }

} // namespace rachis::vulkan
