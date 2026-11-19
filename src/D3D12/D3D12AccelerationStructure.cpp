#include "D3D12/D3D12ResourceInternal.h"

#include "Rachis/Rendering.h"

namespace rachis::d3d12
{

    using namespace resource_detail;

    namespace
    {

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

        [[nodiscard]] constexpr D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS nativeBuildFlags(
            AccelerationStructureBuildFlag flags) noexcept
        {
            auto result = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE;
            if (hasAny(flags, AccelerationStructureBuildFlag::preferFastTrace))
            {
                result |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
            }
            if (hasAny(flags, AccelerationStructureBuildFlag::preferFastBuild))
            {
                result |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
            }
            if (hasAny(flags, AccelerationStructureBuildFlag::minimizeMemory))
            {
                result |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_MINIMIZE_MEMORY;
            }
            if (hasAny(flags, AccelerationStructureBuildFlag::allowUpdate))
            {
                result |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
            }
            if (hasAny(flags, AccelerationStructureBuildFlag::allowCompaction))
            {
                result |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
            }
            return result;
        }

        struct NativeAccelerationStructureVertexFormat final
        {
            DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
            std::uint32_t byteSize = 0;
            std::uint32_t alignment = 0;
        };

        [[nodiscard]] constexpr std::optional<NativeAccelerationStructureVertexFormat>
        nativeAccelerationStructureVertexFormat(AccelerationStructureVertexFormat format) noexcept
        {
            switch (format)
            {
            case AccelerationStructureVertexFormat::float32x2:
                return NativeAccelerationStructureVertexFormat{DXGI_FORMAT_R32G32_FLOAT, 8, 4};
            case AccelerationStructureVertexFormat::float32x3:
                return NativeAccelerationStructureVertexFormat{DXGI_FORMAT_R32G32B32_FLOAT, 12, 4};
            case AccelerationStructureVertexFormat::float16x2:
                return NativeAccelerationStructureVertexFormat{DXGI_FORMAT_R16G16_FLOAT, 4, 2};
            case AccelerationStructureVertexFormat::float16x4:
                return NativeAccelerationStructureVertexFormat{DXGI_FORMAT_R16G16B16A16_FLOAT, 8, 2};
            case AccelerationStructureVertexFormat::snorm16x2:
                return NativeAccelerationStructureVertexFormat{DXGI_FORMAT_R16G16_SNORM, 4, 2};
            case AccelerationStructureVertexFormat::snorm16x4:
                return NativeAccelerationStructureVertexFormat{DXGI_FORMAT_R16G16B16A16_SNORM, 8, 2};
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

        [[nodiscard]] constexpr D3D12_RAYTRACING_GEOMETRY_FLAGS nativeGeometryFlags(
            AccelerationStructureGeometryFlag flags) noexcept
        {
            auto result = D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
            if (hasAny(flags, AccelerationStructureGeometryFlag::opaque))
            {
                result |= D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
            }
            if (hasAny(flags, AccelerationStructureGeometryFlag::noDuplicateAnyHitInvocation))
            {
                result |= D3D12_RAYTRACING_GEOMETRY_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION;
            }
            return result;
        }

        [[nodiscard]] constexpr bool validInstanceFlags(AccelerationStructureInstanceFlag flags) noexcept
        {
            constexpr auto allowed =
                static_cast<std::uint8_t>(AccelerationStructureInstanceFlag::triangleCullDisable) |
                static_cast<std::uint8_t>(AccelerationStructureInstanceFlag::triangleFrontCounterClockwise) |
                static_cast<std::uint8_t>(AccelerationStructureInstanceFlag::forceOpaque) |
                static_cast<std::uint8_t>(AccelerationStructureInstanceFlag::forceNonOpaque);
            return (static_cast<std::uint8_t>(flags) & ~allowed) == 0 &&
                   !(hasAny(flags, AccelerationStructureInstanceFlag::forceOpaque) &&
                     hasAny(flags, AccelerationStructureInstanceFlag::forceNonOpaque));
        }

        [[nodiscard]] constexpr D3D12_RAYTRACING_INSTANCE_FLAGS nativeInstanceFlags(
            AccelerationStructureInstanceFlag flags) noexcept
        {
            auto result = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
            if (hasAny(flags, AccelerationStructureInstanceFlag::triangleCullDisable))
            {
                result |= D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE;
            }
            if (hasAny(flags, AccelerationStructureInstanceFlag::triangleFrontCounterClockwise))
            {
                result |= D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_FRONT_COUNTERCLOCKWISE;
            }
            if (hasAny(flags, AccelerationStructureInstanceFlag::forceOpaque))
            {
                result |= D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE;
            }
            if (hasAny(flags, AccelerationStructureInstanceFlag::forceNonOpaque))
            {
                result |= D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_NON_OPAQUE;
            }
            return result;
        }

        struct RayTracingIndirectApplicationRetention final
        {
            std::vector<std::shared_ptr<void>> owners;
            std::vector<std::shared_ptr<ResidencyRecord>> residencies;
        };

        struct AccelerationStructureBuildRetention final
        {
            std::vector<std::shared_ptr<void>> owners;
            std::vector<std::shared_ptr<ResidencyRecord>> residencies;
        };

        [[nodiscard]] bool updateCompatibleGeometry(const D3D12_RAYTRACING_GEOMETRY_DESC& original,
                                                    const D3D12_RAYTRACING_GEOMETRY_DESC& updated) noexcept
        {
            if (original.Type != updated.Type || original.Flags != updated.Flags)
            {
                return false;
            }
            if (original.Type == D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES)
            {
                return original.Triangles.VertexBuffer.StrideInBytes == updated.Triangles.VertexBuffer.StrideInBytes &&
                       original.Triangles.VertexCount == updated.Triangles.VertexCount &&
                       original.Triangles.VertexFormat == updated.Triangles.VertexFormat &&
                       original.Triangles.IndexCount == updated.Triangles.IndexCount &&
                       original.Triangles.IndexFormat == updated.Triangles.IndexFormat &&
                       (original.Triangles.IndexBuffer != 0) == (updated.Triangles.IndexBuffer != 0) &&
                       (original.Triangles.Transform3x4 != 0) == (updated.Triangles.Transform3x4 != 0);
            }
            if (original.Type == D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS)
            {
                return original.AABBs.AABBCount == updated.AABBs.AABBCount &&
                       original.AABBs.AABBs.StrideInBytes == updated.AABBs.AABBs.StrideInBytes;
            }
            return false;
        }

        [[nodiscard]] bool updateCompatibleGeometryLayout(
            std::span<const D3D12_RAYTRACING_GEOMETRY_DESC> original,
            std::span<const D3D12_RAYTRACING_GEOMETRY_DESC> updated) noexcept
        {
            return original.size() == updated.size() && std::ranges::equal(original, updated, updateCompatibleGeometry);
        }

        [[nodiscard]] constexpr bool validStridedBufferRange(std::uint64_t bufferSize,
                                                             std::uint64_t offset,
                                                             std::uint32_t count,
                                                             std::uint32_t stride,
                                                             std::uint32_t elementSize) noexcept
        {
            return count != 0 && stride >= elementSize && offset <= bufferSize && elementSize <= bufferSize - offset &&
                   static_cast<std::uint64_t>(count - 1U) <= (bufferSize - offset - elementSize) / stride;
        }

        struct AccelerationStructureCommandLists final
        {
            ID3D12GraphicsCommandList4* rayTracing = nullptr;
            ID3D12GraphicsCommandList7* enhanced = nullptr;
        };

        [[nodiscard]] Result<AccelerationStructureCommandLists> prepareAccelerationStructureCommandLists(
            QueueManager& queues, detail::BackendCommandBuffer& commandBuffer, bool enhancedBarriers)
        {
            auto rayTracing = queues.commandList4(commandBuffer);
            if (!rayTracing)
            {
                return std::unexpected(std::move(rayTracing.error()));
            }
            ID3D12GraphicsCommandList7* enhanced = nullptr;
            if (enhancedBarriers)
            {
                auto prepared = queues.commandList7(commandBuffer);
                if (!prepared)
                {
                    return std::unexpected(std::move(prepared.error()));
                }
                enhanced = *prepared;
            }
            return AccelerationStructureCommandLists{.rayTracing = *rayTracing, .enhanced = enhanced};
        }

        void recordAccelerationStructureGlobalBarrier(const AccelerationStructureCommandLists& commandLists,
                                                      const D3D12_GLOBAL_BARRIER& barrier) noexcept
        {
            if (commandLists.enhanced != nullptr)
            {
                const D3D12_BARRIER_GROUP group{
                    .Type = D3D12_BARRIER_TYPE_GLOBAL,
                    .NumBarriers = 1,
                    .pGlobalBarriers = &barrier,
                };
                commandLists.enhanced->Barrier(1, &group);
                test::lastAccelerationStructureGlobalBarrier() = test::GlobalBarrierSnapshot{
                    .syncBefore = static_cast<std::uint32_t>(barrier.SyncBefore),
                    .syncAfter = static_cast<std::uint32_t>(barrier.SyncAfter),
                    .accessBefore = static_cast<std::uint32_t>(barrier.AccessBefore),
                    .accessAfter = static_cast<std::uint32_t>(barrier.AccessAfter),
                };
            }
            else
            {
                const D3D12_RESOURCE_BARRIER legacyBarrier{
                    .Type = D3D12_RESOURCE_BARRIER_TYPE_UAV,
                    .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    .UAV = {.pResource = nullptr},
                };
                commandLists.rayTracing->ResourceBarrier(1, &legacyBarrier);
            }
        }

        void recordAccelerationStructureBuildBarrier(const AccelerationStructureCommandLists& commandLists) noexcept
        {
            recordAccelerationStructureGlobalBarrier(
                commandLists,
                D3D12_GLOBAL_BARRIER{
                    .SyncBefore = D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,
                    .SyncAfter = D3D12_BARRIER_SYNC_ALL_SHADING |
                                 D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE |
                                 D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE |
                                 D3D12_BARRIER_SYNC_EMIT_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO,
                    .AccessBefore = D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE |
                                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    .AccessAfter = D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ |
                                   D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                });
        }

        void recordAccelerationStructureInPlaceUpdateBarrier(
            const AccelerationStructureCommandLists& commandLists) noexcept
        {
            const D3D12_GLOBAL_BARRIER barrier{
                .SyncBefore = D3D12_BARRIER_SYNC_ALL_SHADING |
                              D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE |
                              D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE |
                              D3D12_BARRIER_SYNC_EMIT_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO,
                .SyncAfter = D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,
                .AccessBefore = D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ,
                .AccessAfter = D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE,
            };
            recordAccelerationStructureGlobalBarrier(commandLists, barrier);
            test::lastAccelerationStructureInPlaceUpdateBarrier() = test::GlobalBarrierSnapshot{
                .syncBefore = static_cast<std::uint32_t>(barrier.SyncBefore),
                .syncAfter = static_cast<std::uint32_t>(barrier.SyncAfter),
                .accessBefore = static_cast<std::uint32_t>(barrier.AccessBefore),
                .accessAfter = static_cast<std::uint32_t>(barrier.AccessAfter),
                .legacyUnorderedAccess = commandLists.enhanced == nullptr,
            };
        }

        void recordAccelerationStructureCopyBarrier(const AccelerationStructureCommandLists& commandLists) noexcept
        {
            recordAccelerationStructureGlobalBarrier(
                commandLists,
                D3D12_GLOBAL_BARRIER{
                    .SyncBefore = D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE,
                    .SyncAfter = D3D12_BARRIER_SYNC_ALL_SHADING |
                                 D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE |
                                 D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE | D3D12_BARRIER_SYNC_COPY |
                                 D3D12_BARRIER_SYNC_EMIT_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO,
                    .AccessBefore = D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE |
                                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    .AccessAfter = D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ |
                                   D3D12_BARRIER_ACCESS_UNORDERED_ACCESS | D3D12_BARRIER_ACCESS_SHADER_RESOURCE |
                                   D3D12_BARRIER_ACCESS_COPY_SOURCE,
                });
        }

        using BufferLookupResult = Result<std::shared_ptr<BufferRecord>>;
        using AccelerationStructureLookupResult = Result<std::shared_ptr<AccelerationStructureRecord>>;
        using BufferLookup = std::function<BufferLookupResult(BufferHandle)>;
        using AccelerationStructureLookup =
            std::function<AccelerationStructureLookupResult(AccelerationStructureHandle)>;

        void retainBuildInputBuffer(PreparedBottomLevelInput& prepared, const std::shared_ptr<BufferRecord>& buffer)
        {
            prepared.owners.push_back(buffer);
            if (buffer->residency != nullptr)
            {
                prepared.residencies.push_back(buffer->residency);
            }
        }

        [[nodiscard]] Status describeTriangleVertices(const AccelerationStructureTriangleGeometry& triangles,
                                                      const BufferLookup& findBuffer,
                                                      D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC& native,
                                                      PreparedBottomLevelInput& prepared)
        {
            const auto vertexFormat = nativeAccelerationStructureVertexFormat(triangles.vertexFormat);
            auto vertices = findBuffer(triangles.vertexBuffer);
            if (!vertexFormat || !vertices ||
                !hasAny((*vertices)->desc.usage, BufferUsage::accelerationStructureBuildInput) ||
                triangles.vertexOffset % vertexFormat->alignment != 0 ||
                triangles.vertexStride % vertexFormat->alignment != 0 ||
                !validStridedBufferRange((*vertices)->desc.size,
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
            native.VertexBuffer = {
                .StartAddress = (*vertices)->resource->GetGPUVirtualAddress() + triangles.vertexOffset,
                .StrideInBytes = triangles.vertexStride,
            };
            native.VertexCount = triangles.vertexCount;
            native.VertexFormat = vertexFormat->format;
            retainBuildInputBuffer(prepared, *vertices);
            return {};
        }

        [[nodiscard]] Status describeTriangleIndices(const AccelerationStructureTriangleGeometry& triangles,
                                                     const BufferLookup& findBuffer,
                                                     D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC& native,
                                                     PreparedBottomLevelInput& prepared)
        {
            auto indices = findBuffer(triangles.indexBuffer);
            const std::uint32_t indexSize = triangles.indexFormat == IndexFormat::uint16 ? 2U : 4U;
            const bool validIndexFormat =
                triangles.indexFormat == IndexFormat::uint16 || triangles.indexFormat == IndexFormat::uint32;
            if (!indices || !validIndexFormat || triangles.indexCount == 0 || triangles.indexCount % 3U != 0 ||
                !hasAny((*indices)->desc.usage, BufferUsage::accelerationStructureBuildInput) ||
                triangles.indexOffset % indexSize != 0 ||
                !validStridedBufferRange(
                    (*indices)->desc.size, triangles.indexOffset, triangles.indexCount, indexSize, indexSize))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Triangle indices require a live build-input buffer and an aligned in-bounds triangle list.",
                });
            }
            native.IndexBuffer = (*indices)->resource->GetGPUVirtualAddress() + triangles.indexOffset;
            native.IndexCount = triangles.indexCount;
            native.IndexFormat =
                triangles.indexFormat == IndexFormat::uint16 ? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_R32_UINT;
            prepared.primitiveCount += triangles.indexCount / 3U;
            retainBuildInputBuffer(prepared, *indices);
            return {};
        }

        [[nodiscard]] Status describeNonIndexedTriangles(const AccelerationStructureTriangleGeometry& triangles,
                                                         D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC& native,
                                                         PreparedBottomLevelInput& prepared)
        {
            if (triangles.indexCount != 0 || triangles.vertexCount % 3U != 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Non-indexed BLAS triangles require zero indices and a multiple-of-three vertex count.",
                });
            }
            native.IndexFormat = DXGI_FORMAT_UNKNOWN;
            prepared.primitiveCount += triangles.vertexCount / 3U;
            return {};
        }

        [[nodiscard]] Status describeTriangleTransform(const AccelerationStructureTriangleGeometry& triangles,
                                                       const BufferLookup& findBuffer,
                                                       D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC& native,
                                                       PreparedBottomLevelInput& prepared)
        {
            auto transform = findBuffer(triangles.transformBuffer);
            constexpr std::uint64_t transformSize = 12U * sizeof(float);
            if (!transform || !hasAny((*transform)->desc.usage, BufferUsage::accelerationStructureBuildInput) ||
                triangles.transformOffset % D3D12_RAYTRACING_TRANSFORM3X4_BYTE_ALIGNMENT != 0 ||
                !validRange((*transform)->desc.size, triangles.transformOffset, transformSize))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A triangle transform requires one aligned 3x4 float matrix in a build-input buffer.",
                });
            }
            native.Transform3x4 = (*transform)->resource->GetGPUVirtualAddress() + triangles.transformOffset;
            retainBuildInputBuffer(prepared, *transform);
            return {};
        }

        [[nodiscard]] Status describeTriangleGeometry(const AccelerationStructureTriangleGeometry& triangles,
                                                      const BufferLookup& findBuffer,
                                                      D3D12_RAYTRACING_GEOMETRY_DESC& native,
                                                      PreparedBottomLevelInput& prepared)
        {
            native.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
            auto vertices = describeTriangleVertices(triangles, findBuffer, native.Triangles, prepared);
            if (!vertices)
            {
                return vertices;
            }
            auto indices = triangles.indexBuffer.isValid()
                               ? describeTriangleIndices(triangles, findBuffer, native.Triangles, prepared)
                               : describeNonIndexedTriangles(triangles, native.Triangles, prepared);
            if (!indices)
            {
                return indices;
            }
            if (!triangles.transformBuffer.isValid())
            {
                return {};
            }
            return describeTriangleTransform(triangles, findBuffer, native.Triangles, prepared);
        }

        [[nodiscard]] Status describeAabbGeometry(const AccelerationStructureAabbGeometry& aabbs,
                                                  const BufferLookup& findBuffer,
                                                  D3D12_RAYTRACING_GEOMETRY_DESC& native,
                                                  PreparedBottomLevelInput& prepared)
        {
            auto source = findBuffer(aabbs.buffer);
            constexpr std::uint32_t aabbSize = 6U * sizeof(float);
            if (!source || !hasAny((*source)->desc.usage, BufferUsage::accelerationStructureBuildInput) ||
                aabbs.offset % D3D12_RAYTRACING_AABB_BYTE_ALIGNMENT != 0 ||
                aabbs.stride % D3D12_RAYTRACING_AABB_BYTE_ALIGNMENT != 0 ||
                !validStridedBufferRange((*source)->desc.size, aabbs.offset, aabbs.count, aabbs.stride, aabbSize))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "AABB geometry requires aligned, strided bounds in a live build-input buffer.",
                });
            }
            native.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
            native.AABBs = {
                .AABBCount = aabbs.count,
                .AABBs =
                    {
                        .StartAddress = (*source)->resource->GetGPUVirtualAddress() + aabbs.offset,
                        .StrideInBytes = aabbs.stride,
                    },
            };
            prepared.primitiveCount += aabbs.count;
            retainBuildInputBuffer(prepared, *source);
            return {};
        }

        [[nodiscard]] Status appendBottomLevelGeometry(PreparedBottomLevelInput& prepared,
                                                       const AccelerationStructureGeometry& geometry,
                                                       const BufferLookup& findBuffer)
        {
            if (!validGeometryFlags(geometry.flags))
            {
                return std::unexpected(Error{ErrorCode::invalidArgument, 0, "A BLAS geometry contains unknown flags."});
            }
            const auto* triangles = std::get_if<AccelerationStructureTriangleGeometry>(&geometry.data);
            const auto* aabbs = std::get_if<AccelerationStructureAabbGeometry>(&geometry.data);
            if (triangles == nullptr && aabbs == nullptr)
            {
                return std::unexpected(
                    Error{ErrorCode::invalidArgument, 0, "The acceleration-structure geometry kind is invalid."});
            }
            D3D12_RAYTRACING_GEOMETRY_DESC native{};
            native.Flags = nativeGeometryFlags(geometry.flags);
            auto described = triangles != nullptr ? describeTriangleGeometry(*triangles, findBuffer, native, prepared)
                                                  : describeAabbGeometry(*aabbs, findBuffer, native, prepared);
            if (!described)
            {
                return described;
            }
            if (prepared.primitiveCount > D3D12_RAYTRACING_MAX_PRIMITIVES_PER_BOTTOM_LEVEL_ACCELERATION_STRUCTURE)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The BLAS primitive count exceeds the portable D3D12 limit.",
                });
            }
            prepared.geometries.push_back(native);
            return {};
        }

    } // namespace

    [[nodiscard]] Result<PreparedBottomLevelInput> ResourceManager::Impl::prepareBottomLevel(
        const BottomLevelAccelerationStructureBuildInput& input) const
    {
        if (!rayTracing)
        {
            return std::unexpected(
                Error{ErrorCode::unsupported, 0, "Acceleration-structure builds require ray-tracing support."});
        }
        if (input.geometries.empty() ||
            input.geometries.size() > D3D12_RAYTRACING_MAX_GEOMETRIES_PER_BOTTOM_LEVEL_ACCELERATION_STRUCTURE ||
            !validAccelerationStructureBuildFlags(input.flags))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A BLAS build requires geometry, valid build flags, and a supported geometry count.",
            });
        }

        const BufferLookup findBuffer = [this](BufferHandle handle) { return buffer(handle); };
        PreparedBottomLevelInput result;
        result.geometries.reserve(input.geometries.size());
        for (const auto& geometry : input.geometries)
        {
            auto appended = appendBottomLevelGeometry(result, geometry, findBuffer);
            if (!appended)
            {
                return std::unexpected(std::move(appended.error()));
            }
        }
        return result;
    }

    namespace
    {

        [[nodiscard]] D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS nativeBottomLevelInputs(
            std::span<const D3D12_RAYTRACING_GEOMETRY_DESC> geometries,
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS flags) noexcept
        {
            return D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS{
                .Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                .Flags = flags,
                .NumDescs = static_cast<UINT>(geometries.size()),
                .DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY,
                .pGeometryDescs = geometries.data(),
            };
        }

        [[nodiscard]] D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS nativeTopLevelInputs(
            UINT instanceCount,
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS flags,
            D3D12_GPU_VIRTUAL_ADDRESS instanceDescs) noexcept
        {
            return D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS{
                .Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
                .Flags = flags,
                .NumDescs = instanceCount,
                .DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY,
                .InstanceDescs = instanceDescs,
            };
        }

    } // namespace

    Result<AccelerationStructureBuildSizes> ResourceManager::queryBottomLevelAccelerationStructureBuildSizes(
        const BottomLevelAccelerationStructureBuildInput& input) const
    {
        auto prepared = impl_->prepareBottomLevel(input);
        if (!prepared)
        {
            return std::unexpected(std::move(prepared.error()));
        }
        const auto native = nativeBottomLevelInputs(prepared->geometries, nativeBuildFlags(input.flags));
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO information{};
        impl_->rayTracingDevice->GetRaytracingAccelerationStructurePrebuildInfo(&native, &information);
        if (information.ResultDataMaxSizeInBytes == 0 || information.ScratchDataSizeInBytes == 0)
        {
            return std::unexpected(Error{
                ErrorCode::nativeFailure,
                0,
                "D3D12 returned empty BLAS prebuild sizes for otherwise valid input.",
            });
        }
        return AccelerationStructureBuildSizes{
            .accelerationStructureSize = information.ResultDataMaxSizeInBytes,
            .buildScratchSize = information.ScratchDataSizeInBytes,
            .updateScratchSize = information.UpdateScratchDataSizeInBytes,
        };
    }

    Result<AccelerationStructureBuildSizes> ResourceManager::queryTopLevelAccelerationStructureBuildSizes(
        const TopLevelAccelerationStructureBuildInput& input) const
    {
        if (!impl_->rayTracing)
        {
            return std::unexpected(
                Error{ErrorCode::unsupported, 0, "Acceleration-structure builds require ray-tracing support."});
        }
        if (input.instanceCount == 0 ||
            input.instanceCount > D3D12_RAYTRACING_MAX_INSTANCES_PER_TOP_LEVEL_ACCELERATION_STRUCTURE ||
            !validAccelerationStructureBuildFlags(input.flags))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A TLAS build requires a supported non-zero instance count and valid build flags.",
            });
        }
        const auto native = nativeTopLevelInputs(input.instanceCount, nativeBuildFlags(input.flags), 0);
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO information{};
        impl_->rayTracingDevice->GetRaytracingAccelerationStructurePrebuildInfo(&native, &information);
        if (information.ResultDataMaxSizeInBytes == 0 || information.ScratchDataSizeInBytes == 0)
        {
            return std::unexpected(Error{
                ErrorCode::nativeFailure,
                0,
                "D3D12 returned empty TLAS prebuild sizes for otherwise valid input.",
            });
        }
        return AccelerationStructureBuildSizes{
            .accelerationStructureSize = information.ResultDataMaxSizeInBytes,
            .buildScratchSize = information.ScratchDataSizeInBytes,
            .updateScratchSize = information.UpdateScratchDataSizeInBytes,
        };
    }

    Result<AccelerationStructureHandle> ResourceManager::createAccelerationStructure(
        const AccelerationStructureDesc& desc)
    {
        if (!impl_->rayTracing)
        {
            return std::unexpected(
                Error{ErrorCode::unsupported, 0, "Acceleration structures require ray-tracing support."});
        }
        if (desc.type != AccelerationStructureType::bottomLevel && desc.type != AccelerationStructureType::topLevel)
        {
            return std::unexpected(Error{ErrorCode::invalidArgument, 0, "The acceleration-structure type is invalid."});
        }
        auto storage = impl_->buffer(desc.storage);
        if (!storage)
        {
            return std::unexpected(std::move(storage.error()));
        }
        if ((*storage)->desc.usage != BufferUsage::accelerationStructureStorage ||
            !validRange((*storage)->desc.size, desc.offset, desc.size))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Acceleration-structure storage must be a live, aligned, in-bounds dedicated storage-buffer range.",
            });
        }
        const auto size = resolveSize((*storage)->desc.size, desc.offset, desc.size);
        if (desc.offset % D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT != 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Acceleration-structure storage must be a live, in-bounds dedicated range with an aligned start "
                "address.",
            });
        }

        auto record = std::make_shared<AccelerationStructureRecord>();
        auto storageLease = std::make_shared<AccelerationStructureStorageLease>();
        storageLease->storage = *storage;
        storageLease->offset = desc.offset;
        storageLease->size = size;
        {
            const std::scoped_lock lock(impl_->accelerationStructureStorageLeasesMutex);
            std::erase_if(impl_->accelerationStructureStorageLeases,
                          [](const auto& existing) { return existing.expired(); });
            for (const auto& existingWeak : impl_->accelerationStructureStorageLeases)
            {
                const auto existing = existingWeak.lock();
                if (existing != nullptr && existing->storage == storageLease->storage &&
                    storageLease->offset < existing->offset + existing->size &&
                    existing->offset < storageLease->offset + storageLease->size)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Acceleration-structure storage ranges must not overlap retained acceleration structures.",
                    });
                }
            }
            impl_->accelerationStructureStorageLeases.push_back(storageLease);
        }
        record->type = desc.type;
        record->storage = std::move(*storage);
        record->offset = desc.offset;
        record->size = size;
        record->label = desc.label;
        record->storageLease = std::move(storageLease);
        const std::scoped_lock lock(impl_->accelerationStructuresMutex);
        return impl_->accelerationStructures.emplace(std::move(record));
    }

    Status ResourceManager::destroyAccelerationStructure(AccelerationStructureHandle accelerationStructure)
    {
        const std::scoped_lock lock(impl_->accelerationStructuresMutex);
        if (!impl_->accelerationStructures.erase(accelerationStructure))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The acceleration-structure handle is invalid or stale.",
            });
        }
        return {};
    }

    namespace
    {

        [[nodiscard]] bool isGraphicsOrComputeOutsideRendering(
            const detail::BackendCommandBuffer& commandBuffer) noexcept
        {
            return (commandBuffer.queueType() == QueueType::graphics ||
                    commandBuffer.queueType() == QueueType::compute) &&
                   commandBuffer.graphicsState().renderingOwner == nullptr;
        }

        [[nodiscard]] bool validScratchRange(const BufferRecord& scratch,
                                             std::uint64_t offset,
                                             std::uint64_t size) noexcept
        {
            return scratch.desc.usage == BufferUsage::accelerationStructureScratch &&
                   offset % D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT == 0 &&
                   (size == 0 ? offset < scratch.desc.size : validRange(scratch.desc.size, offset, size));
        }

        struct BuildTarget final
        {
            std::shared_ptr<AccelerationStructureRecord> destination;
            std::shared_ptr<BufferRecord> scratch;
        };

        struct TargetRequirement final
        {
            AccelerationStructureType type = AccelerationStructureType::bottomLevel;
            std::string_view mismatch;
        };

        constexpr TargetRequirement bottomLevelBuildRequirement{
            .type = AccelerationStructureType::bottomLevel,
            .mismatch =
                "A BLAS build requires a sufficiently large BLAS destination and aligned dedicated scratch range.",
        };

        [[nodiscard]] Result<BuildTarget> resolveBuildTarget(const AccelerationStructureLookupResult& destination,
                                                             const BufferLookupResult& scratch,
                                                             std::uint64_t scratchOffset,
                                                             const AccelerationStructureBuildSizes& sizes,
                                                             const TargetRequirement& requirement)
        {
            if (!destination)
            {
                return std::unexpected(destination.error());
            }
            if (!scratch)
            {
                return std::unexpected(scratch.error());
            }
            if ((*destination)->type != requirement.type || (*destination)->size < sizes.accelerationStructureSize ||
                !validScratchRange(**scratch, scratchOffset, sizes.buildScratchSize))
            {
                return std::unexpected(Error{ErrorCode::invalidArgument, 0, std::string(requirement.mismatch)});
            }
            return BuildTarget{.destination = *destination, .scratch = *scratch};
        }

        constexpr std::string_view inPlaceUpdateConsumerConflict =
            "An acceleration structure cannot be read while another command recording updates it in place.";

        [[nodiscard]] Result<AccelerationStructureExecutionLease> acquireProducerLease(
            const AccelerationStructureRecord& destination,
            std::uint64_t recordingId,
            std::span<const std::shared_ptr<AccelerationStructureExecutionTracker>> consumers,
            AccelerationStructureWrite write,
            std::string_view activeExecutionMessage)
        {
            auto lease = AccelerationStructureExecutionLease::tryAcquireProducer(
                destination.executionTracker, recordingId, consumers, write);
            if (!lease)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    std::string(lease.error() == AccelerationStructureLeaseConflict::consumer
                                    ? inPlaceUpdateConsumerConflict
                                    : activeExecutionMessage),
                });
            }
            return std::move(*lease);
        }

        [[nodiscard]] Result<AccelerationStructureExecutionLease> acquireProducerLeaseForRecording(
            QueueManager& queues,
            detail::BackendCommandBuffer& commandBuffer,
            const AccelerationStructureRecord& destination,
            std::span<const std::shared_ptr<AccelerationStructureExecutionTracker>> consumers,
            std::string_view activeExecutionMessage)
        {
            auto recordingId = queues.recordingId(commandBuffer);
            if (!recordingId)
            {
                return std::unexpected(std::move(recordingId.error()));
            }
            return acquireProducerLease(destination,
                                        *recordingId,
                                        consumers,
                                        AccelerationStructureWrite::publishVersion,
                                        activeExecutionMessage);
        }

        [[nodiscard]] Result<AccelerationStructureExecutionLease> acquireConsumerLease(
            const std::shared_ptr<AccelerationStructureExecutionTracker>& tracker, std::uint64_t recordingId)
        {
            auto lease = AccelerationStructureExecutionLease::tryAcquireConsumer(tracker, recordingId);
            if (!lease)
            {
                return std::unexpected(Error{ErrorCode::invalidState, 0, std::string(inPlaceUpdateConsumerConflict)});
            }
            return std::move(*lease);
        }

        [[nodiscard]] std::shared_ptr<AccelerationStructureBuildRetention> makeBottomLevelBuildRetention(
            const BuildTarget& target, const PreparedBottomLevelInput& prepared)
        {
            auto retention = std::make_shared<AccelerationStructureBuildRetention>();
            retention->owners.reserve(prepared.owners.size() + 2U);
            retention->owners.push_back(target.destination);
            retention->owners.push_back(target.scratch);
            retention->owners.insert(retention->owners.end(), prepared.owners.begin(), prepared.owners.end());
            retention->residencies.reserve(prepared.residencies.size() + 2U);
            retention->residencies.push_back(target.destination->storage->residency);
            retention->residencies.push_back(target.scratch->residency);
            retention->residencies.insert(
                retention->residencies.end(), prepared.residencies.begin(), prepared.residencies.end());
            return retention;
        }

        [[nodiscard]] D3D12_GPU_VIRTUAL_ADDRESS storageAddress(const AccelerationStructureRecord& record)
        {
            return record.storage->resource->GetGPUVirtualAddress() + record.offset;
        }

        [[nodiscard]] std::shared_ptr<AccelerationStructureVersion> makeVersionInStorage(
            const AccelerationStructureRecord& destination, D3D12_GPU_VIRTUAL_ADDRESS address)
        {
            auto version = std::make_shared<AccelerationStructureVersion>();
            version->storage = destination.storage;
            version->storageLease = destination.storageLease;
            version->address = address;
            if (destination.storage->residency != nullptr)
            {
                version->residencies.push_back(destination.storage->residency);
            }
            return version;
        }

        [[nodiscard]] std::shared_ptr<AccelerationStructureVersion> makeBottomLevelVersion(
            const AccelerationStructureRecord& destination,
            D3D12_GPU_VIRTUAL_ADDRESS address,
            const std::vector<D3D12_RAYTRACING_GEOMETRY_DESC>& geometryLayout,
            AccelerationStructureBuildFlag buildFlags,
            std::uint64_t resultSize)
        {
            auto version = makeVersionInStorage(destination, address);
            version->geometryLayout = geometryLayout;
            version->buildFlags = buildFlags;
            version->resultSize = resultSize;
            return version;
        }

        [[nodiscard]] AccelerationStructurePublication publicationOf(
            const std::shared_ptr<AccelerationStructureRecord>& destination,
            const std::shared_ptr<AccelerationStructureVersion>& version)
        {
            return AccelerationStructurePublication{
                .record = destination,
                .published = version,
                .previous = destination->currentVersion,
            };
        }

        [[nodiscard]] Status retainBuildDependencies(
            QueueManager& queues,
            detail::BackendCommandBuffer& commandBuffer,
            const std::shared_ptr<AccelerationStructureBuildRetention>& retention,
            AccelerationStructureExecutionLease executionLease,
            AccelerationStructurePublication publication)
        {
            const std::array dependencies{
                QueueManager::ResourceRetention{
                    .owner = retention,
                    .residencySet = retention->residencies,
                },
            };
            return queues.retainAccelerationStructureDependencies(
                commandBuffer, dependencies, std::move(executionLease), std::move(publication));
        }

        [[nodiscard]] Status retainBuildDependencies(
            QueueManager& queues,
            detail::BackendCommandBuffer& commandBuffer,
            const std::shared_ptr<AccelerationStructureBuildRetention>& retention,
            const std::shared_ptr<void>& upload,
            AccelerationStructureExecutionLease executionLease,
            AccelerationStructurePublication publication)
        {
            const std::array dependencies{
                QueueManager::ResourceRetention{
                    .owner = retention,
                    .residencySet = retention->residencies,
                },
                QueueManager::ResourceRetention{
                    .owner = upload,
                    .internal = true,
                },
            };
            return queues.retainAccelerationStructureDependencies(
                commandBuffer, dependencies, std::move(executionLease), std::move(publication));
        }

    } // namespace

    void rollBackAccelerationStructurePublications(std::vector<AccelerationStructurePublication>& publications) noexcept
    {
        for (auto publication = publications.rbegin(); publication != publications.rend(); ++publication)
        {
            const auto record = publication->record.lock();
            const auto published = publication->published.lock();
            if (record == nullptr || published == nullptr)
            {
                continue;
            }
            std::shared_ptr<AccelerationStructureVersion> discarded;
            {
                const std::scoped_lock lock(record->mutex);
                if (record->currentVersion == published)
                {
                    discarded = std::exchange(record->currentVersion, std::move(publication->previous));
                }
            }
        }
        publications.clear();
    }

    Status ResourceManager::buildBottomLevelAccelerationStructure(QueueManager& queues,
                                                                  detail::BackendCommandBuffer& commandBuffer,
                                                                  const BottomLevelAccelerationStructureBuildDesc& desc)
    {
        if (!isGraphicsOrComputeOutsideRendering(commandBuffer))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A BLAS build requires a graphics or compute command buffer outside a rendering scope.",
            });
        }
        auto prepared = impl_->prepareBottomLevel(desc.input);
        if (!prepared)
        {
            return std::unexpected(std::move(prepared.error()));
        }
        auto sizes = queryBottomLevelAccelerationStructureBuildSizes(desc.input);
        if (!sizes)
        {
            return std::unexpected(std::move(sizes.error()));
        }
        const auto destination = impl_->accelerationStructure(desc.destination);
        const auto scratch = impl_->buffer(desc.scratch);
        auto target = resolveBuildTarget(destination, scratch, desc.scratchOffset, *sizes, bottomLevelBuildRequirement);
        if (!target)
        {
            return std::unexpected(std::move(target.error()));
        }
        const std::unique_lock destinationLock(target->destination->mutex);
        if (target->destination->currentVersion != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Acceleration structures are immutable after their first recorded build; create a new destination.",
            });
        }

        auto commandLists = prepareAccelerationStructureCommandLists(queues, commandBuffer, impl_->enhancedBarriers);
        if (!commandLists)
        {
            return std::unexpected(std::move(commandLists.error()));
        }
        auto executionLease = acquireProducerLeaseForRecording(
            queues,
            commandBuffer,
            *target->destination,
            {},
            "A BLAS build cannot overwrite an acceleration structure with active GPU execution.");
        if (!executionLease)
        {
            return std::unexpected(std::move(executionLease.error()));
        }
        const auto retention = makeBottomLevelBuildRetention(*target, *prepared);
        const auto destinationAddress = storageAddress(*target->destination);
        const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC native{
            .DestAccelerationStructureData = destinationAddress,
            .Inputs = nativeBottomLevelInputs(prepared->geometries, nativeBuildFlags(desc.input.flags)),
            .SourceAccelerationStructureData = 0,
            .ScratchAccelerationStructureData = target->scratch->resource->GetGPUVirtualAddress() + desc.scratchOffset,
        };
        auto version = makeBottomLevelVersion(*target->destination,
                                              destinationAddress,
                                              prepared->geometries,
                                              desc.input.flags,
                                              sizes->accelerationStructureSize);
        auto retained = retainBuildDependencies(
            queues, commandBuffer, retention, std::move(*executionLease), publicationOf(target->destination, version));
        if (!retained)
        {
            return retained;
        }

        commandLists->rayTracing->BuildRaytracingAccelerationStructure(&native, 0, nullptr);
        recordAccelerationStructureBuildBarrier(*commandLists);
        target->destination->currentVersion = std::move(version);
        test::noteAccelerationStructureRecorded();
        return {};
    }

    struct TopLevelInstanceBufferRange final
    {
        BufferHandle buffer = {};
        std::uint64_t offset = 0;
        std::uint32_t count = 0;
    };

    using TopLevelInstanceSource =
        std::variant<std::span<const AccelerationStructureInstance>, TopLevelInstanceBufferRange>;

    struct TopLevelBuildRequest final
    {
        AccelerationStructureHandle source = {};
        AccelerationStructureHandle destination = {};
        TopLevelInstanceSource instances = {};
        AccelerationStructureBuildFlag flags = AccelerationStructureBuildFlag::none;
        BufferHandle scratch = {};
        std::uint64_t scratchOffset = 0;
    };

    namespace
    {

        static_assert(sizeof(D3D12_RAYTRACING_INSTANCE_DESC) == sizeof(AccelerationStructureInstanceRecord));
        static_assert(offsetof(D3D12_RAYTRACING_INSTANCE_DESC, AccelerationStructure) ==
                      offsetof(AccelerationStructureInstanceRecord, bottomLevel));
        static_assert(D3D12_RAYTRACING_INSTANCE_DESCS_BYTE_ALIGNMENT == alignof(AccelerationStructureInstanceRecord));
        static_assert(static_cast<std::uint32_t>(AccelerationStructureInstanceFlag::triangleCullDisable) ==
                      static_cast<std::uint32_t>(D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE));
        static_assert(static_cast<std::uint32_t>(AccelerationStructureInstanceFlag::triangleFrontCounterClockwise) ==
                      static_cast<std::uint32_t>(D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_FRONT_COUNTERCLOCKWISE));
        static_assert(static_cast<std::uint32_t>(AccelerationStructureInstanceFlag::forceOpaque) ==
                      static_cast<std::uint32_t>(D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE));
        static_assert(static_cast<std::uint32_t>(AccelerationStructureInstanceFlag::forceNonOpaque) ==
                      static_cast<std::uint32_t>(D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_NON_OPAQUE));

        [[nodiscard]] std::size_t requestedInstanceCount(const TopLevelInstanceSource& instances) noexcept
        {
            if (const auto* span = std::get_if<std::span<const AccelerationStructureInstance>>(&instances))
            {
                return span->size();
            }
            return std::get<TopLevelInstanceBufferRange>(instances).count;
        }

        [[nodiscard]] Status validateTopLevelBuildRequest(const detail::BackendCommandBuffer& commandBuffer,
                                                          const TopLevelBuildRequest& request)
        {
            if (!isGraphicsOrComputeOutsideRendering(commandBuffer))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A TLAS build requires a graphics or compute command buffer outside a rendering scope.",
                });
            }
            if (requestedInstanceCount(request.instances) > std::numeric_limits<std::uint32_t>::max())
            {
                return std::unexpected(
                    Error{ErrorCode::invalidArgument, 0, "The TLAS instance count is not portable."});
            }
            return {};
        }

        constexpr TargetRequirement topLevelBuildRequirement{
            .type = AccelerationStructureType::topLevel,
            .mismatch =
                "A TLAS build requires a sufficiently large TLAS destination and aligned dedicated scratch range.",
        };

        struct TopLevelInstanceErrors final
        {
            std::string_view invalidInstance;
            std::string_view bottomLevelRequired;
            std::string_view unbuiltBottomLevel;
        };

        constexpr TopLevelInstanceErrors topLevelBuildInstanceErrors{
            .invalidInstance = "A TLAS instance contains an invalid transform, packed field, or flag combination.",
            .bottomLevelRequired = "Every TLAS instance must reference a bottom-level acceleration structure.",
            .unbuiltBottomLevel = "Every referenced BLAS must have a recorded build before the TLAS build.",
        };

        struct TopLevelInstances final
        {
            std::vector<D3D12_RAYTRACING_INSTANCE_DESC> native;
            std::vector<std::shared_ptr<AccelerationStructureVersion>> bottomLevels;
            std::vector<std::shared_ptr<AccelerationStructureExecutionTracker>> bottomLevelTrackers;
        };

        [[nodiscard]] std::shared_ptr<AccelerationStructureVersion> snapshotCurrentVersion(
            AccelerationStructureRecord& record)
        {
            const std::scoped_lock lock(record.mutex);
            return record.currentVersion;
        }

        [[nodiscard]] bool validTopLevelInstance(const AccelerationStructureInstance& instance) noexcept
        {
            return instance.instanceId <= 0x00FF'FFFFU && instance.instanceMask <= 0xFFU &&
                   instance.hitGroupOffset <= 0x00FF'FFFFU && validInstanceFlags(instance.flags) &&
                   std::ranges::all_of(instance.transform, [](float value) { return std::isfinite(value); });
        }

        [[nodiscard]] D3D12_RAYTRACING_INSTANCE_DESC nativeInstance(
            const AccelerationStructureInstance& instance, D3D12_GPU_VIRTUAL_ADDRESS bottomLevelAddress) noexcept
        {
            D3D12_RAYTRACING_INSTANCE_DESC native{};
            static_assert(sizeof(native.Transform) == sizeof(instance.transform));
            std::memcpy(static_cast<void*>(native.Transform), instance.transform.data(), sizeof(native.Transform));
            native.InstanceID = instance.instanceId;
            native.InstanceMask = instance.instanceMask;
            native.InstanceContributionToHitGroupIndex = instance.hitGroupOffset;
            native.Flags = nativeInstanceFlags(instance.flags);
            native.AccelerationStructure = bottomLevelAddress;
            return native;
        }

        [[nodiscard]] Result<TopLevelInstances> resolveTopLevelInstances(
            std::span<const AccelerationStructureInstance> instances,
            const AccelerationStructureLookup& findAccelerationStructure,
            const TopLevelInstanceErrors& errors)
        {
            TopLevelInstances resolved;
            resolved.native.reserve(instances.size());
            resolved.bottomLevels.reserve(instances.size());
            resolved.bottomLevelTrackers.reserve(instances.size());
            for (const auto& instance : instances)
            {
                if (!validTopLevelInstance(instance))
                {
                    return std::unexpected(Error{ErrorCode::invalidArgument, 0, std::string(errors.invalidInstance)});
                }
                auto bottomLevel = findAccelerationStructure(instance.bottomLevel);
                if (!bottomLevel)
                {
                    return std::unexpected(std::move(bottomLevel.error()));
                }
                if ((*bottomLevel)->type != AccelerationStructureType::bottomLevel)
                {
                    return std::unexpected(
                        Error{ErrorCode::invalidArgument, 0, std::string(errors.bottomLevelRequired)});
                }
                auto bottomLevelVersion = snapshotCurrentVersion(**bottomLevel);
                if (bottomLevelVersion == nullptr)
                {
                    return std::unexpected(Error{ErrorCode::invalidState, 0, std::string(errors.unbuiltBottomLevel)});
                }
                resolved.native.push_back(nativeInstance(instance, bottomLevelVersion->address));
                resolved.bottomLevels.push_back(std::move(bottomLevelVersion));
                resolved.bottomLevelTrackers.push_back((*bottomLevel)->executionTracker);
            }
            deduplicateSharedPointers(resolved.bottomLevels);
            deduplicateSharedPointers(resolved.bottomLevelTrackers);
            return resolved;
        }

        void appendBottomLevelResidencies(std::vector<std::shared_ptr<ResidencyRecord>>& residencies,
                                          std::span<const std::shared_ptr<AccelerationStructureVersion>> bottomLevels)
        {
            for (const auto& bottomLevel : bottomLevels)
            {
                residencies.insert(residencies.end(), bottomLevel->residencies.begin(), bottomLevel->residencies.end());
            }
        }

        struct PreparedTopLevelInstances final
        {
            D3D12_GPU_VIRTUAL_ADDRESS address = 0;
            std::uint32_t count = 0;
            std::shared_ptr<void> upload;
            std::shared_ptr<BufferRecord> buffer;
            std::vector<std::shared_ptr<AccelerationStructureVersion>> bottomLevels;
            std::vector<std::shared_ptr<AccelerationStructureExecutionTracker>> bottomLevelTrackers;
        };

        using UploadSliceAllocator = std::function<Result<UploadSlice>(std::uint64_t, std::uint64_t)>;

        [[nodiscard]] Result<PreparedTopLevelInstances> prepareUploadedInstances(
            std::span<const AccelerationStructureInstance> instances,
            const AccelerationStructureLookup& findAccelerationStructure,
            const UploadSliceAllocator& allocateUpload,
            const TopLevelInstanceErrors& errors)
        {
            auto resolved = resolveTopLevelInstances(instances, findAccelerationStructure, errors);
            if (!resolved)
            {
                return std::unexpected(std::move(resolved.error()));
            }
            const std::uint64_t uploadBytes = resolved->native.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC);
            auto upload = allocateUpload(uploadBytes, D3D12_RAYTRACING_INSTANCE_DESCS_BYTE_ALIGNMENT);
            if (!upload)
            {
                return std::unexpected(std::move(upload.error()));
            }
            std::memcpy(upload->mappedData, resolved->native.data(), uploadBytes);
            return PreparedTopLevelInstances{
                .address = upload->gpuVirtualAddress,
                .count = static_cast<std::uint32_t>(resolved->native.size()),
                .upload = std::move(upload->owner),
                .buffer = nullptr,
                .bottomLevels = std::move(resolved->bottomLevels),
                .bottomLevelTrackers = std::move(resolved->bottomLevelTrackers),
            };
        }

        [[nodiscard]] Result<PreparedTopLevelInstances> prepareBufferInstances(const TopLevelInstanceBufferRange& range,
                                                                               const BufferLookup& findBuffer)
        {
            auto buffer = findBuffer(range.buffer);
            if (!buffer)
            {
                return std::unexpected(std::move(buffer.error()));
            }
            const std::uint64_t recordBytes =
                static_cast<std::uint64_t>(range.count) * sizeof(D3D12_RAYTRACING_INSTANCE_DESC);
            if (!hasAny((*buffer)->desc.usage, BufferUsage::accelerationStructureBuildInput) ||
                (*buffer)->desc.memoryType == MemoryType::readback ||
                range.offset % D3D12_RAYTRACING_INSTANCE_DESCS_BYTE_ALIGNMENT != 0 ||
                !validRange((*buffer)->desc.size, range.offset, recordBytes))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "TLAS instance records require a live, non-readback build-input buffer that holds every record "
                    "from an aligned offset.",
                });
            }
            return PreparedTopLevelInstances{
                .address = (*buffer)->resource->GetGPUVirtualAddress() + range.offset,
                .count = range.count,
                .upload = nullptr,
                .buffer = std::move(*buffer),
                .bottomLevels = {},
                .bottomLevelTrackers = {},
            };
        }

        [[nodiscard]] Result<PreparedTopLevelInstances> prepareTopLevelInstances(
            const TopLevelInstanceSource& instances,
            const AccelerationStructureLookup& findAccelerationStructure,
            const BufferLookup& findBuffer,
            const UploadSliceAllocator& allocateUpload,
            const TopLevelInstanceErrors& errors)
        {
            if (const auto* range = std::get_if<TopLevelInstanceBufferRange>(&instances))
            {
                return prepareBufferInstances(*range, findBuffer);
            }
            return prepareUploadedInstances(std::get<std::span<const AccelerationStructureInstance>>(instances),
                                            findAccelerationStructure,
                                            allocateUpload,
                                            errors);
        }

        void appendInstanceRetention(AccelerationStructureBuildRetention& retention,
                                     const PreparedTopLevelInstances& instances)
        {
            if (instances.upload != nullptr)
            {
                retention.owners.push_back(instances.upload);
            }
            if (instances.buffer != nullptr)
            {
                retention.owners.push_back(instances.buffer);
                retention.residencies.push_back(instances.buffer->residency);
            }
            retention.owners.insert(
                retention.owners.end(), instances.bottomLevels.begin(), instances.bottomLevels.end());
            appendBottomLevelResidencies(retention.residencies, instances.bottomLevels);
        }

        [[nodiscard]] std::shared_ptr<AccelerationStructureBuildRetention> makeTopLevelBuildRetention(
            const BuildTarget& target, const PreparedTopLevelInstances& instances)
        {
            auto retention = std::make_shared<AccelerationStructureBuildRetention>();
            retention->owners.reserve(instances.bottomLevels.size() + 3U);
            retention->owners.push_back(target.destination);
            retention->owners.push_back(target.scratch);
            retention->residencies.reserve(instances.bottomLevels.size() + 3U);
            retention->residencies.push_back(target.destination->storage->residency);
            retention->residencies.push_back(target.scratch->residency);
            appendInstanceRetention(*retention, instances);
            deduplicateSharedPointers(retention->residencies);
            return retention;
        }

        [[nodiscard]] std::shared_ptr<AccelerationStructureVersion> makeTopLevelVersion(
            const AccelerationStructureRecord& destination,
            D3D12_GPU_VIRTUAL_ADDRESS address,
            const PreparedTopLevelInstances& instances,
            AccelerationStructureBuildFlag buildFlags,
            std::uint64_t resultSize)
        {
            auto version = makeVersionInStorage(destination, address);
            version->dependencies.assign(instances.bottomLevels.begin(), instances.bottomLevels.end());
            version->dependencyTrackers = instances.bottomLevelTrackers;
            version->buildFlags = buildFlags;
            version->instanceCount = instances.count;
            version->resultSize = resultSize;
            appendBottomLevelResidencies(version->residencies, instances.bottomLevels);
            deduplicateSharedPointers(version->residencies);
            return version;
        }

        [[nodiscard]] Status retainTopLevelDependencies(
            QueueManager& queues,
            detail::BackendCommandBuffer& commandBuffer,
            const std::shared_ptr<AccelerationStructureBuildRetention>& retention,
            const PreparedTopLevelInstances& instances,
            AccelerationStructureExecutionLease executionLease,
            AccelerationStructurePublication publication)
        {
            if (instances.upload != nullptr)
            {
                return retainBuildDependencies(queues,
                                               commandBuffer,
                                               retention,
                                               instances.upload,
                                               std::move(executionLease),
                                               std::move(publication));
            }
            return retainBuildDependencies(
                queues, commandBuffer, retention, std::move(executionLease), std::move(publication));
        }

    } // namespace

    Status ResourceManager::buildTopLevelAccelerationStructure(QueueManager& queues,
                                                               detail::BackendCommandBuffer& commandBuffer,
                                                               const TopLevelAccelerationStructureBuildDesc& desc)
    {
        return recordTopLevelBuild(queues,
                                   commandBuffer,
                                   {
                                       .destination = desc.destination,
                                       .instances = desc.instances,
                                       .flags = desc.flags,
                                       .scratch = desc.scratch,
                                       .scratchOffset = desc.scratchOffset,
                                   });
    }

    Status ResourceManager::buildTopLevelAccelerationStructureFromBuffer(
        QueueManager& queues,
        detail::BackendCommandBuffer& commandBuffer,
        const TopLevelAccelerationStructureBufferBuildDesc& desc)
    {
        return recordTopLevelBuild(queues,
                                   commandBuffer,
                                   {
                                       .destination = desc.destination,
                                       .instances =
                                           TopLevelInstanceBufferRange{
                                               .buffer = desc.instanceBuffer,
                                               .offset = desc.instanceOffset,
                                               .count = desc.instanceCount,
                                           },
                                       .flags = desc.flags,
                                       .scratch = desc.scratch,
                                       .scratchOffset = desc.scratchOffset,
                                   });
    }

    Status ResourceManager::recordTopLevelBuild(QueueManager& queues,
                                                detail::BackendCommandBuffer& commandBuffer,
                                                const TopLevelBuildRequest& request)
    {
        auto validRequest = validateTopLevelBuildRequest(commandBuffer, request);
        if (!validRequest)
        {
            return validRequest;
        }
        const TopLevelAccelerationStructureBuildInput input{
            .instanceCount = static_cast<std::uint32_t>(requestedInstanceCount(request.instances)),
            .flags = request.flags,
        };
        auto sizes = queryTopLevelAccelerationStructureBuildSizes(input);
        if (!sizes)
        {
            return std::unexpected(std::move(sizes.error()));
        }
        const auto destination = impl_->accelerationStructure(request.destination);
        const auto scratch = impl_->buffer(request.scratch);
        auto target = resolveBuildTarget(destination, scratch, request.scratchOffset, *sizes, topLevelBuildRequirement);
        if (!target)
        {
            return std::unexpected(std::move(target.error()));
        }
        const std::unique_lock destinationLock(target->destination->mutex);
        if (target->destination->currentVersion != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Acceleration structures are immutable after their first recorded build; create a new destination.",
            });
        }

        auto instances = prepareTopLevelInstances(
            request.instances,
            [this](AccelerationStructureHandle handle) { return impl_->accelerationStructure(handle); },
            [this](BufferHandle handle) { return impl_->buffer(handle); },
            [this](std::uint64_t size, std::uint64_t alignment) {
                return impl_->allocateUploadSlice(size, alignment, "TLAS instances");
            },
            topLevelBuildInstanceErrors);
        if (!instances)
        {
            return std::unexpected(std::move(instances.error()));
        }

        auto commandLists = prepareAccelerationStructureCommandLists(queues, commandBuffer, impl_->enhancedBarriers);
        if (!commandLists)
        {
            return std::unexpected(std::move(commandLists.error()));
        }
        auto executionLease = acquireProducerLeaseForRecording(
            queues,
            commandBuffer,
            *target->destination,
            instances->bottomLevelTrackers,
            "A TLAS build cannot overwrite an acceleration structure with active GPU execution.");
        if (!executionLease)
        {
            return std::unexpected(std::move(executionLease.error()));
        }
        const auto retention = makeTopLevelBuildRetention(*target, *instances);
        const auto destinationAddress = storageAddress(*target->destination);
        const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC native{
            .DestAccelerationStructureData = destinationAddress,
            .Inputs = nativeTopLevelInputs(instances->count, nativeBuildFlags(request.flags), instances->address),
            .SourceAccelerationStructureData = 0,
            .ScratchAccelerationStructureData =
                target->scratch->resource->GetGPUVirtualAddress() + request.scratchOffset,
        };
        auto version = makeTopLevelVersion(
            *target->destination, destinationAddress, *instances, request.flags, sizes->accelerationStructureSize);
        auto retained = retainTopLevelDependencies(queues,
                                                   commandBuffer,
                                                   retention,
                                                   *instances,
                                                   std::move(*executionLease),
                                                   publicationOf(target->destination, version));
        if (!retained)
        {
            return retained;
        }

        commandLists->rayTracing->BuildRaytracingAccelerationStructure(&native, 0, nullptr);
        recordAccelerationStructureBuildBarrier(*commandLists);
        target->destination->currentVersion = std::move(version);
        test::noteAccelerationStructureRecorded();
        return {};
    }

    Status ResourceManager::updateBottomLevelAccelerationStructure(
        QueueManager& queues,
        detail::BackendCommandBuffer& commandBuffer,
        const BottomLevelAccelerationStructureUpdateDesc& desc)
    {
        return updateBottomLevelAccelerationStructure(queues, commandBuffer, desc, false);
    }

    Status ResourceManager::updateBottomLevelAccelerationStructureInPlace(
        QueueManager& queues,
        detail::BackendCommandBuffer& commandBuffer,
        const BottomLevelAccelerationStructureInPlaceUpdateDesc& desc)
    {
        return updateBottomLevelAccelerationStructure(queues,
                                                      commandBuffer,
                                                      {
                                                          .source = desc.target,
                                                          .destination = desc.target,
                                                          .input = desc.input,
                                                          .scratch = desc.scratch,
                                                          .scratchOffset = desc.scratchOffset,
                                                      },
                                                      true);
    }

    namespace
    {

        [[nodiscard]] Status validateBottomLevelUpdateRequest(const detail::BackendCommandBuffer& commandBuffer,
                                                              const BottomLevelAccelerationStructureUpdateDesc& desc,
                                                              bool inPlace)
        {
            if (!isGraphicsOrComputeOutsideRendering(commandBuffer))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A BLAS update requires a graphics or compute command buffer outside a rendering scope.",
                });
            }
            if (inPlace != (desc.source == desc.destination))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    inPlace ? "An in-place BLAS update requires one identical source and destination handle."
                            : "An immutable BLAS update requires distinct source and destination handles.",
                });
            }
            return {};
        }

        struct UpdateTarget final
        {
            std::shared_ptr<AccelerationStructureRecord> source;
            std::shared_ptr<AccelerationStructureRecord> destination;
            std::shared_ptr<BufferRecord> scratch;
        };

        constexpr TargetRequirement bottomLevelUpdateRequirement{
            .type = AccelerationStructureType::bottomLevel,
            .mismatch =
                "A BLAS update requires compatible BLAS handles, destination storage, and aligned update scratch.",
        };

        [[nodiscard]] Result<UpdateTarget> resolveUpdateTarget(const AccelerationStructureLookupResult& source,
                                                               const AccelerationStructureLookupResult& destination,
                                                               const BufferLookupResult& scratch,
                                                               std::uint64_t scratchOffset,
                                                               std::uint64_t updateScratchSize,
                                                               const TargetRequirement& requirement)
        {
            if (!source)
            {
                return std::unexpected(source.error());
            }
            if (!destination)
            {
                return std::unexpected(destination.error());
            }
            if (!scratch)
            {
                return std::unexpected(scratch.error());
            }
            if ((*source)->type != requirement.type || (*destination)->type != requirement.type ||
                !validScratchRange(**scratch, scratchOffset, updateScratchSize))
            {
                return std::unexpected(Error{ErrorCode::invalidArgument, 0, std::string(requirement.mismatch)});
            }
            return UpdateTarget{.source = *source, .destination = *destination, .scratch = *scratch};
        }

        struct UpdateTargetErrors final
        {
            std::string_view unbuiltInPlaceTarget;
            std::string_view unbuiltSourceOrBuiltDestination;
            std::string_view inPlaceTargetExecuting;
            std::string_view destinationTooSmall;
        };

        constexpr UpdateTargetErrors bottomLevelUpdateTargetErrors{
            .unbuiltInPlaceTarget = "An in-place BLAS update requires a built target.",
            .unbuiltSourceOrBuiltDestination =
                "A BLAS update requires a built source and an unbuilt immutable destination.",
            .inPlaceTargetExecuting =
                "An in-place BLAS update cannot overwrite an acceleration structure with active GPU execution.",
            .destinationTooSmall = "A BLAS update destination must fit the source version and the update input's "
                                   "prebuild size, including for compacted sources.",
        };

        struct LockedUpdateTarget final
        {
            std::unique_lock<std::mutex> sourceLock;
            std::unique_lock<std::mutex> destinationLock;
            std::shared_ptr<AccelerationStructureVersion> sourceVersion;
            std::uint64_t recordingId = 0;
        };

        [[nodiscard]] Result<LockedUpdateTarget> lockAndValidateUpdateTarget(
            QueueManager& queues,
            detail::BackendCommandBuffer& commandBuffer,
            const UpdateTarget& target,
            bool inPlace,
            std::uint64_t updateResultSize,
            const UpdateTargetErrors& errors)
        {
            auto recordingId = queues.recordingId(commandBuffer);
            if (!recordingId)
            {
                return std::unexpected(std::move(recordingId.error()));
            }
            if (inPlace)
            {
                queues.collectGarbage();
            }

            LockedUpdateTarget locked;
            locked.recordingId = *recordingId;
            locked.sourceLock = std::unique_lock(target.source->mutex, std::defer_lock);
            locked.destinationLock = std::unique_lock(target.destination->mutex, std::defer_lock);
            if (inPlace)
            {
                locked.sourceLock.lock();
            }
            else
            {
                std::lock(locked.sourceLock, locked.destinationLock);
            }
            locked.sourceVersion = target.source->currentVersion;
            if (locked.sourceVersion == nullptr || (!inPlace && target.destination->currentVersion != nullptr))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    std::string(inPlace ? errors.unbuiltInPlaceTarget : errors.unbuiltSourceOrBuiltDestination),
                });
            }
            if (inPlace && !target.source->executionTracker->canUpdateInPlace(locked.recordingId))
            {
                return std::unexpected(Error{ErrorCode::invalidState, 0, std::string(errors.inPlaceTargetExecuting)});
            }
            if (target.destination->size < locked.sourceVersion->resultSize ||
                target.destination->size < updateResultSize)
            {
                return std::unexpected(Error{ErrorCode::invalidArgument, 0, std::string(errors.destinationTooSmall)});
            }
            return locked;
        }

        [[nodiscard]] Status validateBottomLevelUpdateCompatibility(
            const AccelerationStructureVersion& source,
            AccelerationStructureBuildFlag flags,
            std::span<const D3D12_RAYTRACING_GEOMETRY_DESC> geometries)
        {
            if (!hasAny(source.buildFlags, AccelerationStructureBuildFlag::allowUpdate) || flags != source.buildFlags ||
                !updateCompatibleGeometryLayout(source.geometryLayout, geometries))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A BLAS update must preserve the update-enabled source flags and complete geometry layout.",
                });
            }
            return {};
        }

        [[nodiscard]] Result<AccelerationStructureExecutionLease> acquireBottomLevelUpdateLease(
            const UpdateTarget& target, std::uint64_t recordingId, bool inPlace)
        {
            const std::array sourceConsumers{target.source->executionTracker};
            return acquireProducerLease(
                *target.destination,
                recordingId,
                inPlace ? std::span<const std::shared_ptr<AccelerationStructureExecutionTracker>>{} : sourceConsumers,
                inPlace ? AccelerationStructureWrite::updateInPlace : AccelerationStructureWrite::publishVersion,
                "A BLAS update cannot overwrite an acceleration structure with active GPU execution.");
        }

        [[nodiscard]] std::shared_ptr<AccelerationStructureBuildRetention> makeBottomLevelUpdateRetention(
            const UpdateTarget& target,
            const std::shared_ptr<AccelerationStructureVersion>& sourceVersion,
            const PreparedBottomLevelInput& prepared)
        {
            auto retention = std::make_shared<AccelerationStructureBuildRetention>();
            retention->owners.reserve(prepared.owners.size() + 3U);
            retention->owners.push_back(target.destination);
            retention->owners.push_back(target.scratch);
            retention->owners.push_back(sourceVersion);
            retention->owners.insert(retention->owners.end(), prepared.owners.begin(), prepared.owners.end());
            retention->residencies.reserve(prepared.residencies.size() + sourceVersion->residencies.size() + 2U);
            retention->residencies.push_back(target.destination->storage->residency);
            retention->residencies.push_back(target.scratch->residency);
            retention->residencies.insert(
                retention->residencies.end(), sourceVersion->residencies.begin(), sourceVersion->residencies.end());
            retention->residencies.insert(
                retention->residencies.end(), prepared.residencies.begin(), prepared.residencies.end());
            return retention;
        }

    } // namespace

    Status ResourceManager::updateBottomLevelAccelerationStructure(
        QueueManager& queues,
        detail::BackendCommandBuffer& commandBuffer,
        const BottomLevelAccelerationStructureUpdateDesc& desc,
        bool inPlace)
    {
        auto validRequest = validateBottomLevelUpdateRequest(commandBuffer, desc, inPlace);
        if (!validRequest)
        {
            return validRequest;
        }
        auto prepared = impl_->prepareBottomLevel(desc.input);
        if (!prepared)
        {
            return std::unexpected(std::move(prepared.error()));
        }
        auto sizes = queryBottomLevelAccelerationStructureBuildSizes(desc.input);
        if (!sizes)
        {
            return std::unexpected(std::move(sizes.error()));
        }
        const auto source = impl_->accelerationStructure(desc.source);
        const auto destination = impl_->accelerationStructure(desc.destination);
        const auto scratch = impl_->buffer(desc.scratch);
        auto target = resolveUpdateTarget(
            source, destination, scratch, desc.scratchOffset, sizes->updateScratchSize, bottomLevelUpdateRequirement);
        if (!target)
        {
            return std::unexpected(std::move(target.error()));
        }

        auto lockedTarget = lockAndValidateUpdateTarget(
            queues, commandBuffer, *target, inPlace, sizes->accelerationStructureSize, bottomLevelUpdateTargetErrors);
        if (!lockedTarget)
        {
            return std::unexpected(std::move(lockedTarget.error()));
        }
        auto compatible = validateBottomLevelUpdateCompatibility(
            *lockedTarget->sourceVersion, desc.input.flags, prepared->geometries);
        if (!compatible)
        {
            return compatible;
        }

        auto commandLists = prepareAccelerationStructureCommandLists(queues, commandBuffer, impl_->enhancedBarriers);
        if (!commandLists)
        {
            return std::unexpected(std::move(commandLists.error()));
        }
        auto executionLease = acquireBottomLevelUpdateLease(*target, lockedTarget->recordingId, inPlace);
        if (!executionLease)
        {
            return std::unexpected(std::move(executionLease.error()));
        }
        const auto retention = makeBottomLevelUpdateRetention(*target, lockedTarget->sourceVersion, *prepared);
        const auto destinationAddress = storageAddress(*target->destination);
        const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC native{
            .DestAccelerationStructureData = destinationAddress,
            .Inputs = nativeBottomLevelInputs(prepared->geometries,
                                              nativeBuildFlags(desc.input.flags) |
                                                  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE),
            .SourceAccelerationStructureData = lockedTarget->sourceVersion->address,
            .ScratchAccelerationStructureData = target->scratch->resource->GetGPUVirtualAddress() + desc.scratchOffset,
        };
        auto version = makeBottomLevelVersion(*target->destination,
                                              destinationAddress,
                                              prepared->geometries,
                                              desc.input.flags,
                                              lockedTarget->sourceVersion->resultSize);
        auto retained = retainBuildDependencies(
            queues, commandBuffer, retention, std::move(*executionLease), publicationOf(target->destination, version));
        if (!retained)
        {
            return retained;
        }

        if (inPlace)
        {
            recordAccelerationStructureInPlaceUpdateBarrier(*commandLists);
        }
        commandLists->rayTracing->BuildRaytracingAccelerationStructure(&native, 0, nullptr);
        recordAccelerationStructureBuildBarrier(*commandLists);
        target->destination->currentVersion = std::move(version);
        test::noteAccelerationStructureRecorded();
        return {};
    }

    Status ResourceManager::updateTopLevelAccelerationStructure(QueueManager& queues,
                                                                detail::BackendCommandBuffer& commandBuffer,
                                                                const TopLevelAccelerationStructureUpdateDesc& desc)
    {
        return recordTopLevelUpdate(queues,
                                    commandBuffer,
                                    {
                                        .source = desc.source,
                                        .destination = desc.destination,
                                        .instances = desc.instances,
                                        .flags = desc.flags,
                                        .scratch = desc.scratch,
                                        .scratchOffset = desc.scratchOffset,
                                    },
                                    false);
    }

    Status ResourceManager::updateTopLevelAccelerationStructureInPlace(
        QueueManager& queues,
        detail::BackendCommandBuffer& commandBuffer,
        const TopLevelAccelerationStructureInPlaceUpdateDesc& desc)
    {
        return recordTopLevelUpdate(queues,
                                    commandBuffer,
                                    {
                                        .source = desc.target,
                                        .destination = desc.target,
                                        .instances = desc.instances,
                                        .flags = desc.flags,
                                        .scratch = desc.scratch,
                                        .scratchOffset = desc.scratchOffset,
                                    },
                                    true);
    }

    Status ResourceManager::updateTopLevelAccelerationStructureFromBuffer(
        QueueManager& queues,
        detail::BackendCommandBuffer& commandBuffer,
        const TopLevelAccelerationStructureBufferUpdateDesc& desc)
    {
        return recordTopLevelUpdate(queues,
                                    commandBuffer,
                                    {
                                        .source = desc.source,
                                        .destination = desc.destination,
                                        .instances =
                                            TopLevelInstanceBufferRange{
                                                .buffer = desc.instanceBuffer,
                                                .offset = desc.instanceOffset,
                                                .count = desc.instanceCount,
                                            },
                                        .flags = desc.flags,
                                        .scratch = desc.scratch,
                                        .scratchOffset = desc.scratchOffset,
                                    },
                                    false);
    }

    Status ResourceManager::updateTopLevelAccelerationStructureInPlaceFromBuffer(
        QueueManager& queues,
        detail::BackendCommandBuffer& commandBuffer,
        const TopLevelAccelerationStructureBufferInPlaceUpdateDesc& desc)
    {
        return recordTopLevelUpdate(queues,
                                    commandBuffer,
                                    {
                                        .source = desc.target,
                                        .destination = desc.target,
                                        .instances =
                                            TopLevelInstanceBufferRange{
                                                .buffer = desc.instanceBuffer,
                                                .offset = desc.instanceOffset,
                                                .count = desc.instanceCount,
                                            },
                                        .flags = desc.flags,
                                        .scratch = desc.scratch,
                                        .scratchOffset = desc.scratchOffset,
                                    },
                                    true);
    }

    namespace
    {

        [[nodiscard]] Status validateTopLevelUpdateRequest(const detail::BackendCommandBuffer& commandBuffer,
                                                           const TopLevelBuildRequest& desc,
                                                           bool inPlace)
        {
            if (!isGraphicsOrComputeOutsideRendering(commandBuffer))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "A TLAS update requires a graphics or compute command buffer outside a rendering scope.",
                });
            }
            if (inPlace != (desc.source == desc.destination) ||
                requestedInstanceCount(desc.instances) > std::numeric_limits<std::uint32_t>::max())
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    inPlace ? "An in-place TLAS update requires one target and a portable instance count."
                            : "A TLAS update requires distinct handles and a portable instance count.",
                });
            }
            return {};
        }

        constexpr TargetRequirement topLevelUpdateRequirement{
            .type = AccelerationStructureType::topLevel,
            .mismatch =
                "A TLAS update requires compatible TLAS handles, destination storage, and aligned update scratch.",
        };

        constexpr UpdateTargetErrors topLevelUpdateTargetErrors{
            .unbuiltInPlaceTarget = "An in-place TLAS update requires a built target.",
            .unbuiltSourceOrBuiltDestination =
                "A TLAS update requires a built source and an unbuilt immutable destination.",
            .inPlaceTargetExecuting =
                "An in-place TLAS update cannot overwrite an acceleration structure with active GPU execution.",
            .destinationTooSmall = "A TLAS update destination must fit the source version and the update input's "
                                   "prebuild size, including for compacted sources.",
        };

        [[nodiscard]] Status validateTopLevelUpdateCompatibility(const AccelerationStructureVersion& source,
                                                                 const TopLevelBuildRequest& desc)
        {
            if (!hasAny(source.buildFlags, AccelerationStructureBuildFlag::allowUpdate) ||
                desc.flags != source.buildFlags || requestedInstanceCount(desc.instances) != source.instanceCount)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A TLAS update must preserve the update-enabled source flags and instance count.",
                });
            }
            return {};
        }

        constexpr TopLevelInstanceErrors topLevelUpdateInstanceErrors{
            .invalidInstance =
                "A TLAS update instance contains an invalid transform, packed field, or flag combination.",
            .bottomLevelRequired = "Every TLAS update instance must reference a bottom-level acceleration structure.",
            .unbuiltBottomLevel = "Every referenced BLAS must have a recorded build before the TLAS update.",
        };

        [[nodiscard]] Result<AccelerationStructureExecutionLease> acquireTopLevelUpdateLease(
            const UpdateTarget& target,
            std::uint64_t recordingId,
            const std::vector<std::shared_ptr<AccelerationStructureExecutionTracker>>& bottomLevelTrackers,
            bool inPlace)
        {
            std::vector<std::shared_ptr<AccelerationStructureExecutionTracker>> consumerTrackers = bottomLevelTrackers;
            if (!inPlace)
            {
                consumerTrackers.push_back(target.source->executionTracker);
            }
            return acquireProducerLease(
                *target.destination,
                recordingId,
                consumerTrackers,
                inPlace ? AccelerationStructureWrite::updateInPlace : AccelerationStructureWrite::publishVersion,
                "A TLAS update cannot overwrite an acceleration structure with active GPU execution.");
        }

        [[nodiscard]] std::shared_ptr<AccelerationStructureBuildRetention> makeTopLevelUpdateRetention(
            const UpdateTarget& target,
            const std::shared_ptr<AccelerationStructureVersion>& sourceVersion,
            const PreparedTopLevelInstances& instances)
        {
            auto retention = std::make_shared<AccelerationStructureBuildRetention>();
            retention->owners.reserve(instances.bottomLevels.size() + 4U);
            retention->owners.push_back(target.destination);
            retention->owners.push_back(target.scratch);
            retention->owners.push_back(sourceVersion);
            retention->residencies.reserve(instances.bottomLevels.size() + sourceVersion->residencies.size() + 3U);
            retention->residencies.push_back(target.destination->storage->residency);
            retention->residencies.push_back(target.scratch->residency);
            retention->residencies.insert(
                retention->residencies.end(), sourceVersion->residencies.begin(), sourceVersion->residencies.end());
            appendInstanceRetention(*retention, instances);
            deduplicateSharedPointers(retention->residencies);
            return retention;
        }

    } // namespace

    Status ResourceManager::recordTopLevelUpdate(QueueManager& queues,
                                                 detail::BackendCommandBuffer& commandBuffer,
                                                 const TopLevelBuildRequest& desc,
                                                 bool inPlace)
    {
        auto validRequest = validateTopLevelUpdateRequest(commandBuffer, desc, inPlace);
        if (!validRequest)
        {
            return validRequest;
        }
        const TopLevelAccelerationStructureBuildInput input{
            .instanceCount = static_cast<std::uint32_t>(requestedInstanceCount(desc.instances)),
            .flags = desc.flags,
        };
        auto sizes = queryTopLevelAccelerationStructureBuildSizes(input);
        if (!sizes)
        {
            return std::unexpected(std::move(sizes.error()));
        }
        const auto source = impl_->accelerationStructure(desc.source);
        const auto destination = impl_->accelerationStructure(desc.destination);
        const auto scratch = impl_->buffer(desc.scratch);
        auto target = resolveUpdateTarget(
            source, destination, scratch, desc.scratchOffset, sizes->updateScratchSize, topLevelUpdateRequirement);
        if (!target)
        {
            return std::unexpected(std::move(target.error()));
        }

        auto lockedTarget = lockAndValidateUpdateTarget(
            queues, commandBuffer, *target, inPlace, sizes->accelerationStructureSize, topLevelUpdateTargetErrors);
        if (!lockedTarget)
        {
            return std::unexpected(std::move(lockedTarget.error()));
        }
        auto compatible = validateTopLevelUpdateCompatibility(*lockedTarget->sourceVersion, desc);
        if (!compatible)
        {
            return compatible;
        }

        auto instances = prepareTopLevelInstances(
            desc.instances,
            [this](AccelerationStructureHandle handle) { return impl_->accelerationStructure(handle); },
            [this](BufferHandle handle) { return impl_->buffer(handle); },
            [this](std::uint64_t size, std::uint64_t alignment) {
                return impl_->allocateUploadSlice(size, alignment, "TLAS update instances");
            },
            topLevelUpdateInstanceErrors);
        if (!instances)
        {
            return std::unexpected(std::move(instances.error()));
        }

        auto commandLists = prepareAccelerationStructureCommandLists(queues, commandBuffer, impl_->enhancedBarriers);
        if (!commandLists)
        {
            return std::unexpected(std::move(commandLists.error()));
        }
        auto executionLease =
            acquireTopLevelUpdateLease(*target, lockedTarget->recordingId, instances->bottomLevelTrackers, inPlace);
        if (!executionLease)
        {
            return std::unexpected(std::move(executionLease.error()));
        }
        const auto retention = makeTopLevelUpdateRetention(*target, lockedTarget->sourceVersion, *instances);
        const auto destinationAddress = storageAddress(*target->destination);
        const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC native{
            .DestAccelerationStructureData = destinationAddress,
            .Inputs = nativeTopLevelInputs(instances->count,
                                           nativeBuildFlags(desc.flags) |
                                               D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE,
                                           instances->address),
            .SourceAccelerationStructureData = lockedTarget->sourceVersion->address,
            .ScratchAccelerationStructureData = target->scratch->resource->GetGPUVirtualAddress() + desc.scratchOffset,
        };
        auto version = makeTopLevelVersion(
            *target->destination, destinationAddress, *instances, desc.flags, lockedTarget->sourceVersion->resultSize);
        auto retained = retainTopLevelDependencies(queues,
                                                   commandBuffer,
                                                   retention,
                                                   *instances,
                                                   std::move(*executionLease),
                                                   publicationOf(target->destination, version));
        if (!retained)
        {
            return retained;
        }

        if (inPlace)
        {
            recordAccelerationStructureInPlaceUpdateBarrier(*commandLists);
        }
        commandLists->rayTracing->BuildRaytracingAccelerationStructure(&native, 0, nullptr);
        recordAccelerationStructureBuildBarrier(*commandLists);
        target->destination->currentVersion = std::move(version);
        test::noteAccelerationStructureRecorded();
        return {};
    }

    namespace
    {

        [[nodiscard]] Status validateCompactionRequest(const detail::BackendCommandBuffer& commandBuffer,
                                                       const AccelerationStructureCompactionDesc& desc)
        {
            if (!isGraphicsOrComputeOutsideRendering(commandBuffer))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Acceleration-structure compaction requires a graphics or compute command buffer outside "
                    "rendering.",
                });
            }
            if (desc.source == desc.destination || desc.compactedSize == 0)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Acceleration-structure compaction requires distinct handles and a non-zero queried size.",
                });
            }
            return {};
        }

        struct CompactionTarget final
        {
            std::shared_ptr<AccelerationStructureRecord> source;
            std::shared_ptr<AccelerationStructureRecord> destination;
        };

        [[nodiscard]] Result<CompactionTarget> resolveCompactionTarget(
            const AccelerationStructureLookupResult& source,
            const AccelerationStructureLookupResult& destination,
            std::uint64_t compactedSize)
        {
            if (!source)
            {
                return std::unexpected(source.error());
            }
            if (!destination)
            {
                return std::unexpected(destination.error());
            }
            if ((*source)->type != (*destination)->type || (*destination)->size != compactedSize)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "A compact destination must match the source type and exact queried compacted-size range.",
                });
            }
            return CompactionTarget{.source = *source, .destination = *destination};
        }

        [[nodiscard]] Status validateCompactionSource(
            const std::shared_ptr<AccelerationStructureVersion>& sourceVersion,
            const AccelerationStructureRecord& destination,
            std::uint64_t compactedSize)
        {
            if (sourceVersion == nullptr || destination.currentVersion != nullptr)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidState,
                    0,
                    "Compaction requires a built source and an unbuilt immutable destination.",
                });
            }
            if (!hasAny(sourceVersion->buildFlags, AccelerationStructureBuildFlag::allowCompaction) ||
                compactedSize > sourceVersion->resultSize)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The source must allow compaction and the queried size cannot exceed its current version size.",
                });
            }
            return {};
        }

        [[nodiscard]] std::shared_ptr<AccelerationStructureBuildRetention> makeCompactionRetention(
            const std::shared_ptr<AccelerationStructureRecord>& destination,
            const std::shared_ptr<AccelerationStructureVersion>& sourceVersion)
        {
            auto retention = std::make_shared<AccelerationStructureBuildRetention>();
            retention->owners.reserve(2);
            retention->owners.push_back(destination);
            retention->owners.push_back(sourceVersion);
            retention->residencies.reserve(sourceVersion->residencies.size() + 1U);
            retention->residencies.push_back(destination->storage->residency);
            retention->residencies.insert(
                retention->residencies.end(), sourceVersion->residencies.begin(), sourceVersion->residencies.end());
            deduplicateSharedPointers(retention->residencies);
            return retention;
        }

        [[nodiscard]] std::shared_ptr<AccelerationStructureVersion> makeCompactedVersion(
            const AccelerationStructureRecord& destination,
            D3D12_GPU_VIRTUAL_ADDRESS address,
            const AccelerationStructureVersion& source,
            std::uint64_t compactedSize)
        {
            auto version = makeVersionInStorage(destination, address);
            version->dependencies = source.dependencies;
            version->dependencyTrackers = source.dependencyTrackers;
            version->geometryLayout = source.geometryLayout;
            version->buildFlags = source.buildFlags;
            version->instanceCount = source.instanceCount;
            version->resultSize = compactedSize;
            for (const auto& dependency : version->dependencies)
            {
                const auto accelerationStructure = std::static_pointer_cast<AccelerationStructureVersion>(dependency);
                version->residencies.insert(version->residencies.end(),
                                            accelerationStructure->residencies.begin(),
                                            accelerationStructure->residencies.end());
            }
            deduplicateSharedPointers(version->residencies);
            return version;
        }

    } // namespace

    Status ResourceManager::compactAccelerationStructure(QueueManager& queues,
                                                         detail::BackendCommandBuffer& commandBuffer,
                                                         const AccelerationStructureCompactionDesc& desc)
    {
        auto validRequest = validateCompactionRequest(commandBuffer, desc);
        if (!validRequest)
        {
            return validRequest;
        }

        const auto source = impl_->accelerationStructure(desc.source);
        const auto destination = impl_->accelerationStructure(desc.destination);
        auto target = resolveCompactionTarget(source, destination, desc.compactedSize);
        if (!target)
        {
            return std::unexpected(std::move(target.error()));
        }

        const std::scoped_lock versionLocks(target->source->mutex, target->destination->mutex);
        const auto sourceVersion = target->source->currentVersion;
        auto validSource = validateCompactionSource(sourceVersion, *target->destination, desc.compactedSize);
        if (!validSource)
        {
            return validSource;
        }

        auto commandLists = prepareAccelerationStructureCommandLists(queues, commandBuffer, impl_->enhancedBarriers);
        if (!commandLists)
        {
            return std::unexpected(std::move(commandLists.error()));
        }
        const std::array compactionConsumers{target->source->executionTracker};
        auto executionLease = acquireProducerLeaseForRecording(
            queues,
            commandBuffer,
            *target->destination,
            compactionConsumers,
            "Compaction cannot overwrite an acceleration structure with active GPU execution.");
        if (!executionLease)
        {
            return std::unexpected(std::move(executionLease.error()));
        }
        const auto retention = makeCompactionRetention(target->destination, sourceVersion);
        const auto destinationAddress = storageAddress(*target->destination);
        auto version =
            makeCompactedVersion(*target->destination, destinationAddress, *sourceVersion, desc.compactedSize);
        auto retained = retainBuildDependencies(
            queues, commandBuffer, retention, std::move(*executionLease), publicationOf(target->destination, version));
        if (!retained)
        {
            return retained;
        }

        commandLists->rayTracing->CopyRaytracingAccelerationStructure(
            destinationAddress, sourceVersion->address, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT);
        recordAccelerationStructureCopyBarrier(*commandLists);
        target->destination->currentVersion = std::move(version);
        test::noteAccelerationStructureRecorded();
        return {};
    }

    Status ResourceManager::serializeAccelerationStructure(QueueManager& queues,
                                                           detail::BackendCommandBuffer& commandBuffer,
                                                           const AccelerationStructureSerializationDesc& desc)
    {
        if (!isGraphicsOrComputeOutsideRendering(commandBuffer))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Acceleration-structure serialization requires a graphics or compute command buffer outside rendering.",
            });
        }
        if (desc.serializedSize < sizeof(D3D12_SERIALIZED_RAYTRACING_ACCELERATION_STRUCTURE_HEADER) ||
            desc.destinationOffset % D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT != 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Acceleration-structure serialization requires a non-empty queried size and a 256-byte-aligned "
                "destination.",
            });
        }

        auto recordingId = queues.recordingId(commandBuffer);
        if (!recordingId)
        {
            return std::unexpected(std::move(recordingId.error()));
        }
        auto source = accelerationStructureSerializationSource(desc.source, *recordingId);
        if (!source)
        {
            return std::unexpected(std::move(source.error()));
        }
        auto destination = impl_->buffer(desc.destination);
        if (!destination)
        {
            return std::unexpected(std::move(destination.error()));
        }
        if ((*destination)->desc.memoryType != MemoryType::deviceLocal ||
            !hasAny((*destination)->desc.usage, BufferUsage::storage) ||
            !hasAny((*destination)->desc.usage, BufferUsage::copySource) ||
            !validRange((*destination)->desc.size, desc.destinationOffset, desc.serializedSize))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Serialized acceleration structures require an in-bounds device-local storage and copy-source range.",
            });
        }

        auto commandLists = prepareAccelerationStructureCommandLists(queues, commandBuffer, impl_->enhancedBarriers);
        if (!commandLists)
        {
            return std::unexpected(std::move(commandLists.error()));
        }
        auto retention = std::make_shared<AccelerationStructureBuildRetention>();
        retention->owners.reserve(2);
        retention->owners.push_back(source->owner);
        retention->owners.push_back(*destination);
        retention->residencies = source->residencies;
        if ((*destination)->residency != nullptr)
        {
            retention->residencies.push_back((*destination)->residency);
        }
        auto retained = retainBuildDependencies(queues, commandBuffer, retention, std::move(source->lease), {});
        if (!retained)
        {
            return retained;
        }

        commandLists->rayTracing->CopyRaytracingAccelerationStructure(
            (*destination)->resource->GetGPUVirtualAddress() + desc.destinationOffset,
            source->address,
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_SERIALIZE);
        recordAccelerationStructureCopyBarrier(*commandLists);
        test::noteAccelerationStructureRecorded();
        return {};
    }

    Result<AccelerationStructureArchiveInfo> ResourceManager::inspectAccelerationStructureArchive(
        std::span<const std::byte> archive) const
    {
        if (!impl_->rayTracing || impl_->rayTracingDevice == nullptr)
        {
            return std::unexpected(
                Error{ErrorCode::unsupported, 0, "Acceleration-structure archive inspection requires ray tracing."});
        }
        if (archive.size() < sizeof(D3D12_SERIALIZED_RAYTRACING_ACCELERATION_STRUCTURE_HEADER))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The acceleration-structure archive is smaller than its native header.",
            });
        }

        D3D12_SERIALIZED_RAYTRACING_ACCELERATION_STRUCTURE_HEADER header{};
        std::memcpy(&header, archive.data(), sizeof(header));
        if (header.SerializedSizeInBytesIncludingHeader != archive.size() || header.DeserializedSizeInBytes == 0)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The acceleration-structure archive has inconsistent serialized or deserialized sizes.",
            });
        }
        const auto pointerBytesAvailable = archive.size() - sizeof(header);
        if (header.NumBottomLevelAccelerationStructurePointersAfterHeader >
            pointerBytesAvailable / sizeof(D3D12_GPU_VIRTUAL_ADDRESS))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "The acceleration-structure archive pointer relocation table exceeds the archive.",
            });
        }

        AccelerationStructureArchiveCompatibility compatibility{};
        switch (impl_->rayTracingDevice->CheckDriverMatchingIdentifier(
            D3D12_SERIALIZED_DATA_RAYTRACING_ACCELERATION_STRUCTURE, &header.DriverMatchingIdentifier))
        {
        case D3D12_DRIVER_MATCHING_IDENTIFIER_COMPATIBLE_WITH_DEVICE:
            compatibility = AccelerationStructureArchiveCompatibility::compatible;
            break;
        case D3D12_DRIVER_MATCHING_IDENTIFIER_UNSUPPORTED_TYPE:
            compatibility = AccelerationStructureArchiveCompatibility::unsupportedType;
            break;
        case D3D12_DRIVER_MATCHING_IDENTIFIER_UNRECOGNIZED:
            compatibility = AccelerationStructureArchiveCompatibility::unrecognized;
            break;
        case D3D12_DRIVER_MATCHING_IDENTIFIER_INCOMPATIBLE_VERSION:
            compatibility = AccelerationStructureArchiveCompatibility::incompatibleVersion;
            break;
        case D3D12_DRIVER_MATCHING_IDENTIFIER_INCOMPATIBLE_TYPE:
            compatibility = AccelerationStructureArchiveCompatibility::incompatibleType;
            break;
        default:
            return std::unexpected(Error{
                ErrorCode::nativeFailure,
                0,
                "D3D12 returned an unknown acceleration-structure archive compatibility status.",
            });
        }
        return AccelerationStructureArchiveInfo{
            .compatibility = compatibility,
            .serializedSize = header.SerializedSizeInBytesIncludingHeader,
            .deserializedSize = header.DeserializedSizeInBytes,
            .bottomLevelPointerCount = header.NumBottomLevelAccelerationStructurePointersAfterHeader,
        };
    }

    namespace
    {

        [[nodiscard]] Status validateDeserializableArchive(const AccelerationStructureArchiveInfo& archive,
                                                           std::size_t relocationCount)
        {
            if (archive.compatibility != AccelerationStructureArchiveCompatibility::compatible)
            {
                return std::unexpected(Error{
                    ErrorCode::unsupported,
                    0,
                    "The acceleration-structure archive is incompatible with the active device and driver.",
                });
            }
            if (relocationCount != archive.bottomLevelPointerCount)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "The acceleration-structure archive requires an exact ordered BLAS relocation table.",
                });
            }
            return {};
        }

        [[nodiscard]] Result<std::shared_ptr<AccelerationStructureRecord>> resolveDeserializationDestination(
            const AccelerationStructureLookupResult& destination,
            const AccelerationStructureArchiveInfo& archive,
            bool hasRelocations)
        {
            if (!destination)
            {
                return std::unexpected(destination.error());
            }
            if ((*destination)->size != archive.deserializedSize ||
                ((*destination)->type == AccelerationStructureType::bottomLevel && hasRelocations))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Deserialization requires the exact archive size, destination type, and relocation-table shape.",
                });
            }
            return *destination;
        }

        struct RelocationTargets final
        {
            std::vector<std::shared_ptr<AccelerationStructureVersion>> bottomLevels;
            std::vector<std::shared_ptr<AccelerationStructureExecutionTracker>> bottomLevelTrackers;
        };

        [[nodiscard]] Result<RelocationTargets> resolveRelocationTargets(
            std::span<const AccelerationStructureHandle> handles,
            const AccelerationStructureLookup& findAccelerationStructure,
            AccelerationStructureType destinationType)
        {
            RelocationTargets targets;
            targets.bottomLevels.reserve(handles.size());
            targets.bottomLevelTrackers.reserve(handles.size());
            for (const auto handle : handles)
            {
                auto bottomLevel = findAccelerationStructure(handle);
                if (!bottomLevel)
                {
                    return std::unexpected(std::move(bottomLevel.error()));
                }
                if (destinationType != AccelerationStructureType::topLevel ||
                    (*bottomLevel)->type != AccelerationStructureType::bottomLevel)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidArgument,
                        0,
                        "Only a TLAS archive may relocate live bottom-level acceleration structures.",
                    });
                }
                auto version = snapshotCurrentVersion(**bottomLevel);
                if (version == nullptr)
                {
                    return std::unexpected(Error{
                        ErrorCode::invalidState,
                        0,
                        "Every deserialized TLAS relocation target must be a built BLAS.",
                    });
                }
                targets.bottomLevels.push_back(std::move(version));
                targets.bottomLevelTrackers.push_back((*bottomLevel)->executionTracker);
            }
            return targets;
        }

        void writeRelocatedArchive(std::byte* mappedBytes,
                                   std::span<const std::byte> archive,
                                   std::span<const std::shared_ptr<AccelerationStructureVersion>> bottomLevels)
        {
            std::memcpy(mappedBytes, archive.data(), archive.size());
            for (std::size_t index = 0; index < bottomLevels.size(); ++index)
            {
                std::memcpy(mappedBytes + sizeof(D3D12_SERIALIZED_RAYTRACING_ACCELERATION_STRUCTURE_HEADER) +
                                index * sizeof(D3D12_GPU_VIRTUAL_ADDRESS),
                            &bottomLevels[index]->address,
                            sizeof(D3D12_GPU_VIRTUAL_ADDRESS));
            }
        }

        [[nodiscard]] std::shared_ptr<AccelerationStructureVersion> makeDeserializedVersion(
            const AccelerationStructureRecord& destination,
            D3D12_GPU_VIRTUAL_ADDRESS address,
            const RelocationTargets& targets,
            std::uint64_t deserializedSize)
        {
            auto version = makeVersionInStorage(destination, address);
            version->dependencies.insert(
                version->dependencies.end(), targets.bottomLevels.begin(), targets.bottomLevels.end());
            version->dependencyTrackers = targets.bottomLevelTrackers;
            version->resultSize = deserializedSize;
            appendBottomLevelResidencies(version->residencies, targets.bottomLevels);
            deduplicateSharedPointers(version->residencies);
            return version;
        }

        [[nodiscard]] std::shared_ptr<AccelerationStructureBuildRetention> makeDeserializationRetention(
            const std::shared_ptr<AccelerationStructureRecord>& destination,
            const std::shared_ptr<void>& archiveUpload,
            std::span<const std::shared_ptr<AccelerationStructureVersion>> bottomLevels)
        {
            auto retention = std::make_shared<AccelerationStructureBuildRetention>();
            retention->owners.reserve(bottomLevels.size() + 2U);
            retention->owners.push_back(destination);
            retention->owners.push_back(archiveUpload);
            retention->owners.insert(retention->owners.end(), bottomLevels.begin(), bottomLevels.end());
            if (destination->storage->residency != nullptr)
            {
                retention->residencies.push_back(destination->storage->residency);
            }
            appendBottomLevelResidencies(retention->residencies, bottomLevels);
            deduplicateSharedPointers(retention->residencies);
            return retention;
        }

    } // namespace

    Status ResourceManager::deserializeAccelerationStructure(QueueManager& queues,
                                                             detail::BackendCommandBuffer& commandBuffer,
                                                             const AccelerationStructureDeserializationDesc& desc)
    {
        if (!isGraphicsOrComputeOutsideRendering(commandBuffer))
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Acceleration-structure deserialization requires a graphics or compute command buffer outside "
                "rendering.",
            });
        }
        auto archiveInfo = inspectAccelerationStructureArchive(desc.archive);
        if (!archiveInfo)
        {
            return std::unexpected(std::move(archiveInfo.error()));
        }
        auto validArchive = validateDeserializableArchive(*archiveInfo, desc.bottomLevels.size());
        if (!validArchive)
        {
            return validArchive;
        }

        auto destination = resolveDeserializationDestination(
            impl_->accelerationStructure(desc.destination), *archiveInfo, !desc.bottomLevels.empty());
        if (!destination)
        {
            return std::unexpected(std::move(destination.error()));
        }

        const std::unique_lock destinationLock((*destination)->mutex);
        if ((*destination)->currentVersion != nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Acceleration structures are immutable after their first recorded build or deserialization.",
            });
        }

        auto relocation = resolveRelocationTargets(
            desc.bottomLevels,
            [this](AccelerationStructureHandle handle) { return impl_->accelerationStructure(handle); },
            (*destination)->type);
        if (!relocation)
        {
            return std::unexpected(std::move(relocation.error()));
        }
        auto archiveUpload = impl_->allocateUploadSlice(archiveInfo->serializedSize,
                                                        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT,
                                                        "acceleration-structure archive");
        if (!archiveUpload)
        {
            return std::unexpected(std::move(archiveUpload.error()));
        }
        writeRelocatedArchive(archiveUpload->mappedData, desc.archive, relocation->bottomLevels);
        deduplicateSharedPointers(relocation->bottomLevels);
        deduplicateSharedPointers(relocation->bottomLevelTrackers);

        auto commandLists = prepareAccelerationStructureCommandLists(queues, commandBuffer, impl_->enhancedBarriers);
        if (!commandLists)
        {
            return std::unexpected(std::move(commandLists.error()));
        }
        auto executionLease = acquireProducerLeaseForRecording(
            queues,
            commandBuffer,
            **destination,
            relocation->bottomLevelTrackers,
            "Deserialization cannot overwrite an acceleration structure with active GPU execution.");
        if (!executionLease)
        {
            return std::unexpected(std::move(executionLease.error()));
        }

        const auto destinationAddress = storageAddress(**destination);
        auto version =
            makeDeserializedVersion(**destination, destinationAddress, *relocation, archiveInfo->deserializedSize);
        const auto retention =
            makeDeserializationRetention(*destination, archiveUpload->owner, relocation->bottomLevels);
        auto retained = retainBuildDependencies(queues,
                                                commandBuffer,
                                                retention,
                                                archiveUpload->owner,
                                                std::move(*executionLease),
                                                publicationOf(*destination, version));
        if (!retained)
        {
            return retained;
        }

        commandLists->rayTracing->CopyRaytracingAccelerationStructure(
            destinationAddress,
            archiveUpload->gpuVirtualAddress,
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_DESERIALIZE);
        recordAccelerationStructureCopyBarrier(*commandLists);
        (*destination)->currentVersion = std::move(version);
        test::noteAccelerationStructureRecorded();
        return {};
    }

    namespace
    {

        constexpr std::uint64_t portableDispatchArgumentSize = sizeof(DispatchIndirectArguments);
        constexpr std::uint64_t nativeDispatchArgumentAlignment = 16;

        [[nodiscard]] Status validateIndirectDispatchSources(const BufferBindingSource& arguments,
                                                             std::uint64_t argumentOffset,
                                                             std::span<const BufferBindingSource* const> shaderTables)
        {
            if (!hasAny(arguments.desc.usage, BufferUsage::indirect) ||
                arguments.desc.memoryType == MemoryType::readback || argumentOffset % indirectArgumentAlignment != 0 ||
                argumentOffset > arguments.desc.size ||
                portableDispatchArgumentSize > arguments.desc.size - argumentOffset)
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Indirect ray dispatch requires a live non-readback indirect buffer with an aligned in-bounds "
                    "width/height/depth record.",
                });
            }
            if (std::ranges::any_of(shaderTables, [](const auto* table) {
                    return table == nullptr || table->owner == nullptr || table->resource == nullptr;
                }))
            {
                return std::unexpected(Error{
                    ErrorCode::invalidArgument,
                    0,
                    "Indirect ray dispatch received an invalid prepared shader-table dependency.",
                });
            }
            return {};
        }

        [[nodiscard]] std::shared_ptr<RayTracingIndirectApplicationRetention> makeIndirectDispatchRetention(
            std::span<const BufferBindingSource* const> shaderTables, const BufferBindingSource& arguments)
        {
            auto retention = std::make_shared<RayTracingIndirectApplicationRetention>();
            retention->owners.reserve(shaderTables.size() + 3U);
            retention->residencies.reserve(shaderTables.size() + 1U);
            for (const auto* table : shaderTables)
            {
                retention->owners.push_back(table->owner);
                if (table->residency != nullptr)
                {
                    retention->residencies.push_back(table->residency);
                }
            }
            retention->owners.push_back(arguments.owner);
            if (arguments.residency != nullptr)
            {
                retention->residencies.push_back(arguments.residency);
            }
            return retention;
        }

        struct BufferBarrierState final
        {
            D3D12_BARRIER_SYNC sync = D3D12_BARRIER_SYNC_NONE;
            D3D12_BARRIER_ACCESS access = D3D12_BARRIER_ACCESS_NO_ACCESS;
            D3D12_RESOURCE_STATES legacyState = D3D12_RESOURCE_STATE_COMMON;
        };

        constexpr BufferBarrierState copyDestinationState{
            .sync = D3D12_BARRIER_SYNC_COPY,
            .access = D3D12_BARRIER_ACCESS_COPY_DEST,
            .legacyState = D3D12_RESOURCE_STATE_COPY_DEST,
        };

        constexpr BufferBarrierState copySourceState{
            .sync = D3D12_BARRIER_SYNC_COPY,
            .access = D3D12_BARRIER_ACCESS_COPY_SOURCE,
            .legacyState = D3D12_RESOURCE_STATE_COPY_SOURCE,
        };

        constexpr BufferBarrierState indirectArgumentState{
            .sync = D3D12_BARRIER_SYNC_EXECUTE_INDIRECT,
            .access = D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT,
            .legacyState = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        };

        constexpr BufferBarrierState callerIndirectArgumentState{
            .sync = D3D12_BARRIER_SYNC_ALL,
            .access = D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT,
            .legacyState = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        };

        struct BufferTransition final
        {
            ID3D12Resource* resource = nullptr;
            std::uint64_t size = 0;
            BufferBarrierState before;
            BufferBarrierState after;
        };

        [[nodiscard]] D3D12_BUFFER_BARRIER enhancedBufferBarrier(const BufferTransition& transition) noexcept
        {
            return D3D12_BUFFER_BARRIER{
                .SyncBefore = transition.before.sync,
                .SyncAfter = transition.after.sync,
                .AccessBefore = transition.before.access,
                .AccessAfter = transition.after.access,
                .pResource = transition.resource,
                .Offset = 0,
                .Size = transition.size,
            };
        }

        [[nodiscard]] D3D12_RESOURCE_BARRIER legacyBufferBarrier(const BufferTransition& transition) noexcept
        {
            return D3D12_RESOURCE_BARRIER{
                .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
                .Transition =
                    {
                        .pResource = transition.resource,
                        .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        .StateBefore = transition.before.legacyState,
                        .StateAfter = transition.after.legacyState,
                    },
            };
        }

        template<std::size_t Count>
        void recordBufferTransitions(const AccelerationStructureCommandLists& commandLists,
                                     const std::array<BufferTransition, Count>& transitions) noexcept
        {
            if (commandLists.enhanced != nullptr)
            {
                std::array<D3D12_BUFFER_BARRIER, Count> barriers{};
                std::ranges::transform(transitions, barriers.begin(), enhancedBufferBarrier);
                const D3D12_BARRIER_GROUP group{
                    .Type = D3D12_BARRIER_TYPE_BUFFER,
                    .NumBarriers = static_cast<UINT32>(Count),
                    .pBufferBarriers = barriers.data(),
                };
                commandLists.enhanced->Barrier(1, &group);
            }
            else
            {
                std::array<D3D12_RESOURCE_BARRIER, Count> barriers{};
                std::ranges::transform(transitions, barriers.begin(), legacyBufferBarrier);
                commandLists.rayTracing->ResourceBarrier(static_cast<UINT>(Count), barriers.data());
            }
        }

        void recordNativeDispatchArguments(const AccelerationStructureCommandLists& commandLists,
                                           const IndirectArgumentSlice& nativeArguments,
                                           const UploadSlice& nativeTemplateUpload,
                                           const BufferBindingSource& arguments,
                                           std::uint64_t argumentOffset)
        {
            constexpr std::uint64_t nativeSize = sizeof(D3D12_DISPATCH_RAYS_DESC);
            const BufferTransition nativeArgumentsToCopyDestination{
                .resource = nativeArguments.resource,
                .size = nativeArguments.resourceSize,
                .before = indirectArgumentState,
                .after = copyDestinationState,
            };
            const BufferTransition argumentsToCopySource{
                .resource = arguments.resource,
                .size = arguments.desc.size,
                .before = callerIndirectArgumentState,
                .after = copySourceState,
            };
            const BufferTransition argumentsToIndirect{
                .resource = arguments.resource,
                .size = arguments.desc.size,
                .before = copySourceState,
                .after = indirectArgumentState,
            };
            const BufferTransition nativeArgumentsToIndirect{
                .resource = nativeArguments.resource,
                .size = nativeArguments.resourceSize,
                .before = copyDestinationState,
                .after = indirectArgumentState,
            };
            const bool synchronizeArguments =
                commandLists.enhanced != nullptr && arguments.desc.memoryType == MemoryType::deviceLocal;

            if (synchronizeArguments)
            {
                recordBufferTransitions(commandLists,
                                        std::array{nativeArgumentsToCopyDestination, argumentsToCopySource});
            }
            else
            {
                recordBufferTransitions(commandLists, std::array{nativeArgumentsToCopyDestination});
            }
            commandLists.rayTracing->CopyBufferRegion(nativeArguments.resource,
                                                      nativeArguments.offset,
                                                      nativeTemplateUpload.resource,
                                                      nativeTemplateUpload.offset,
                                                      nativeSize);
            commandLists.rayTracing->CopyBufferRegion(nativeArguments.resource,
                                                      nativeArguments.offset +
                                                          offsetof(D3D12_DISPATCH_RAYS_DESC, Width),
                                                      arguments.resource,
                                                      argumentOffset,
                                                      portableDispatchArgumentSize);
            if (synchronizeArguments)
            {
                recordBufferTransitions(commandLists, std::array{nativeArgumentsToIndirect, argumentsToIndirect});
            }
            else
            {
                recordBufferTransitions(commandLists, std::array{nativeArgumentsToIndirect});
            }
        }

    } // namespace

    Result<RayTracingIndirectDispatchSource> ResourceManager::recordRayTracingIndirectDispatchArguments(
        QueueManager& queues,
        detail::BackendCommandBuffer& commandBuffer,
        const D3D12_DISPATCH_RAYS_DESC& nativeTemplate,
        BufferHandle argumentBuffer,
        std::uint64_t argumentOffset,
        std::span<const BufferBindingSource* const> shaderTables)
    {
        auto arguments = bufferBindingSource(argumentBuffer);
        if (!arguments)
        {
            return std::unexpected(std::move(arguments.error()));
        }
        auto validSources = validateIndirectDispatchSources(*arguments, argumentOffset, shaderTables);
        if (!validSources)
        {
            return std::unexpected(std::move(validSources.error()));
        }

        auto nativeTemplateUpload =
            impl_->allocateUploadSlice(sizeof(D3D12_DISPATCH_RAYS_DESC), 16U, "indirect ray upload");
        if (!nativeTemplateUpload)
        {
            return std::unexpected(std::move(nativeTemplateUpload.error()));
        }
        auto nativeArguments = impl_->allocateIndirectArgumentSlice(commandBuffer.queueType(),
                                                                    sizeof(D3D12_DISPATCH_RAYS_DESC),
                                                                    nativeDispatchArgumentAlignment,
                                                                    "indirect ray arguments");
        if (!nativeArguments)
        {
            return std::unexpected(std::move(nativeArguments.error()));
        }
        std::memcpy(nativeTemplateUpload->mappedData, &nativeTemplate, sizeof(nativeTemplate));
        auto commandLists = prepareAccelerationStructureCommandLists(queues, commandBuffer, impl_->enhancedBarriers);
        if (!commandLists)
        {
            return std::unexpected(std::move(commandLists.error()));
        }

        const auto retention = makeIndirectDispatchRetention(shaderTables, *arguments);
        const std::array dependencies{
            QueueManager::ResourceRetention{
                .owner = retention,
                .residencySet = retention->residencies,
            },
            QueueManager::ResourceRetention{
                .owner = nativeTemplateUpload->owner,
                .internal = true,
            },
            QueueManager::ResourceRetention{
                .owner = nativeArguments->owner,
                .internal = true,
            },
        };
        auto retained = queues.retainCommandDependencies(commandBuffer, dependencies, {});
        if (!retained)
        {
            return std::unexpected(std::move(retained.error()));
        }

        recordNativeDispatchArguments(
            *commandLists, *nativeArguments, *nativeTemplateUpload, *arguments, argumentOffset);
        return RayTracingIndirectDispatchSource{
            .argumentResource = nativeArguments->resource,
            .argumentOffset = nativeArguments->offset,
        };
    }

    Result<AccelerationStructureBindingSource> ResourceManager::accelerationStructureBindingSource(
        AccelerationStructureHandle accelerationStructure)
    {
        auto found = impl_->accelerationStructure(accelerationStructure);
        if (!found)
        {
            return std::unexpected(std::move(found.error()));
        }
        if ((*found)->type != AccelerationStructureType::topLevel)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Only a top-level acceleration structure can be bound for shader traversal.",
            });
        }
        const auto version = snapshotCurrentVersion(**found);
        if (version == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "An acceleration structure must have a recorded build before it can be bound.",
            });
        }
        return AccelerationStructureBindingSource{
            .record = *found,
            .address = version->address,
        };
    }

    Result<AccelerationStructureBindingSnapshot> ResourceManager::captureAccelerationStructureBinding(
        const AccelerationStructureBindingSource& source, std::uint64_t recordingId)
    {
        const auto record = std::static_pointer_cast<AccelerationStructureRecord>(source.record);
        if (record == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A bound acceleration-structure descriptor is missing its backing record.",
            });
        }
        const std::scoped_lock lock(record->mutex);
        const auto version = record->currentVersion;
        if (version == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A bound acceleration-structure descriptor has no built version.",
            });
        }
        std::vector<std::shared_ptr<AccelerationStructureExecutionTracker>> trackers;
        trackers.reserve(version->dependencyTrackers.size() + 1U);
        const auto lessByAddress = [](const std::shared_ptr<AccelerationStructureExecutionTracker>& lhs,
                                      const std::shared_ptr<AccelerationStructureExecutionTracker>& rhs) noexcept {
            return std::less<const AccelerationStructureExecutionTracker*>{}(lhs.get(), rhs.get());
        };
        const auto insertionPoint =
            std::ranges::lower_bound(version->dependencyTrackers, record->executionTracker, lessByAddress);
        trackers.insert(trackers.end(), version->dependencyTrackers.begin(), insertionPoint);
        if (record->executionTracker != nullptr &&
            (insertionPoint == version->dependencyTrackers.end() || *insertionPoint != record->executionTracker))
        {
            trackers.push_back(record->executionTracker);
        }
        trackers.insert(trackers.end(), insertionPoint, version->dependencyTrackers.end());
        auto lease = AccelerationStructureExecutionLease::tryAcquireConsumers(trackers, recordingId);
        if (!lease)
        {
            return std::unexpected(Error{ErrorCode::invalidState, 0, std::string(inPlaceUpdateConsumerConflict)});
        }
        return AccelerationStructureBindingSnapshot{
            .owner = version,
            .residencies = version->residencies,
            .lease = std::move(*lease),
        };
    }

    Result<AccelerationStructureCompactionSource> ResourceManager::accelerationStructureCompactionSource(
        AccelerationStructureHandle accelerationStructure, std::uint64_t recordingId)
    {
        auto found = impl_->accelerationStructure(accelerationStructure);
        if (!found)
        {
            return std::unexpected(std::move(found.error()));
        }
        const std::scoped_lock lock((*found)->mutex);
        const auto version = (*found)->currentVersion;
        if (version == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "A compacted-size query requires a previously built acceleration structure.",
            });
        }
        if (!hasAny(version->buildFlags, AccelerationStructureBuildFlag::allowCompaction))
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "A compacted-size query requires an acceleration structure built with allowCompaction.",
            });
        }
        auto lease = acquireConsumerLease((*found)->executionTracker, recordingId);
        if (!lease)
        {
            return std::unexpected(std::move(lease.error()));
        }
        return AccelerationStructureCompactionSource{
            .owner = version,
            .residencies = version->residencies,
            .address = version->address,
            .lease = std::move(*lease),
        };
    }

    Result<AccelerationStructureSerializationSource> ResourceManager::accelerationStructureSerializationSource(
        AccelerationStructureHandle accelerationStructure, std::uint64_t recordingId)
    {
        auto found = impl_->accelerationStructure(accelerationStructure);
        if (!found)
        {
            return std::unexpected(std::move(found.error()));
        }
        const std::scoped_lock lock((*found)->mutex);
        const auto version = (*found)->currentVersion;
        if (version == nullptr)
        {
            return std::unexpected(Error{
                ErrorCode::invalidState,
                0,
                "Serialization requires a previously built acceleration structure.",
            });
        }
        auto lease = acquireConsumerLease((*found)->executionTracker, recordingId);
        if (!lease)
        {
            return std::unexpected(std::move(lease.error()));
        }
        return AccelerationStructureSerializationSource{
            .owner = version,
            .residencies = version->residencies,
            .address = version->address,
            .lease = std::move(*lease),
        };
    }

    Result<AccelerationStructureInstanceLayout> ResourceManager::queryAccelerationStructureInstanceLayout() const
    {
        if (!impl_->rayTracing)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Acceleration-structure instance records require ray-tracing support.",
            });
        }
        return AccelerationStructureInstanceLayout{
            .format = AccelerationStructureInstanceFormat::packed64,
            .size = sizeof(D3D12_RAYTRACING_INSTANCE_DESC),
            .alignment = D3D12_RAYTRACING_INSTANCE_DESCS_BYTE_ALIGNMENT,
        };
    }

    Result<std::uint64_t> ResourceManager::queryAccelerationStructureReference(
        AccelerationStructureHandle bottomLevel) const
    {
        if (!impl_->rayTracing)
        {
            return std::unexpected(Error{
                ErrorCode::unsupported,
                0,
                "Acceleration-structure references require ray-tracing support.",
            });
        }
        auto found = impl_->accelerationStructure(bottomLevel);
        if (!found)
        {
            return std::unexpected(std::move(found.error()));
        }
        if ((*found)->type != AccelerationStructureType::bottomLevel)
        {
            return std::unexpected(Error{
                ErrorCode::invalidArgument,
                0,
                "Only a bottom-level acceleration structure has a reference that instance records can name.",
            });
        }
        return storageAddress(**found);
    }

} // namespace rachis::d3d12
