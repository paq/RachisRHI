#include "D3D12/D3D12TestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <d3d12.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <span>
#include <vector>

namespace
{

    class FailResourceRetention final
    {
    public:
        explicit FailResourceRetention(std::int32_t countdown)
            : previous_(rachis::d3d12::test::resourceRetentionFailureCountdown().exchange(countdown,
                                                                                          std::memory_order_relaxed))
        {
        }

        ~FailResourceRetention()
        {
            rachis::d3d12::test::resourceRetentionFailureCountdown().store(previous_, std::memory_order_relaxed);
        }

        FailResourceRetention(const FailResourceRetention&) = delete;
        FailResourceRetention& operator=(const FailResourceRetention&) = delete;
        FailResourceRetention(FailResourceRetention&&) = delete;
        FailResourceRetention& operator=(FailResourceRetention&&) = delete;

    private:
        std::int32_t previous_ = 0;
    };

    void printDiagnostic(const rachis::DiagnosticMessage& message, void*) noexcept
    {
        static_cast<void>(std::fprintf(stderr, "D3D12 [%u]: ", static_cast<unsigned int>(message.nativeId)));
        if (!message.description.empty())
        {
            static_cast<void>(std::fwrite(message.description.data(), 1, message.description.size(), stderr));
        }
        static_cast<void>(std::fputc('\n', stderr));
    }

    void skipWithoutHardwareAdapter(const rachis::Error& error)
    {
        const auto classification = rachis::test::classifyDeviceCreateError(error.code);
        if (classification == rachis::test::DeviceCreateClassification::adapterNotFound ||
            classification == rachis::test::DeviceCreateClassification::backendUnavailable)
        {
            SKIP("No D3D12 hardware adapter is available: " << error.message);
        }
        FAIL("Creating the D3D12 hardware device failed: " << error.message);
    }

    [[nodiscard]] std::vector<std::byte> loadShader()
    {
        std::ifstream stream(RACHIS_TEST_RAY_QUERY_SHADER_PATH, std::ios::binary | std::ios::ate);
        const auto end = stream.tellg();
        if (end <= 0)
        {
            return {};
        }
        std::vector<std::byte> result(static_cast<std::size_t>(end));
        stream.seekg(0, std::ios::beg);
        stream.read(reinterpret_cast<char*>(result.data()), end);
        return stream ? result : std::vector<std::byte>{};
    }

    constexpr auto allShadingSync = static_cast<std::uint32_t>(D3D12_BARRIER_SYNC_ALL_SHADING);
    constexpr auto buildSync = static_cast<std::uint32_t>(D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE);
    constexpr auto accelerationStructureCopySync =
        static_cast<std::uint32_t>(D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE);
    constexpr auto postbuildInfoSync =
        static_cast<std::uint32_t>(D3D12_BARRIER_SYNC_EMIT_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO);
    constexpr auto copySync = static_cast<std::uint32_t>(D3D12_BARRIER_SYNC_COPY);
    constexpr auto accelerationStructureWriteAccess =
        static_cast<std::uint32_t>(D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE);
    constexpr auto accelerationStructureReadAccess =
        static_cast<std::uint32_t>(D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ);
    constexpr auto unorderedAccess = static_cast<std::uint32_t>(D3D12_BARRIER_ACCESS_UNORDERED_ACCESS);
    constexpr auto shaderResourceAccess = static_cast<std::uint32_t>(D3D12_BARRIER_ACCESS_SHADER_RESOURCE);
    constexpr auto copySourceAccess = static_cast<std::uint32_t>(D3D12_BARRIER_ACCESS_COPY_SOURCE);

    [[nodiscard]] rachis::d3d12::test::GlobalBarrierSnapshot takeAccelerationStructureBarrier()
    {
        auto& recorded = rachis::d3d12::test::lastAccelerationStructureGlobalBarrier();
        REQUIRE(recorded.has_value());
        const auto barrier = *recorded;
        recorded.reset();
        return barrier;
    }

    void checkAccelerationStructureBuildBarrier(const char* operation)
    {
        INFO(operation);
        const auto barrier = takeAccelerationStructureBarrier();
        CHECK(barrier.syncBefore == buildSync);
        CHECK(barrier.syncAfter == (allShadingSync | buildSync | accelerationStructureCopySync | postbuildInfoSync));
        CHECK(barrier.accessBefore == (accelerationStructureWriteAccess | unorderedAccess));
        CHECK(barrier.accessAfter == (accelerationStructureReadAccess | unorderedAccess));
    }

    void checkAccelerationStructureInPlaceUpdateBarrier(const char* operation, bool expected)
    {
        INFO(operation);
        auto& recorded = rachis::d3d12::test::lastAccelerationStructureInPlaceUpdateBarrier();
        REQUIRE(recorded.has_value() == expected);
        if (!expected)
        {
            return;
        }
        const auto barrier = *recorded;
        recorded.reset();
        CHECK(barrier.syncBefore == (allShadingSync | buildSync | accelerationStructureCopySync | postbuildInfoSync));
        CHECK(barrier.syncAfter == buildSync);
        CHECK(barrier.accessBefore == accelerationStructureReadAccess);
        CHECK(barrier.accessAfter == accelerationStructureWriteAccess);
        CHECK_FALSE(barrier.legacyUnorderedAccess);
    }

    void checkAccelerationStructureCopyBarrier(const char* operation)
    {
        INFO(operation);
        const auto barrier = takeAccelerationStructureBarrier();
        CHECK(barrier.syncBefore == accelerationStructureCopySync);
        CHECK(barrier.syncAfter ==
              (allShadingSync | buildSync | accelerationStructureCopySync | copySync | postbuildInfoSync));
        CHECK(barrier.accessBefore == (accelerationStructureWriteAccess | unorderedAccess));
        CHECK(barrier.accessAfter ==
              (accelerationStructureReadAccess | unorderedAccess | shaderResourceAccess | copySourceAccess));
    }

    struct BottomLevelBuildTarget final
    {
        rachis::BufferHandle storage = {};
        rachis::BufferHandle scratch = {};
        rachis::AccelerationStructureHandle structure = {};
    };

    struct VertexStrideLayout final
    {
        rachis::AccelerationStructureVertexFormat format = rachis::AccelerationStructureVertexFormat::float32x3;
        std::uint64_t offset = 0;
        std::uint32_t stride = 0;
    };

    [[nodiscard]] BottomLevelBuildTarget createBottomLevelBuildTarget(
        rachis::Device& device, const rachis::AccelerationStructureBuildSizes& sizes)
    {
        auto storage = device.createBuffer({
            .size = sizes.accelerationStructureSize,
            .usage = rachis::BufferUsage::accelerationStructureStorage,
            .memoryType = rachis::MemoryType::deviceLocal,
            .label = "Vertex-stride BLAS storage",
        });
        auto scratch = device.createBuffer({
            .size = sizes.buildScratchSize,
            .usage = rachis::BufferUsage::accelerationStructureScratch,
            .memoryType = rachis::MemoryType::deviceLocal,
            .label = "Vertex-stride BLAS scratch",
        });
        REQUIRE(storage.has_value());
        REQUIRE(scratch.has_value());
        auto structure = device.createAccelerationStructure({
            .type = rachis::AccelerationStructureType::bottomLevel,
            .storage = *storage,
            .label = "Vertex-stride BLAS",
        });
        REQUIRE(structure.has_value());
        return {.storage = *storage, .scratch = *scratch, .structure = *structure};
    }

    void requireTriangleVertexStridesAlignedToComponentSize(rachis::Device& device)
    {
        if (!rachis::test::requireCapabilities(device).rayTracing)
        {
            SKIP("The selected adapter does not expose DXR acceleration structures.");
        }

        constexpr std::uint64_t packedHalfOffset = 64;
        const std::array paddedVertices{
            std::array{-1.0F, -1.0F, 0.0F, 0.0F},
            std::array{0.0F, 1.0F, 0.0F, 0.0F},
            std::array{1.0F, -1.0F, 0.0F, 0.0F},
        };
        const std::array packedHalfVertices{
            std::array<std::uint16_t, 3>{0xBC00U, 0xBC00U, 0U},
            std::array<std::uint16_t, 3>{0x0000U, 0x3C00U, 0U},
            std::array<std::uint16_t, 3>{0x3C00U, 0xBC00U, 0U},
        };
        auto vertexBuffer = device.createBuffer({
            .size = 256,
            .usage = rachis::BufferUsage::accelerationStructureBuildInput,
            .memoryType = rachis::MemoryType::upload,
            .label = "Vertex-stride BLAS vertices",
        });
        REQUIRE(vertexBuffer.has_value());
        REQUIRE(device.writeBuffer(*vertexBuffer, 0, std::as_bytes(std::span(paddedVertices))).has_value());
        REQUIRE(device.writeBuffer(*vertexBuffer, packedHalfOffset, std::as_bytes(std::span(packedHalfVertices)))
                    .has_value());

        const auto triangles = [&](const VertexStrideLayout& layout) {
            return std::array{
                rachis::AccelerationStructureGeometry{
                    .data =
                        rachis::AccelerationStructureTriangleGeometry{
                            .vertexBuffer = *vertexBuffer,
                            .vertexOffset = layout.offset,
                            .vertexCount = 3,
                            .vertexStride = layout.stride,
                            .vertexFormat = layout.format,
                        },
                    .flags = rachis::AccelerationStructureGeometryFlag::opaque,
                },
            };
        };
        constexpr std::array alignedLayouts{
            VertexStrideLayout{rachis::AccelerationStructureVertexFormat::float32x3, 0, 16},
            VertexStrideLayout{rachis::AccelerationStructureVertexFormat::float16x2, packedHalfOffset, 6},
        };
        constexpr std::array misalignedLayouts{
            VertexStrideLayout{rachis::AccelerationStructureVertexFormat::float32x3, 0, 14},
            VertexStrideLayout{rachis::AccelerationStructureVertexFormat::float16x2, packedHalfOffset, 5},
        };

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command.has_value());
        std::vector<BottomLevelBuildTarget> targets;
        std::vector<rachis::AccelerationStructureBuildSizes> alignedSizes;
        for (const auto& layout : alignedLayouts)
        {
            INFO("aligned vertex stride " << layout.stride);
            const auto geometry = triangles(layout);
            const rachis::BottomLevelAccelerationStructureBuildInput input{
                .geometries = geometry,
                .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace,
            };
            auto sizes = device.queryBottomLevelAccelerationStructureBuildSizes(input);
            if (!sizes)
            {
                UNSCOPED_INFO("size query failed: " << sizes.error().message);
            }
            REQUIRE(sizes.has_value());
            alignedSizes.push_back(*sizes);
            targets.push_back(createBottomLevelBuildTarget(device, *sizes));
            const auto built = device.buildBottomLevelAccelerationStructure(*command,
                                                                            {
                                                                                .destination = targets.back().structure,
                                                                                .input = input,
                                                                                .scratch = targets.back().scratch,
                                                                            });
            if (!built)
            {
                UNSCOPED_INFO("build failed: " << built.error().message);
            }
            REQUIRE(built.has_value());
        }
        for (const auto& layout : misalignedLayouts)
        {
            INFO("misaligned vertex stride " << layout.stride);
            const auto geometry = triangles(layout);
            const rachis::BottomLevelAccelerationStructureBuildInput input{
                .geometries = geometry,
                .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace,
            };
            const auto sizes = device.queryBottomLevelAccelerationStructureBuildSizes(input);
            REQUIRE_FALSE(sizes.has_value());
            CHECK(sizes.error().code == rachis::ErrorCode::invalidArgument);
            targets.push_back(createBottomLevelBuildTarget(device, alignedSizes.front()));
            const auto built = device.buildBottomLevelAccelerationStructure(*command,
                                                                            {
                                                                                .destination = targets.back().structure,
                                                                                .input = input,
                                                                                .scratch = targets.back().scratch,
                                                                            });
            REQUIRE_FALSE(built.has_value());
            CHECK(built.error().code == rachis::ErrorCode::invalidArgument);
        }

        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());
        for (const auto& target : targets)
        {
            REQUIRE(device.destroyAccelerationStructure(target.structure).has_value());
            REQUIRE(device.destroyBuffer(target.scratch).has_value());
            REQUIRE(device.destroyBuffer(target.storage).has_value());
        }
        REQUIRE(device.destroyBuffer(*vertexBuffer).has_value());
        device.collectGarbage();

        const auto diagnostics = device.diagnosticCounts();
        CHECK(diagnostics.warnings == 0);
        CHECK(diagnostics.errors == 0);
        CHECK(diagnostics.corruptions == 0);
    }

} // namespace

TEST_CASE("acceleration-structure capability state is coherent on WARP", "[size:medium][rhi][ray-query][warp]")
{
    auto created = rachis::createDevice({
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .enableDred = true,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    const auto capabilities = *device.capabilities();
    if (!capabilities.rayTracing)
    {
        auto storage = device.createBuffer({
            .size = 256,
            .usage = rachis::BufferUsage::accelerationStructureStorage,
            .memoryType = rachis::MemoryType::deviceLocal,
        });
        REQUIRE_FALSE(storage.has_value());
        CHECK(storage.error().code == rachis::ErrorCode::unsupported);
    }
}

TEST_CASE("in-place BLAS updates accept a zero-byte update scratch requirement on WARP",
          "[size:medium][rhi][ray-query][warp]")
{
    auto created = rachis::createDevice({
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    if (!device.capabilities()->rayTracing)
    {
        SKIP("WARP does not expose DXR on this host.");
    }

    const std::array vertices{
        std::array{-1.0F, -1.0F, 0.0F},
        std::array{0.0F, 1.0F, 0.0F},
        std::array{1.0F, -1.0F, 0.0F},
    };
    auto vertexBuffer = device.createBuffer({
        .size = sizeof(vertices),
        .usage = rachis::BufferUsage::accelerationStructureBuildInput,
        .memoryType = rachis::MemoryType::upload,
        .label = "Zero-scratch update vertices",
    });
    REQUIRE(vertexBuffer.has_value());
    REQUIRE(device.writeBuffer(*vertexBuffer, 0, std::as_bytes(std::span(vertices))).has_value());
    const std::array geometry{
        rachis::AccelerationStructureGeometry{
            .data =
                rachis::AccelerationStructureTriangleGeometry{
                    .vertexBuffer = *vertexBuffer,
                    .vertexCount = 3,
                    .vertexStride = sizeof(vertices[0]),
                    .vertexFormat = rachis::AccelerationStructureVertexFormat::float32x3,
                },
            .flags = rachis::AccelerationStructureGeometryFlag::opaque,
        },
    };
    const rachis::BottomLevelAccelerationStructureBuildInput input{
        .geometries = geometry,
        .flags = rachis::AccelerationStructureBuildFlag::allowUpdate,
    };
    auto sizes = device.queryBottomLevelAccelerationStructureBuildSizes(input);
    REQUIRE(sizes.has_value());
    CAPTURE(sizes->buildScratchSize, sizes->updateScratchSize);
    auto storage = device.createBuffer({
        .size = sizes->accelerationStructureSize,
        .usage = rachis::BufferUsage::accelerationStructureStorage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Zero-scratch update BLAS storage",
    });
    auto scratch = device.createBuffer({
        .size = (std::max)(sizes->buildScratchSize, sizes->updateScratchSize),
        .usage = rachis::BufferUsage::accelerationStructureScratch,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Zero-scratch update scratch",
    });
    REQUIRE(storage.has_value());
    REQUIRE(scratch.has_value());
    auto bottomLevel = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::bottomLevel,
        .storage = *storage,
        .label = "Zero-scratch update BLAS",
    });
    REQUIRE(bottomLevel.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    REQUIRE(device
                .buildBottomLevelAccelerationStructure(*command,
                                                       {
                                                           .destination = *bottomLevel,
                                                           .input = input,
                                                           .scratch = *scratch,
                                                       })
                .has_value());
    auto updated = device.updateBottomLevelAccelerationStructureInPlace(*command,
                                                                        {
                                                                            .target = *bottomLevel,
                                                                            .input = input,
                                                                            .scratch = *scratch,
                                                                        });
    if (!updated)
    {
        UNSCOPED_INFO(updated.error().message);
    }
    REQUIRE(updated.has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
    CHECK(device.diagnosticCounts().corruptions == 0);
}

TEST_CASE("BLAS updates from a compacted source need a destination of the input's prebuild size on WARP",
          "[size:medium][rhi][ray-query][warp]")
{
    auto created = rachis::createDevice({
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    if (!device.capabilities()->rayTracing)
    {
        SKIP("WARP does not expose DXR on this host.");
    }

    const std::array vertices{
        std::array{-1.0F, -1.0F, 0.0F},
        std::array{0.0F, 1.0F, 0.0F},
        std::array{1.0F, -1.0F, 0.0F},
    };
    auto vertexBuffer = device.createBuffer({
        .size = sizeof(vertices),
        .usage = rachis::BufferUsage::accelerationStructureBuildInput,
        .memoryType = rachis::MemoryType::upload,
        .label = "Compacted-update vertices",
    });
    REQUIRE(vertexBuffer.has_value());
    REQUIRE(device.writeBuffer(*vertexBuffer, 0, std::as_bytes(std::span(vertices))).has_value());
    const std::array geometry{
        rachis::AccelerationStructureGeometry{
            .data =
                rachis::AccelerationStructureTriangleGeometry{
                    .vertexBuffer = *vertexBuffer,
                    .vertexCount = 3,
                    .vertexStride = sizeof(vertices[0]),
                    .vertexFormat = rachis::AccelerationStructureVertexFormat::float32x3,
                },
            .flags = rachis::AccelerationStructureGeometryFlag::opaque,
        },
    };
    const rachis::BottomLevelAccelerationStructureBuildInput input{
        .geometries = geometry,
        .flags = rachis::AccelerationStructureBuildFlag::allowUpdate |
                 rachis::AccelerationStructureBuildFlag::allowCompaction,
    };
    auto sizes = device.queryBottomLevelAccelerationStructureBuildSizes(input);
    REQUIRE(sizes.has_value());
    const auto createStorage = [&](std::uint64_t size, const char* label) {
        auto storage = device.createBuffer({
            .size = size,
            .usage = rachis::BufferUsage::accelerationStructureStorage,
            .memoryType = rachis::MemoryType::deviceLocal,
            .label = label,
        });
        REQUIRE(storage.has_value());
        auto structure = device.createAccelerationStructure({
            .type = rachis::AccelerationStructureType::bottomLevel,
            .storage = *storage,
            .label = label,
        });
        REQUIRE(structure.has_value());
        return *structure;
    };
    auto scratch = device.createBuffer({
        .size = (std::max)(sizes->buildScratchSize, sizes->updateScratchSize),
        .usage = rachis::BufferUsage::accelerationStructureScratch,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Compacted-update scratch",
    });
    REQUIRE(scratch.has_value());
    const auto built = createStorage(sizes->accelerationStructureSize, "Compacted-update source BLAS");

    auto compactionPool = device.createAccelerationStructureCompactedSizeQueryPool({
        .queueType = rachis::QueueType::compute,
        .queryCount = 1,
        .label = "Compacted-update compacted size",
    });
    auto compactionReadback = device.createBuffer({
        .size = rachis::accelerationStructureCompactedSizeQueryResultSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Compacted-update compacted-size readback",
    });
    REQUIRE(compactionPool.has_value());
    REQUIRE(compactionReadback.has_value());
    auto buildCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(buildCommand.has_value());
    REQUIRE(device
                .buildBottomLevelAccelerationStructure(*buildCommand,
                                                       {
                                                           .destination = built,
                                                           .input = input,
                                                           .scratch = *scratch,
                                                       })
                .has_value());
    REQUIRE(device.writeAccelerationStructureCompactedSize(*buildCommand, *compactionPool, 0, built).has_value());
    REQUIRE(device
                .resolveAccelerationStructureCompactedSizes(*buildCommand,
                                                            {
                                                                .queryPool = *compactionPool,
                                                                .firstQuery = 0,
                                                                .queryCount = 1,
                                                                .destination = *compactionReadback,
                                                            })
                .has_value());
    auto buildSubmitted = device.submit(std::move(*buildCommand));
    REQUIRE(buildSubmitted.has_value());
    REQUIRE(device.wait(*buildSubmitted).has_value());
    std::uint64_t compactedSize = 0;
    REQUIRE(
        device.readBuffer(*compactionReadback, 0, std::as_writable_bytes(std::span(&compactedSize, 1))).has_value());
    CAPTURE(compactedSize, sizes->accelerationStructureSize);
    REQUIRE(compactedSize > 0);
    if (compactedSize >= sizes->accelerationStructureSize)
    {
        SKIP("Compaction does not shrink this BLAS on this WARP version.");
    }

    const auto compacted = createStorage(compactedSize, "Compacted-update compacted BLAS");
    const auto compactedSizeDestination = createStorage(compactedSize, "Compacted-update compacted-size destination");
    const auto prebuildSizeDestination =
        createStorage(sizes->accelerationStructureSize, "Compacted-update prebuild-size destination");
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    REQUIRE(device
                .compactAccelerationStructure(*command,
                                              {
                                                  .source = built,
                                                  .destination = compacted,
                                                  .compactedSize = compactedSize,
                                              })
                .has_value());
    auto undersized = device.updateBottomLevelAccelerationStructure(*command,
                                                                    {
                                                                        .source = compacted,
                                                                        .destination = compactedSizeDestination,
                                                                        .input = input,
                                                                        .scratch = *scratch,
                                                                    });
    REQUIRE_FALSE(undersized.has_value());
    CHECK(undersized.error().code == rachis::ErrorCode::invalidArgument);
    auto undersizedInPlace = device.updateBottomLevelAccelerationStructureInPlace(*command,
                                                                                  {
                                                                                      .target = compacted,
                                                                                      .input = input,
                                                                                      .scratch = *scratch,
                                                                                  });
    REQUIRE_FALSE(undersizedInPlace.has_value());
    CHECK(undersizedInPlace.error().code == rachis::ErrorCode::invalidArgument);
    auto fitted = device.updateBottomLevelAccelerationStructure(*command,
                                                                {
                                                                    .source = compacted,
                                                                    .destination = prebuildSizeDestination,
                                                                    .input = input,
                                                                    .scratch = *scratch,
                                                                });
    if (!fitted)
    {
        UNSCOPED_INFO(fitted.error().message);
    }
    REQUIRE(fitted.has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
    CHECK(device.diagnosticCounts().corruptions == 0);
}

TEST_CASE("resetting an acceleration-structure postbuild query pool orders earlier postbuild writes",
          "[size:medium][rhi][ray-query][warp]")
{
    const bool forceLegacy = GENERATE(false, true);
    CAPTURE(forceLegacy);
    const bool previousLegacy =
        rachis::d3d12::test::forceLegacyBarriers().exchange(forceLegacy, std::memory_order_relaxed);
    auto created = rachis::createDevice({
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    rachis::d3d12::test::forceLegacyBarriers().store(previousLegacy, std::memory_order_relaxed);
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    if (forceLegacy)
    {
        CHECK_FALSE(device.capabilities()->enhancedBarriers);
    }

    auto compactionPool = device.createAccelerationStructureCompactedSizeQueryPool({
        .queueType = rachis::QueueType::graphics,
        .queryCount = 2,
        .label = "Postbuild reset compacted sizes",
    });
    REQUIRE(compactionPool.has_value());
    auto serializationPool = device.createAccelerationStructureSerializationInfoQueryPool({
        .queueType = rachis::QueueType::compute,
        .queryCount = 2,
        .label = "Postbuild reset serialization information",
    });
    REQUIRE(serializationPool.has_value());
    auto timestampPool = device.createTimestampQueryPool({
        .queueType = rachis::QueueType::graphics,
        .queryCount = 2,
        .label = "Postbuild reset timestamps",
    });
    REQUIRE(timestampPool.has_value());
    auto computeTimestampPool = device.createTimestampQueryPool({
        .queueType = rachis::QueueType::compute,
        .queryCount = 1,
        .label = "Postbuild reset compute timestamps",
    });
    REQUIRE(computeTimestampPool.has_value());

    const auto barriersBefore = rachis::d3d12::test::postbuildQueryResetBarrierCount().load(std::memory_order_relaxed);
    auto graphics = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(graphics.has_value());
    REQUIRE(device.resetQueryPool(*graphics, *compactionPool, 0, 2).has_value());
    CHECK(rachis::d3d12::test::postbuildQueryResetBarrierCount().load(std::memory_order_relaxed) == barriersBefore + 1);
    REQUIRE(device.resetQueryPool(*graphics, *compactionPool, 1, 1).has_value());
    CHECK(rachis::d3d12::test::postbuildQueryResetBarrierCount().load(std::memory_order_relaxed) == barriersBefore + 2);
    REQUIRE(device.resetQueryPool(*graphics, *timestampPool, 0, 2).has_value());
    CHECK(rachis::d3d12::test::postbuildQueryResetBarrierCount().load(std::memory_order_relaxed) == barriersBefore + 2);
    REQUIRE(device.writeTimestamp(*graphics, *timestampPool, 0).has_value());
    auto graphicsSubmitted = device.submit(std::move(*graphics));
    REQUIRE(graphicsSubmitted.has_value());

    auto compute = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(compute.has_value());
    REQUIRE(device.resetQueryPool(*compute, *serializationPool, 0, 1).has_value());
    CHECK(rachis::d3d12::test::postbuildQueryResetBarrierCount().load(std::memory_order_relaxed) == barriersBefore + 3);
    REQUIRE(device.resetQueryPool(*compute, *computeTimestampPool, 0, 1).has_value());
    REQUIRE(device.writeTimestamp(*compute, *computeTimestampPool, 0).has_value());
    CHECK(rachis::d3d12::test::postbuildQueryResetBarrierCount().load(std::memory_order_relaxed) == barriersBefore + 3);
    auto computeSubmitted = device.submit(std::move(*compute));
    REQUIRE(computeSubmitted.has_value());

    REQUIRE(device.wait(*graphicsSubmitted).has_value());
    REQUIRE(device.wait(*computeSubmitted).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
    CHECK(device.diagnosticCounts().corruptions == 0);
    REQUIRE(device.destroyQueryPool(*computeTimestampPool).has_value());
    REQUIRE(device.destroyQueryPool(*timestampPool).has_value());
    REQUIRE(device.destroyQueryPool(*serializationPool).has_value());
    REQUIRE(device.destroyQueryPool(*compactionPool).has_value());
}

TEST_CASE("BLAS triangle vertex strides must be multiples of the vertex component size on WARP",
          "[size:medium][rhi][ray-query][warp]")
{
    auto created = rachis::createDevice({
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    requireTriangleVertexStridesAlignedToComponentSize(device);
}

TEST_CASE("BLAS triangle vertex strides must be multiples of the vertex component size",
          "[size:medium][rhi][ray-query][hardware]")
{
    auto created = rachis::createDevice({
        .adapterPreference = rachis::AdapterPreference::highPerformance,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        skipWithoutHardwareAdapter(created.error());
    }
    auto device = std::move(*created);
    requireTriangleVertexStridesAlignedToComponentSize(device);
}

TEST_CASE("inline ray query traverses retained BLAS and TLAS versions", "[size:medium][rhi][ray-query][hardware]")
{
    auto created = rachis::createDevice({
        .adapterPreference = rachis::AdapterPreference::highPerformance,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        skipWithoutHardwareAdapter(created.error());
    }
    auto device = std::move(*created);
    const auto capabilities = *device.capabilities();
    if (!capabilities.rayQuery)
    {
        SKIP("The selected hardware adapter does not expose DXR 1.1 inline ray queries.");
    }
    REQUIRE(capabilities.rayTracing);

    const std::array vertices{
        std::array{-1.0F, -1.0F, 0.0F},
        std::array{0.0F, 1.0F, 0.0F},
        std::array{1.0F, -1.0F, 0.0F},
    };
    auto vertexBuffer = device.createBuffer({
        .size = sizeof(vertices),
        .usage = rachis::BufferUsage::accelerationStructureBuildInput,
        .memoryType = rachis::MemoryType::upload,
        .label = "Ray-query triangle vertices",
    });
    REQUIRE(vertexBuffer.has_value());
    REQUIRE(device.writeBuffer(*vertexBuffer, 0, std::as_bytes(std::span(vertices))).has_value());

    const std::array geometry{
        rachis::AccelerationStructureGeometry{
            .data =
                rachis::AccelerationStructureTriangleGeometry{
                    .vertexBuffer = *vertexBuffer,
                    .vertexCount = 3,
                    .vertexStride = sizeof(vertices[0]),
                    .vertexFormat = rachis::AccelerationStructureVertexFormat::float32x3,
                },
            .flags = rachis::AccelerationStructureGeometryFlag::opaque,
        },
    };
    const rachis::BottomLevelAccelerationStructureBuildInput bottomInput{
        .geometries = geometry,
        .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace |
                 rachis::AccelerationStructureBuildFlag::allowUpdate |
                 rachis::AccelerationStructureBuildFlag::allowCompaction,
    };
    const auto topBuildFlags = rachis::AccelerationStructureBuildFlag::preferFastTrace |
                               rachis::AccelerationStructureBuildFlag::allowUpdate |
                               rachis::AccelerationStructureBuildFlag::allowCompaction;
    CHECK_FALSE(device.queryBottomLevelAccelerationStructureBuildSizes({}).has_value());
    auto bottomSizes = device.queryBottomLevelAccelerationStructureBuildSizes(bottomInput);
    REQUIRE(bottomSizes.has_value());
    auto topSizes = device.queryTopLevelAccelerationStructureBuildSizes({
        .instanceCount = 1,
        .flags = topBuildFlags,
    });
    REQUIRE(topSizes.has_value());
    auto bottomStorage = device.createBuffer({
        .size = bottomSizes->accelerationStructureSize,
        .usage = rachis::BufferUsage::accelerationStructureStorage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Ray-query BLAS storage",
    });
    auto topStorage = device.createBuffer({
        .size = topSizes->accelerationStructureSize,
        .usage = rachis::BufferUsage::accelerationStructureStorage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Ray-query TLAS storage",
    });
    auto updatedBottomStorage = device.createBuffer({
        .size = bottomSizes->accelerationStructureSize,
        .usage = rachis::BufferUsage::accelerationStructureStorage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Updated ray-query BLAS storage",
    });
    auto updatedTopStorage = device.createBuffer({
        .size = topSizes->accelerationStructureSize,
        .usage = rachis::BufferUsage::accelerationStructureStorage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Updated ray-query TLAS storage",
    });
    const auto scratchSize = (std::max)({bottomSizes->buildScratchSize,
                                         bottomSizes->updateScratchSize,
                                         topSizes->buildScratchSize,
                                         topSizes->updateScratchSize});
    auto scratch = device.createBuffer({
        .size = scratchSize,
        .usage = rachis::BufferUsage::accelerationStructureScratch,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Ray-query build scratch",
    });
    REQUIRE(bottomStorage.has_value());
    REQUIRE(topStorage.has_value());
    REQUIRE(updatedBottomStorage.has_value());
    REQUIRE(updatedTopStorage.has_value());
    REQUIRE(scratch.has_value());

    auto bottomLevel = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::bottomLevel,
        .storage = *bottomStorage,
        .label = "Ray-query BLAS",
    });
    auto topLevel = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::topLevel,
        .storage = *topStorage,
        .label = "Ray-query TLAS",
    });
    auto updatedBottomLevel = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::bottomLevel,
        .storage = *updatedBottomStorage,
        .label = "Updated ray-query BLAS",
    });
    auto updatedTopLevel = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::topLevel,
        .storage = *updatedTopStorage,
        .label = "Updated ray-query TLAS",
    });
    if (!bottomLevel)
    {
        UNSCOPED_INFO("BLAS creation failed: " << bottomLevel.error().message);
    }
    if (!topLevel)
    {
        UNSCOPED_INFO("TLAS creation failed: " << topLevel.error().message);
    }
    if (!updatedBottomLevel)
    {
        UNSCOPED_INFO("Updated BLAS creation failed: " << updatedBottomLevel.error().message);
    }
    if (!updatedTopLevel)
    {
        UNSCOPED_INFO("Updated TLAS creation failed: " << updatedTopLevel.error().message);
    }
    REQUIRE(bottomLevel.has_value());
    REQUIRE(topLevel.has_value());
    REQUIRE(updatedBottomLevel.has_value());
    REQUIRE(updatedTopLevel.has_value());
    auto overlappingBottomLevel = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::bottomLevel,
        .storage = *bottomStorage,
        .label = "Overlapping ray-query BLAS",
    });
    REQUIRE_FALSE(overlappingBottomLevel.has_value());
    CHECK(overlappingBottomLevel.error().code == rachis::ErrorCode::invalidArgument);

    auto invalidCompactionPool = device.createAccelerationStructureCompactedSizeQueryPool({
        .queueType = rachis::QueueType::copy,
        .queryCount = 1,
    });
    REQUIRE_FALSE(invalidCompactionPool.has_value());
    CHECK(invalidCompactionPool.error().code == rachis::ErrorCode::invalidArgument);
    auto compactionPool = device.createAccelerationStructureCompactedSizeQueryPool({
        .queueType = rachis::QueueType::compute,
        .queryCount = 2,
        .label = "Ray-query compacted sizes",
    });
    auto compactionReadback = device.createBuffer({
        .size = 2 * rachis::accelerationStructureCompactedSizeQueryResultSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Ray-query compacted-size readback",
    });
    REQUIRE(compactionPool.has_value());
    REQUIRE(compactionReadback.has_value());
    auto invalidSerializationPool = device.createAccelerationStructureSerializationInfoQueryPool({
        .queueType = rachis::QueueType::copy,
        .queryCount = 1,
    });
    REQUIRE_FALSE(invalidSerializationPool.has_value());
    CHECK(invalidSerializationPool.error().code == rachis::ErrorCode::invalidArgument);
    auto serializationPool = device.createAccelerationStructureSerializationInfoQueryPool({
        .queueType = rachis::QueueType::compute,
        .queryCount = 2,
        .label = "Ray-query serialization information",
    });
    auto serializationInfoReadback = device.createBuffer({
        .size = 2 * rachis::accelerationStructureSerializationInfoQueryResultSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Ray-query serialization-info readback",
    });
    REQUIRE(serializationPool.has_value());
    REQUIRE(serializationInfoReadback.has_value());

    auto output = device.createBuffer({
        .size = 2 * sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Ray-query output",
    });
    auto readback = device.createBuffer({
        .size = 2 * sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Ray-query readback",
    });
    REQUIRE(output.has_value());
    REQUIRE(readback.has_value());

    const auto bytecode = loadShader();
    REQUIRE_FALSE(bytecode.empty());
    const std::array shaderBindings{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::accelerationStructure, 1},
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
    };
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Inline ray-query shader",
        .shaderInterface =
            rachis::ShaderInterfaceDesc{
                .bindings = shaderBindings,
                .workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1},
            },
    });
    REQUIRE(shader.has_value());
    const std::array layoutEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::accelerationStructure, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    auto bindingLayout = device.createBindingLayout({.entries = layoutEntries, .label = "Ray-query resources"});
    REQUIRE(bindingLayout.has_value());
    const std::array bindingLayouts{*bindingLayout};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = bindingLayouts, .label = "Ray-query layout"});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Ray-query pipeline",
    });
    REQUIRE(pipeline.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    REQUIRE(device
                .buildBottomLevelAccelerationStructure(*command,
                                                       {
                                                           .destination = *bottomLevel,
                                                           .input = bottomInput,
                                                           .scratch = *scratch,
                                                       })
                .has_value());
    REQUIRE(device
                .updateBottomLevelAccelerationStructureInPlace(*command,
                                                               {
                                                                   .target = *bottomLevel,
                                                                   .input = bottomInput,
                                                                   .scratch = *scratch,
                                                               })
                .has_value());
    auto rebuiltBottomLevel = device.buildBottomLevelAccelerationStructure(*command,
                                                                           {
                                                                               .destination = *bottomLevel,
                                                                               .input = bottomInput,
                                                                               .scratch = *scratch,
                                                                           });
    REQUIRE_FALSE(rebuiltBottomLevel.has_value());
    CHECK(rebuiltBottomLevel.error().code == rachis::ErrorCode::invalidState);
    const std::array instances{
        rachis::AccelerationStructureInstance{
            .bottomLevel = *bottomLevel,
            .flags = rachis::AccelerationStructureInstanceFlag::triangleCullDisable,
        },
    };
    const auto dependenciesBeforeRejectedTlasBuild = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(dependenciesBeforeRejectedTlasBuild.has_value());
    const auto recordsBeforeRejectedTlasBuild =
        rachis::d3d12::test::accelerationStructureRecordCount().load(std::memory_order_relaxed);
    {
        const FailResourceRetention failPrivateUploadPublication(2);
        auto rejectedTlasBuild = device.buildTopLevelAccelerationStructure(*command,
                                                                           {
                                                                               .destination = *topLevel,
                                                                               .instances = instances,
                                                                               .flags = topBuildFlags,
                                                                               .scratch = *scratch,
                                                                           });
        REQUIRE_FALSE(rejectedTlasBuild.has_value());
        CHECK(rejectedTlasBuild.error().code == rachis::ErrorCode::outOfMemory);
    }
    const auto dependenciesAfterRejectedTlasBuild = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(dependenciesAfterRejectedTlasBuild.has_value());
    CHECK(*dependenciesAfterRejectedTlasBuild == *dependenciesBeforeRejectedTlasBuild);
    CHECK(rachis::d3d12::test::accelerationStructureRecordCount().load(std::memory_order_relaxed) ==
          recordsBeforeRejectedTlasBuild);
    REQUIRE(device
                .buildTopLevelAccelerationStructure(*command,
                                                    {
                                                        .destination = *topLevel,
                                                        .instances = instances,
                                                        .flags = topBuildFlags,
                                                        .scratch = *scratch,
                                                    })
                .has_value());
    CHECK(rachis::d3d12::test::accelerationStructureRecordCount().load(std::memory_order_relaxed) ==
          recordsBeforeRejectedTlasBuild + 1);
    {
        auto foreignCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(foreignCommand.has_value());
        auto retainedBottomLevelInPlaceUpdate =
            device.updateBottomLevelAccelerationStructureInPlace(*foreignCommand,
                                                                 {
                                                                     .target = *bottomLevel,
                                                                     .input = bottomInput,
                                                                     .scratch = *scratch,
                                                                 });
        REQUIRE_FALSE(retainedBottomLevelInPlaceUpdate.has_value());
        CHECK(retainedBottomLevelInPlaceUpdate.error().code == rachis::ErrorCode::invalidState);
    }
    REQUIRE(device
                .updateTopLevelAccelerationStructureInPlace(*command,
                                                            {
                                                                .target = *topLevel,
                                                                .instances = instances,
                                                                .flags = topBuildFlags,
                                                                .scratch = *scratch,
                                                            })
                .has_value());
    auto unbuiltCompactedSize =
        device.writeAccelerationStructureCompactedSize(*command, *compactionPool, 0, *updatedBottomLevel);
    REQUIRE_FALSE(unbuiltCompactedSize.has_value());
    CHECK(unbuiltCompactedSize.error().code == rachis::ErrorCode::invalidState);
    auto outOfBoundsCompactedSize =
        device.writeAccelerationStructureCompactedSize(*command, *compactionPool, 2, *bottomLevel);
    REQUIRE_FALSE(outOfBoundsCompactedSize.has_value());
    CHECK(outOfBoundsCompactedSize.error().code == rachis::ErrorCode::invalidArgument);
    auto unbuiltSerializationInfo =
        device.writeAccelerationStructureSerializationInfo(*command, *serializationPool, 0, *updatedBottomLevel);
    REQUIRE_FALSE(unbuiltSerializationInfo.has_value());
    CHECK(unbuiltSerializationInfo.error().code == rachis::ErrorCode::invalidState);
    auto outOfBoundsSerializationInfo =
        device.writeAccelerationStructureSerializationInfo(*command, *serializationPool, 2, *bottomLevel);
    REQUIRE_FALSE(outOfBoundsSerializationInfo.has_value());
    CHECK(outOfBoundsSerializationInfo.error().code == rachis::ErrorCode::invalidArgument);
    auto inPlaceUpdate = device.updateBottomLevelAccelerationStructure(*command,
                                                                       {
                                                                           .source = *bottomLevel,
                                                                           .destination = *bottomLevel,
                                                                           .input = bottomInput,
                                                                           .scratch = *scratch,
                                                                       });
    REQUIRE_FALSE(inPlaceUpdate.has_value());
    CHECK(inPlaceUpdate.error().code == rachis::ErrorCode::invalidArgument);
    const std::array incompatibleGeometry{
        rachis::AccelerationStructureGeometry{
            .data = geometry.front().data,
            .flags = rachis::AccelerationStructureGeometryFlag::none,
        },
    };
    auto incompatibleLayout = device.updateBottomLevelAccelerationStructure(
        *command,
        {
            .source = *bottomLevel,
            .destination = *updatedBottomLevel,
            .input = {.geometries = incompatibleGeometry, .flags = bottomInput.flags},
            .scratch = *scratch,
        });
    REQUIRE_FALSE(incompatibleLayout.has_value());
    CHECK(incompatibleLayout.error().code == rachis::ErrorCode::invalidArgument);
    auto incompatibleFlags = device.updateBottomLevelAccelerationStructure(
        *command,
        {
            .source = *bottomLevel,
            .destination = *updatedBottomLevel,
            .input =
                {
                    .geometries = geometry,
                    .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace,
                },
            .scratch = *scratch,
        });
    REQUIRE_FALSE(incompatibleFlags.has_value());
    CHECK(incompatibleFlags.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(device
                .updateBottomLevelAccelerationStructure(*command,
                                                        {
                                                            .source = *bottomLevel,
                                                            .destination = *updatedBottomLevel,
                                                            .input = bottomInput,
                                                            .scratch = *scratch,
                                                        })
                .has_value());
    const std::array updatedInstances{
        rachis::AccelerationStructureInstance{
            .bottomLevel = *updatedBottomLevel,
            .flags = rachis::AccelerationStructureInstanceFlag::triangleCullDisable,
        },
    };
    const std::array incompatibleInstances{updatedInstances.front(), updatedInstances.front()};
    auto incompatibleInstanceCount = device.updateTopLevelAccelerationStructure(*command,
                                                                                {
                                                                                    .source = *topLevel,
                                                                                    .destination = *updatedTopLevel,
                                                                                    .instances = incompatibleInstances,
                                                                                    .flags = topBuildFlags,
                                                                                    .scratch = *scratch,
                                                                                });
    REQUIRE_FALSE(incompatibleInstanceCount.has_value());
    CHECK(incompatibleInstanceCount.error().code == rachis::ErrorCode::invalidArgument);
    const auto dependenciesBeforeRejectedTlasUpdate = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(dependenciesBeforeRejectedTlasUpdate.has_value());
    const auto recordsBeforeRejectedTlasUpdate =
        rachis::d3d12::test::accelerationStructureRecordCount().load(std::memory_order_relaxed);
    {
        const FailResourceRetention failPrivateUploadPublication(2);
        auto rejectedTlasUpdate = device.updateTopLevelAccelerationStructure(*command,
                                                                             {
                                                                                 .source = *topLevel,
                                                                                 .destination = *updatedTopLevel,
                                                                                 .instances = updatedInstances,
                                                                                 .flags = topBuildFlags,
                                                                                 .scratch = *scratch,
                                                                             });
        REQUIRE_FALSE(rejectedTlasUpdate.has_value());
        CHECK(rejectedTlasUpdate.error().code == rachis::ErrorCode::outOfMemory);
    }
    const auto dependenciesAfterRejectedTlasUpdate = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(dependenciesAfterRejectedTlasUpdate.has_value());
    CHECK(*dependenciesAfterRejectedTlasUpdate == *dependenciesBeforeRejectedTlasUpdate);
    CHECK(rachis::d3d12::test::accelerationStructureRecordCount().load(std::memory_order_relaxed) ==
          recordsBeforeRejectedTlasUpdate);
    REQUIRE(device
                .updateTopLevelAccelerationStructure(*command,
                                                     {
                                                         .source = *topLevel,
                                                         .destination = *updatedTopLevel,
                                                         .instances = updatedInstances,
                                                         .flags = topBuildFlags,
                                                         .scratch = *scratch,
                                                     })
                .has_value());
    CHECK(rachis::d3d12::test::accelerationStructureRecordCount().load(std::memory_order_relaxed) ==
          recordsBeforeRejectedTlasUpdate + 1);
    {
        const FailResourceRetention failPoolPublication(2);
        auto failedWrite =
            device.writeAccelerationStructureCompactedSize(*command, *compactionPool, 0, *updatedBottomLevel);
        REQUIRE_FALSE(failedWrite.has_value());
        CHECK(failedWrite.error().code == rachis::ErrorCode::outOfMemory);
    }
    REQUIRE(
        device.writeAccelerationStructureCompactedSize(*command, *compactionPool, 0, *updatedBottomLevel).has_value());
    auto unresolvedRewrite =
        device.writeAccelerationStructureCompactedSize(*command, *compactionPool, 0, *updatedBottomLevel);
    REQUIRE_FALSE(unresolvedRewrite.has_value());
    CHECK(unresolvedRewrite.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.writeAccelerationStructureCompactedSize(*command, *compactionPool, 1, *updatedTopLevel).has_value());
    {
        const FailResourceRetention failPoolPublication(2);
        auto failedResolve = device.resolveAccelerationStructureCompactedSizes(*command,
                                                                               {
                                                                                   .queryPool = *compactionPool,
                                                                                   .firstQuery = 0,
                                                                                   .queryCount = 2,
                                                                                   .destination = *compactionReadback,
                                                                               });
        REQUIRE_FALSE(failedResolve.has_value());
        CHECK(failedResolve.error().code == rachis::ErrorCode::outOfMemory);
    }
    REQUIRE(device
                .resolveAccelerationStructureCompactedSizes(*command,
                                                            {
                                                                .queryPool = *compactionPool,
                                                                .firstQuery = 0,
                                                                .queryCount = 2,
                                                                .destination = *compactionReadback,
                                                            })
                .has_value());
    auto duplicateResolve = device.resolveAccelerationStructureCompactedSizes(*command,
                                                                              {
                                                                                  .queryPool = *compactionPool,
                                                                                  .firstQuery = 0,
                                                                                  .queryCount = 2,
                                                                                  .destination = *compactionReadback,
                                                                              });
    REQUIRE_FALSE(duplicateResolve.has_value());
    CHECK(duplicateResolve.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.writeAccelerationStructureSerializationInfo(*command, *serializationPool, 0, *updatedBottomLevel)
                .has_value());
    auto unresolvedSerializationRewrite =
        device.writeAccelerationStructureSerializationInfo(*command, *serializationPool, 0, *updatedBottomLevel);
    REQUIRE_FALSE(unresolvedSerializationRewrite.has_value());
    CHECK(unresolvedSerializationRewrite.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.writeAccelerationStructureSerializationInfo(*command, *serializationPool, 1, *updatedTopLevel)
                .has_value());
    REQUIRE(device
                .resolveAccelerationStructureSerializationInfo(*command,
                                                               {
                                                                   .queryPool = *serializationPool,
                                                                   .firstQuery = 0,
                                                                   .queryCount = 2,
                                                                   .destination = *serializationInfoReadback,
                                                               })
                .has_value());
    auto duplicateSerializationResolve =
        device.resolveAccelerationStructureSerializationInfo(*command,
                                                             {
                                                                 .queryPool = *serializationPool,
                                                                 .firstQuery = 0,
                                                                 .queryCount = 2,
                                                                 .destination = *serializationInfoReadback,
                                                             });
    REQUIRE_FALSE(duplicateSerializationResolve.has_value());
    CHECK(duplicateSerializationResolve.error().code == rachis::ErrorCode::invalidState);

    const std::array writes{
        rachis::BindingWrite{.binding = 0, .resource = *updatedTopLevel},
        rachis::BindingWrite{
            .binding = 1,
            .resource =
                rachis::BufferBinding{
                    .buffer = *output,
                    .size = 2 * sizeof(std::uint32_t),
                    .structureByteStride = 2 * sizeof(std::uint32_t),
                },
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device,
                                                     {
                                                         .layout = *bindingLayout,
                                                         .writes = writes,
                                                         .label = "Ray-query binding set",
                                                     });
    REQUIRE(bindingSet.has_value());
    {
        auto foreignCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(foreignCommand.has_value());
        auto retainedTopLevelInPlaceUpdate =
            device.updateTopLevelAccelerationStructureInPlace(*foreignCommand,
                                                              {
                                                                  .target = *updatedTopLevel,
                                                                  .instances = updatedInstances,
                                                                  .flags = topBuildFlags,
                                                                  .scratch = *scratch,
                                                              });
        REQUIRE_FALSE(retainedTopLevelInPlaceUpdate.has_value());
        CHECK(retainedTopLevelInPlaceUpdate.error().code == rachis::ErrorCode::invalidState);
    }
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *output,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *output,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device.copyBuffer(*command, {.source = *output, .destination = *readback, .size = 8}).has_value());

    REQUIRE(device.destroyAccelerationStructure(*bottomLevel).has_value());
    REQUIRE(device.destroyAccelerationStructure(*topLevel).has_value());
    auto retainedBottomLevel = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::bottomLevel,
        .storage = *bottomStorage,
        .label = "Retained-overlap ray-query BLAS",
    });
    REQUIRE_FALSE(retainedBottomLevel.has_value());
    CHECK(retainedBottomLevel.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(device.destroyBuffer(*bottomStorage).has_value());
    REQUIRE(device.destroyBuffer(*topStorage).has_value());
    REQUIRE(device.destroyBuffer(*output).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyQueryPool(*compactionPool).has_value());
    REQUIRE(device.destroyQueryPool(*serializationPool).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    std::array<std::uint32_t, 2> result{};
    REQUIRE(device.readBuffer(*readback, 0, std::as_writable_bytes(std::span(result))).has_value());
    CHECK(result[0] == 1);
    CHECK(std::bit_cast<float>(result[1]) == 1.0F);
    REQUIRE(device.destroyBuffer(*readback).has_value());

    std::array<std::uint64_t, 2> compactedSizes{};
    REQUIRE(device.readBuffer(*compactionReadback, 0, std::as_writable_bytes(std::span(compactedSizes))).has_value());
    REQUIRE(compactedSizes[0] > 1);
    REQUIRE(compactedSizes[1] > 1);
    CHECK(compactedSizes[0] <= bottomSizes->accelerationStructureSize);
    CHECK(compactedSizes[1] <= topSizes->accelerationStructureSize);
    REQUIRE(device.destroyBuffer(*compactionReadback).has_value());

    std::array<rachis::AccelerationStructureSerializationInfo, 2> serializationInfo{};
    REQUIRE(device.readBuffer(*serializationInfoReadback, 0, std::as_writable_bytes(std::span(serializationInfo)))
                .has_value());
    REQUIRE(serializationInfo[0].serializedSize >= 56);
    REQUIRE(serializationInfo[1].serializedSize >= 64);
    CHECK(serializationInfo[0].bottomLevelPointerCount == 0);
    CHECK(serializationInfo[1].bottomLevelPointerCount >= 1);
    REQUIRE(device.destroyBuffer(*serializationInfoReadback).has_value());

    std::array<rachis::BufferHandle, 2> serializedBuffers;
    std::array<rachis::BufferHandle, 2> serializedReadbacks;
    for (std::size_t index = 0; index < serializationInfo.size(); ++index)
    {
        auto serialized = device.createBuffer({
            .size = serializationInfo[index].serializedSize,
            .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::deviceLocal,
            .label = "Serialized acceleration structure",
        });
        auto serializedReadback = device.createBuffer({
            .size = serializationInfo[index].serializedSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Serialized acceleration-structure readback",
        });
        REQUIRE(serialized.has_value());
        REQUIRE(serializedReadback.has_value());
        serializedBuffers[index] = *serialized;
        serializedReadbacks[index] = *serializedReadback;
    }
    auto serializeCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(serializeCommand.has_value());
    const std::array serializedSources{*updatedBottomLevel, *updatedTopLevel};
    for (std::size_t index = 0; index < serializationInfo.size(); ++index)
    {
        REQUIRE(device
                    .bufferBarrier(*serializeCommand,
                                   {
                                       .buffer = serializedBuffers[index],
                                       .stagesBefore = rachis::PipelineStage::all,
                                       .stagesAfter = rachis::PipelineStage::accelerationStructureCopy,
                                       .accessBefore = rachis::Access::common,
                                       .accessAfter = rachis::Access::unorderedAccess,
                                   })
                    .has_value());
        REQUIRE(device
                    .serializeAccelerationStructure(*serializeCommand,
                                                    {
                                                        .source = serializedSources[index],
                                                        .destination = serializedBuffers[index],
                                                        .serializedSize = serializationInfo[index].serializedSize,
                                                    })
                    .has_value());
        REQUIRE(device
                    .bufferBarrier(*serializeCommand,
                                   {
                                       .buffer = serializedBuffers[index],
                                       .stagesBefore = rachis::PipelineStage::accelerationStructureCopy,
                                       .stagesAfter = rachis::PipelineStage::copy,
                                       .accessBefore = rachis::Access::unorderedAccess,
                                       .accessAfter = rachis::Access::copySource,
                                   })
                    .has_value());
        REQUIRE(device
                    .copyBuffer(*serializeCommand,
                                {
                                    .source = serializedBuffers[index],
                                    .destination = serializedReadbacks[index],
                                    .size = serializationInfo[index].serializedSize,
                                })
                    .has_value());
    }
    auto serializeSubmitted = device.submit(std::move(*serializeCommand));
    REQUIRE(serializeSubmitted.has_value());
    REQUIRE(device.wait(*serializeSubmitted).has_value());
    device.collectGarbage();
    std::array<std::vector<std::byte>, 2> archives;
    std::array<rachis::AccelerationStructureArchiveInfo, 2> archiveInfo;
    const std::array serializedSourceSizes{bottomSizes->accelerationStructureSize, topSizes->accelerationStructureSize};
    for (std::size_t index = 0; index < serializationInfo.size(); ++index)
    {
        archives[index].resize(static_cast<std::size_t>(serializationInfo[index].serializedSize));
        REQUIRE(device.readBuffer(serializedReadbacks[index], 0, archives[index]).has_value());
        CHECK(std::ranges::any_of(archives[index], [](std::byte value) { return value != std::byte{}; }));
        auto inspected = device.inspectAccelerationStructureArchive(archives[index]);
        REQUIRE(inspected.has_value());
        archiveInfo[index] = *inspected;
        CHECK(archiveInfo[index].compatibility == rachis::AccelerationStructureArchiveCompatibility::compatible);
        CHECK(archiveInfo[index].serializedSize == serializationInfo[index].serializedSize);
        CHECK(archiveInfo[index].bottomLevelPointerCount == serializationInfo[index].bottomLevelPointerCount);
        CHECK(archiveInfo[index].deserializedSize > 0);
        CHECK(archiveInfo[index].deserializedSize <= serializedSourceSizes[index]);
        REQUIRE(device.destroyBuffer(serializedBuffers[index]).has_value());
        REQUIRE(device.destroyBuffer(serializedReadbacks[index]).has_value());
    }
    auto truncatedArchive = device.inspectAccelerationStructureArchive(
        std::span<const std::byte>(archives[0]).first(archives[0].size() - 1));
    REQUIRE_FALSE(truncatedArchive.has_value());
    CHECK(truncatedArchive.error().code == rachis::ErrorCode::invalidArgument);

    auto importedBottomStorage = device.createBuffer({
        .size = archiveInfo[0].deserializedSize,
        .usage = rachis::BufferUsage::accelerationStructureStorage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Imported ray-query BLAS storage",
    });
    auto importedTopStorage = device.createBuffer({
        .size = archiveInfo[1].deserializedSize,
        .usage = rachis::BufferUsage::accelerationStructureStorage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Imported ray-query TLAS storage",
    });
    auto oversizedImportStorage = device.createBuffer({
        .size = archiveInfo[0].deserializedSize + 256,
        .usage = rachis::BufferUsage::accelerationStructureStorage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Oversized imported ray-query BLAS storage",
    });
    auto wrongTypeImportStorage = device.createBuffer({
        .size = archiveInfo[1].deserializedSize,
        .usage = rachis::BufferUsage::accelerationStructureStorage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Wrong-type imported ray-query storage",
    });
    REQUIRE(importedBottomStorage.has_value());
    REQUIRE(importedTopStorage.has_value());
    REQUIRE(oversizedImportStorage.has_value());
    REQUIRE(wrongTypeImportStorage.has_value());
    auto bottomArchiveTopStorage = device.createBuffer({
        .size = archiveInfo[0].deserializedSize,
        .usage = rachis::BufferUsage::accelerationStructureStorage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "BLAS-archive TLAS import storage",
    });
    REQUIRE(bottomArchiveTopStorage.has_value());
    auto importedBottomLevel = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::bottomLevel,
        .storage = *importedBottomStorage,
        .label = "Imported ray-query BLAS",
    });
    auto importedTopLevel = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::topLevel,
        .storage = *importedTopStorage,
        .label = "Imported ray-query TLAS",
    });
    auto oversizedImport = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::bottomLevel,
        .storage = *oversizedImportStorage,
        .label = "Oversized imported ray-query BLAS",
    });
    auto wrongTypeImport = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::bottomLevel,
        .storage = *wrongTypeImportStorage,
        .label = "Wrong-type imported ray-query acceleration structure",
    });
    REQUIRE(importedBottomLevel.has_value());
    REQUIRE(importedTopLevel.has_value());
    REQUIRE(oversizedImport.has_value());
    REQUIRE(wrongTypeImport.has_value());
    auto bottomArchiveTopImport = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::topLevel,
        .storage = *bottomArchiveTopStorage,
        .label = "BLAS-archive TLAS import",
    });
    REQUIRE(bottomArchiveTopImport.has_value());

    auto importedOutput = device.createBuffer({
        .size = 2 * sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Imported ray-query output",
    });
    auto importedReadback = device.createBuffer({
        .size = 2 * sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Imported ray-query readback",
    });
    REQUIRE(importedOutput.has_value());
    REQUIRE(importedReadback.has_value());

    auto importCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(importCommand.has_value());
    auto oversizedDestination = device.deserializeAccelerationStructure(*importCommand,
                                                                        {
                                                                            .destination = *oversizedImport,
                                                                            .archive = archives[0],
                                                                        });
    REQUIRE_FALSE(oversizedDestination.has_value());
    CHECK(oversizedDestination.error().code == rachis::ErrorCode::invalidArgument);
    const auto dependenciesBeforeRejectedImport = rachis::d3d12::test::commandDependencySnapshot(*importCommand);
    REQUIRE(dependenciesBeforeRejectedImport.has_value());
    const auto recordsBeforeRejectedImport =
        rachis::d3d12::test::accelerationStructureRecordCount().load(std::memory_order_relaxed);
    {
        const FailResourceRetention failPrivateUploadPublication(2);
        auto rejectedImport = device.deserializeAccelerationStructure(*importCommand,
                                                                      {
                                                                          .destination = *importedBottomLevel,
                                                                          .archive = archives[0],
                                                                      });
        REQUIRE_FALSE(rejectedImport.has_value());
        CHECK(rejectedImport.error().code == rachis::ErrorCode::outOfMemory);
    }
    const auto dependenciesAfterRejectedImport = rachis::d3d12::test::commandDependencySnapshot(*importCommand);
    REQUIRE(dependenciesAfterRejectedImport.has_value());
    CHECK(*dependenciesAfterRejectedImport == *dependenciesBeforeRejectedImport);
    CHECK(rachis::d3d12::test::accelerationStructureRecordCount().load(std::memory_order_relaxed) ==
          recordsBeforeRejectedImport);
    REQUIRE(device
                .deserializeAccelerationStructure(*importCommand,
                                                  {
                                                      .destination = *importedBottomLevel,
                                                      .archive = archives[0],
                                                  })
                .has_value());
    CHECK(rachis::d3d12::test::accelerationStructureRecordCount().load(std::memory_order_relaxed) ==
          recordsBeforeRejectedImport + 1);
    auto duplicateImport = device.deserializeAccelerationStructure(*importCommand,
                                                                   {
                                                                       .destination = *importedBottomLevel,
                                                                       .archive = archives[0],
                                                                   });
    REQUIRE_FALSE(duplicateImport.has_value());
    CHECK(duplicateImport.error().code == rachis::ErrorCode::invalidState);
    std::vector<rachis::AccelerationStructureHandle> importedBottomLevels(
        static_cast<std::size_t>(archiveInfo[1].bottomLevelPointerCount), *importedBottomLevel);
    auto missingRelocations = device.deserializeAccelerationStructure(*importCommand,
                                                                      {
                                                                          .destination = *importedTopLevel,
                                                                          .archive = archives[1],
                                                                      });
    REQUIRE_FALSE(missingRelocations.has_value());
    CHECK(missingRelocations.error().code == rachis::ErrorCode::invalidArgument);
    auto wrongDestinationType = device.deserializeAccelerationStructure(*importCommand,
                                                                        {
                                                                            .destination = *wrongTypeImport,
                                                                            .archive = archives[1],
                                                                            .bottomLevels = importedBottomLevels,
                                                                        });
    REQUIRE_FALSE(wrongDestinationType.has_value());
    CHECK(wrongDestinationType.error().code == rachis::ErrorCode::invalidArgument);
    auto bottomArchiveAsTop = device.deserializeAccelerationStructure(*importCommand,
                                                                      {
                                                                          .destination = *bottomArchiveTopImport,
                                                                          .archive = archives[0],
                                                                      });
    REQUIRE_FALSE(bottomArchiveAsTop.has_value());
    CHECK(bottomArchiveAsTop.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(device
                .deserializeAccelerationStructure(*importCommand,
                                                  {
                                                      .destination = *importedTopLevel,
                                                      .archive = archives[1],
                                                      .bottomLevels = importedBottomLevels,
                                                  })
                .has_value());

    const std::array importedWrites{
        rachis::BindingWrite{.binding = 0, .resource = *importedTopLevel},
        rachis::BindingWrite{
            .binding = 1,
            .resource =
                rachis::BufferBinding{
                    .buffer = *importedOutput,
                    .size = 2 * sizeof(std::uint32_t),
                    .structureByteStride = 2 * sizeof(std::uint32_t),
                },
        },
    };
    auto importedBindingSet = rachis::test::createBindingSet(device,
                                                             {
                                                                 .layout = *bindingLayout,
                                                                 .writes = importedWrites,
                                                                 .label = "Imported ray-query binding set",
                                                             });
    REQUIRE(importedBindingSet.has_value());
    REQUIRE(device
                .bufferBarrier(*importCommand,
                               {
                                   .buffer = *importedOutput,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*importCommand, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*importCommand, 0, *importedBindingSet).has_value());
    REQUIRE(device.dispatch(*importCommand, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*importCommand,
                               {
                                   .buffer = *importedOutput,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device.copyBuffer(*importCommand, {.source = *importedOutput, .destination = *importedReadback, .size = 8})
                .has_value());

    REQUIRE(device.destroyAccelerationStructure(*importedBottomLevel).has_value());
    REQUIRE(device.destroyAccelerationStructure(*importedTopLevel).has_value());
    REQUIRE(device.destroyAccelerationStructure(*oversizedImport).has_value());
    REQUIRE(device.destroyAccelerationStructure(*wrongTypeImport).has_value());
    REQUIRE(device.destroyAccelerationStructure(*bottomArchiveTopImport).has_value());
    REQUIRE(device.destroyBuffer(*importedBottomStorage).has_value());
    REQUIRE(device.destroyBuffer(*importedTopStorage).has_value());
    REQUIRE(device.destroyBuffer(*oversizedImportStorage).has_value());
    REQUIRE(device.destroyBuffer(*wrongTypeImportStorage).has_value());
    REQUIRE(device.destroyBuffer(*bottomArchiveTopStorage).has_value());
    REQUIRE(device.destroyBuffer(*importedOutput).has_value());
    REQUIRE(device.destroyBindingSet(*importedBindingSet).has_value());
    auto importSubmitted = device.submit(std::move(*importCommand));
    REQUIRE(importSubmitted.has_value());
    REQUIRE(device.wait(*importSubmitted).has_value());
    device.collectGarbage();

    std::array<std::uint32_t, 2> importedResult{};
    REQUIRE(device.readBuffer(*importedReadback, 0, std::as_writable_bytes(std::span(importedResult))).has_value());
    CHECK(importedResult[0] == 1);
    CHECK(std::bit_cast<float>(importedResult[1]) == 1.0F);
    REQUIRE(device.destroyBuffer(*importedReadback).has_value());

    auto compactedBottomStorage = device.createBuffer({
        .size = compactedSizes[0],
        .usage = rachis::BufferUsage::accelerationStructureStorage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Compacted ray-query BLAS storage",
    });
    auto compactedTopStorage = device.createBuffer({
        .size = compactedSizes[1],
        .usage = rachis::BufferUsage::accelerationStructureStorage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Compacted ray-query TLAS storage",
    });
    auto compactedUpdatedBottomStorage = device.createBuffer({
        .size = bottomSizes->accelerationStructureSize,
        .usage = rachis::BufferUsage::accelerationStructureStorage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Updated compacted ray-query BLAS storage",
    });
    REQUIRE(compactedBottomStorage.has_value());
    REQUIRE(compactedTopStorage.has_value());
    REQUIRE(compactedUpdatedBottomStorage.has_value());
    auto compactedBottomLevel = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::bottomLevel,
        .storage = *compactedBottomStorage,
        .label = "Compacted ray-query BLAS",
    });
    auto compactedTopLevel = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::topLevel,
        .storage = *compactedTopStorage,
        .label = "Compacted ray-query TLAS",
    });
    auto compactedUpdatedBottomLevel = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::bottomLevel,
        .storage = *compactedUpdatedBottomStorage,
        .label = "Updated compacted ray-query BLAS",
    });
    REQUIRE(compactedBottomLevel.has_value());
    REQUIRE(compactedTopLevel.has_value());
    REQUIRE(compactedUpdatedBottomLevel.has_value());

    auto compactedOutput = device.createBuffer({
        .size = 2 * sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Compacted ray-query output",
    });
    auto compactedReadback = device.createBuffer({
        .size = 2 * sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Compacted ray-query readback",
    });
    REQUIRE(compactedOutput.has_value());
    REQUIRE(compactedReadback.has_value());

    auto compactCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(compactCommand.has_value());
    auto inPlaceCompaction = device.compactAccelerationStructure(*compactCommand,
                                                                 {
                                                                     .source = *updatedBottomLevel,
                                                                     .destination = *updatedBottomLevel,
                                                                     .compactedSize = compactedSizes[0],
                                                                 });
    REQUIRE_FALSE(inPlaceCompaction.has_value());
    CHECK(inPlaceCompaction.error().code == rachis::ErrorCode::invalidArgument);
    auto mismatchedCompactedSize = device.compactAccelerationStructure(*compactCommand,
                                                                       {
                                                                           .source = *updatedBottomLevel,
                                                                           .destination = *compactedBottomLevel,
                                                                           .compactedSize = compactedSizes[0] - 1,
                                                                       });
    REQUIRE_FALSE(mismatchedCompactedSize.has_value());
    CHECK(mismatchedCompactedSize.error().code == rachis::ErrorCode::invalidArgument);
    auto mismatchedCompactionType = device.compactAccelerationStructure(*compactCommand,
                                                                        {
                                                                            .source = *updatedBottomLevel,
                                                                            .destination = *compactedTopLevel,
                                                                            .compactedSize = compactedSizes[1],
                                                                        });
    REQUIRE_FALSE(mismatchedCompactionType.has_value());
    CHECK(mismatchedCompactionType.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(device
                .compactAccelerationStructure(*compactCommand,
                                              {
                                                  .source = *updatedBottomLevel,
                                                  .destination = *compactedBottomLevel,
                                                  .compactedSize = compactedSizes[0],
                                              })
                .has_value());
    REQUIRE(device
                .compactAccelerationStructure(*compactCommand,
                                              {
                                                  .source = *updatedTopLevel,
                                                  .destination = *compactedTopLevel,
                                                  .compactedSize = compactedSizes[1],
                                              })
                .has_value());
    if (compactedSizes[0] < bottomSizes->accelerationStructureSize)
    {
        auto compactedSizeStorage = device.createBuffer({
            .size = compactedSizes[0],
            .usage = rachis::BufferUsage::accelerationStructureStorage,
            .memoryType = rachis::MemoryType::deviceLocal,
            .label = "Compacted-size update destination storage",
        });
        REQUIRE(compactedSizeStorage.has_value());
        auto compactedSizeDestination = device.createAccelerationStructure({
            .type = rachis::AccelerationStructureType::bottomLevel,
            .storage = *compactedSizeStorage,
            .label = "Compacted-size update destination",
        });
        REQUIRE(compactedSizeDestination.has_value());
        auto undersizedUpdate =
            device.updateBottomLevelAccelerationStructure(*compactCommand,
                                                          {
                                                              .source = *compactedBottomLevel,
                                                              .destination = *compactedSizeDestination,
                                                              .input = bottomInput,
                                                              .scratch = *scratch,
                                                          });
        REQUIRE_FALSE(undersizedUpdate.has_value());
        CHECK(undersizedUpdate.error().code == rachis::ErrorCode::invalidArgument);
        REQUIRE(device.destroyAccelerationStructure(*compactedSizeDestination).has_value());
        REQUIRE(device.destroyBuffer(*compactedSizeStorage).has_value());
    }
    REQUIRE(device
                .updateBottomLevelAccelerationStructure(*compactCommand,
                                                        {
                                                            .source = *compactedBottomLevel,
                                                            .destination = *compactedUpdatedBottomLevel,
                                                            .input = bottomInput,
                                                            .scratch = *scratch,
                                                        })
                .has_value());
    auto duplicateCompaction = device.compactAccelerationStructure(*compactCommand,
                                                                   {
                                                                       .source = *updatedTopLevel,
                                                                       .destination = *compactedTopLevel,
                                                                       .compactedSize = compactedSizes[1],
                                                                   });
    REQUIRE_FALSE(duplicateCompaction.has_value());
    CHECK(duplicateCompaction.error().code == rachis::ErrorCode::invalidState);

    const std::array compactedWrites{
        rachis::BindingWrite{.binding = 0, .resource = *compactedTopLevel},
        rachis::BindingWrite{
            .binding = 1,
            .resource =
                rachis::BufferBinding{
                    .buffer = *compactedOutput,
                    .size = 2 * sizeof(std::uint32_t),
                    .structureByteStride = 2 * sizeof(std::uint32_t),
                },
        },
    };
    auto compactedBindingSet = rachis::test::createBindingSet(device,
                                                              {
                                                                  .layout = *bindingLayout,
                                                                  .writes = compactedWrites,
                                                                  .label = "Compacted ray-query binding set",
                                                              });
    REQUIRE(compactedBindingSet.has_value());
    REQUIRE(device
                .bufferBarrier(*compactCommand,
                               {
                                   .buffer = *compactedOutput,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*compactCommand, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*compactCommand, 0, *compactedBindingSet).has_value());
    REQUIRE(device.dispatch(*compactCommand, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*compactCommand,
                               {
                                   .buffer = *compactedOutput,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(
        device.copyBuffer(*compactCommand, {.source = *compactedOutput, .destination = *compactedReadback, .size = 8})
            .has_value());

    REQUIRE(device.destroyAccelerationStructure(*updatedBottomLevel).has_value());
    REQUIRE(device.destroyAccelerationStructure(*updatedTopLevel).has_value());
    REQUIRE(device.destroyAccelerationStructure(*compactedBottomLevel).has_value());
    REQUIRE(device.destroyAccelerationStructure(*compactedTopLevel).has_value());
    REQUIRE(device.destroyAccelerationStructure(*compactedUpdatedBottomLevel).has_value());
    REQUIRE(device.destroyBuffer(*vertexBuffer).has_value());
    REQUIRE(device.destroyBuffer(*scratch).has_value());
    REQUIRE(device.destroyBuffer(*updatedBottomStorage).has_value());
    REQUIRE(device.destroyBuffer(*updatedTopStorage).has_value());
    REQUIRE(device.destroyBuffer(*compactedBottomStorage).has_value());
    REQUIRE(device.destroyBuffer(*compactedTopStorage).has_value());
    REQUIRE(device.destroyBuffer(*compactedUpdatedBottomStorage).has_value());
    REQUIRE(device.destroyBuffer(*compactedOutput).has_value());
    REQUIRE(device.destroyBindingSet(*compactedBindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());

    auto compactSubmitted = device.submit(std::move(*compactCommand));
    REQUIRE(compactSubmitted.has_value());
    REQUIRE(device.wait(*compactSubmitted).has_value());
    device.collectGarbage();

    std::array<std::uint32_t, 2> compactedResult{};
    REQUIRE(device.readBuffer(*compactedReadback, 0, std::as_writable_bytes(std::span(compactedResult))).has_value());
    CHECK(compactedResult[0] == 1);
    CHECK(std::bit_cast<float>(compactedResult[1]) == 1.0F);
    REQUIRE(device.destroyBuffer(*compactedReadback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("in-place acceleration-structure refits and serialization barriers stay coherent across frames",
          "[size:medium][rhi][ray-query][hardware]")
{
    auto created = rachis::createDevice({
        .adapterPreference = rachis::AdapterPreference::highPerformance,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        skipWithoutHardwareAdapter(created.error());
    }
    auto device = std::move(*created);
    const auto capabilities = *device.capabilities();
    if (!capabilities.rayQuery)
    {
        SKIP("The selected hardware adapter does not expose DXR 1.1 inline ray queries.");
    }

    const auto createVertexBuffer = [&](float zDepth, const char* label) {
        const std::array vertices{
            std::array{-1.0F, -1.0F, zDepth},
            std::array{0.0F, 1.0F, zDepth},
            std::array{1.0F, -1.0F, zDepth},
        };
        auto buffer = device.createBuffer({
            .size = sizeof(vertices),
            .usage = rachis::BufferUsage::accelerationStructureBuildInput,
            .memoryType = rachis::MemoryType::upload,
            .label = label,
        });
        REQUIRE(buffer.has_value());
        REQUIRE(device.writeBuffer(*buffer, 0, std::as_bytes(std::span(vertices))).has_value());
        return *buffer;
    };

    const auto initialVertices = createVertexBuffer(0.0F, "Animated ray-query vertices (frame 0)");
    const auto refitVertices = createVertexBuffer(2.0F, "Animated ray-query vertices (frame 1)");
    const auto finalVertices = createVertexBuffer(1.5F, "Animated ray-query vertices (frame 2)");

    constexpr std::uint32_t vertexStride = 3U * sizeof(float);
    const auto makeGeometry = [&](rachis::BufferHandle vertexBuffer) {
        return std::array{
            rachis::AccelerationStructureGeometry{
                .data =
                    rachis::AccelerationStructureTriangleGeometry{
                        .vertexBuffer = vertexBuffer,
                        .vertexCount = 3,
                        .vertexStride = vertexStride,
                        .vertexFormat = rachis::AccelerationStructureVertexFormat::float32x3,
                    },
                .flags = rachis::AccelerationStructureGeometryFlag::opaque,
            },
        };
    };

    const auto initialGeometry = makeGeometry(initialVertices);
    const auto refitGeometry = makeGeometry(refitVertices);
    const auto finalGeometry = makeGeometry(finalVertices);
    constexpr auto buildFlags =
        rachis::AccelerationStructureBuildFlag::preferFastTrace | rachis::AccelerationStructureBuildFlag::allowUpdate;

    const rachis::BottomLevelAccelerationStructureBuildInput initialBottomInput{
        .geometries = initialGeometry,
        .flags = buildFlags,
    };
    const rachis::BottomLevelAccelerationStructureBuildInput refitBottomInput{
        .geometries = refitGeometry,
        .flags = buildFlags,
    };
    const rachis::BottomLevelAccelerationStructureBuildInput finalBottomInput{
        .geometries = finalGeometry,
        .flags = buildFlags,
    };

    auto bottomSizes = device.queryBottomLevelAccelerationStructureBuildSizes(initialBottomInput);
    auto topSizes = device.queryTopLevelAccelerationStructureBuildSizes({
        .instanceCount = 1,
        .flags = buildFlags,
    });
    REQUIRE(bottomSizes.has_value());
    REQUIRE(topSizes.has_value());

    const auto scratchSize = (std::max)({bottomSizes->buildScratchSize,
                                         bottomSizes->updateScratchSize,
                                         topSizes->buildScratchSize,
                                         topSizes->updateScratchSize});
    auto bottomStorage = device.createBuffer({
        .size = bottomSizes->accelerationStructureSize,
        .usage = rachis::BufferUsage::accelerationStructureStorage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Animated BLAS storage",
    });
    auto topStorage = device.createBuffer({
        .size = topSizes->accelerationStructureSize,
        .usage = rachis::BufferUsage::accelerationStructureStorage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Animated TLAS storage",
    });
    auto rebuiltTopStorage = device.createBuffer({
        .size = topSizes->accelerationStructureSize,
        .usage = rachis::BufferUsage::accelerationStructureStorage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Rebuilt TLAS storage",
    });
    auto scratch = device.createBuffer({
        .size = scratchSize,
        .usage = rachis::BufferUsage::accelerationStructureScratch,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Animated AS scratch",
    });
    REQUIRE(bottomStorage.has_value());
    REQUIRE(topStorage.has_value());
    REQUIRE(rebuiltTopStorage.has_value());
    REQUIRE(scratch.has_value());

    auto bottomLevel = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::bottomLevel,
        .storage = *bottomStorage,
        .label = "Animated BLAS",
    });
    auto topLevel = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::topLevel,
        .storage = *topStorage,
        .label = "Animated TLAS",
    });
    auto rebuiltTopLevel = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::topLevel,
        .storage = *rebuiltTopStorage,
        .label = "Rebuilt TLAS from refitted BLAS",
    });
    REQUIRE(bottomLevel.has_value());
    REQUIRE(topLevel.has_value());
    REQUIRE(rebuiltTopLevel.has_value());

    auto output = device.createBuffer({
        .size = 2 * sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Animated ray-query output",
    });
    auto rebuiltOutput = device.createBuffer({
        .size = 2 * sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Rebuilt TLAS ray-query output",
    });
    auto readback = device.createBuffer({
        .size = 2 * sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Animated ray-query readback",
    });
    auto rebuiltReadback = device.createBuffer({
        .size = 2 * sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Rebuilt TLAS ray-query readback",
    });
    REQUIRE(output.has_value());
    REQUIRE(rebuiltOutput.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(rebuiltReadback.has_value());

    const auto bytecode = loadShader();
    REQUIRE_FALSE(bytecode.empty());
    const std::array shaderBindings{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::accelerationStructure, 1},
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
    };
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Animated inline ray-query shader",
        .shaderInterface =
            rachis::ShaderInterfaceDesc{
                .bindings = shaderBindings,
                .workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1},
            },
    });
    REQUIRE(shader.has_value());
    const std::array layoutEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::accelerationStructure, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    auto bindingLayout = device.createBindingLayout({.entries = layoutEntries, .label = "Animated ray-query layout"});
    REQUIRE(bindingLayout.has_value());
    const std::array bindingLayouts{*bindingLayout};
    auto pipelineLayout =
        device.createPipelineLayout({.bindingLayouts = bindingLayouts, .label = "Animated ray-query pipeline layout"});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Animated ray-query pipeline",
    });
    REQUIRE(pipeline.has_value());

    const std::array instances{
        rachis::AccelerationStructureInstance{
            .bottomLevel = *bottomLevel,
            .flags = rachis::AccelerationStructureInstanceFlag::triangleCullDisable,
        },
    };

    const auto recordRayQuery = [&](rachis::CommandBuffer& commandBuffer,
                                    rachis::BindingSetHandle targetBindingSet,
                                    rachis::BufferHandle targetOutput,
                                    rachis::BufferHandle targetReadback,
                                    rachis::PipelineStage stagesBeforeOutput,
                                    rachis::Access accessBeforeOutput) {
        REQUIRE(device
                    .bufferBarrier(commandBuffer,
                                   {
                                       .buffer = targetOutput,
                                       .stagesBefore = stagesBeforeOutput,
                                       .stagesAfter = rachis::PipelineStage::computeShader,
                                       .accessBefore = accessBeforeOutput,
                                       .accessAfter = rachis::Access::unorderedAccess,
                                   })
                    .has_value());
        REQUIRE(device.bindComputePipeline(commandBuffer, *pipeline).has_value());
        REQUIRE(device.bindComputeBindingSet(commandBuffer, 0, targetBindingSet).has_value());
        REQUIRE(device.dispatch(commandBuffer, 1).has_value());
        REQUIRE(device
                    .bufferBarrier(commandBuffer,
                                   {
                                       .buffer = targetOutput,
                                       .stagesBefore = rachis::PipelineStage::computeShader,
                                       .stagesAfter = rachis::PipelineStage::copy,
                                       .accessBefore = rachis::Access::unorderedAccess,
                                       .accessAfter = rachis::Access::copySource,
                                   })
                    .has_value());
        REQUIRE(device.copyBuffer(commandBuffer, {.source = targetOutput, .destination = targetReadback, .size = 8})
                    .has_value());
    };

    auto frame0Command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(frame0Command.has_value());
    REQUIRE(device
                .buildBottomLevelAccelerationStructure(*frame0Command,
                                                       {
                                                           .destination = *bottomLevel,
                                                           .input = initialBottomInput,
                                                           .scratch = *scratch,
                                                       })
                .has_value());
    REQUIRE(device
                .buildTopLevelAccelerationStructure(*frame0Command,
                                                    {
                                                        .destination = *topLevel,
                                                        .instances = instances,
                                                        .flags = buildFlags,
                                                        .scratch = *scratch,
                                                    })
                .has_value());

    const std::array bindingWrites{
        rachis::BindingWrite{.binding = 0, .resource = *topLevel},
        rachis::BindingWrite{
            .binding = 1,
            .resource =
                rachis::BufferBinding{
                    .buffer = *output,
                    .size = 2 * sizeof(std::uint32_t),
                    .structureByteStride = 2 * sizeof(std::uint32_t),
                },
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device,
                                                     {
                                                         .layout = *bindingLayout,
                                                         .writes = bindingWrites,
                                                         .label = "Persistent animated ray-query binding set",
                                                     });
    REQUIRE(bindingSet.has_value());

    recordRayQuery(*frame0Command, *bindingSet, *output, *readback, rachis::PipelineStage::all, rachis::Access::common);
    auto frame0Submitted = device.submit(std::move(*frame0Command));
    REQUIRE(frame0Submitted.has_value());
    REQUIRE(device.wait(*frame0Submitted).has_value());

    std::array<std::uint32_t, 2> frame0Result{};
    REQUIRE(device.readBuffer(*readback, 0, std::as_writable_bytes(std::span(frame0Result))).has_value());
    CHECK(frame0Result[0] == 1);
    CHECK(std::bit_cast<float>(frame0Result[1]) == 1.0F);

    auto frame1Command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(frame1Command.has_value());
    REQUIRE(device
                .updateBottomLevelAccelerationStructureInPlace(*frame1Command,
                                                               {
                                                                   .target = *bottomLevel,
                                                                   .input = refitBottomInput,
                                                                   .scratch = *scratch,
                                                               })
                .has_value());
    REQUIRE(device
                .updateTopLevelAccelerationStructureInPlace(*frame1Command,
                                                            {
                                                                .target = *topLevel,
                                                                .instances = instances,
                                                                .flags = buildFlags,
                                                                .scratch = *scratch,
                                                            })
                .has_value());
    REQUIRE(device
                .buildTopLevelAccelerationStructure(*frame1Command,
                                                    {
                                                        .destination = *rebuiltTopLevel,
                                                        .instances = instances,
                                                        .flags = buildFlags,
                                                        .scratch = *scratch,
                                                    })
                .has_value());

    const std::array rebuiltWrites{
        rachis::BindingWrite{.binding = 0, .resource = *rebuiltTopLevel},
        rachis::BindingWrite{
            .binding = 1,
            .resource =
                rachis::BufferBinding{
                    .buffer = *rebuiltOutput,
                    .size = 2 * sizeof(std::uint32_t),
                    .structureByteStride = 2 * sizeof(std::uint32_t),
                },
        },
    };
    auto rebuiltBindingSet = rachis::test::createBindingSet(device,
                                                            {
                                                                .layout = *bindingLayout,
                                                                .writes = rebuiltWrites,
                                                                .label = "Rebuilt TLAS ray-query binding set",
                                                            });
    REQUIRE(rebuiltBindingSet.has_value());

    recordRayQuery(
        *frame1Command, *bindingSet, *output, *readback, rachis::PipelineStage::copy, rachis::Access::copySource);
    recordRayQuery(*frame1Command,
                   *rebuiltBindingSet,
                   *rebuiltOutput,
                   *rebuiltReadback,
                   rachis::PipelineStage::all,
                   rachis::Access::common);
    auto frame1Submitted = device.submit(std::move(*frame1Command));
    REQUIRE(frame1Submitted.has_value());
    REQUIRE(device.wait(*frame1Submitted).has_value());

    std::array<std::uint32_t, 2> frame1Result{};
    std::array<std::uint32_t, 2> rebuiltResult{};
    REQUIRE(device.readBuffer(*readback, 0, std::as_writable_bytes(std::span(frame1Result))).has_value());
    REQUIRE(device.readBuffer(*rebuiltReadback, 0, std::as_writable_bytes(std::span(rebuiltResult))).has_value());
    CHECK(frame1Result[0] == 1);
    CHECK(std::bit_cast<float>(frame1Result[1]) == 3.0F);
    CHECK(rebuiltResult[0] == 1);
    CHECK(std::bit_cast<float>(rebuiltResult[1]) == 3.0F);

    auto resettablePool = device.createCommandPool({
        .queueType = rachis::QueueType::compute,
        .maxCommandBuffers = 2,
        .label = "AS lease rejection pool",
        .flags = rachis::CommandPoolFlag::allowIndividualReset,
    });
    REQUIRE(resettablePool.has_value());
    {
        auto holdingCommand = device.acquireCommandBuffer(*resettablePool);
        auto attemptCommand = device.acquireCommandBuffer(*resettablePool);
        REQUIRE(holdingCommand.has_value());
        REQUIRE(attemptCommand.has_value());

        REQUIRE(device.bindComputePipeline(*holdingCommand, *pipeline).has_value());
        REQUIRE(device.bindComputeBindingSet(*holdingCommand, 0, *bindingSet).has_value());

        auto blasBlockedByBoundSet = device.updateBottomLevelAccelerationStructureInPlace(*attemptCommand,
                                                                                          {
                                                                                              .target = *bottomLevel,
                                                                                              .input = finalBottomInput,
                                                                                              .scratch = *scratch,
                                                                                          });
        REQUIRE_FALSE(blasBlockedByBoundSet.has_value());
        CHECK(blasBlockedByBoundSet.error().code == rachis::ErrorCode::invalidState);

        auto tlasBlockedByBoundSet = device.updateTopLevelAccelerationStructureInPlace(*attemptCommand,
                                                                                       {
                                                                                           .target = *topLevel,
                                                                                           .instances = instances,
                                                                                           .flags = buildFlags,
                                                                                           .scratch = *scratch,
                                                                                       });
        REQUIRE_FALSE(tlasBlockedByBoundSet.has_value());
        CHECK(tlasBlockedByBoundSet.error().code == rachis::ErrorCode::invalidState);

        auto sameRecordingAfterBind =
            device.updateBottomLevelAccelerationStructureInPlace(*holdingCommand,
                                                                 {
                                                                     .target = *bottomLevel,
                                                                     .input = finalBottomInput,
                                                                     .scratch = *scratch,
                                                                 });
        CHECK(sameRecordingAfterBind.has_value());

        REQUIRE(device.resetCommandBuffer(*holdingCommand).has_value());
        REQUIRE(device
                    .updateBottomLevelAccelerationStructureInPlace(*holdingCommand,
                                                                   {
                                                                       .target = *bottomLevel,
                                                                       .input = finalBottomInput,
                                                                       .scratch = *scratch,
                                                                   })
                    .has_value());

        auto crossRecordingBlockedByProducer =
            device.updateBottomLevelAccelerationStructureInPlace(*attemptCommand,
                                                                 {
                                                                     .target = *bottomLevel,
                                                                     .input = finalBottomInput,
                                                                     .scratch = *scratch,
                                                                 });
        REQUIRE_FALSE(crossRecordingBlockedByProducer.has_value());
        CHECK(crossRecordingBlockedByProducer.error().code == rachis::ErrorCode::invalidState);

        holdingCommand = std::unexpected(rachis::Error{});
        REQUIRE(device
                    .updateBottomLevelAccelerationStructureInPlace(*attemptCommand,
                                                                   {
                                                                       .target = *bottomLevel,
                                                                       .input = finalBottomInput,
                                                                       .scratch = *scratch,
                                                                   })
                    .has_value());
        REQUIRE(device
                    .updateTopLevelAccelerationStructureInPlace(*attemptCommand,
                                                                {
                                                                    .target = *topLevel,
                                                                    .instances = instances,
                                                                    .flags = buildFlags,
                                                                    .scratch = *scratch,
                                                                })
                    .has_value());
        auto finalRefitSubmitted = device.submit(std::move(*attemptCommand));
        REQUIRE(finalRefitSubmitted.has_value());
        REQUIRE(device.wait(*finalRefitSubmitted).has_value());
    }
    REQUIRE(device.destroyCommandPool(*resettablePool).has_value());

    auto serializationPool = device.createAccelerationStructureSerializationInfoQueryPool({
        .queueType = rachis::QueueType::compute,
        .queryCount = 2,
        .label = "Refitted AS serialization info",
    });
    auto serializationInfoReadback = device.createBuffer({
        .size = 2 * rachis::accelerationStructureSerializationInfoQueryResultSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Refitted AS serialization-info readback",
    });
    REQUIRE(serializationPool.has_value());
    REQUIRE(serializationInfoReadback.has_value());

    auto queryCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(queryCommand.has_value());
    REQUIRE(device.writeAccelerationStructureSerializationInfo(*queryCommand, *serializationPool, 0, *bottomLevel)
                .has_value());
    REQUIRE(device.writeAccelerationStructureSerializationInfo(*queryCommand, *serializationPool, 1, *topLevel)
                .has_value());
    REQUIRE(device
                .resolveAccelerationStructureSerializationInfo(*queryCommand,
                                                               {
                                                                   .queryPool = *serializationPool,
                                                                   .firstQuery = 0,
                                                                   .queryCount = 2,
                                                                   .destination = *serializationInfoReadback,
                                                               })
                .has_value());
    auto querySubmitted = device.submit(std::move(*queryCommand));
    REQUIRE(querySubmitted.has_value());
    REQUIRE(device.wait(*querySubmitted).has_value());

    std::array<rachis::AccelerationStructureSerializationInfo, 2> serializationInfo{};
    REQUIRE(device.readBuffer(*serializationInfoReadback, 0, std::as_writable_bytes(std::span(serializationInfo)))
                .has_value());
    REQUIRE(serializationInfo[0].serializedSize >= 56);
    REQUIRE(serializationInfo[1].serializedSize >= 64);

    constexpr std::uint64_t archiveAlignment = 256;
    const std::uint64_t topArchiveOffset =
        ((serializationInfo[0].serializedSize + archiveAlignment - 1U) / archiveAlignment) * archiveAlignment;
    const std::uint64_t combinedArchiveSize = topArchiveOffset + serializationInfo[1].serializedSize;

    auto serializedBuffer = device.createBuffer({
        .size = combinedArchiveSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Combined serialized AS buffer",
    });
    auto serializedReadback = device.createBuffer({
        .size = combinedArchiveSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Combined serialized AS readback",
    });
    REQUIRE(serializedBuffer.has_value());
    REQUIRE(serializedReadback.has_value());

    auto serializeCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(serializeCommand.has_value());
    REQUIRE(device
                .bufferBarrier(*serializeCommand,
                               {
                                   .buffer = *serializedBuffer,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::accelerationStructureCopy,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device
                .serializeAccelerationStructure(*serializeCommand,
                                                {
                                                    .source = *bottomLevel,
                                                    .destination = *serializedBuffer,
                                                    .destinationOffset = 0,
                                                    .serializedSize = serializationInfo[0].serializedSize,
                                                })
                .has_value());
    REQUIRE(device
                .serializeAccelerationStructure(*serializeCommand,
                                                {
                                                    .source = *topLevel,
                                                    .destination = *serializedBuffer,
                                                    .destinationOffset = topArchiveOffset,
                                                    .serializedSize = serializationInfo[1].serializedSize,
                                                })
                .has_value());
    if (!capabilities.enhancedBarriers)
    {
        REQUIRE(device
                    .bufferBarrier(*serializeCommand,
                                   {
                                       .buffer = *serializedBuffer,
                                       .stagesBefore = rachis::PipelineStage::accelerationStructureCopy,
                                       .stagesAfter = rachis::PipelineStage::copy,
                                       .accessBefore = rachis::Access::unorderedAccess,
                                       .accessAfter = rachis::Access::copySource,
                                   })
                    .has_value());
    }
    REQUIRE(device
                .copyBuffer(*serializeCommand,
                            {
                                .source = *serializedBuffer,
                                .destination = *serializedReadback,
                                .size = combinedArchiveSize,
                            })
                .has_value());
    auto serializeSubmitted = device.submit(std::move(*serializeCommand));
    REQUIRE(serializeSubmitted.has_value());
    REQUIRE(device.wait(*serializeSubmitted).has_value());

    std::vector<std::byte> combinedArchive(static_cast<std::size_t>(combinedArchiveSize));
    REQUIRE(device.readBuffer(*serializedReadback, 0, combinedArchive).has_value());
    const auto bottomArchive = std::span<const std::byte>(combinedArchive)
                                   .subspan(0, static_cast<std::size_t>(serializationInfo[0].serializedSize));
    const auto topArchive = std::span<const std::byte>(combinedArchive)
                                .subspan(static_cast<std::size_t>(topArchiveOffset),
                                         static_cast<std::size_t>(serializationInfo[1].serializedSize));
    auto bottomArchiveInfo = device.inspectAccelerationStructureArchive(bottomArchive);
    auto topArchiveInfo = device.inspectAccelerationStructureArchive(topArchive);
    REQUIRE(bottomArchiveInfo.has_value());
    REQUIRE(topArchiveInfo.has_value());
    CHECK(bottomArchiveInfo->compatibility == rachis::AccelerationStructureArchiveCompatibility::compatible);
    CHECK(topArchiveInfo->compatibility == rachis::AccelerationStructureArchiveCompatibility::compatible);

    auto roundTripBottomStorage = device.createBuffer({
        .size = bottomArchiveInfo->deserializedSize,
        .usage = rachis::BufferUsage::accelerationStructureStorage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Round-trip deserialized BLAS storage",
    });
    auto roundTripTopStorage = device.createBuffer({
        .size = topArchiveInfo->deserializedSize,
        .usage = rachis::BufferUsage::accelerationStructureStorage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Round-trip deserialized TLAS storage",
    });
    REQUIRE(roundTripBottomStorage.has_value());
    REQUIRE(roundTripTopStorage.has_value());
    auto roundTripBottomLevel = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::bottomLevel,
        .storage = *roundTripBottomStorage,
        .label = "Round-trip deserialized BLAS",
    });
    auto roundTripTopLevel = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::topLevel,
        .storage = *roundTripTopStorage,
        .label = "Round-trip deserialized TLAS",
    });
    REQUIRE(roundTripBottomLevel.has_value());
    REQUIRE(roundTripTopLevel.has_value());

    const std::vector<rachis::AccelerationStructureHandle> relocatedBottomLevels(
        static_cast<std::size_t>(topArchiveInfo->bottomLevelPointerCount), *roundTripBottomLevel);

    auto deserializeCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(deserializeCommand.has_value());
    REQUIRE(device
                .deserializeAccelerationStructure(*deserializeCommand,
                                                  {
                                                      .destination = *roundTripBottomLevel,
                                                      .archive = bottomArchive,
                                                  })
                .has_value());
    REQUIRE(device
                .deserializeAccelerationStructure(*deserializeCommand,
                                                  {
                                                      .destination = *roundTripTopLevel,
                                                      .archive = topArchive,
                                                      .bottomLevels = relocatedBottomLevels,
                                                  })
                .has_value());

    const std::array roundTripWrites{
        rachis::BindingWrite{.binding = 0, .resource = *roundTripTopLevel},
        rachis::BindingWrite{
            .binding = 1,
            .resource =
                rachis::BufferBinding{
                    .buffer = *output,
                    .size = 2 * sizeof(std::uint32_t),
                    .structureByteStride = 2 * sizeof(std::uint32_t),
                },
        },
    };
    auto roundTripBindingSet = rachis::test::createBindingSet(device,
                                                              {
                                                                  .layout = *bindingLayout,
                                                                  .writes = roundTripWrites,
                                                                  .label = "Round-trip deserialized binding set",
                                                              });
    REQUIRE(roundTripBindingSet.has_value());

    recordRayQuery(*deserializeCommand,
                   *roundTripBindingSet,
                   *output,
                   *readback,
                   rachis::PipelineStage::copy,
                   rachis::Access::copySource);
    auto deserializeSubmitted = device.submit(std::move(*deserializeCommand));
    REQUIRE(deserializeSubmitted.has_value());
    REQUIRE(device.wait(*deserializeSubmitted).has_value());

    std::array<std::uint32_t, 2> roundTripResult{};
    REQUIRE(device.readBuffer(*readback, 0, std::as_writable_bytes(std::span(roundTripResult))).has_value());
    CHECK(roundTripResult[0] == 1);
    CHECK(std::bit_cast<float>(roundTripResult[1]) == 2.5F);

    REQUIRE(device.destroyBindingSet(*roundTripBindingSet).has_value());
    REQUIRE(device.destroyBindingSet(*rebuiltBindingSet).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyAccelerationStructure(*roundTripTopLevel).has_value());
    REQUIRE(device.destroyAccelerationStructure(*roundTripBottomLevel).has_value());
    REQUIRE(device.destroyAccelerationStructure(*rebuiltTopLevel).has_value());
    REQUIRE(device.destroyAccelerationStructure(*topLevel).has_value());
    REQUIRE(device.destroyAccelerationStructure(*bottomLevel).has_value());
    REQUIRE(device.destroyQueryPool(*serializationPool).has_value());
    REQUIRE(device.destroyBuffer(*serializedReadback).has_value());
    REQUIRE(device.destroyBuffer(*serializedBuffer).has_value());
    REQUIRE(device.destroyBuffer(*serializationInfoReadback).has_value());
    REQUIRE(device.destroyBuffer(*roundTripTopStorage).has_value());
    REQUIRE(device.destroyBuffer(*roundTripBottomStorage).has_value());
    REQUIRE(device.destroyBuffer(*rebuiltReadback).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*rebuiltOutput).has_value());
    REQUIRE(device.destroyBuffer(*output).has_value());
    REQUIRE(device.destroyBuffer(*scratch).has_value());
    REQUIRE(device.destroyBuffer(*rebuiltTopStorage).has_value());
    REQUIRE(device.destroyBuffer(*topStorage).has_value());
    REQUIRE(device.destroyBuffer(*bottomStorage).has_value());
    REQUIRE(device.destroyBuffer(finalVertices).has_value());
    REQUIRE(device.destroyBuffer(refitVertices).has_value());
    REQUIRE(device.destroyBuffer(initialVertices).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    device.collectGarbage();

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("discarded command recordings roll back unsubmitted acceleration-structure versions",
          "[size:medium][rhi][ray-query][hardware]")
{
    auto created = rachis::createDevice({
        .adapterPreference = rachis::AdapterPreference::highPerformance,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        skipWithoutHardwareAdapter(created.error());
    }
    auto device = std::move(*created);
    const auto capabilities = *device.capabilities();
    if (!capabilities.rayQuery)
    {
        SKIP("The selected hardware adapter does not expose DXR 1.1 inline ray queries.");
    }

    const std::array vertices{
        std::array{-1.0F, -1.0F, 0.0F},
        std::array{0.0F, 1.0F, 0.0F},
        std::array{1.0F, -1.0F, 0.0F},
    };
    auto vertexBuffer = device.createBuffer({
        .size = sizeof(vertices),
        .usage = rachis::BufferUsage::accelerationStructureBuildInput,
        .memoryType = rachis::MemoryType::upload,
        .label = "Discarded-build vertices",
    });
    REQUIRE(vertexBuffer.has_value());
    REQUIRE(device.writeBuffer(*vertexBuffer, 0, std::as_bytes(std::span(vertices))).has_value());
    const std::array geometry{
        rachis::AccelerationStructureGeometry{
            .data =
                rachis::AccelerationStructureTriangleGeometry{
                    .vertexBuffer = *vertexBuffer,
                    .vertexCount = 3,
                    .vertexStride = sizeof(vertices[0]),
                    .vertexFormat = rachis::AccelerationStructureVertexFormat::float32x3,
                },
            .flags = rachis::AccelerationStructureGeometryFlag::opaque,
        },
    };
    constexpr auto buildFlags =
        rachis::AccelerationStructureBuildFlag::preferFastTrace | rachis::AccelerationStructureBuildFlag::allowUpdate;
    const rachis::BottomLevelAccelerationStructureBuildInput bottomInput{
        .geometries = geometry,
        .flags = buildFlags,
    };
    auto bottomSizes = device.queryBottomLevelAccelerationStructureBuildSizes(bottomInput);
    auto topSizes = device.queryTopLevelAccelerationStructureBuildSizes({.instanceCount = 1, .flags = buildFlags});
    REQUIRE(bottomSizes.has_value());
    REQUIRE(topSizes.has_value());

    const auto createBuffer = [&](std::uint64_t size, rachis::BufferUsage usage, const char* label) {
        auto buffer = device.createBuffer({
            .size = size,
            .usage = usage,
            .memoryType = rachis::MemoryType::deviceLocal,
            .label = label,
        });
        REQUIRE(buffer.has_value());
        return *buffer;
    };
    const auto createAccelerationStructure =
        [&](rachis::AccelerationStructureType type, rachis::BufferHandle storage, const char* label) {
            auto accelerationStructure = device.createAccelerationStructure({
                .type = type,
                .storage = storage,
                .label = label,
            });
            REQUIRE(accelerationStructure.has_value());
            return *accelerationStructure;
        };
    const auto firstStorage = createBuffer(bottomSizes->accelerationStructureSize,
                                           rachis::BufferUsage::accelerationStructureStorage,
                                           "Discarded-build first BLAS storage");
    const auto secondStorage = createBuffer(bottomSizes->accelerationStructureSize,
                                            rachis::BufferUsage::accelerationStructureStorage,
                                            "Discarded-build second BLAS storage");
    const auto topStorage = createBuffer(topSizes->accelerationStructureSize,
                                         rachis::BufferUsage::accelerationStructureStorage,
                                         "Discarded-build TLAS storage");
    const auto scratch = createBuffer((std::max)({bottomSizes->buildScratchSize,
                                                  bottomSizes->updateScratchSize,
                                                  topSizes->buildScratchSize,
                                                  topSizes->updateScratchSize}),
                                      rachis::BufferUsage::accelerationStructureScratch,
                                      "Discarded-build scratch");
    const auto output = createBuffer(2 * sizeof(std::uint32_t),
                                     rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
                                     "Discarded-build ray-query output");
    const auto firstBottomLevel = createAccelerationStructure(
        rachis::AccelerationStructureType::bottomLevel, firstStorage, "Discarded-build first BLAS");
    const auto secondBottomLevel = createAccelerationStructure(
        rachis::AccelerationStructureType::bottomLevel, secondStorage, "Discarded-build second BLAS");
    const auto topLevel =
        createAccelerationStructure(rachis::AccelerationStructureType::topLevel, topStorage, "Discarded-build TLAS");

    const auto bytecode = loadShader();
    REQUIRE_FALSE(bytecode.empty());
    const std::array shaderBindings{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::accelerationStructure, 1},
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
    };
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Discarded-build ray-query shader",
        .shaderInterface =
            rachis::ShaderInterfaceDesc{
                .bindings = shaderBindings,
                .workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1},
            },
    });
    REQUIRE(shader.has_value());
    const std::array layoutEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::accelerationStructure, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    auto bindingLayout = device.createBindingLayout({.entries = layoutEntries, .label = "Discarded-build layout"});
    REQUIRE(bindingLayout.has_value());
    const std::array bindingLayouts{*bindingLayout};
    auto pipelineLayout =
        device.createPipelineLayout({.bindingLayouts = bindingLayouts, .label = "Discarded-build pipeline layout"});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Discarded-build pipeline",
    });
    REQUIRE(pipeline.has_value());

    auto pool = device.createCommandPool({
        .queueType = rachis::QueueType::compute,
        .maxCommandBuffers = 3,
        .label = "Discarded-build pool",
        .flags = rachis::CommandPoolFlag::allowIndividualReset,
    });
    REQUIRE(pool.has_value());

    const std::array firstInstances{
        rachis::AccelerationStructureInstance{
            .bottomLevel = firstBottomLevel,
            .flags = rachis::AccelerationStructureInstanceFlag::triangleCullDisable,
        },
    };
    const std::array secondInstances{
        rachis::AccelerationStructureInstance{
            .bottomLevel = secondBottomLevel,
            .flags = rachis::AccelerationStructureInstanceFlag::triangleCullDisable,
        },
    };
    const auto buildBottomLevel = [&](rachis::CommandBuffer& command, rachis::AccelerationStructureHandle target) {
        return device.buildBottomLevelAccelerationStructure(command,
                                                            {
                                                                .destination = target,
                                                                .input = bottomInput,
                                                                .scratch = scratch,
                                                            });
    };
    const auto buildTopLevel = [&](rachis::CommandBuffer& command,
                                   std::span<const rachis::AccelerationStructureInstance> instances) {
        return device.buildTopLevelAccelerationStructure(command,
                                                         {
                                                             .destination = topLevel,
                                                             .instances = instances,
                                                             .flags = buildFlags,
                                                             .scratch = scratch,
                                                         });
    };
    const auto refitBottomLevel = [&](rachis::CommandBuffer& command, rachis::AccelerationStructureHandle target) {
        return device.updateBottomLevelAccelerationStructureInPlace(command,
                                                                    {
                                                                        .target = target,
                                                                        .input = bottomInput,
                                                                        .scratch = scratch,
                                                                    });
    };

    {
        auto abandoned = device.acquireCommandBuffer(*pool);
        REQUIRE(abandoned.has_value());
        REQUIRE(buildBottomLevel(*abandoned, firstBottomLevel).has_value());
    }
    auto command = device.acquireCommandBuffer(*pool);
    REQUIRE(command.has_value());
    auto topLevelOverAbandonedBuild = buildTopLevel(*command, firstInstances);
    REQUIRE_FALSE(topLevelOverAbandonedBuild.has_value());
    CHECK(topLevelOverAbandonedBuild.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(buildBottomLevel(*command, firstBottomLevel).has_value());

    REQUIRE(buildBottomLevel(*command, secondBottomLevel).has_value());
    REQUIRE(device.resetCommandBuffer(*command).has_value());
    auto topLevelOverResetBuild = buildTopLevel(*command, secondInstances);
    REQUIRE_FALSE(topLevelOverResetBuild.has_value());
    CHECK(topLevelOverResetBuild.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(buildBottomLevel(*command, firstBottomLevel).has_value());
    REQUIRE(buildBottomLevel(*command, secondBottomLevel).has_value());

    REQUIRE(device.resetCommandPool(*pool).has_value());
    auto topLevelOverPoolReset = buildTopLevel(*command, firstInstances);
    REQUIRE_FALSE(topLevelOverPoolReset.has_value());
    CHECK(topLevelOverPoolReset.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(buildBottomLevel(*command, firstBottomLevel).has_value());
    REQUIRE(buildBottomLevel(*command, secondBottomLevel).has_value());
    REQUIRE(buildTopLevel(*command, firstInstances).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    {
        auto abandonedRefit = device.acquireCommandBuffer(*pool);
        REQUIRE(abandonedRefit.has_value());
        REQUIRE(device
                    .updateTopLevelAccelerationStructureInPlace(*abandonedRefit,
                                                                {
                                                                    .target = topLevel,
                                                                    .instances = secondInstances,
                                                                    .flags = buildFlags,
                                                                    .scratch = scratch,
                                                                })
                    .has_value());
    }

    const std::array writes{
        rachis::BindingWrite{.binding = 0, .resource = topLevel},
        rachis::BindingWrite{
            .binding = 1,
            .resource =
                rachis::BufferBinding{
                    .buffer = output,
                    .size = 2 * sizeof(std::uint32_t),
                    .structureByteStride = 2 * sizeof(std::uint32_t),
                },
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device,
                                                     {
                                                         .layout = *bindingLayout,
                                                         .writes = writes,
                                                         .label = "Discarded-build binding set",
                                                     });
    REQUIRE(bindingSet.has_value());
    {
        auto binder = device.acquireCommandBuffer(*pool);
        auto refitter = device.acquireCommandBuffer(*pool);
        REQUIRE(binder.has_value());
        REQUIRE(refitter.has_value());
        REQUIRE(device.bindComputePipeline(*binder, *pipeline).has_value());
        REQUIRE(device.bindComputeBindingSet(*binder, 0, *bindingSet).has_value());

        auto boundBottomLevelRefit = refitBottomLevel(*refitter, firstBottomLevel);
        REQUIRE_FALSE(boundBottomLevelRefit.has_value());
        CHECK(boundBottomLevelRefit.error().code == rachis::ErrorCode::invalidState);
        REQUIRE(refitBottomLevel(*refitter, secondBottomLevel).has_value());
    }

    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyCommandPool(*pool).has_value());
    REQUIRE(device.destroyAccelerationStructure(topLevel).has_value());
    REQUIRE(device.destroyAccelerationStructure(secondBottomLevel).has_value());
    REQUIRE(device.destroyAccelerationStructure(firstBottomLevel).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBuffer(output).has_value());
    REQUIRE(device.destroyBuffer(scratch).has_value());
    REQUIRE(device.destroyBuffer(topStorage).has_value());
    REQUIRE(device.destroyBuffer(secondStorage).has_value());
    REQUIRE(device.destroyBuffer(firstStorage).has_value());
    REQUIRE(device.destroyBuffer(*vertexBuffer).has_value());
    device.collectGarbage();

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("in-place acceleration-structure updates exclude consumers in other recordings until they retire",
          "[size:medium][rhi][ray-query][hardware]")
{
    auto created = rachis::createDevice({
        .adapterPreference = rachis::AdapterPreference::highPerformance,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        skipWithoutHardwareAdapter(created.error());
    }
    auto device = std::move(*created);
    const auto capabilities = *device.capabilities();
    if (!capabilities.rayQuery)
    {
        SKIP("The selected hardware adapter does not expose DXR 1.1 inline ray queries.");
    }

    const std::array vertices{
        std::array{-1.0F, -1.0F, 0.0F},
        std::array{0.0F, 1.0F, 0.0F},
        std::array{1.0F, -1.0F, 0.0F},
    };
    auto vertexBuffer = device.createBuffer({
        .size = sizeof(vertices),
        .usage = rachis::BufferUsage::accelerationStructureBuildInput,
        .memoryType = rachis::MemoryType::upload,
        .label = "Exclusive-refit vertices",
    });
    REQUIRE(vertexBuffer.has_value());
    REQUIRE(device.writeBuffer(*vertexBuffer, 0, std::as_bytes(std::span(vertices))).has_value());
    const std::array geometry{
        rachis::AccelerationStructureGeometry{
            .data =
                rachis::AccelerationStructureTriangleGeometry{
                    .vertexBuffer = *vertexBuffer,
                    .vertexCount = 3,
                    .vertexStride = sizeof(vertices[0]),
                    .vertexFormat = rachis::AccelerationStructureVertexFormat::float32x3,
                },
            .flags = rachis::AccelerationStructureGeometryFlag::opaque,
        },
    };
    constexpr auto buildFlags = rachis::AccelerationStructureBuildFlag::preferFastTrace |
                                rachis::AccelerationStructureBuildFlag::allowUpdate |
                                rachis::AccelerationStructureBuildFlag::allowCompaction;
    const rachis::BottomLevelAccelerationStructureBuildInput bottomInput{
        .geometries = geometry,
        .flags = buildFlags,
    };
    auto bottomSizes = device.queryBottomLevelAccelerationStructureBuildSizes(bottomInput);
    auto topSizes = device.queryTopLevelAccelerationStructureBuildSizes({.instanceCount = 1, .flags = buildFlags});
    REQUIRE(bottomSizes.has_value());
    REQUIRE(topSizes.has_value());

    const auto createBuffer = [&](std::uint64_t size, rachis::BufferUsage usage, const char* label) {
        auto buffer = device.createBuffer({
            .size = size,
            .usage = usage,
            .memoryType = rachis::MemoryType::deviceLocal,
            .label = label,
        });
        REQUIRE(buffer.has_value());
        return *buffer;
    };
    const auto createAccelerationStructure =
        [&](rachis::AccelerationStructureType type, rachis::BufferHandle storage, const char* label) {
            auto accelerationStructure = device.createAccelerationStructure({
                .type = type,
                .storage = storage,
                .label = label,
            });
            REQUIRE(accelerationStructure.has_value());
            return *accelerationStructure;
        };
    const auto createStorage = [&](std::uint64_t size, const char* label) {
        return createBuffer(size, rachis::BufferUsage::accelerationStructureStorage, label);
    };
    const auto bottomStorage = createStorage(bottomSizes->accelerationStructureSize, "Exclusive-refit BLAS storage");
    const auto updatedBottomStorage =
        createStorage(bottomSizes->accelerationStructureSize, "Exclusive-refit updated BLAS storage");
    const auto compactedBottomStorage =
        createStorage(bottomSizes->accelerationStructureSize, "Exclusive-refit compacted BLAS storage");
    const auto topStorage = createStorage(topSizes->accelerationStructureSize, "Exclusive-refit TLAS storage");
    const auto foreignTopStorage =
        createStorage(topSizes->accelerationStructureSize, "Exclusive-refit foreign TLAS storage");
    const auto sameRecordingTopStorage =
        createStorage(topSizes->accelerationStructureSize, "Exclusive-refit same-recording TLAS storage");
    const auto scratch = createBuffer((std::max)({bottomSizes->buildScratchSize,
                                                  bottomSizes->updateScratchSize,
                                                  topSizes->buildScratchSize,
                                                  topSizes->updateScratchSize}),
                                      rachis::BufferUsage::accelerationStructureScratch,
                                      "Exclusive-refit scratch");
    const auto output = createBuffer(2 * sizeof(std::uint32_t),
                                     rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
                                     "Exclusive-refit ray-query output");
    constexpr std::uint64_t serializedSize = 4096;
    const auto serialized = createBuffer(serializedSize,
                                         rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
                                         "Exclusive-refit serialized archive");
    const auto bottomLevel =
        createAccelerationStructure(rachis::AccelerationStructureType::bottomLevel, bottomStorage, "Exclusive BLAS");
    const auto updatedBottomLevel = createAccelerationStructure(
        rachis::AccelerationStructureType::bottomLevel, updatedBottomStorage, "Exclusive updated BLAS");
    const auto compactedBottomLevel = createAccelerationStructure(
        rachis::AccelerationStructureType::bottomLevel, compactedBottomStorage, "Exclusive compacted BLAS");
    const auto topLevel =
        createAccelerationStructure(rachis::AccelerationStructureType::topLevel, topStorage, "Exclusive TLAS");
    const auto foreignTopLevel = createAccelerationStructure(
        rachis::AccelerationStructureType::topLevel, foreignTopStorage, "Exclusive foreign TLAS");
    const auto sameRecordingTopLevel = createAccelerationStructure(
        rachis::AccelerationStructureType::topLevel, sameRecordingTopStorage, "Exclusive same-recording TLAS");

    auto compactionPool = device.createAccelerationStructureCompactedSizeQueryPool({
        .queueType = rachis::QueueType::compute,
        .queryCount = 1,
        .label = "Exclusive-refit compacted sizes",
    });
    auto serializationPool = device.createAccelerationStructureSerializationInfoQueryPool({
        .queueType = rachis::QueueType::compute,
        .queryCount = 1,
        .label = "Exclusive-refit serialization information",
    });
    REQUIRE(compactionPool.has_value());
    REQUIRE(serializationPool.has_value());

    const auto bytecode = loadShader();
    REQUIRE_FALSE(bytecode.empty());
    const std::array shaderBindings{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::accelerationStructure, 1},
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
    };
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Exclusive-refit ray-query shader",
        .shaderInterface =
            rachis::ShaderInterfaceDesc{
                .bindings = shaderBindings,
                .workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1},
            },
    });
    REQUIRE(shader.has_value());
    const std::array layoutEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::accelerationStructure, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    auto bindingLayout = device.createBindingLayout({.entries = layoutEntries, .label = "Exclusive-refit layout"});
    REQUIRE(bindingLayout.has_value());
    const std::array bindingLayouts{*bindingLayout};
    auto pipelineLayout =
        device.createPipelineLayout({.bindingLayouts = bindingLayouts, .label = "Exclusive-refit pipeline layout"});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Exclusive-refit pipeline",
    });
    REQUIRE(pipeline.has_value());

    const std::array instances{
        rachis::AccelerationStructureInstance{
            .bottomLevel = bottomLevel,
            .flags = rachis::AccelerationStructureInstanceFlag::triangleCullDisable,
        },
    };
    const auto buildTopLevel = [&](rachis::CommandBuffer& command, rachis::AccelerationStructureHandle target) {
        return device.buildTopLevelAccelerationStructure(command,
                                                         {
                                                             .destination = target,
                                                             .instances = instances,
                                                             .flags = buildFlags,
                                                             .scratch = scratch,
                                                         });
    };
    const auto refitTopLevel = [&](rachis::CommandBuffer& command) {
        return device.updateTopLevelAccelerationStructureInPlace(command,
                                                                 {
                                                                     .target = topLevel,
                                                                     .instances = instances,
                                                                     .flags = buildFlags,
                                                                     .scratch = scratch,
                                                                 });
    };

    auto initialCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(initialCommand.has_value());
    REQUIRE(device
                .buildBottomLevelAccelerationStructure(*initialCommand,
                                                       {
                                                           .destination = bottomLevel,
                                                           .input = bottomInput,
                                                           .scratch = scratch,
                                                       })
                .has_value());
    REQUIRE(buildTopLevel(*initialCommand, topLevel).has_value());
    auto initialSubmitted = device.submit(std::move(*initialCommand));
    REQUIRE(initialSubmitted.has_value());
    REQUIRE(device.wait(*initialSubmitted).has_value());

    const std::array writes{
        rachis::BindingWrite{.binding = 0, .resource = topLevel},
        rachis::BindingWrite{
            .binding = 1,
            .resource =
                rachis::BufferBinding{
                    .buffer = output,
                    .size = 2 * sizeof(std::uint32_t),
                    .structureByteStride = 2 * sizeof(std::uint32_t),
                },
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device,
                                                     {
                                                         .layout = *bindingLayout,
                                                         .writes = writes,
                                                         .label = "Exclusive-refit binding set",
                                                     });
    REQUIRE(bindingSet.has_value());

    auto pool = device.createCommandPool({
        .queueType = rachis::QueueType::compute,
        .maxCommandBuffers = 2,
        .label = "Exclusive-refit pool",
    });
    REQUIRE(pool.has_value());
    auto refit = device.acquireCommandBuffer(*pool);
    auto foreign = device.acquireCommandBuffer(*pool);
    REQUIRE(refit.has_value());
    REQUIRE(foreign.has_value());
    REQUIRE(device
                .updateBottomLevelAccelerationStructureInPlace(*refit,
                                                               {
                                                                   .target = bottomLevel,
                                                                   .input = bottomInput,
                                                                   .scratch = scratch,
                                                               })
                .has_value());

    const auto requireInvalidState = [](const rachis::Status& status) {
        REQUIRE_FALSE(status.has_value());
        CHECK(status.error().code == rachis::ErrorCode::invalidState);
    };
    requireInvalidState(buildTopLevel(*foreign, foreignTopLevel));
    requireInvalidState(refitTopLevel(*foreign));
    requireInvalidState(device.updateBottomLevelAccelerationStructure(*foreign,
                                                                      {
                                                                          .source = bottomLevel,
                                                                          .destination = updatedBottomLevel,
                                                                          .input = bottomInput,
                                                                          .scratch = scratch,
                                                                      }));
    requireInvalidState(device.compactAccelerationStructure(*foreign,
                                                            {
                                                                .source = bottomLevel,
                                                                .destination = compactedBottomLevel,
                                                                .compactedSize = bottomSizes->accelerationStructureSize,
                                                            }));
    requireInvalidState(device.serializeAccelerationStructure(*foreign,
                                                              {
                                                                  .source = bottomLevel,
                                                                  .destination = serialized,
                                                                  .serializedSize = serializedSize,
                                                              }));
    requireInvalidState(device.writeAccelerationStructureCompactedSize(*foreign, *compactionPool, 0, bottomLevel));
    requireInvalidState(
        device.writeAccelerationStructureSerializationInfo(*foreign, *serializationPool, 0, bottomLevel));
    REQUIRE(device.bindComputePipeline(*foreign, *pipeline).has_value());
    requireInvalidState(device.bindComputeBindingSet(*foreign, 0, *bindingSet));

    REQUIRE(refitTopLevel(*refit).has_value());
    REQUIRE(buildTopLevel(*refit, sameRecordingTopLevel).has_value());
    REQUIRE(device.bindComputePipeline(*refit, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*refit, 0, *bindingSet).has_value());
    auto refitSubmitted = device.submit(std::move(*refit));
    REQUIRE(refitSubmitted.has_value());

    requireInvalidState(device.bindComputeBindingSet(*foreign, 0, *bindingSet));
    requireInvalidState(buildTopLevel(*foreign, foreignTopLevel));

    REQUIRE(device.wait(*refitSubmitted).has_value());
    REQUIRE(device.bindComputeBindingSet(*foreign, 0, *bindingSet).has_value());
    REQUIRE(buildTopLevel(*foreign, foreignTopLevel).has_value());
    auto foreignSubmitted = device.submit(std::move(*foreign));
    REQUIRE(foreignSubmitted.has_value());
    REQUIRE(device.wait(*foreignSubmitted).has_value());

    REQUIRE(device.destroyCommandPool(*pool).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyQueryPool(*serializationPool).has_value());
    REQUIRE(device.destroyQueryPool(*compactionPool).has_value());
    REQUIRE(device.destroyAccelerationStructure(sameRecordingTopLevel).has_value());
    REQUIRE(device.destroyAccelerationStructure(foreignTopLevel).has_value());
    REQUIRE(device.destroyAccelerationStructure(topLevel).has_value());
    REQUIRE(device.destroyAccelerationStructure(compactedBottomLevel).has_value());
    REQUIRE(device.destroyAccelerationStructure(updatedBottomLevel).has_value());
    REQUIRE(device.destroyAccelerationStructure(bottomLevel).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBuffer(serialized).has_value());
    REQUIRE(device.destroyBuffer(output).has_value());
    REQUIRE(device.destroyBuffer(scratch).has_value());
    REQUIRE(device.destroyBuffer(sameRecordingTopStorage).has_value());
    REQUIRE(device.destroyBuffer(foreignTopStorage).has_value());
    REQUIRE(device.destroyBuffer(topStorage).has_value());
    REQUIRE(device.destroyBuffer(compactedBottomStorage).has_value());
    REQUIRE(device.destroyBuffer(updatedBottomStorage).has_value());
    REQUIRE(device.destroyBuffer(bottomStorage).has_value());
    REQUIRE(device.destroyBuffer(*vertexBuffer).has_value());
    device.collectGarbage();

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("enhanced acceleration-structure barriers order postbuild queries and copies after every write",
          "[size:medium][rhi][ray-query][hardware]")
{
    auto created = rachis::createDevice({
        .adapterPreference = rachis::AdapterPreference::highPerformance,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        skipWithoutHardwareAdapter(created.error());
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (!capabilities.rayTracing || !capabilities.enhancedBarriers)
    {
        SKIP("The selected hardware adapter does not expose DXR acceleration structures with enhanced barriers.");
    }

    const std::array vertices{
        std::array{-1.0F, -1.0F, 0.0F},
        std::array{0.0F, 1.0F, 0.0F},
        std::array{1.0F, -1.0F, 0.0F},
    };
    auto vertexBuffer = device.createBuffer({
        .size = sizeof(vertices),
        .usage = rachis::BufferUsage::accelerationStructureBuildInput,
        .memoryType = rachis::MemoryType::upload,
        .label = "Barrier-scope triangle vertices",
    });
    REQUIRE(vertexBuffer.has_value());
    REQUIRE(device.writeBuffer(*vertexBuffer, 0, std::as_bytes(std::span(vertices))).has_value());
    const std::array geometry{
        rachis::AccelerationStructureGeometry{
            .data =
                rachis::AccelerationStructureTriangleGeometry{
                    .vertexBuffer = *vertexBuffer,
                    .vertexCount = 3,
                    .vertexStride = sizeof(vertices[0]),
                    .vertexFormat = rachis::AccelerationStructureVertexFormat::float32x3,
                },
            .flags = rachis::AccelerationStructureGeometryFlag::opaque,
        },
    };
    constexpr auto buildFlags = rachis::AccelerationStructureBuildFlag::preferFastTrace |
                                rachis::AccelerationStructureBuildFlag::allowUpdate |
                                rachis::AccelerationStructureBuildFlag::allowCompaction;
    const rachis::BottomLevelAccelerationStructureBuildInput bottomInput{
        .geometries = geometry,
        .flags = buildFlags,
    };
    auto bottomSizes = device.queryBottomLevelAccelerationStructureBuildSizes(bottomInput);
    auto topSizes = device.queryTopLevelAccelerationStructureBuildSizes({
        .instanceCount = 1,
        .flags = buildFlags,
    });
    REQUIRE(bottomSizes.has_value());
    REQUIRE(topSizes.has_value());

    std::vector<rachis::BufferHandle> buffers;
    std::vector<rachis::AccelerationStructureHandle> accelerationStructures;
    const auto createBuffer = [&](const rachis::BufferDesc& desc) {
        auto buffer = device.createBuffer(desc);
        REQUIRE(buffer.has_value());
        buffers.push_back(*buffer);
        return *buffer;
    };
    const auto createAccelerationStructure =
        [&](rachis::AccelerationStructureType type, std::uint64_t size, const char* label) {
            const auto storage = createBuffer({
                .size = size,
                .usage = rachis::BufferUsage::accelerationStructureStorage,
                .memoryType = rachis::MemoryType::deviceLocal,
                .label = label,
            });
            auto accelerationStructure = device.createAccelerationStructure({
                .type = type,
                .storage = storage,
                .label = label,
            });
            REQUIRE(accelerationStructure.has_value());
            accelerationStructures.push_back(*accelerationStructure);
            return *accelerationStructure;
        };

    const auto bottomLevel = createAccelerationStructure(
        rachis::AccelerationStructureType::bottomLevel, bottomSizes->accelerationStructureSize, "Barrier-scope BLAS");
    const auto updatedBottomLevel = createAccelerationStructure(rachis::AccelerationStructureType::bottomLevel,
                                                                bottomSizes->accelerationStructureSize,
                                                                "Barrier-scope updated BLAS");
    const auto topLevel = createAccelerationStructure(
        rachis::AccelerationStructureType::topLevel, topSizes->accelerationStructureSize, "Barrier-scope TLAS");
    const auto updatedTopLevel = createAccelerationStructure(
        rachis::AccelerationStructureType::topLevel, topSizes->accelerationStructureSize, "Barrier-scope updated TLAS");
    const auto scratch = createBuffer({
        .size = (std::max)({bottomSizes->buildScratchSize,
                            bottomSizes->updateScratchSize,
                            topSizes->buildScratchSize,
                            topSizes->updateScratchSize}),
        .usage = rachis::BufferUsage::accelerationStructureScratch,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Barrier-scope scratch",
    });
    const auto compactionReadback = createBuffer({
        .size = rachis::accelerationStructureCompactedSizeQueryResultSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Barrier-scope compacted-size readback",
    });
    const auto serializationInfoReadback = createBuffer({
        .size = rachis::accelerationStructureSerializationInfoQueryResultSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Barrier-scope serialization-info readback",
    });
    auto compactionPool = device.createAccelerationStructureCompactedSizeQueryPool({
        .queueType = rachis::QueueType::compute,
        .queryCount = 1,
        .label = "Barrier-scope compacted size",
    });
    auto serializationPool = device.createAccelerationStructureSerializationInfoQueryPool({
        .queueType = rachis::QueueType::compute,
        .queryCount = 1,
        .label = "Barrier-scope serialization info",
    });
    REQUIRE(compactionPool.has_value());
    REQUIRE(serializationPool.has_value());

    const std::array instances{
        rachis::AccelerationStructureInstance{
            .bottomLevel = bottomLevel,
            .flags = rachis::AccelerationStructureInstanceFlag::triangleCullDisable,
        },
    };
    rachis::d3d12::test::lastAccelerationStructureGlobalBarrier().reset();
    rachis::d3d12::test::lastAccelerationStructureInPlaceUpdateBarrier().reset();

    auto buildCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(buildCommand.has_value());
    REQUIRE(device
                .buildBottomLevelAccelerationStructure(*buildCommand,
                                                       {
                                                           .destination = bottomLevel,
                                                           .input = bottomInput,
                                                           .scratch = scratch,
                                                       })
                .has_value());
    checkAccelerationStructureBuildBarrier("BLAS build");
    checkAccelerationStructureInPlaceUpdateBarrier("BLAS build", false);
    REQUIRE(device
                .updateBottomLevelAccelerationStructureInPlace(*buildCommand,
                                                               {
                                                                   .target = bottomLevel,
                                                                   .input = bottomInput,
                                                                   .scratch = scratch,
                                                               })
                .has_value());
    checkAccelerationStructureBuildBarrier("BLAS in-place update");
    checkAccelerationStructureInPlaceUpdateBarrier("BLAS in-place update", true);
    REQUIRE(device
                .updateBottomLevelAccelerationStructure(*buildCommand,
                                                        {
                                                            .source = bottomLevel,
                                                            .destination = updatedBottomLevel,
                                                            .input = bottomInput,
                                                            .scratch = scratch,
                                                        })
                .has_value());
    checkAccelerationStructureBuildBarrier("BLAS update");
    checkAccelerationStructureInPlaceUpdateBarrier("BLAS update", false);
    REQUIRE(device
                .buildTopLevelAccelerationStructure(*buildCommand,
                                                    {
                                                        .destination = topLevel,
                                                        .instances = instances,
                                                        .flags = buildFlags,
                                                        .scratch = scratch,
                                                    })
                .has_value());
    checkAccelerationStructureBuildBarrier("TLAS build");
    checkAccelerationStructureInPlaceUpdateBarrier("TLAS build", false);
    REQUIRE(device
                .updateTopLevelAccelerationStructureInPlace(*buildCommand,
                                                            {
                                                                .target = topLevel,
                                                                .instances = instances,
                                                                .flags = buildFlags,
                                                                .scratch = scratch,
                                                            })
                .has_value());
    checkAccelerationStructureBuildBarrier("TLAS in-place update");
    checkAccelerationStructureInPlaceUpdateBarrier("TLAS in-place update", true);
    REQUIRE(device
                .updateTopLevelAccelerationStructure(*buildCommand,
                                                     {
                                                         .source = topLevel,
                                                         .destination = updatedTopLevel,
                                                         .instances = instances,
                                                         .flags = buildFlags,
                                                         .scratch = scratch,
                                                     })
                .has_value());
    checkAccelerationStructureBuildBarrier("TLAS update");
    checkAccelerationStructureInPlaceUpdateBarrier("TLAS update", false);
    REQUIRE(device.writeAccelerationStructureCompactedSize(*buildCommand, *compactionPool, 0, bottomLevel).has_value());
    REQUIRE(device.writeAccelerationStructureSerializationInfo(*buildCommand, *serializationPool, 0, bottomLevel)
                .has_value());
    REQUIRE(device
                .resolveAccelerationStructureCompactedSizes(*buildCommand,
                                                            {
                                                                .queryPool = *compactionPool,
                                                                .firstQuery = 0,
                                                                .queryCount = 1,
                                                                .destination = compactionReadback,
                                                            })
                .has_value());
    REQUIRE(device
                .resolveAccelerationStructureSerializationInfo(*buildCommand,
                                                               {
                                                                   .queryPool = *serializationPool,
                                                                   .firstQuery = 0,
                                                                   .queryCount = 1,
                                                                   .destination = serializationInfoReadback,
                                                               })
                .has_value());
    auto buildSubmitted = device.submit(std::move(*buildCommand));
    REQUIRE(buildSubmitted.has_value());
    REQUIRE(device.wait(*buildSubmitted).has_value());

    std::array<std::uint64_t, 1> compactedSize{};
    std::array<rachis::AccelerationStructureSerializationInfo, 1> serializationInfo{};
    REQUIRE(device.readBuffer(compactionReadback, 0, std::as_writable_bytes(std::span(compactedSize))).has_value());
    REQUIRE(device.readBuffer(serializationInfoReadback, 0, std::as_writable_bytes(std::span(serializationInfo)))
                .has_value());
    REQUIRE(compactedSize[0] > 0);
    REQUIRE(compactedSize[0] <= bottomSizes->accelerationStructureSize);
    REQUIRE(serializationInfo[0].serializedSize > 0);

    const auto compactedBottomLevel = createAccelerationStructure(
        rachis::AccelerationStructureType::bottomLevel, compactedSize[0], "Barrier-scope compacted BLAS");
    const auto serialized = createBuffer({
        .size = serializationInfo[0].serializedSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Barrier-scope serialized BLAS",
    });
    const auto serializedReadback = createBuffer({
        .size = serializationInfo[0].serializedSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Barrier-scope serialized BLAS readback",
    });

    auto copyCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(copyCommand.has_value());
    REQUIRE(device
                .compactAccelerationStructure(*copyCommand,
                                              {
                                                  .source = bottomLevel,
                                                  .destination = compactedBottomLevel,
                                                  .compactedSize = compactedSize[0],
                                              })
                .has_value());
    checkAccelerationStructureCopyBarrier("compaction");
    REQUIRE(device
                .bufferBarrier(*copyCommand,
                               {
                                   .buffer = serialized,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::accelerationStructureCopy,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device
                .serializeAccelerationStructure(*copyCommand,
                                                {
                                                    .source = bottomLevel,
                                                    .destination = serialized,
                                                    .serializedSize = serializationInfo[0].serializedSize,
                                                })
                .has_value());
    checkAccelerationStructureCopyBarrier("serialization");
    REQUIRE(device
                .bufferBarrier(*copyCommand,
                               {
                                   .buffer = serialized,
                                   .stagesBefore = rachis::PipelineStage::accelerationStructureCopy,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device
                .copyBuffer(*copyCommand,
                            {
                                .source = serialized,
                                .destination = serializedReadback,
                                .size = serializationInfo[0].serializedSize,
                            })
                .has_value());
    auto copySubmitted = device.submit(std::move(*copyCommand));
    REQUIRE(copySubmitted.has_value());
    REQUIRE(device.wait(*copySubmitted).has_value());

    std::vector<std::byte> archive(static_cast<std::size_t>(serializationInfo[0].serializedSize));
    REQUIRE(device.readBuffer(serializedReadback, 0, archive).has_value());
    auto archiveInfo = device.inspectAccelerationStructureArchive(archive);
    REQUIRE(archiveInfo.has_value());
    REQUIRE(archiveInfo->compatibility == rachis::AccelerationStructureArchiveCompatibility::compatible);
    const auto importedBottomLevel = createAccelerationStructure(
        rachis::AccelerationStructureType::bottomLevel, archiveInfo->deserializedSize, "Barrier-scope imported BLAS");

    auto importCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(importCommand.has_value());
    REQUIRE(device
                .deserializeAccelerationStructure(*importCommand,
                                                  {
                                                      .destination = importedBottomLevel,
                                                      .archive = archive,
                                                  })
                .has_value());
    checkAccelerationStructureCopyBarrier("deserialization");
    auto importSubmitted = device.submit(std::move(*importCommand));
    REQUIRE(importSubmitted.has_value());
    REQUIRE(device.wait(*importSubmitted).has_value());

    for (const auto accelerationStructure : accelerationStructures)
    {
        REQUIRE(device.destroyAccelerationStructure(accelerationStructure).has_value());
    }
    for (const auto buffer : buffers)
    {
        REQUIRE(device.destroyBuffer(buffer).has_value());
    }
    REQUIRE(device.destroyQueryPool(*serializationPool).has_value());
    REQUIRE(device.destroyQueryPool(*compactionPool).has_value());
    REQUIRE(device.destroyBuffer(*vertexBuffer).has_value());
    device.collectGarbage();

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("same-list postbuild queries of a large compactable BLAS match fenced queries on every queue",
          "[size:medium][rhi][ray-query][hardware]")
{
    auto created = rachis::createDevice({
        .adapterPreference = rachis::AdapterPreference::highPerformance,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        skipWithoutHardwareAdapter(created.error());
    }
    auto device = std::move(*created);
    if (!rachis::test::requireCapabilities(device).rayTracing)
    {
        SKIP("The selected hardware adapter does not expose DXR acceleration structures.");
    }

    constexpr std::uint32_t triangleCount = 1U << 20U;
    std::vector<std::array<float, 3>> vertices(static_cast<std::size_t>(triangleCount) * 3U);
    std::uint32_t seed = 0x2545F491U;
    const auto nextCoordinate = [&seed] {
        seed = seed * 1664525U + 1013904223U;
        return static_cast<float>(seed >> 8U) * (2.0F / 16777216.0F) - 1.0F;
    };
    for (auto& vertex : vertices)
    {
        vertex = {nextCoordinate(), nextCoordinate(), nextCoordinate()};
    }
    auto vertexBuffer = device.createBuffer({
        .size = vertices.size() * sizeof(vertices[0]),
        .usage = rachis::BufferUsage::accelerationStructureBuildInput,
        .memoryType = rachis::MemoryType::upload,
        .label = "Large compactable BLAS vertices",
    });
    REQUIRE(vertexBuffer.has_value());
    REQUIRE(device.writeBuffer(*vertexBuffer, 0, std::as_bytes(std::span(vertices))).has_value());
    const std::array geometry{
        rachis::AccelerationStructureGeometry{
            .data =
                rachis::AccelerationStructureTriangleGeometry{
                    .vertexBuffer = *vertexBuffer,
                    .vertexCount = static_cast<std::uint32_t>(vertices.size()),
                    .vertexStride = sizeof(vertices[0]),
                    .vertexFormat = rachis::AccelerationStructureVertexFormat::float32x3,
                },
            .flags = rachis::AccelerationStructureGeometryFlag::opaque,
        },
    };
    const rachis::BottomLevelAccelerationStructureBuildInput input{
        .geometries = geometry,
        .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace |
                 rachis::AccelerationStructureBuildFlag::allowCompaction,
    };
    auto sizes = device.queryBottomLevelAccelerationStructureBuildSizes(input);
    REQUIRE(sizes.has_value());

    for (const auto queueType : {rachis::QueueType::compute, rachis::QueueType::graphics})
    {
        INFO((queueType == rachis::QueueType::compute ? "compute queue" : "direct queue"));
        auto storage = device.createBuffer({
            .size = sizes->accelerationStructureSize,
            .usage = rachis::BufferUsage::accelerationStructureStorage,
            .memoryType = rachis::MemoryType::deviceLocal,
            .label = "Large compactable BLAS storage",
        });
        auto scratch = device.createBuffer({
            .size = sizes->buildScratchSize,
            .usage = rachis::BufferUsage::accelerationStructureScratch,
            .memoryType = rachis::MemoryType::deviceLocal,
            .label = "Large compactable BLAS scratch",
        });
        auto compactionReadback = device.createBuffer({
            .size = 2 * rachis::accelerationStructureCompactedSizeQueryResultSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Large compactable BLAS compacted-size readback",
        });
        auto serializationInfoReadback = device.createBuffer({
            .size = 2 * rachis::accelerationStructureSerializationInfoQueryResultSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Large compactable BLAS serialization-info readback",
        });
        REQUIRE(storage.has_value());
        REQUIRE(scratch.has_value());
        REQUIRE(compactionReadback.has_value());
        REQUIRE(serializationInfoReadback.has_value());
        auto bottomLevel = device.createAccelerationStructure({
            .type = rachis::AccelerationStructureType::bottomLevel,
            .storage = *storage,
            .label = "Large compactable BLAS",
        });
        auto compactionPool = device.createAccelerationStructureCompactedSizeQueryPool({
            .queueType = queueType,
            .queryCount = 2,
            .label = "Large compactable BLAS compacted sizes",
        });
        auto serializationPool = device.createAccelerationStructureSerializationInfoQueryPool({
            .queueType = queueType,
            .queryCount = 2,
            .label = "Large compactable BLAS serialization info",
        });
        REQUIRE(bottomLevel.has_value());
        REQUIRE(compactionPool.has_value());
        REQUIRE(serializationPool.has_value());

        const auto recordQueries = [&](rachis::CommandBuffer& commandBuffer, std::uint32_t queryIndex) {
            REQUIRE(
                device.writeAccelerationStructureCompactedSize(commandBuffer, *compactionPool, queryIndex, *bottomLevel)
                    .has_value());
            REQUIRE(device
                        .writeAccelerationStructureSerializationInfo(
                            commandBuffer, *serializationPool, queryIndex, *bottomLevel)
                        .has_value());
            REQUIRE(
                device
                    .resolveAccelerationStructureCompactedSizes(
                        commandBuffer,
                        {
                            .queryPool = *compactionPool,
                            .firstQuery = queryIndex,
                            .queryCount = 1,
                            .destination = *compactionReadback,
                            .destinationOffset = queryIndex * rachis::accelerationStructureCompactedSizeQueryResultSize,
                        })
                    .has_value());
            REQUIRE(device
                        .resolveAccelerationStructureSerializationInfo(
                            commandBuffer,
                            {
                                .queryPool = *serializationPool,
                                .firstQuery = queryIndex,
                                .queryCount = 1,
                                .destination = *serializationInfoReadback,
                                .destinationOffset =
                                    queryIndex * rachis::accelerationStructureSerializationInfoQueryResultSize,
                            })
                        .has_value());
        };

        auto buildCommand = rachis::test::acquireCommandBuffer(device, queueType);
        REQUIRE(buildCommand.has_value());
        REQUIRE(device
                    .buildBottomLevelAccelerationStructure(*buildCommand,
                                                           {
                                                               .destination = *bottomLevel,
                                                               .input = input,
                                                               .scratch = *scratch,
                                                           })
                    .has_value());
        recordQueries(*buildCommand, 0);
        auto buildSubmitted = device.submit(std::move(*buildCommand));
        REQUIRE(buildSubmitted.has_value());
        REQUIRE(device.wait(*buildSubmitted).has_value());

        auto fencedCommand = rachis::test::acquireCommandBuffer(device, queueType);
        REQUIRE(fencedCommand.has_value());
        recordQueries(*fencedCommand, 1);
        auto fencedSubmitted = device.submit(std::move(*fencedCommand));
        REQUIRE(fencedSubmitted.has_value());
        REQUIRE(device.wait(*fencedSubmitted).has_value());

        std::array<std::uint64_t, 2> compactedSizes{};
        std::array<rachis::AccelerationStructureSerializationInfo, 2> serializationInfo{};
        REQUIRE(
            device.readBuffer(*compactionReadback, 0, std::as_writable_bytes(std::span(compactedSizes))).has_value());
        REQUIRE(device.readBuffer(*serializationInfoReadback, 0, std::as_writable_bytes(std::span(serializationInfo)))
                    .has_value());
        CHECK(compactedSizes[1] > 0);
        CHECK(compactedSizes[1] <= sizes->accelerationStructureSize);
        CHECK(compactedSizes[0] == compactedSizes[1]);
        CHECK(serializationInfo[1].serializedSize > 0);
        CHECK(serializationInfo[0].serializedSize == serializationInfo[1].serializedSize);
        CHECK(serializationInfo[0].bottomLevelPointerCount == serializationInfo[1].bottomLevelPointerCount);

        REQUIRE(device.destroyQueryPool(*serializationPool).has_value());
        REQUIRE(device.destroyQueryPool(*compactionPool).has_value());
        REQUIRE(device.destroyAccelerationStructure(*bottomLevel).has_value());
        REQUIRE(device.destroyBuffer(*serializationInfoReadback).has_value());
        REQUIRE(device.destroyBuffer(*compactionReadback).has_value());
        REQUIRE(device.destroyBuffer(*scratch).has_value());
        REQUIRE(device.destroyBuffer(*storage).has_value());
        device.collectGarbage();
    }
    REQUIRE(device.destroyBuffer(*vertexBuffer).has_value());
    device.collectGarbage();

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
