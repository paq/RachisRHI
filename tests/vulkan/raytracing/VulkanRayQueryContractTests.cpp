#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include "Internal/VulkanResources.h"
#include "Vulkan/VulkanTestHooks.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <volk.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

    [[nodiscard]] rachis::DeviceDesc vulkanDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::vulkan,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
    }

    [[nodiscard]] std::vector<std::byte> loadBytes(const std::filesystem::path& path)
    {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        const auto end = stream.tellg();
        if (end <= 0)
        {
            return {};
        }
        std::vector<std::byte> bytes(static_cast<std::size_t>(end));
        stream.seekg(0, std::ios::beg);
        stream.read(reinterpret_cast<char*>(bytes.data()), end);
        return stream ? bytes : std::vector<std::byte>{};
    }

#ifdef RACHIS_SPIRV_DIR
    [[nodiscard]] std::vector<std::byte> loadSpirv(const char* name)
    {
        return loadBytes(std::filesystem::path(RACHIS_SPIRV_DIR) / name);
    }
#endif

} // namespace

#ifdef RACHIS_SPIRV_DIR
TEST_CASE("Vulkan inline ray query traverses a built triangle TLAS", "[size:medium][rhi][vulkan][ray-query]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (!capabilities.rayQuery)
    {
        SKIP("The Vulkan adapter does not advertise inline ray queries.");
    }

    const auto bytecode = loadSpirv("RayQueryWorkload.spv");
    REQUIRE_FALSE(bytecode.empty());
    const std::array shaderBindings{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::accelerationStructure, 1},
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
    };
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Vulkan inline ray-query shader",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, shaderBindings),
    });
    if (!shader)
    {
        INFO(shader.error().message);
    }
    REQUIRE(shader.has_value());

    const std::array vertices{
        std::array{-1.0F, -1.0F, 0.0F},
        std::array{0.0F, 1.0F, 0.0F},
        std::array{1.0F, -1.0F, 0.0F},
    };
    auto vertexBuffer = device.createBuffer({
        .size = sizeof(vertices),
        .usage = rachis::BufferUsage::accelerationStructureBuildInput,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan ray-query triangle vertices",
    });
    if (!vertexBuffer)
    {
        INFO(vertexBuffer.error().message);
    }
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
        .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace,
    };
    CHECK_FALSE(device.queryBottomLevelAccelerationStructureBuildSizes({}).has_value());
    auto bottomSizes = device.queryBottomLevelAccelerationStructureBuildSizes(bottomInput);
    REQUIRE(bottomSizes.has_value());
    auto topSizes = device.queryTopLevelAccelerationStructureBuildSizes({
        .instanceCount = 1,
        .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace,
    });
    REQUIRE(topSizes.has_value());

    auto bottomStorage = device.createBuffer({
        .size = bottomSizes->accelerationStructureSize,
        .usage = rachis::BufferUsage::accelerationStructureStorage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan ray-query BLAS storage",
    });
    auto topStorage = device.createBuffer({
        .size = topSizes->accelerationStructureSize,
        .usage = rachis::BufferUsage::accelerationStructureStorage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan ray-query TLAS storage",
    });
    const auto scratchSize = (std::max)(bottomSizes->buildScratchSize, topSizes->buildScratchSize);
    auto scratch = device.createBuffer({
        .size = scratchSize,
        .usage = rachis::BufferUsage::accelerationStructureScratch,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan ray-query build scratch",
    });
    REQUIRE(bottomStorage.has_value());
    REQUIRE(topStorage.has_value());
    REQUIRE(scratch.has_value());

    auto bottomLevel = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::bottomLevel,
        .storage = *bottomStorage,
        .label = "Vulkan ray-query BLAS",
    });
    auto topLevel = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::topLevel,
        .storage = *topStorage,
        .label = "Vulkan ray-query TLAS",
    });
    REQUIRE(bottomLevel.has_value());
    REQUIRE(topLevel.has_value());
    auto overlapping = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::bottomLevel,
        .storage = *bottomStorage,
        .label = "Overlapping Vulkan ray-query BLAS",
    });
    REQUIRE_FALSE(overlapping.has_value());
    CHECK(overlapping.error().code == rachis::ErrorCode::invalidArgument);

    auto output = device.createBuffer({
        .size = 2 * sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan ray-query output",
    });
    auto readback = device.createBuffer({
        .size = 2 * sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan ray-query readback",
    });
    REQUIRE(output.has_value());
    REQUIRE(readback.has_value());

    const std::array layoutEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::accelerationStructure, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    auto bindingLayout = device.createBindingLayout({.entries = layoutEntries, .label = "Vulkan ray-query resources"});
    REQUIRE(bindingLayout.has_value());
    const std::array bindingLayouts{*bindingLayout};
    auto pipelineLayout =
        device.createPipelineLayout({.bindingLayouts = bindingLayouts, .label = "Vulkan ray-query layout"});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Vulkan ray-query pipeline",
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
    auto rebuilt = device.buildBottomLevelAccelerationStructure(*command,
                                                                {
                                                                    .destination = *bottomLevel,
                                                                    .input = bottomInput,
                                                                    .scratch = *scratch,
                                                                });
    REQUIRE_FALSE(rebuilt.has_value());
    CHECK(rebuilt.error().code == rachis::ErrorCode::invalidState);
    const std::array instances{
        rachis::AccelerationStructureInstance{
            .bottomLevel = *bottomLevel,
            .flags = rachis::AccelerationStructureInstanceFlag::triangleCullDisable,
        },
    };
    REQUIRE(
        device
            .buildTopLevelAccelerationStructure(*command,
                                                {
                                                    .destination = *topLevel,
                                                    .instances = instances,
                                                    .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace,
                                                    .scratch = *scratch,
                                                })
            .has_value());

    const std::array writes{
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
                                                         .writes = writes,
                                                         .label = "Vulkan ray-query binding set",
                                                     });
    if (!bindingSet)
    {
        INFO(bindingSet.error().message);
    }
    REQUIRE(bindingSet.has_value());
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

    const std::array bottomLevelWrites{
        rachis::BindingWrite{.binding = 0, .resource = *bottomLevel},
        writes[1],
    };
    auto bottomLevelSet = rachis::test::createBindingSet(device,
                                                         {
                                                             .layout = *bindingLayout,
                                                             .writes = bottomLevelWrites,
                                                             .label = "Vulkan ray-query BLAS binding set",
                                                         });
    REQUIRE_FALSE(bottomLevelSet.has_value());
    CHECK(bottomLevelSet.error().code == rachis::ErrorCode::invalidArgument);
    const std::array bottomLevelUpdate{rachis::BindingWrite{.binding = 0, .resource = *bottomLevel}};
    auto bottomLevelUpdated = device.updateBindingSet(*bindingSet, bottomLevelUpdate);
    REQUIRE_FALSE(bottomLevelUpdated.has_value());
    CHECK(bottomLevelUpdated.error().code == rachis::ErrorCode::invalidArgument);

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
    REQUIRE(device.destroyBuffer(*vertexBuffer).has_value());
    REQUIRE(device.destroyBuffer(*scratch).has_value());
    REQUIRE(device.destroyBuffer(*bottomStorage).has_value());
    REQUIRE(device.destroyBuffer(*topStorage).has_value());
    REQUIRE(device.destroyBuffer(*output).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    std::array<std::uint32_t, 2> result{};
    REQUIRE(device.readBuffer(*readback, 0, std::as_writable_bytes(std::span(result))).has_value());
    CHECK(result[0] == 1);
    CHECK(std::bit_cast<float>(result[1]) == 1.0F);
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
#endif

TEST_CASE("Vulkan acceleration-structure builds reject destinations smaller than the build requires",
          "[size:medium][rhi][vulkan][ray-query]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (!capabilities.rayTracing && !capabilities.rayQuery)
    {
        SKIP("The Vulkan adapter does not advertise acceleration structures.");
    }

    std::vector<std::array<float, 3>> vertices(768);
    for (std::size_t index = 0; index < vertices.size(); ++index)
    {
        vertices[index] = {
            static_cast<float>(index % 3U),
            static_cast<float>((index + 1U) % 3U),
            static_cast<float>(index / 3U),
        };
    }
    auto vertexBuffer = device.createBuffer({
        .size = vertices.size() * sizeof(vertices[0]),
        .usage = rachis::BufferUsage::accelerationStructureBuildInput,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan undersized-destination vertices",
    });
    REQUIRE(vertexBuffer.has_value());
    REQUIRE(device.writeBuffer(*vertexBuffer, 0, std::as_bytes(std::span(vertices))).has_value());

    const auto triangles = [&](std::uint32_t vertexCount) {
        return rachis::AccelerationStructureGeometry{
            .data =
                rachis::AccelerationStructureTriangleGeometry{
                    .vertexBuffer = *vertexBuffer,
                    .vertexCount = vertexCount,
                    .vertexStride = sizeof(vertices[0]),
                    .vertexFormat = rachis::AccelerationStructureVertexFormat::float32x3,
                },
            .flags = rachis::AccelerationStructureGeometryFlag::opaque,
        };
    };
    const std::array smallGeometry{triangles(3)};
    const std::array largeGeometry{triangles(static_cast<std::uint32_t>(vertices.size()))};
    const rachis::BottomLevelAccelerationStructureBuildInput smallInput{
        .geometries = smallGeometry,
        .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace,
    };
    const rachis::BottomLevelAccelerationStructureBuildInput largeInput{
        .geometries = largeGeometry,
        .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace,
    };
    constexpr std::uint32_t largeInstanceCount = 64;
    auto smallBottomSizes = device.queryBottomLevelAccelerationStructureBuildSizes(smallInput);
    auto largeBottomSizes = device.queryBottomLevelAccelerationStructureBuildSizes(largeInput);
    auto smallTopSizes = device.queryTopLevelAccelerationStructureBuildSizes({
        .instanceCount = 1,
        .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace,
    });
    auto largeTopSizes = device.queryTopLevelAccelerationStructureBuildSizes({
        .instanceCount = largeInstanceCount,
        .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace,
    });
    REQUIRE(smallBottomSizes.has_value());
    REQUIRE(largeBottomSizes.has_value());
    REQUIRE(smallTopSizes.has_value());
    REQUIRE(largeTopSizes.has_value());
    REQUIRE(largeBottomSizes->accelerationStructureSize > smallBottomSizes->accelerationStructureSize);
    REQUIRE(largeTopSizes->accelerationStructureSize > smallTopSizes->accelerationStructureSize);

    std::vector<rachis::BufferHandle> storageBuffers;
    std::vector<rachis::AccelerationStructureHandle> destinations;
    const auto createDestination = [&](rachis::AccelerationStructureType type, std::uint64_t size) {
        auto storage = device.createBuffer({
            .size = size,
            .usage = rachis::BufferUsage::accelerationStructureStorage,
            .memoryType = rachis::MemoryType::deviceLocal,
            .label = "Vulkan sized acceleration-structure storage",
        });
        REQUIRE(storage.has_value());
        storageBuffers.push_back(*storage);
        auto destination = device.createAccelerationStructure({
            .type = type,
            .storage = *storage,
            .label = "Vulkan sized acceleration structure",
        });
        REQUIRE(destination.has_value());
        destinations.push_back(*destination);
        return *destination;
    };
    const auto undersizedBottom =
        createDestination(rachis::AccelerationStructureType::bottomLevel, smallBottomSizes->accelerationStructureSize);
    const auto fittingBottom =
        createDestination(rachis::AccelerationStructureType::bottomLevel, largeBottomSizes->accelerationStructureSize);
    const auto undersizedTop =
        createDestination(rachis::AccelerationStructureType::topLevel, smallTopSizes->accelerationStructureSize);
    const auto fittingTop =
        createDestination(rachis::AccelerationStructureType::topLevel, largeTopSizes->accelerationStructureSize);
    const auto scratchSize = (std::max)({
        smallBottomSizes->buildScratchSize,
        largeBottomSizes->buildScratchSize,
        smallTopSizes->buildScratchSize,
        largeTopSizes->buildScratchSize,
    });
    auto scratch = device.createBuffer({
        .size = scratchSize,
        .usage = rachis::BufferUsage::accelerationStructureScratch,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan undersized-destination scratch",
    });
    REQUIRE(scratch.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    const auto buildBottom = [&](rachis::AccelerationStructureHandle destination,
                                 const rachis::BottomLevelAccelerationStructureBuildInput& input) {
        return device.buildBottomLevelAccelerationStructure(*command,
                                                            {
                                                                .destination = destination,
                                                                .input = input,
                                                                .scratch = *scratch,
                                                            });
    };
    auto oversizedBottom = buildBottom(undersizedBottom, largeInput);
    REQUIRE_FALSE(oversizedBottom.has_value());
    CHECK(oversizedBottom.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(buildBottom(undersizedBottom, smallInput).has_value());
    REQUIRE(buildBottom(fittingBottom, largeInput).has_value());

    const std::vector<rachis::AccelerationStructureInstance> largeInstances(
        largeInstanceCount, rachis::AccelerationStructureInstance{.bottomLevel = fittingBottom});
    const std::array smallInstances{rachis::AccelerationStructureInstance{.bottomLevel = undersizedBottom}};
    const auto buildTop = [&](rachis::AccelerationStructureHandle destination,
                              std::span<const rachis::AccelerationStructureInstance> instances) {
        return device.buildTopLevelAccelerationStructure(
            *command,
            {
                .destination = destination,
                .instances = instances,
                .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace,
                .scratch = *scratch,
            });
    };
    auto oversizedTop = buildTop(undersizedTop, largeInstances);
    REQUIRE_FALSE(oversizedTop.has_value());
    CHECK(oversizedTop.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(buildTop(undersizedTop, smallInstances).has_value());
    REQUIRE(buildTop(fittingTop, largeInstances).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    for (const auto destination : destinations)
    {
        REQUIRE(device.destroyAccelerationStructure(destination).has_value());
    }
    for (const auto storage : storageBuffers)
    {
        REQUIRE(device.destroyBuffer(storage).has_value());
    }
    REQUIRE(device.destroyBuffer(*scratch).has_value());
    REQUIRE(device.destroyBuffer(*vertexBuffer).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan acceleration-structure build inputs reject offsets that wrap past the buffer end",
          "[size:medium][rhi][vulkan][ray-query]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (!capabilities.rayTracing && !capabilities.rayQuery)
    {
        SKIP("The Vulkan adapter does not advertise acceleration structures.");
    }

    const std::array vertices{
        std::array{-1.0F, -1.0F, 0.0F},
        std::array{0.0F, 1.0F, 0.0F},
        std::array{1.0F, -1.0F, 0.0F},
    };
    const std::array<std::uint32_t, 3> indices{0, 1, 2};
    const std::array bounds{-1.0F, -1.0F, -1.0F, 1.0F, 1.0F, 1.0F};
    const auto createInput = [&](std::span<const std::byte> bytes, const char* label) {
        auto buffer = device.createBuffer({
            .size = bytes.size(),
            .usage = rachis::BufferUsage::accelerationStructureBuildInput,
            .memoryType = rachis::MemoryType::upload,
            .label = label,
        });
        REQUIRE(buffer.has_value());
        REQUIRE(device.writeBuffer(*buffer, 0, bytes).has_value());
        return *buffer;
    };
    const auto vertexBuffer = createInput(std::as_bytes(std::span(vertices)), "Vulkan wrapped-range vertices");
    const auto indexBuffer = createInput(std::as_bytes(std::span(indices)), "Vulkan wrapped-range indices");
    const auto aabbBuffer = createInput(std::as_bytes(std::span(bounds)), "Vulkan wrapped-range bounds");

    const auto querySizes = [&](const rachis::AccelerationStructureGeometryData& data) {
        const std::array geometry{
            rachis::AccelerationStructureGeometry{
                .data = data,
                .flags = rachis::AccelerationStructureGeometryFlag::opaque,
            },
        };
        return device.queryBottomLevelAccelerationStructureBuildSizes({
            .geometries = geometry,
            .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace,
        });
    };
    const rachis::AccelerationStructureTriangleGeometry triangles{
        .vertexBuffer = vertexBuffer,
        .vertexCount = 3,
        .vertexStride = sizeof(vertices[0]),
        .vertexFormat = rachis::AccelerationStructureVertexFormat::float32x3,
    };
    auto indexed = triangles;
    indexed.indexBuffer = indexBuffer;
    indexed.indexCount = 3;
    indexed.indexFormat = rachis::IndexFormat::uint32;
    const rachis::AccelerationStructureAabbGeometry aabbs{.buffer = aabbBuffer, .count = 1};
    REQUIRE(querySizes(triangles).has_value());
    REQUIRE(querySizes(indexed).has_value());
    REQUIRE(querySizes(aabbs).has_value());

    constexpr auto lastByte = std::numeric_limits<std::uint64_t>::max();
    auto wrappedVertices = triangles;
    wrappedVertices.vertexOffset = lastByte - 3U;
    auto wrappedIndices = indexed;
    wrappedIndices.indexOffset = lastByte - 3U;
    auto wrappedAabbs = aabbs;
    wrappedAabbs.offset = lastByte - 7U;
    const std::array wrapped{
        rachis::AccelerationStructureGeometryData{wrappedVertices},
        rachis::AccelerationStructureGeometryData{wrappedIndices},
        rachis::AccelerationStructureGeometryData{wrappedAabbs},
    };
    for (const auto& data : wrapped)
    {
        auto sizes = querySizes(data);
        REQUIRE_FALSE(sizes.has_value());
        CHECK(sizes.error().code == rachis::ErrorCode::invalidArgument);
    }

    REQUIRE(device.destroyBuffer(vertexBuffer).has_value());
    REQUIRE(device.destroyBuffer(indexBuffer).has_value());
    REQUIRE(device.destroyBuffer(aabbBuffer).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
}

TEST_CASE("Vulkan rejected TLAS rebuilds leave no instance uploads behind", "[size:medium][rhi][vulkan][ray-query]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (!capabilities.rayTracing && !capabilities.rayQuery)
    {
        SKIP("The Vulkan adapter does not advertise acceleration structures.");
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
        .label = "Vulkan TLAS rebuild vertices",
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
        .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace,
    };
    constexpr std::uint32_t instanceCount = 1024;
    auto bottomSizes = device.queryBottomLevelAccelerationStructureBuildSizes(bottomInput);
    auto topSizes = device.queryTopLevelAccelerationStructureBuildSizes({
        .instanceCount = instanceCount,
        .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace,
    });
    REQUIRE(bottomSizes.has_value());
    REQUIRE(topSizes.has_value());

    auto bottomStorage = device.createBuffer({
        .size = bottomSizes->accelerationStructureSize,
        .usage = rachis::BufferUsage::accelerationStructureStorage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan TLAS rebuild BLAS storage",
    });
    auto topStorage = device.createBuffer({
        .size = topSizes->accelerationStructureSize,
        .usage = rachis::BufferUsage::accelerationStructureStorage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan TLAS rebuild TLAS storage",
    });
    auto scratch = device.createBuffer({
        .size = (std::max)(bottomSizes->buildScratchSize, topSizes->buildScratchSize),
        .usage = rachis::BufferUsage::accelerationStructureScratch,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan TLAS rebuild scratch",
    });
    REQUIRE(bottomStorage.has_value());
    REQUIRE(topStorage.has_value());
    REQUIRE(scratch.has_value());
    auto bottomLevel = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::bottomLevel,
        .storage = *bottomStorage,
        .label = "Vulkan TLAS rebuild BLAS",
    });
    auto topLevel = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::topLevel,
        .storage = *topStorage,
        .label = "Vulkan TLAS rebuild TLAS",
    });
    REQUIRE(bottomLevel.has_value());
    REQUIRE(topLevel.has_value());

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
    const std::vector<rachis::AccelerationStructureInstance> instances(
        instanceCount, rachis::AccelerationStructureInstance{.bottomLevel = *bottomLevel});
    const auto buildTop = [&]() {
        return device.buildTopLevelAccelerationStructure(
            *command,
            {
                .destination = *topLevel,
                .instances = instances,
                .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace,
                .scratch = *scratch,
            });
    };
    REQUIRE(buildTop().has_value());

    auto baseline = device.captureResourceMemorySnapshot();
    REQUIRE(baseline.has_value());
    for (int attempt = 0; attempt < 64; ++attempt)
    {
        auto rebuilt = buildTop();
        REQUIRE_FALSE(rebuilt.has_value());
        CHECK(rebuilt.error().code == rachis::ErrorCode::invalidState);
    }
    auto rejected = device.captureResourceMemorySnapshot();
    REQUIRE(rejected.has_value());
    CHECK(rejected->liveAllocationCount == baseline->liveAllocationCount);
    CHECK(rejected->liveAllocationBytes == baseline->liveAllocationBytes);

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();
    auto drained = device.captureResourceMemorySnapshot();
    REQUIRE(drained.has_value());
    CHECK(drained->liveAllocationCount == baseline->liveAllocationCount);
    CHECK(drained->liveAllocationBytes == baseline->liveAllocationBytes);

    REQUIRE(device.destroyAccelerationStructure(*topLevel).has_value());
    REQUIRE(device.destroyAccelerationStructure(*bottomLevel).has_value());
    REQUIRE(device.destroyBuffer(*scratch).has_value());
    REQUIRE(device.destroyBuffer(*topStorage).has_value());
    REQUIRE(device.destroyBuffer(*bottomStorage).has_value());
    REQUIRE(device.destroyBuffer(*vertexBuffer).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan acceleration-structure builds record aligned scratch addresses after small allocations",
          "[size:medium][rhi][vulkan][ray-query]")
{
    const auto priority = GENERATE(rachis::ResidencyPriority::automatic, rachis::ResidencyPriority::high);
    CAPTURE(static_cast<int>(priority));
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (!capabilities.rayTracing && !capabilities.rayQuery)
    {
        SKIP("The Vulkan adapter does not advertise acceleration structures.");
    }
    if (priority != rachis::ResidencyPriority::automatic && !capabilities.residencyPriorities)
    {
        SKIP("The Vulkan device does not support explicit residency priorities.");
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
        .label = "Vulkan scratch-alignment vertices",
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
        .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace,
    };
    auto bottomSizes = device.queryBottomLevelAccelerationStructureBuildSizes(bottomInput);
    auto topSizes = device.queryTopLevelAccelerationStructureBuildSizes({
        .instanceCount = 1,
        .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace,
    });
    REQUIRE(bottomSizes.has_value());
    REQUIRE(topSizes.has_value());

    std::vector<rachis::BufferHandle> buffers;
    std::vector<rachis::AccelerationStructureHandle> structures;
    const auto createDestination = [&](rachis::AccelerationStructureType type, std::uint64_t size) {
        auto storage = device.createBuffer({
            .size = size,
            .usage = rachis::BufferUsage::accelerationStructureStorage,
            .memoryType = rachis::MemoryType::deviceLocal,
            .label = "Vulkan scratch-alignment storage",
        });
        REQUIRE(storage.has_value());
        buffers.push_back(*storage);
        auto destination = device.createAccelerationStructure({
            .type = type,
            .storage = *storage,
            .label = "Vulkan scratch-alignment destination",
        });
        REQUIRE(destination.has_value());
        structures.push_back(*destination);
        return *destination;
    };
    const auto createScratch = [&](std::uint32_t paddingCount, std::uint64_t size, rachis::ResidencyPriority priority) {
        for (std::uint32_t index = 0; index < paddingCount; ++index)
        {
            auto padding = device.createBuffer({
                .size = 48,
                .usage = rachis::BufferUsage::storage,
                .memoryType = rachis::MemoryType::deviceLocal,
                .residencyPriority = priority,
                .label = "Vulkan scratch-alignment padding",
            });
            REQUIRE(padding.has_value());
            buffers.push_back(*padding);
        }
        auto scratch = device.createBuffer({
            .size = size,
            .usage = rachis::BufferUsage::accelerationStructureScratch,
            .memoryType = rachis::MemoryType::deviceLocal,
            .residencyPriority = priority,
            .label = "Vulkan scratch-alignment scratch",
        });
        REQUIRE(scratch.has_value());
        buffers.push_back(*scratch);
        return *scratch;
    };
    const auto recordAndSubmit = [&](std::string_view build, const auto& record) {
        INFO(build);
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command.has_value());
        const auto errorsBefore = device.diagnosticCounts().errors;
        REQUIRE(record(*command).has_value());
        CHECK(device.diagnosticCounts().errors == errorsBefore);
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());
    };

    for (std::uint32_t paddingCount = 0; paddingCount < 8; ++paddingCount)
    {
        CAPTURE(paddingCount);
        const auto bottomLevel =
            createDestination(rachis::AccelerationStructureType::bottomLevel, bottomSizes->accelerationStructureSize);
        const auto bottomScratch = createScratch(paddingCount, bottomSizes->buildScratchSize, priority);
        recordAndSubmit("BLAS", [&](rachis::CommandBuffer& command) {
            return device.buildBottomLevelAccelerationStructure(command,
                                                                {
                                                                    .destination = bottomLevel,
                                                                    .input = bottomInput,
                                                                    .scratch = bottomScratch,
                                                                });
        });

        const auto topLevel =
            createDestination(rachis::AccelerationStructureType::topLevel, topSizes->accelerationStructureSize);
        const auto topScratch = createScratch(paddingCount, topSizes->buildScratchSize, priority);
        const std::array instances{rachis::AccelerationStructureInstance{.bottomLevel = bottomLevel}};
        recordAndSubmit("TLAS", [&](rachis::CommandBuffer& command) {
            return device.buildTopLevelAccelerationStructure(
                command,
                {
                    .destination = topLevel,
                    .instances = instances,
                    .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace,
                    .scratch = topScratch,
                });
        });
    }
    device.collectGarbage();

    for (const auto structure : structures)
    {
        REQUIRE(device.destroyAccelerationStructure(structure).has_value());
    }
    for (const auto buffer : buffers)
    {
        REQUIRE(device.destroyBuffer(buffer).has_value());
    }
    REQUIRE(device.destroyBuffer(*vertexBuffer).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan acceleration-structure creation classifies native create failures by their VkResult",
          "[size:medium][rhi][vulkan][ray-query]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (!capabilities.rayTracing && !capabilities.rayQuery)
    {
        SKIP("The Vulkan adapter does not advertise acceleration structures.");
    }

    auto sizes = device.queryTopLevelAccelerationStructureBuildSizes({
        .instanceCount = 1,
        .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace,
    });
    REQUIRE(sizes.has_value());
    auto storage = device.createBuffer({
        .size = sizes->accelerationStructureSize,
        .usage = rachis::BufferUsage::accelerationStructureStorage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan create-failure acceleration-structure storage",
    });
    REQUIRE(storage.has_value());
    const rachis::AccelerationStructureDesc desc{
        .type = rachis::AccelerationStructureType::topLevel,
        .storage = *storage,
        .label = "Vulkan create-failure acceleration structure",
    };

    const std::array failures{
        std::pair{VK_ERROR_OUT_OF_DEVICE_MEMORY, rachis::ErrorCode::outOfMemory},
        std::pair{VK_ERROR_OUT_OF_HOST_MEMORY, rachis::ErrorCode::outOfMemory},
        std::pair{VK_ERROR_UNKNOWN, rachis::ErrorCode::nativeFailure},
    };
    for (const auto& [result, code] : failures)
    {
        CAPTURE(result);
        REQUIRE(rachis::vulkan::test::injectCreateResult(result));
        auto structure = device.createAccelerationStructure(desc);
        REQUIRE(rachis::vulkan::test::injectCreateResult(VK_SUCCESS));

        REQUIRE_FALSE(structure.has_value());
        CHECK(structure.error().code == code);
        CHECK(structure.error().nativeCode == result);
        CHECK(structure.error().message.contains("vkCreateAccelerationStructureKHR"));
        CHECK(device.operationalState() == rachis::DeviceOperationalState::active);
        CHECK_FALSE(device.deviceLossError().has_value());
    }

    auto structure = device.createAccelerationStructure(desc);
    if (!structure)
    {
        INFO(structure.error().message);
    }
    REQUIRE(structure.has_value());

    REQUIRE(device.destroyAccelerationStructure(*structure).has_value());
    REQUIRE(device.destroyBuffer(*storage).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan discarded command recordings roll back unsubmitted acceleration-structure builds",
          "[size:medium][rhi][vulkan][ray-query]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (!capabilities.rayQuery)
    {
        SKIP("The Vulkan adapter does not advertise inline ray queries.");
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
        .label = "Vulkan discarded-build vertices",
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
        .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace,
    };
    auto bottomSizes = device.queryBottomLevelAccelerationStructureBuildSizes(bottomInput);
    auto topSizes = device.queryTopLevelAccelerationStructureBuildSizes({
        .instanceCount = 1,
        .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace,
    });
    REQUIRE(bottomSizes.has_value());
    REQUIRE(topSizes.has_value());
    auto bottomStorage = device.createBuffer({
        .size = bottomSizes->accelerationStructureSize,
        .usage = rachis::BufferUsage::accelerationStructureStorage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan discarded-build BLAS storage",
    });
    auto topStorage = device.createBuffer({
        .size = topSizes->accelerationStructureSize,
        .usage = rachis::BufferUsage::accelerationStructureStorage,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan discarded-build TLAS storage",
    });
    auto scratch = device.createBuffer({
        .size = (std::max)(bottomSizes->buildScratchSize, topSizes->buildScratchSize),
        .usage = rachis::BufferUsage::accelerationStructureScratch,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan discarded-build scratch",
    });
    REQUIRE(bottomStorage.has_value());
    REQUIRE(topStorage.has_value());
    REQUIRE(scratch.has_value());
    auto bottomLevel = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::bottomLevel,
        .storage = *bottomStorage,
        .label = "Vulkan discarded-build BLAS",
    });
    auto topLevel = device.createAccelerationStructure({
        .type = rachis::AccelerationStructureType::topLevel,
        .storage = *topStorage,
        .label = "Vulkan discarded-build TLAS",
    });
    REQUIRE(bottomLevel.has_value());
    REQUIRE(topLevel.has_value());

    auto pool = device.createCommandPool({
        .queueType = rachis::QueueType::compute,
        .maxCommandBuffers = 2,
        .label = "Vulkan discarded-build pool",
        .flags = rachis::CommandPoolFlag::allowIndividualReset,
    });
    REQUIRE(pool.has_value());

    const std::array instances{
        rachis::AccelerationStructureInstance{
            .bottomLevel = *bottomLevel,
            .flags = rachis::AccelerationStructureInstanceFlag::triangleCullDisable,
        },
    };
    const auto buildBottomLevel = [&](rachis::CommandBuffer& command) {
        return device.buildBottomLevelAccelerationStructure(command,
                                                            {
                                                                .destination = *bottomLevel,
                                                                .input = bottomInput,
                                                                .scratch = *scratch,
                                                            });
    };
    const auto buildTopLevel = [&](rachis::CommandBuffer& command) {
        return device.buildTopLevelAccelerationStructure(
            command,
            {
                .destination = *topLevel,
                .instances = instances,
                .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace,
                .scratch = *scratch,
            });
    };

    {
        auto abandoned = device.acquireCommandBuffer(*pool);
        REQUIRE(abandoned.has_value());
        REQUIRE(buildBottomLevel(*abandoned).has_value());
    }
    auto command = device.acquireCommandBuffer(*pool);
    REQUIRE(command.has_value());
    auto topLevelOverAbandonedBuild = buildTopLevel(*command);
    REQUIRE_FALSE(topLevelOverAbandonedBuild.has_value());
    CHECK(topLevelOverAbandonedBuild.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(buildBottomLevel(*command).has_value());

    REQUIRE(device.resetCommandBuffer(*command).has_value());
    auto topLevelOverResetBuild = buildTopLevel(*command);
    REQUIRE_FALSE(topLevelOverResetBuild.has_value());
    CHECK(topLevelOverResetBuild.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(buildBottomLevel(*command).has_value());

    REQUIRE(device.resetCommandPool(*pool).has_value());
    auto topLevelOverPoolReset = buildTopLevel(*command);
    REQUIRE_FALSE(topLevelOverPoolReset.has_value());
    CHECK(topLevelOverPoolReset.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(buildBottomLevel(*command).has_value());

    rachis::vulkan::test::nextEndCommandBufferResult().store(-2, std::memory_order_relaxed);
    auto rejectedSubmission = device.submit(std::move(*command));
    rachis::vulkan::test::nextEndCommandBufferResult().store(0, std::memory_order_relaxed);
    REQUIRE_FALSE(rejectedSubmission.has_value());

    auto retried = device.acquireCommandBuffer(*pool);
    REQUIRE(retried.has_value());
    auto topLevelOverRejectedSubmission = buildTopLevel(*retried);
    REQUIRE_FALSE(topLevelOverRejectedSubmission.has_value());
    CHECK(topLevelOverRejectedSubmission.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(buildBottomLevel(*retried).has_value());
    REQUIRE(buildTopLevel(*retried).has_value());
    auto submitted = device.submit(std::move(*retried));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto afterSubmission = device.acquireCommandBuffer(*pool);
    REQUIRE(afterSubmission.has_value());
    auto rebuiltAfterSubmission = buildBottomLevel(*afterSubmission);
    REQUIRE_FALSE(rebuiltAfterSubmission.has_value());
    CHECK(rebuiltAfterSubmission.error().code == rachis::ErrorCode::invalidState);

    REQUIRE(device.destroyCommandPool(*pool).has_value());
    REQUIRE(device.destroyAccelerationStructure(*topLevel).has_value());
    REQUIRE(device.destroyAccelerationStructure(*bottomLevel).has_value());
    REQUIRE(device.destroyBuffer(*scratch).has_value());
    REQUIRE(device.destroyBuffer(*topStorage).has_value());
    REQUIRE(device.destroyBuffer(*bottomStorage).has_value());
    REQUIRE(device.destroyBuffer(*vertexBuffer).has_value());
    device.collectGarbage();

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan BLAS triangle vertex strides must be multiples of the vertex component size",
          "[size:medium][rhi][vulkan][ray-query]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (!capabilities.rayTracing && !capabilities.rayQuery)
    {
        SKIP("The Vulkan adapter does not advertise acceleration structures.");
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
        .label = "Vulkan vertex-stride BLAS vertices",
    });
    REQUIRE(vertexBuffer.has_value());
    REQUIRE(device.writeBuffer(*vertexBuffer, 0, std::as_bytes(std::span(paddedVertices))).has_value());
    REQUIRE(
        device.writeBuffer(*vertexBuffer, packedHalfOffset, std::as_bytes(std::span(packedHalfVertices))).has_value());

    struct VertexStrideLayout final
    {
        rachis::AccelerationStructureVertexFormat format = rachis::AccelerationStructureVertexFormat::float32x3;
        std::uint64_t offset = 0;
        std::uint32_t stride = 0;
    };
    struct BuildTarget final
    {
        rachis::BufferHandle storage = {};
        rachis::BufferHandle scratch = {};
        rachis::AccelerationStructureHandle structure = {};
    };
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
    const auto createTarget = [&](const rachis::AccelerationStructureBuildSizes& sizes) {
        auto storage = device.createBuffer({
            .size = sizes.accelerationStructureSize,
            .usage = rachis::BufferUsage::accelerationStructureStorage,
            .memoryType = rachis::MemoryType::deviceLocal,
            .label = "Vulkan vertex-stride BLAS storage",
        });
        auto scratch = device.createBuffer({
            .size = sizes.buildScratchSize,
            .usage = rachis::BufferUsage::accelerationStructureScratch,
            .memoryType = rachis::MemoryType::deviceLocal,
            .label = "Vulkan vertex-stride BLAS scratch",
        });
        REQUIRE(storage.has_value());
        REQUIRE(scratch.has_value());
        auto structure = device.createAccelerationStructure({
            .type = rachis::AccelerationStructureType::bottomLevel,
            .storage = *storage,
            .label = "Vulkan vertex-stride BLAS",
        });
        REQUIRE(structure.has_value());
        return BuildTarget{.storage = *storage, .scratch = *scratch, .structure = *structure};
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
    std::vector<BuildTarget> targets;
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
        targets.push_back(createTarget(*sizes));
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
        targets.push_back(createTarget(alignedSizes.front()));
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

#ifdef RACHIS_SPIRV_DIR
namespace
{

    constexpr auto topLevelFlags = rachis::AccelerationStructureBuildFlag::allowUpdate;

    template<typename T>
    [[nodiscard]] T require(rachis::Result<T> created)
    {
        if (!created)
        {
            UNSCOPED_INFO(created.error().message);
        }
        REQUIRE(created.has_value());
        return *created;
    }

    class TopLevelLeaseFixture final
    {
    public:
        explicit TopLevelLeaseFixture(rachis::Device& device) : device_(device)
        {
            const auto bytecode = loadSpirv("RayQueryWorkload.spv");
            REQUIRE_FALSE(bytecode.empty());
            const std::array shaderBindings{
                rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::accelerationStructure, 1},
                rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
            };
            shader_ = require(device.createShaderModule({
                .stage = rachis::ShaderStage::compute,
                .bytecode = bytecode,
                .entryPoint = "main",
                .label = "Vulkan TLAS lease ray query",
                .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, shaderBindings),
            }));

            const std::array vertices{
                std::array{-1.0F, -1.0F, 0.0F},
                std::array{0.0F, 1.0F, 0.0F},
                std::array{1.0F, -1.0F, 0.0F},
            };
            vertices_ = require(device.createBuffer({
                .size = sizeof(vertices),
                .usage = rachis::BufferUsage::accelerationStructureBuildInput,
                .memoryType = rachis::MemoryType::upload,
                .label = "Vulkan TLAS lease triangle",
            }));
            REQUIRE(device.writeBuffer(vertices_, 0, std::as_bytes(std::span(vertices))).has_value());
            geometry_ = {
                rachis::AccelerationStructureGeometry{
                    .data =
                        rachis::AccelerationStructureTriangleGeometry{
                            .vertexBuffer = vertices_,
                            .vertexCount = 3,
                            .vertexStride = sizeof(vertices[0]),
                            .vertexFormat = rachis::AccelerationStructureVertexFormat::float32x3,
                        },
                    .flags = rachis::AccelerationStructureGeometryFlag::opaque,
                },
            };
            const auto bottomSizes = require(device.queryBottomLevelAccelerationStructureBuildSizes(bottomInput()));
            const auto updatableSizes = require(device.queryTopLevelAccelerationStructureBuildSizes({
                .instanceCount = 1,
                .flags = topLevelFlags,
            }));
            const auto fastTraceSizes = require(device.queryTopLevelAccelerationStructureBuildSizes({
                .instanceCount = 1,
                .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace,
            }));
            const rachis::AccelerationStructureBuildSizes topSizes{
                .accelerationStructureSize =
                    std::max(updatableSizes.accelerationStructureSize, fastTraceSizes.accelerationStructureSize),
                .buildScratchSize = std::max(updatableSizes.buildScratchSize, fastTraceSizes.buildScratchSize),
                .updateScratchSize = updatableSizes.updateScratchSize,
            };
            bottomSize_ = bottomSizes.accelerationStructureSize;
            bottomStorage_ = require(device.createBuffer({
                .size = bottomSize_,
                .usage = rachis::BufferUsage::accelerationStructureStorage,
                .memoryType = rachis::MemoryType::deviceLocal,
                .label = "Vulkan TLAS lease BLAS storage",
            }));
            for (auto& storage : topStorage_)
            {
                storage = require(device.createBuffer({
                    .size = topSizes.accelerationStructureSize,
                    .usage = rachis::BufferUsage::accelerationStructureStorage,
                    .memoryType = rachis::MemoryType::deviceLocal,
                    .label = "Vulkan TLAS lease TLAS storage",
                }));
            }
            scratch_ = require(device.createBuffer({
                .size = std::max({bottomSizes.buildScratchSize,
                                  topSizes.buildScratchSize,
                                  topSizes.updateScratchSize,
                                  std::uint64_t{256}}),
                .usage = rachis::BufferUsage::accelerationStructureScratch,
                .memoryType = rachis::MemoryType::deviceLocal,
                .label = "Vulkan TLAS lease scratch",
            }));
            bottomLevel_ = require(device.createAccelerationStructure({
                .type = rachis::AccelerationStructureType::bottomLevel,
                .storage = bottomStorage_,
                .label = "Vulkan TLAS lease BLAS",
            }));
            for (std::size_t index = 0; index < topLevels_.size(); ++index)
            {
                topLevels_[index] = require(device.createAccelerationStructure({
                    .type = rachis::AccelerationStructureType::topLevel,
                    .storage = topStorage_[index],
                    .label = "Vulkan TLAS lease TLAS",
                }));
            }

            output_ = require(device.createBuffer({
                .size = 2 * sizeof(std::uint32_t),
                .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
                .memoryType = rachis::MemoryType::deviceLocal,
                .label = "Vulkan TLAS lease output",
            }));
            readback_ = require(device.createBuffer({
                .size = 2 * sizeof(std::uint32_t),
                .usage = rachis::BufferUsage::copyDestination,
                .memoryType = rachis::MemoryType::readback,
                .label = "Vulkan TLAS lease readback",
            }));
            const std::array layoutEntries{
                rachis::BindingLayoutEntry{
                    0, rachis::BindingType::accelerationStructure, rachis::ShaderStage::compute, 1},
                rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
            };
            bindingLayout_ = require(device.createBindingLayout({.entries = layoutEntries}));
            const std::array bindingLayouts{bindingLayout_};
            pipelineLayout_ = require(device.createPipelineLayout({.bindingLayouts = bindingLayouts}));
            pipeline_ = require(device.createComputePipeline({.layout = pipelineLayout_, .computeShader = shader_}));
        }

        TopLevelLeaseFixture(const TopLevelLeaseFixture&) = delete;
        TopLevelLeaseFixture& operator=(const TopLevelLeaseFixture&) = delete;
        TopLevelLeaseFixture(TopLevelLeaseFixture&&) = delete;
        TopLevelLeaseFixture& operator=(TopLevelLeaseFixture&&) = delete;

        ~TopLevelLeaseFixture()
        {
            device_.collectGarbage();
            if (bindingSet_.isValid())
            {
                static_cast<void>(device_.destroyBindingSet(bindingSet_));
            }
            static_cast<void>(device_.destroyPipeline(pipeline_));
            static_cast<void>(device_.destroyPipelineLayout(pipelineLayout_));
            static_cast<void>(device_.destroyBindingLayout(bindingLayout_));
            static_cast<void>(device_.destroyShaderModule(shader_));
            static_cast<void>(device_.destroyAccelerationStructure(bottomLevel_));
            for (const auto topLevel : topLevels_)
            {
                static_cast<void>(device_.destroyAccelerationStructure(topLevel));
            }
            for (const auto buffer :
                 {vertices_, bottomStorage_, topStorage_[0], topStorage_[1], scratch_, output_, readback_})
            {
                static_cast<void>(device_.destroyBuffer(buffer));
            }
            device_.collectGarbage();
        }

        [[nodiscard]] rachis::BottomLevelAccelerationStructureBuildInput bottomInput() const noexcept
        {
            return {.geometries = geometry_, .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace};
        }

        [[nodiscard]] std::array<rachis::AccelerationStructureInstance, 1> instances() const noexcept
        {
            return {rachis::AccelerationStructureInstance{
                .bottomLevel = bottomLevel_,
                .flags = rachis::AccelerationStructureInstanceFlag::triangleCullDisable,
            }};
        }

        void recordBuilds(rachis::CommandBuffer& command,
                          rachis::AccelerationStructureBuildFlag flags = topLevelFlags) const
        {
            REQUIRE(device_
                        .buildBottomLevelAccelerationStructure(command,
                                                               {
                                                                   .destination = bottomLevel_,
                                                                   .input = bottomInput(),
                                                                   .scratch = scratch_,
                                                               })
                        .has_value());
            const auto built = instances();
            const auto topLevel = device_.buildTopLevelAccelerationStructure(command,
                                                                             {
                                                                                 .destination = topLevels_[0],
                                                                                 .instances = built,
                                                                                 .flags = flags,
                                                                                 .scratch = scratch_,
                                                                             });
            if (!topLevel)
            {
                UNSCOPED_INFO(topLevel.error().message);
            }
            REQUIRE(topLevel.has_value());
        }

        void build() const
        {
            auto command = rachis::test::acquireCommandBuffer(device_, rachis::QueueType::compute);
            REQUIRE(command.has_value());
            recordBuilds(*command);
            auto submitted = device_.submit(std::move(*command));
            REQUIRE(submitted.has_value());
            REQUIRE(device_.wait(*submitted).has_value());
            device_.collectGarbage();
        }

        [[nodiscard]] rachis::Status updateInPlace(rachis::CommandBuffer& command) const
        {
            const auto updated = instances();
            return device_.updateTopLevelAccelerationStructureInPlace(command,
                                                                      {
                                                                          .target = topLevels_[0],
                                                                          .instances = updated,
                                                                          .flags = topLevelFlags,
                                                                          .scratch = scratch_,
                                                                      });
        }

        void createBindingSet()
        {
            const std::array writes{
                rachis::BindingWrite{.binding = 0, .resource = topLevels_[0]},
                rachis::BindingWrite{
                    .binding = 1,
                    .resource =
                        rachis::BufferBinding{
                            .buffer = output_,
                            .size = 2 * sizeof(std::uint32_t),
                            .structureByteStride = 2 * sizeof(std::uint32_t),
                        },
                },
            };
            bindingSet_ =
                require(rachis::test::createBindingSet(device_, {.layout = bindingLayout_, .writes = writes}));
        }

        void bindPipeline(rachis::CommandBuffer& command) const
        {
            REQUIRE(device_.bindComputePipeline(command, pipeline_).has_value());
        }

        [[nodiscard]] rachis::Status bindSet(rachis::CommandBuffer& command) const
        {
            return device_.bindComputeBindingSet(command, 0, bindingSet_);
        }

        void trace(rachis::CommandBuffer& command) const
        {
            REQUIRE(device_
                        .bufferBarrier(command,
                                       {
                                           .buffer = output_,
                                           .stagesBefore = rachis::PipelineStage::all,
                                           .stagesAfter = rachis::PipelineStage::computeShader,
                                           .accessBefore = rachis::Access::common,
                                           .accessAfter = rachis::Access::unorderedAccess,
                                       })
                        .has_value());
            bindPipeline(command);
            REQUIRE(bindSet(command).has_value());
            REQUIRE(device_.dispatch(command, 1).has_value());
        }

        [[nodiscard]] std::array<std::uint32_t, 2> finish(rachis::CommandBuffer command) const
        {
            REQUIRE(device_
                        .bufferBarrier(command,
                                       {
                                           .buffer = output_,
                                           .stagesBefore = rachis::PipelineStage::computeShader,
                                           .stagesAfter = rachis::PipelineStage::copy,
                                           .accessBefore = rachis::Access::unorderedAccess,
                                           .accessAfter = rachis::Access::copySource,
                                       })
                        .has_value());
            REQUIRE(device_.copyBuffer(command, {.source = output_, .destination = readback_, .size = 8}).has_value());
            auto submitted = device_.submit(std::move(command));
            REQUIRE(submitted.has_value());
            REQUIRE(device_.wait(*submitted).has_value());
            device_.collectGarbage();
            std::array<std::uint32_t, 2> result{};
            REQUIRE(device_.readBuffer(readback_, 0, std::as_writable_bytes(std::span(result))).has_value());
            return result;
        }

        [[nodiscard]] rachis::AccelerationStructureHandle bottomLevel() const noexcept
        {
            return bottomLevel_;
        }

        [[nodiscard]] rachis::AccelerationStructureHandle topLevel(std::size_t index) const noexcept
        {
            return topLevels_[index];
        }

        [[nodiscard]] rachis::BufferHandle bottomStorage() const noexcept
        {
            return bottomStorage_;
        }

        [[nodiscard]] rachis::BufferHandle scratch() const noexcept
        {
            return scratch_;
        }

        void forgetBottomLevel() noexcept
        {
            bottomLevel_ = {};
        }

        void forgetTopLevel(std::size_t index) noexcept
        {
            topLevels_[index] = {};
        }

    private:
        rachis::Device& device_;
        rachis::ShaderModuleHandle shader_{};
        rachis::BufferHandle vertices_{};
        std::array<rachis::AccelerationStructureGeometry, 1> geometry_{};
        std::uint64_t bottomSize_ = 0;
        rachis::BufferHandle bottomStorage_{};
        std::array<rachis::BufferHandle, 2> topStorage_{};
        rachis::BufferHandle scratch_{};
        rachis::AccelerationStructureHandle bottomLevel_{};
        std::array<rachis::AccelerationStructureHandle, 2> topLevels_{};
        rachis::BufferHandle output_{};
        rachis::BufferHandle readback_{};
        rachis::BindingLayoutHandle bindingLayout_{};
        rachis::BindingSetHandle bindingSet_{};
        rachis::PipelineLayoutHandle pipelineLayout_{};
        rachis::PipelineHandle pipeline_{};
    };

    [[nodiscard]] std::optional<rachis::Device> createRayQueryDevice()
    {
        auto created = rachis::createDevice(rachis::test::vulkanDeviceDesc());
        if (rachis::test::skipWithoutCreatedDevice(created))
        {
            return std::nullopt;
        }
        if (!rachis::test::requireCapabilities(*created).rayQuery)
        {
            SKIP("The Vulkan adapter does not advertise inline ray queries.");
        }
        return std::move(*created);
    }

    void checkNoDiagnostics(const rachis::Device& device)
    {
        const auto diagnostics = device.diagnosticCounts();
        CHECK(diagnostics.errors == 0);
        CHECK(diagnostics.corruptions == 0);
    }

} // namespace

TEST_CASE("Vulkan in-place TLAS updates follow reads of the same TLAS in their own command buffer only",
          "[size:medium][rhi][vulkan][ray-query][acceleration-structure]")
{
    auto device = createRayQueryDevice();
    if (!device)
    {
        return;
    }
    {
        TopLevelLeaseFixture fixture(*device);
        fixture.build();
        fixture.createBindingSet();

        auto command = rachis::test::acquireCommandBuffer(*device, rachis::QueueType::compute);
        REQUIRE(command.has_value());
        fixture.trace(*command);
        REQUIRE(device
                    ->globalBarrier(*command,
                                    {
                                        .stagesBefore = rachis::PipelineStage::computeShader,
                                        .stagesAfter = rachis::PipelineStage::accelerationStructureBuild,
                                        .accessBefore = rachis::Access::shaderResource,
                                        .accessAfter = rachis::Access::accelerationStructureBuildInput,
                                    })
                    .has_value());
        {
            auto foreign = rachis::test::acquireCommandBuffer(*device, rachis::QueueType::compute);
            REQUIRE(foreign.has_value());
            const auto blocked = fixture.updateInPlace(*foreign);
            REQUIRE_FALSE(blocked.has_value());
            CHECK(blocked.error().code == rachis::ErrorCode::invalidState);
        }
        const auto updated = fixture.updateInPlace(*command);
        if (!updated)
        {
            UNSCOPED_INFO(updated.error().message);
        }
        REQUIRE(updated.has_value());
        fixture.trace(*command);
        const auto result = fixture.finish(std::move(*command));
        CHECK(result[0] == 1);
        CHECK(std::bit_cast<float>(result[1]) == 1.0F);

        auto later = rachis::test::acquireCommandBuffer(*device, rachis::QueueType::compute);
        REQUIRE(later.has_value());
        CHECK(fixture.updateInPlace(*later).has_value());
    }
    checkNoDiagnostics(*device);
}

TEST_CASE("Vulkan BLAS storage stays leased while a built TLAS still references the destroyed BLAS",
          "[size:medium][rhi][vulkan][ray-query][acceleration-structure]")
{
    auto device = createRayQueryDevice();
    if (!device)
    {
        return;
    }
    {
        TopLevelLeaseFixture fixture(*device);
        fixture.build();

        REQUIRE(device->destroyAccelerationStructure(fixture.bottomLevel()).has_value());
        fixture.forgetBottomLevel();
        device->collectGarbage();
        const rachis::AccelerationStructureDesc reuse{
            .type = rachis::AccelerationStructureType::bottomLevel,
            .storage = fixture.bottomStorage(),
            .label = "Vulkan BLAS reusing leased storage",
        };
        const auto overlapping = device->createAccelerationStructure(reuse);
        REQUIRE_FALSE(overlapping.has_value());
        CHECK(overlapping.error().code == rachis::ErrorCode::invalidArgument);

        REQUIRE(device->destroyAccelerationStructure(fixture.topLevel(0)).has_value());
        fixture.forgetTopLevel(0);
        device->collectGarbage();
        const auto reused = device->createAccelerationStructure(reuse);
        REQUIRE(reused.has_value());
        REQUIRE(device->destroyAccelerationStructure(*reused).has_value());
    }
    checkNoDiagnostics(*device);
}

TEST_CASE("Vulkan TLAS updates fail when their source is rebuilt while the update records",
          "[size:medium][rhi][vulkan][ray-query][acceleration-structure]")
{
    auto device = createRayQueryDevice();
    if (!device)
    {
        return;
    }
    {
        TopLevelLeaseFixture fixture(*device);
        auto original = rachis::test::acquireCommandBuffer(*device, rachis::QueueType::compute);
        REQUIRE(original.has_value());
        fixture.recordBuilds(*original);

        std::optional<rachis::CommandBuffer> rebuild;
        bool rebuilt = false;
        rachis::vulkan::test::beforeTopLevelUpdatePublication() = [&] {
            *original = rachis::CommandBuffer{};
            auto command = rachis::test::acquireCommandBuffer(*device, rachis::QueueType::compute);
            if (!command)
            {
                return;
            }
            rebuild = std::move(*command);
            fixture.recordBuilds(*rebuild, rachis::AccelerationStructureBuildFlag::preferFastTrace);
            rebuilt = true;
        };

        auto update = rachis::test::acquireCommandBuffer(*device, rachis::QueueType::compute);
        REQUIRE(update.has_value());
        const auto instances = fixture.instances();
        const auto updated = device->updateTopLevelAccelerationStructure(*update,
                                                                         {
                                                                             .source = fixture.topLevel(0),
                                                                             .destination = fixture.topLevel(1),
                                                                             .instances = instances,
                                                                             .flags = topLevelFlags,
                                                                             .scratch = fixture.scratch(),
                                                                         });
        rachis::vulkan::test::beforeTopLevelUpdatePublication() = {};
        REQUIRE(rebuilt);
        REQUIRE_FALSE(updated.has_value());
        CHECK(updated.error().code == rachis::ErrorCode::invalidState);
    }
    checkNoDiagnostics(*device);
}

TEST_CASE("Vulkan acceleration-structure creation that fails to allocate its record leaks no native object",
          "[size:medium][rhi][vulkan][ray-query][acceleration-structure]")
{
    auto device = createRayQueryDevice();
    if (!device)
    {
        return;
    }
    {
        TopLevelLeaseFixture fixture(*device);
        REQUIRE(device->destroyAccelerationStructure(fixture.bottomLevel()).has_value());
        fixture.forgetBottomLevel();
        device->collectGarbage();
        const rachis::AccelerationStructureDesc desc{
            .type = rachis::AccelerationStructureType::bottomLevel,
            .storage = fixture.bottomStorage(),
            .label = "Vulkan BLAS whose record allocation fails",
        };

        const auto liveBefore = rachis::vulkan::test::liveAccelerationStructures().load();
        rachis::vulkan::injectVulkanBadAlloc(rachis::vulkan::VulkanBadAllocPoint::accelerationStructureRecord, true);
        const auto failed = device->createAccelerationStructure(desc);
        rachis::vulkan::injectVulkanBadAlloc(rachis::vulkan::VulkanBadAllocPoint::accelerationStructureRecord, false);
        REQUIRE_FALSE(failed.has_value());
        CHECK(failed.error().code == rachis::ErrorCode::outOfMemory);
        CHECK(rachis::vulkan::test::liveAccelerationStructures().load() == liveBefore);

        const auto created = device->createAccelerationStructure(desc);
        REQUIRE(created.has_value());
        CHECK(rachis::vulkan::test::liveAccelerationStructures().load() == liveBefore + 1);
        REQUIRE(device->destroyAccelerationStructure(*created).has_value());
        device->collectGarbage();
        CHECK(rachis::vulkan::test::liveAccelerationStructures().load() == liveBefore);
    }
    checkNoDiagnostics(*device);
}
#endif
