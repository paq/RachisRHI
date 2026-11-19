#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
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

    struct OwnedAccelerationStructure final
    {
        rachis::AccelerationStructureHandle structure = {};
        rachis::BufferHandle storage = {};
    };

    [[nodiscard]] rachis::BufferHandle requireBuffer(rachis::Device& device, const rachis::BufferDesc& desc)
    {
        auto buffer = device.createBuffer(desc);
        if (!buffer)
        {
            UNSCOPED_INFO(desc.label << ": " << buffer.error().message);
        }
        REQUIRE(buffer.has_value());
        return *buffer;
    }

    [[nodiscard]] OwnedAccelerationStructure requireAccelerationStructure(rachis::Device& device,
                                                                          rachis::AccelerationStructureType type,
                                                                          std::uint64_t size,
                                                                          const char* label)
    {
        const auto storage = requireBuffer(device,
                                           {
                                               .size = size,
                                               .usage = rachis::BufferUsage::accelerationStructureStorage,
                                               .memoryType = rachis::MemoryType::deviceLocal,
                                               .label = label,
                                           });
        auto structure = device.createAccelerationStructure({.type = type, .storage = storage, .label = label});
        if (!structure)
        {
            UNSCOPED_INFO(label << ": " << structure.error().message);
        }
        REQUIRE(structure.has_value());
        return {.structure = *structure, .storage = storage};
    }

    void destroyAccelerationStructures(rachis::Device& device, std::span<const OwnedAccelerationStructure> owned)
    {
        for (const auto& entry : owned)
        {
            REQUIRE(device.destroyAccelerationStructure(entry.structure).has_value());
            REQUIRE(device.destroyBuffer(entry.storage).has_value());
        }
    }

    constexpr auto topLevelFlags =
        rachis::AccelerationStructureBuildFlag::preferFastTrace | rachis::AccelerationStructureBuildFlag::allowUpdate;

} // namespace

#ifdef RACHIS_SPIRV_DIR
namespace
{

    [[nodiscard]] std::vector<std::byte> loadSpirv(const char* name)
    {
        std::ifstream stream(std::filesystem::path(RACHIS_SPIRV_DIR) / name, std::ios::binary | std::ios::ate);
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

    struct InstanceInput final
    {
        std::uint32_t referenceLow = 0;
        std::uint32_t referenceHigh = 0;
        std::uint32_t translationX = 0;
        std::uint32_t instanceId = 0;
    };

    [[nodiscard]] InstanceInput instanceInput(std::uint64_t reference, float translationX, std::uint32_t instanceId)
    {
        return InstanceInput{
            .referenceLow = static_cast<std::uint32_t>(reference),
            .referenceHigh = static_cast<std::uint32_t>(reference >> 32U),
            .translationX = std::bit_cast<std::uint32_t>(translationX),
            .instanceId = instanceId,
        };
    }

    struct RayResult final
    {
        std::uint32_t hit = 0;
        std::uint32_t instanceId = 0;
        std::uint32_t rayT = 0;
        std::uint32_t instanceIndex = 0;
    };

    void checkHit(const RayResult& result, std::uint32_t instanceId, float rayT, std::uint32_t instanceIndex)
    {
        CHECK(result.hit == 1);
        CHECK(result.instanceId == instanceId);
        CHECK(std::abs(std::bit_cast<float>(result.rayT) - rayT) < 1.0e-4F);
        CHECK(result.instanceIndex == instanceIndex);
    }

    void checkMiss(const RayResult& result)
    {
        CHECK(result.hit == 0);
    }

    constexpr std::uint32_t instanceRecordCount = 3;
    constexpr std::uint64_t instanceRecordOffset = 256;
    constexpr std::uint64_t instanceInputBytes = 256;

    [[nodiscard]] rachis::PipelineHandle requireComputePipeline(
        rachis::Device& device,
        const char* shaderName,
        std::span<const rachis::ShaderBindingRequirement> shaderBindings,
        rachis::PipelineLayoutHandle layout,
        const char* label)
    {
        const auto bytecode = loadSpirv(shaderName);
        REQUIRE_FALSE(bytecode.empty());
        auto shader = device.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = bytecode,
            .entryPoint = "main",
            .label = label,
            .shaderInterface = rachis::test::computeShaderInterface({3, 1, 1}, shaderBindings),
        });
        if (!shader)
        {
            UNSCOPED_INFO(label << ": " << shader.error().message);
        }
        REQUIRE(shader.has_value());
        auto pipeline = device.createComputePipeline({.layout = layout, .computeShader = *shader, .label = label});
        if (!pipeline)
        {
            UNSCOPED_INFO(label << ": " << pipeline.error().message);
        }
        REQUIRE(pipeline.has_value());
        REQUIRE(device.destroyShaderModule(*shader).has_value());
        return *pipeline;
    }

    void requireBarrier(rachis::Device& device,
                        rachis::CommandBuffer& command,
                        rachis::BufferHandle buffer,
                        rachis::PipelineStage stagesBefore,
                        rachis::Access accessBefore,
                        rachis::PipelineStage stagesAfter,
                        rachis::Access accessAfter)
    {
        REQUIRE(device
                    .bufferBarrier(command,
                                   {
                                       .buffer = buffer,
                                       .stagesBefore = stagesBefore,
                                       .stagesAfter = stagesAfter,
                                       .accessBefore = accessBefore,
                                       .accessAfter = accessAfter,
                                   })
                    .has_value());
    }

} // namespace

TEST_CASE("Vulkan TLAS builds and updates read instance records that a compute pass writes",
          "[size:medium][rhi][vulkan][ray-query]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    if (!rachis::test::requireCapabilities(device).rayQuery)
    {
        SKIP("The Vulkan adapter does not advertise inline ray queries.");
    }

    auto layout = device.queryAccelerationStructureInstanceLayout();
    REQUIRE(layout.has_value());
    CHECK(layout->format == rachis::AccelerationStructureInstanceFormat::packed64);
    CHECK(layout->size == sizeof(rachis::AccelerationStructureInstanceRecord));
    CHECK(layout->alignment == alignof(rachis::AccelerationStructureInstanceRecord));

    const std::array vertices{
        std::array{-1.0F, -1.0F, 0.0F},
        std::array{0.0F, 1.0F, 0.0F},
        std::array{1.0F, -1.0F, 0.0F},
        std::array{-1.0F, -1.0F, 0.5F},
        std::array{0.0F, 1.0F, 0.5F},
        std::array{1.0F, -1.0F, 0.5F},
    };
    const auto vertexBuffer = requireBuffer(device,
                                            {
                                                .size = sizeof(vertices),
                                                .usage = rachis::BufferUsage::accelerationStructureBuildInput,
                                                .memoryType = rachis::MemoryType::upload,
                                                .label = "Vulkan instance-buffer BLAS vertices",
                                            });
    REQUIRE(device.writeBuffer(vertexBuffer, 0, std::as_bytes(std::span(vertices))).has_value());
    const auto triangleAt = [&](std::uint64_t vertexOffset) {
        return std::array{
            rachis::AccelerationStructureGeometry{
                .data =
                    rachis::AccelerationStructureTriangleGeometry{
                        .vertexBuffer = vertexBuffer,
                        .vertexOffset = vertexOffset,
                        .vertexCount = 3,
                        .vertexStride = sizeof(vertices[0]),
                        .vertexFormat = rachis::AccelerationStructureVertexFormat::float32x3,
                    },
                .flags = rachis::AccelerationStructureGeometryFlag::opaque,
            },
        };
    };
    const auto nearGeometry = triangleAt(0);
    const auto farGeometry = triangleAt(3 * sizeof(vertices[0]));
    const rachis::BottomLevelAccelerationStructureBuildInput nearInput{
        .geometries = nearGeometry,
        .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace,
    };
    const rachis::BottomLevelAccelerationStructureBuildInput farInput{
        .geometries = farGeometry,
        .flags = rachis::AccelerationStructureBuildFlag::preferFastTrace,
    };
    auto bottomSizes = device.queryBottomLevelAccelerationStructureBuildSizes(nearInput);
    auto topSizes = device.queryTopLevelAccelerationStructureBuildSizes({
        .instanceCount = instanceRecordCount,
        .flags = topLevelFlags,
    });
    REQUIRE(bottomSizes.has_value());
    REQUIRE(topSizes.has_value());

    const std::array structures{
        requireAccelerationStructure(device,
                                     rachis::AccelerationStructureType::bottomLevel,
                                     bottomSizes->accelerationStructureSize,
                                     "Vulkan near BLAS"),
        requireAccelerationStructure(device,
                                     rachis::AccelerationStructureType::bottomLevel,
                                     bottomSizes->accelerationStructureSize,
                                     "Vulkan far BLAS"),
        requireAccelerationStructure(device,
                                     rachis::AccelerationStructureType::topLevel,
                                     topSizes->accelerationStructureSize,
                                     "Vulkan GPU-written TLAS"),
        requireAccelerationStructure(device,
                                     rachis::AccelerationStructureType::topLevel,
                                     topSizes->accelerationStructureSize,
                                     "Vulkan update destination TLAS"),
        requireAccelerationStructure(device,
                                     rachis::AccelerationStructureType::topLevel,
                                     topSizes->accelerationStructureSize,
                                     "Vulkan CPU-written TLAS"),
        requireAccelerationStructure(device,
                                     rachis::AccelerationStructureType::topLevel,
                                     topSizes->accelerationStructureSize,
                                     "Vulkan span-updated TLAS"),
    };
    const auto nearBottomLevel = structures[0].structure;
    const auto farBottomLevel = structures[1].structure;
    const auto builtTopLevel = structures[2].structure;
    const auto updatedTopLevel = structures[3].structure;
    const auto uploadedTopLevel = structures[4].structure;
    const auto spanTopLevel = structures[5].structure;
    const auto scratch = requireBuffer(
        device,
        {
            .size =
                (std::max)({bottomSizes->buildScratchSize, topSizes->buildScratchSize, topSizes->updateScratchSize}),
            .usage = rachis::BufferUsage::accelerationStructureScratch,
            .memoryType = rachis::MemoryType::deviceLocal,
            .label = "Vulkan instance-buffer scratch",
        });

    auto nearReference = device.queryAccelerationStructureReference(nearBottomLevel);
    auto farReference = device.queryAccelerationStructureReference(farBottomLevel);
    REQUIRE(nearReference.has_value());
    REQUIRE(farReference.has_value());
    CHECK(*nearReference != 0);
    CHECK(*farReference != 0);
    CHECK(*nearReference != *farReference);
    auto topLevelReference = device.queryAccelerationStructureReference(builtTopLevel);
    REQUIRE_FALSE(topLevelReference.has_value());
    CHECK(topLevelReference.error().code == rachis::ErrorCode::invalidArgument);

    const auto instanceInputs = requireBuffer(device,
                                              {
                                                  .size = instanceInputBytes,
                                                  .usage = rachis::BufferUsage::constant,
                                                  .memoryType = rachis::MemoryType::upload,
                                                  .label = "Vulkan instance-writer inputs",
                                              });
    const auto instanceRecords =
        requireBuffer(device,
                      {
                          .size = instanceRecordOffset + instanceRecordCount * layout->size,
                          .usage = rachis::BufferUsage::storage | rachis::BufferUsage::accelerationStructureBuildInput,
                          .memoryType = rachis::MemoryType::deviceLocal,
                          .label = "Vulkan GPU-written instance records",
                      });
    const auto rayResults = requireBuffer(device,
                                          {
                                              .size = instanceRecordCount * sizeof(RayResult),
                                              .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
                                              .memoryType = rachis::MemoryType::deviceLocal,
                                              .label = "Vulkan instance-buffer ray results",
                                          });
    const auto readback = requireBuffer(device,
                                        {
                                            .size = instanceRecordCount * sizeof(RayResult),
                                            .usage = rachis::BufferUsage::copyDestination,
                                            .memoryType = rachis::MemoryType::readback,
                                            .label = "Vulkan instance-buffer ray readback",
                                        });

    const std::array writerEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    const std::array writerShaderBindings{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1},
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
    };
    auto writerBindingLayout =
        device.createBindingLayout({.entries = writerEntries, .label = "Vulkan instance writer"});
    REQUIRE(writerBindingLayout.has_value());
    const std::array writerBindingLayouts{*writerBindingLayout};
    auto writerPipelineLayout =
        device.createPipelineLayout({.bindingLayouts = writerBindingLayouts, .label = "Vulkan instance writer"});
    REQUIRE(writerPipelineLayout.has_value());
    const auto writerPipeline = requireComputePipeline(
        device, "InstanceRecordWriter.spv", writerShaderBindings, *writerPipelineLayout, "Vulkan instance writer");
    const std::array writerWrites{
        rachis::BindingWrite{
            .binding = 0,
            .resource = rachis::BufferBinding{.buffer = instanceInputs, .size = instanceInputBytes},
        },
        rachis::BindingWrite{
            .binding = 1,
            .resource =
                rachis::BufferBinding{
                    .buffer = instanceRecords,
                    .offset = instanceRecordOffset,
                    .size = instanceRecordCount * layout->size,
                    .structureByteStride = layout->size,
                },
        },
    };
    auto writerSet = rachis::test::createBindingSet(
        device, {.layout = *writerBindingLayout, .writes = writerWrites, .label = "Vulkan instance writer"});
    if (!writerSet)
    {
        UNSCOPED_INFO(writerSet.error().message);
    }
    REQUIRE(writerSet.has_value());

    const std::array queryEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::accelerationStructure, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    const std::array queryShaderBindings{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::accelerationStructure, 1},
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
    };
    auto queryBindingLayout =
        device.createBindingLayout({.entries = queryEntries, .label = "Vulkan instance ray query"});
    REQUIRE(queryBindingLayout.has_value());
    const std::array queryBindingLayouts{*queryBindingLayout};
    auto queryPipelineLayout =
        device.createPipelineLayout({.bindingLayouts = queryBindingLayouts, .label = "Vulkan instance ray query"});
    REQUIRE(queryPipelineLayout.has_value());
    const auto queryPipeline = requireComputePipeline(
        device, "InstanceBufferRayQuery.spv", queryShaderBindings, *queryPipelineLayout, "Vulkan instance ray query");

    std::vector<rachis::BindingSetHandle> querySets;
    const auto traceAndRead = [&](rachis::CommandBuffer command, rachis::AccelerationStructureHandle topLevel) {
        const std::array queryWrites{
            rachis::BindingWrite{.binding = 0, .resource = topLevel},
            rachis::BindingWrite{
                .binding = 1,
                .resource =
                    rachis::BufferBinding{
                        .buffer = rayResults,
                        .size = instanceRecordCount * sizeof(RayResult),
                        .structureByteStride = sizeof(RayResult),
                    },
            },
        };
        auto querySet = rachis::test::createBindingSet(
            device, {.layout = *queryBindingLayout, .writes = queryWrites, .label = "Vulkan instance ray query"});
        if (!querySet)
        {
            UNSCOPED_INFO(querySet.error().message);
        }
        REQUIRE(querySet.has_value());
        querySets.push_back(*querySet);
        requireBarrier(device,
                       command,
                       rayResults,
                       rachis::PipelineStage::all,
                       rachis::Access::common,
                       rachis::PipelineStage::computeShader,
                       rachis::Access::unorderedAccess);
        REQUIRE(device.bindComputePipeline(command, queryPipeline).has_value());
        REQUIRE(device.bindComputeBindingSet(command, 0, *querySet).has_value());
        REQUIRE(device.dispatch(command, 1).has_value());
        requireBarrier(device,
                       command,
                       rayResults,
                       rachis::PipelineStage::computeShader,
                       rachis::Access::unorderedAccess,
                       rachis::PipelineStage::copy,
                       rachis::Access::copySource);
        REQUIRE(device
                    .copyBuffer(command,
                                {
                                    .source = rayResults,
                                    .destination = readback,
                                    .size = instanceRecordCount * sizeof(RayResult),
                                })
                    .has_value());
        auto submitted = device.submit(std::move(command));
        if (!submitted)
        {
            UNSCOPED_INFO(submitted.error().message);
        }
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());
        std::array<RayResult, instanceRecordCount> results{};
        REQUIRE(device.readBuffer(readback, 0, std::as_writable_bytes(std::span(results))).has_value());
        return results;
    };

    const auto writeInputs = [&](const std::array<InstanceInput, instanceRecordCount>& inputs) {
        REQUIRE(device.writeBuffer(instanceInputs, 0, std::as_bytes(std::span(inputs))).has_value());
    };
    const auto recordWriterPass =
        [&](rachis::CommandBuffer& command, rachis::PipelineStage stagesBefore, rachis::Access accessBefore) {
            requireBarrier(device,
                           command,
                           instanceRecords,
                           stagesBefore,
                           accessBefore,
                           rachis::PipelineStage::computeShader,
                           rachis::Access::unorderedAccess);
            REQUIRE(device.bindComputePipeline(command, writerPipeline).has_value());
            REQUIRE(device.bindComputeBindingSet(command, 0, *writerSet).has_value());
            REQUIRE(device.dispatch(command, 1).has_value());
            requireBarrier(device,
                           command,
                           instanceRecords,
                           rachis::PipelineStage::computeShader,
                           rachis::Access::unorderedAccess,
                           rachis::PipelineStage::accelerationStructureBuild,
                           rachis::Access::accelerationStructureBuildInput);
        };

    writeInputs({
        instanceInput(*nearReference, -2.0F, 7),
        instanceInput(0, 0.0F, 11),
        instanceInput(*farReference, 2.0F, 9),
    });
    {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .buildBottomLevelAccelerationStructure(
                        *command, {.destination = nearBottomLevel, .input = nearInput, .scratch = scratch})
                    .has_value());
        REQUIRE(device
                    .buildBottomLevelAccelerationStructure(
                        *command, {.destination = farBottomLevel, .input = farInput, .scratch = scratch})
                    .has_value());
        recordWriterPass(*command, rachis::PipelineStage::all, rachis::Access::common);
        const rachis::TopLevelAccelerationStructureBufferBuildDesc build{
            .destination = builtTopLevel,
            .instanceBuffer = instanceRecords,
            .instanceOffset = instanceRecordOffset,
            .instanceCount = instanceRecordCount,
            .flags = topLevelFlags,
            .scratch = scratch,
        };
        auto built = device.buildTopLevelAccelerationStructureFromBuffer(*command, build);
        if (!built)
        {
            UNSCOPED_INFO(built.error().message);
        }
        REQUIRE(built.has_value());
        auto rebuilt = device.buildTopLevelAccelerationStructureFromBuffer(*command, build);
        REQUIRE_FALSE(rebuilt.has_value());
        CHECK(rebuilt.error().code == rachis::ErrorCode::invalidState);
        const auto results = traceAndRead(std::move(*command), builtTopLevel);
        INFO("build from GPU-written records");
        checkHit(results[0], 7, 1.0F, 0);
        checkMiss(results[1]);
        checkHit(results[2], 9, 1.5F, 2);
    }

    writeInputs({
        instanceInput(*farReference, -2.0F, 7),
        instanceInput(0, 0.0F, 11),
        instanceInput(*nearReference, 2.0F, 9),
    });
    {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command.has_value());
        recordWriterPass(*command,
                         rachis::PipelineStage::accelerationStructureBuild,
                         rachis::Access::accelerationStructureBuildInput);
        rachis::TopLevelAccelerationStructureBufferUpdateDesc update{
            .source = builtTopLevel,
            .destination = updatedTopLevel,
            .instanceBuffer = instanceRecords,
            .instanceOffset = instanceRecordOffset,
            .instanceCount = instanceRecordCount - 1,
            .flags = topLevelFlags,
            .scratch = scratch,
        };
        auto mismatchedCount = device.updateTopLevelAccelerationStructureFromBuffer(*command, update);
        REQUIRE_FALSE(mismatchedCount.has_value());
        CHECK(mismatchedCount.error().code == rachis::ErrorCode::invalidArgument);
        update.instanceCount = instanceRecordCount;
        auto changedFlags = update;
        changedFlags.flags = rachis::AccelerationStructureBuildFlag::allowUpdate;
        auto mismatchedFlags = device.updateTopLevelAccelerationStructureFromBuffer(*command, changedFlags);
        REQUIRE_FALSE(mismatchedFlags.has_value());
        CHECK(mismatchedFlags.error().code == rachis::ErrorCode::invalidArgument);
        auto updated = device.updateTopLevelAccelerationStructureFromBuffer(*command, update);
        if (!updated)
        {
            UNSCOPED_INFO(updated.error().message);
        }
        REQUIRE(updated.has_value());
        auto updatedAgain = device.updateTopLevelAccelerationStructureFromBuffer(*command, update);
        REQUIRE_FALSE(updatedAgain.has_value());
        CHECK(updatedAgain.error().code == rachis::ErrorCode::invalidState);
        const auto results = traceAndRead(std::move(*command), updatedTopLevel);
        INFO("out-of-place update from GPU-written records");
        checkHit(results[0], 7, 1.5F, 0);
        checkMiss(results[1]);
        checkHit(results[2], 9, 1.0F, 2);
    }

    const std::array readerWrites{
        rachis::BindingWrite{.binding = 0, .resource = updatedTopLevel},
        rachis::BindingWrite{
            .binding = 1,
            .resource =
                rachis::BufferBinding{
                    .buffer = rayResults,
                    .size = instanceRecordCount * sizeof(RayResult),
                    .structureByteStride = sizeof(RayResult),
                },
        },
    };
    auto readerSet = rachis::test::createBindingSet(
        device, {.layout = *queryBindingLayout, .writes = readerWrites, .label = "Vulkan concurrent TLAS reader"});
    REQUIRE(readerSet.has_value());
    const auto recordReader = [&](rachis::CommandBuffer& reader) {
        REQUIRE(device.bindComputePipeline(reader, queryPipeline).has_value());
        return device.bindComputeBindingSet(reader, 0, *readerSet);
    };

    writeInputs({
        instanceInput(*farReference, 2.0F, 5),
        instanceInput(0, 0.0F, 11),
        instanceInput(*nearReference, -2.0F, 3),
    });
    {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command.has_value());
        recordWriterPass(*command,
                         rachis::PipelineStage::accelerationStructureBuild,
                         rachis::Access::accelerationStructureBuildInput);
        const rachis::TopLevelAccelerationStructureBufferInPlaceUpdateDesc update{
            .target = updatedTopLevel,
            .instanceBuffer = instanceRecords,
            .instanceOffset = instanceRecordOffset,
            .instanceCount = instanceRecordCount,
            .flags = topLevelFlags,
            .scratch = scratch,
        };
        {
            auto reader = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
            REQUIRE(reader.has_value());
            REQUIRE(recordReader(*reader).has_value());
            auto blocked = device.updateTopLevelAccelerationStructureInPlaceFromBuffer(*command, update);
            REQUIRE_FALSE(blocked.has_value());
            CHECK(blocked.error().code == rachis::ErrorCode::invalidState);
        }
        auto updated = device.updateTopLevelAccelerationStructureInPlaceFromBuffer(*command, update);
        if (!updated)
        {
            UNSCOPED_INFO(updated.error().message);
        }
        REQUIRE(updated.has_value());
        {
            auto reader = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
            REQUIRE(reader.has_value());
            auto blockedReader = recordReader(*reader);
            REQUIRE_FALSE(blockedReader.has_value());
            CHECK(blockedReader.error().code == rachis::ErrorCode::invalidState);
        }
        const auto results = traceAndRead(std::move(*command), updatedTopLevel);
        INFO("in-place update from GPU-written records");
        checkHit(results[0], 3, 1.0F, 2);
        checkMiss(results[1]);
        checkHit(results[2], 5, 1.5F, 0);
    }
    device.collectGarbage();
    {
        auto reader = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(reader.has_value());
        CHECK(recordReader(*reader).has_value());
    }
    REQUIRE(device.destroyBindingSet(*readerSet).has_value());

    {
        const auto uploadedRecords =
            requireBuffer(device,
                          {
                              .size = instanceRecordCount * sizeof(rachis::AccelerationStructureInstanceRecord),
                              .usage = rachis::BufferUsage::accelerationStructureBuildInput,
                              .memoryType = rachis::MemoryType::upload,
                              .label = "Vulkan CPU-written instance records",
                          });
        std::array<rachis::AccelerationStructureInstanceRecord, instanceRecordCount> records{};
        records[0].transform[3] = 2.0F;
        records[0].instanceIdAndMask = 0xFF00'0000U | 21U;
        records[0].hitGroupOffsetAndFlags =
            static_cast<std::uint32_t>(rachis::AccelerationStructureInstanceFlag::triangleCullDisable) << 24U;
        records[0].bottomLevel = *nearReference;
        records[1].instanceIdAndMask = 22U;
        records[1].bottomLevel = *farReference;
        records[2].transform[3] = -2.0F;
        records[2].instanceIdAndMask = 0xFF00'0000U | 23U;
        records[2].hitGroupOffsetAndFlags =
            static_cast<std::uint32_t>(rachis::AccelerationStructureInstanceFlag::triangleCullDisable) << 24U;
        records[2].bottomLevel = *farReference;
        REQUIRE(device.writeBuffer(uploadedRecords, 0, std::as_bytes(std::span(records))).has_value());

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .buildTopLevelAccelerationStructureFromBuffer(*command,
                                                                  {
                                                                      .destination = uploadedTopLevel,
                                                                      .instanceBuffer = uploadedRecords,
                                                                      .instanceCount = instanceRecordCount,
                                                                      .flags = topLevelFlags,
                                                                      .scratch = scratch,
                                                                  })
                    .has_value());
        REQUIRE(device.destroyBuffer(uploadedRecords).has_value());
        const auto results = traceAndRead(std::move(*command), uploadedTopLevel);
        INFO("build from CPU-written AccelerationStructureInstanceRecord values");
        checkHit(results[0], 23, 1.5F, 2);
        checkMiss(results[1]);
        checkHit(results[2], 21, 1.0F, 0);
    }

    const auto spanInstance = [](rachis::AccelerationStructureHandle bottomLevel,
                                 float translationX,
                                 std::uint32_t instanceId,
                                 std::uint32_t mask) {
        rachis::AccelerationStructureInstance instance{
            .bottomLevel = bottomLevel,
            .instanceId = instanceId,
            .instanceMask = mask,
            .flags = rachis::AccelerationStructureInstanceFlag::triangleCullDisable,
        };
        instance.transform[3] = translationX;
        return instance;
    };
    {
        const std::array instances{
            spanInstance(nearBottomLevel, -2.0F, 31, 0xFF),
            spanInstance(farBottomLevel, 0.0F, 32, 0),
            spanInstance(farBottomLevel, 2.0F, 33, 0xFF),
        };
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command.has_value());
        auto mismatchedCount =
            device.updateTopLevelAccelerationStructure(*command,
                                                       {
                                                           .source = uploadedTopLevel,
                                                           .destination = spanTopLevel,
                                                           .instances = std::span(instances).first(2),
                                                           .flags = topLevelFlags,
                                                           .scratch = scratch,
                                                       });
        REQUIRE_FALSE(mismatchedCount.has_value());
        CHECK(mismatchedCount.error().code == rachis::ErrorCode::invalidArgument);
        auto sameHandles = device.updateTopLevelAccelerationStructure(*command,
                                                                      {
                                                                          .source = uploadedTopLevel,
                                                                          .destination = uploadedTopLevel,
                                                                          .instances = instances,
                                                                          .flags = topLevelFlags,
                                                                          .scratch = scratch,
                                                                      });
        REQUIRE_FALSE(sameHandles.has_value());
        CHECK(sameHandles.error().code == rachis::ErrorCode::invalidArgument);
        auto updated = device.updateTopLevelAccelerationStructure(*command,
                                                                  {
                                                                      .source = uploadedTopLevel,
                                                                      .destination = spanTopLevel,
                                                                      .instances = instances,
                                                                      .flags = topLevelFlags,
                                                                      .scratch = scratch,
                                                                  });
        if (!updated)
        {
            UNSCOPED_INFO(updated.error().message);
        }
        REQUIRE(updated.has_value());
        const auto results = traceAndRead(std::move(*command), spanTopLevel);
        INFO("out-of-place span update of a structure built from a buffer");
        checkHit(results[0], 31, 1.0F, 0);
        checkMiss(results[1]);
        checkHit(results[2], 33, 1.5F, 2);
    }
    {
        const std::array instances{
            spanInstance(farBottomLevel, -2.0F, 41, 0xFF),
            spanInstance(nearBottomLevel, 0.0F, 42, 0),
            spanInstance(nearBottomLevel, 2.0F, 43, 0xFF),
        };
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command.has_value());
        auto updated = device.updateTopLevelAccelerationStructureInPlace(*command,
                                                                         {
                                                                             .target = spanTopLevel,
                                                                             .instances = instances,
                                                                             .flags = topLevelFlags,
                                                                             .scratch = scratch,
                                                                         });
        if (!updated)
        {
            UNSCOPED_INFO(updated.error().message);
        }
        REQUIRE(updated.has_value());
        const auto results = traceAndRead(std::move(*command), spanTopLevel);
        INFO("in-place span update");
        checkHit(results[0], 41, 1.5F, 0);
        checkMiss(results[1]);
        checkHit(results[2], 43, 1.0F, 2);
    }

    for (const auto set : querySets)
    {
        REQUIRE(device.destroyBindingSet(set).has_value());
    }
    REQUIRE(device.destroyBindingSet(*writerSet).has_value());
    REQUIRE(device.destroyPipeline(writerPipeline).has_value());
    REQUIRE(device.destroyPipeline(queryPipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*writerPipelineLayout).has_value());
    REQUIRE(device.destroyPipelineLayout(*queryPipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*writerBindingLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*queryBindingLayout).has_value());
    destroyAccelerationStructures(device, structures);
    for (const auto buffer : {vertexBuffer, scratch, instanceInputs, instanceRecords, rayResults, readback})
    {
        REQUIRE(device.destroyBuffer(buffer).has_value());
    }
    device.collectGarbage();

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
#endif

TEST_CASE("Vulkan TLAS builds from instance buffers validate the buffer range and handles",
          "[size:medium][rhi][vulkan][ray-query]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (!capabilities.rayQuery && !capabilities.rayTracing)
    {
        auto layout = device.queryAccelerationStructureInstanceLayout();
        REQUIRE_FALSE(layout.has_value());
        CHECK(layout.error().code == rachis::ErrorCode::unsupported);
        SKIP("The Vulkan adapter does not advertise acceleration structures.");
    }

    auto layout = device.queryAccelerationStructureInstanceLayout();
    REQUIRE(layout.has_value());
    CHECK(layout->format == rachis::AccelerationStructureInstanceFormat::packed64);
    CHECK(layout->size == 64);
    CHECK(layout->alignment == 16);

    auto topSizes = device.queryTopLevelAccelerationStructureBuildSizes({.instanceCount = 2, .flags = topLevelFlags});
    REQUIRE(topSizes.has_value());
    const std::array structures{
        requireAccelerationStructure(device,
                                     rachis::AccelerationStructureType::topLevel,
                                     topSizes->accelerationStructureSize,
                                     "Vulkan validated TLAS"),
    };
    const auto topLevel = structures[0].structure;
    const auto bottomOwned = requireAccelerationStructure(device,
                                                          rachis::AccelerationStructureType::bottomLevel,
                                                          topSizes->accelerationStructureSize,
                                                          "Vulkan unbuilt BLAS");
    const auto bottomLevel = bottomOwned.structure;
    const auto scratch = requireBuffer(device,
                                       {
                                           .size = topSizes->buildScratchSize,
                                           .usage = rachis::BufferUsage::accelerationStructureScratch,
                                           .memoryType = rachis::MemoryType::deviceLocal,
                                           .label = "Vulkan validated TLAS scratch",
                                       });
    const auto records = requireBuffer(device,
                                       {
                                           .size = 2 * layout->size,
                                           .usage = rachis::BufferUsage::accelerationStructureBuildInput,
                                           .memoryType = rachis::MemoryType::upload,
                                           .label = "Vulkan inactive instance records",
                                       });
    const std::array<rachis::AccelerationStructureInstanceRecord, 2> inactiveRecords{};
    REQUIRE(device.writeBuffer(records, 0, std::as_bytes(std::span(inactiveRecords))).has_value());
    const auto storageOnly = requireBuffer(device,
                                           {
                                               .size = 2 * layout->size,
                                               .usage = rachis::BufferUsage::storage,
                                               .memoryType = rachis::MemoryType::deviceLocal,
                                               .label = "Vulkan instance records without build-input usage",
                                           });

    auto bottomLevelReference = device.queryAccelerationStructureReference(bottomLevel);
    REQUIRE(bottomLevelReference.has_value());
    CHECK(*bottomLevelReference != 0);
    auto nullReference = device.queryAccelerationStructureReference({});
    REQUIRE_FALSE(nullReference.has_value());
    CHECK(nullReference.error().code == rachis::ErrorCode::invalidArgument);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    const rachis::TopLevelAccelerationStructureBufferBuildDesc valid{
        .destination = topLevel,
        .instanceBuffer = records,
        .instanceCount = 2,
        .flags = topLevelFlags,
        .scratch = scratch,
    };
    const auto expectBuildFailure =
        [&](rachis::TopLevelAccelerationStructureBufferBuildDesc desc, rachis::ErrorCode code, const char* reason) {
            INFO(reason);
            auto built = device.buildTopLevelAccelerationStructureFromBuffer(*command, desc);
            REQUIRE_FALSE(built.has_value());
            CHECK(built.error().code == code);
        };
    auto withBuffer = valid;
    withBuffer.instanceBuffer = {};
    expectBuildFailure(withBuffer, rachis::ErrorCode::invalidArgument, "null instance buffer");
    auto withCount = valid;
    withCount.instanceCount = 0;
    expectBuildFailure(withCount, rachis::ErrorCode::invalidArgument, "zero instances");
    auto withOffset = valid;
    withOffset.instanceOffset = 8;
    withOffset.instanceCount = 1;
    expectBuildFailure(withOffset, rachis::ErrorCode::invalidArgument, "misaligned offset");
    auto pastEnd = valid;
    pastEnd.instanceOffset = layout->size;
    expectBuildFailure(pastEnd, rachis::ErrorCode::invalidArgument, "records past the buffer end");
    auto withoutUsage = valid;
    withoutUsage.instanceBuffer = storageOnly;
    expectBuildFailure(withoutUsage, rachis::ErrorCode::invalidArgument, "buffer without build-input usage");
    auto intoBottomLevel = valid;
    intoBottomLevel.destination = bottomLevel;
    expectBuildFailure(intoBottomLevel, rachis::ErrorCode::invalidArgument, "bottom-level destination");

    REQUIRE(device.buildTopLevelAccelerationStructureFromBuffer(*command, valid).has_value());
    expectBuildFailure(valid, rachis::ErrorCode::invalidState, "second build of one destination");
    auto sameHandles = device.updateTopLevelAccelerationStructureFromBuffer(*command,
                                                                            {
                                                                                .source = topLevel,
                                                                                .destination = topLevel,
                                                                                .instanceBuffer = records,
                                                                                .instanceCount = 2,
                                                                                .flags = topLevelFlags,
                                                                                .scratch = scratch,
                                                                            });
    REQUIRE_FALSE(sameHandles.has_value());
    CHECK(sameHandles.error().code == rachis::ErrorCode::invalidArgument);

    REQUIRE(device.destroyBuffer(records).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    destroyAccelerationStructures(device, std::span(&bottomOwned, 1));
    auto staleReference = device.queryAccelerationStructureReference(bottomLevel);
    REQUIRE_FALSE(staleReference.has_value());
    CHECK(staleReference.error().code == rachis::ErrorCode::invalidArgument);
    destroyAccelerationStructures(device, structures);
    REQUIRE(device.destroyBuffer(scratch).has_value());
    REQUIRE(device.destroyBuffer(storageOnly).has_value());
    device.collectGarbage();

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
