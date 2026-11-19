#include "D3D12/D3D12TestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
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

    class ForceLegacyBarriers final
    {
    public:
        explicit ForceLegacyBarriers(bool enabled)
            : previous_(rachis::d3d12::test::forceLegacyBarriers().exchange(enabled, std::memory_order_relaxed))
        {
        }

        ~ForceLegacyBarriers()
        {
            rachis::d3d12::test::forceLegacyBarriers().store(previous_, std::memory_order_relaxed);
        }

        ForceLegacyBarriers(const ForceLegacyBarriers&) = delete;
        ForceLegacyBarriers& operator=(const ForceLegacyBarriers&) = delete;
        ForceLegacyBarriers(ForceLegacyBarriers&&) = delete;
        ForceLegacyBarriers& operator=(ForceLegacyBarriers&&) = delete;

    private:
        bool previous_ = false;
    };

    struct IndirectArgumentReadScope final
    {
        rachis::PipelineStage stages = rachis::PipelineStage::none;
        rachis::Access access = rachis::Access::none;
    };

    [[nodiscard]] std::optional<std::uint64_t> deviceLocalAllocationCount(const rachis::Device& device)
    {
        auto budgets = device.queryMemoryBudgets();
        if (!budgets)
        {
            return std::nullopt;
        }
        const auto deviceLocal = std::ranges::find_if(*budgets, [](const rachis::MemoryBudget& budget) {
            return hasAny(budget.properties, rachis::MemoryHeapProperty::deviceLocal) &&
                   hasAny(budget.features, rachis::MemoryBudgetFeature::allocationStatistics);
        });
        if (deviceLocal == budgets->end())
        {
            return std::nullopt;
        }
        return deviceLocal->allocationCount;
    }

    [[nodiscard]] std::vector<std::byte> loadShader(const char* path = RACHIS_TEST_RAY_TRACING_PIPELINE_SHADER_PATH)
    {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
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

    constexpr rachis::ShaderRecordConstantRange materialRecordConstants{
        .shaderRegister = 1,
        .registerSpace = rachis::pushConstantRegisterSpace,
        .size = sizeof(std::uint32_t),
    };
    constexpr rachis::ShaderRecordConstantRange extendedMaterialRecordConstants{
        .shaderRegister = 1,
        .registerSpace = rachis::pushConstantRegisterSpace,
        .size = 2 * sizeof(std::uint32_t),
    };
    constexpr rachis::ShaderRecordConstantRange alphaTestRecordConstants{
        .shaderRegister = 2,
        .registerSpace = rachis::pushConstantRegisterSpace,
        .size = 2 * sizeof(std::uint32_t),
    };

    struct SharedHitShaderScene final
    {
        rachis::ShaderModuleHandle rayGeneration = {};
        rachis::ShaderModuleHandle miss = {};
        rachis::ShaderModuleHandle recordClosestHit = {};
        rachis::ShaderModuleHandle plainClosestHit = {};
        rachis::ShaderModuleHandle materialAnyHit = {};
        rachis::ShaderModuleHandle alphaTestAnyHit = {};
        rachis::BindingLayoutHandle bindingLayout = {};
        rachis::PipelineLayoutHandle pipelineLayout = {};
        rachis::BufferHandle vertices = {};
        rachis::BufferHandle bottomStorage = {};
        rachis::BufferHandle topStorage = {};
        rachis::BufferHandle scratch = {};
        rachis::AccelerationStructureHandle bottomLevel = {};
        rachis::AccelerationStructureHandle topLevel = {};
    };

    [[nodiscard]] SharedHitShaderScene createSharedHitShaderScene(rachis::Device& device)
    {
        SharedHitShaderScene scene;
        const auto bytecode = loadShader();
        REQUIRE_FALSE(bytecode.empty());
        const std::array rayGenerationBindings{
            rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1},
            rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::accelerationStructure, 1},
        };
        auto rayGeneration = device.createShaderModule({
            .stage = rachis::ShaderStage::rayGeneration,
            .bytecode = bytecode,
            .entryPoint = "rayGenerationTracePerHitGroup",
            .label = "Per-hit-group tracing ray generation",
            .shaderInterface = rachis::ShaderInterfaceDesc{.bindings = rayGenerationBindings},
        });
        auto miss = device.createShaderModule({
            .stage = rachis::ShaderStage::miss,
            .bytecode = bytecode,
            .entryPoint = "missShader",
            .label = "Shared-hit miss",
        });
        auto recordClosestHit = device.createShaderModule({
            .stage = rachis::ShaderStage::closestHit,
            .bytecode = bytecode,
            .entryPoint = "closestHitRecordValue",
            .label = "Shared record-reading closest hit",
        });
        auto plainClosestHit = device.createShaderModule({
            .stage = rachis::ShaderStage::closestHit,
            .bytecode = bytecode,
            .entryPoint = "closestHitShader",
            .label = "Shared record-free closest hit",
        });
        auto materialAnyHit = device.createShaderModule({
            .stage = rachis::ShaderStage::anyHit,
            .bytecode = bytecode,
            .entryPoint = "anyHitRecordValue",
            .label = "Material any hit",
            .shaderInterface = rachis::ShaderInterfaceDesc{.shaderRecordConstants = materialRecordConstants},
        });
        auto alphaTestAnyHit = device.createShaderModule({
            .stage = rachis::ShaderStage::anyHit,
            .bytecode = bytecode,
            .entryPoint = "anyHitAlphaTest",
            .label = "Alpha-test any hit",
            .shaderInterface = rachis::ShaderInterfaceDesc{.shaderRecordConstants = alphaTestRecordConstants},
        });
        REQUIRE(rayGeneration.has_value());
        REQUIRE(miss.has_value());
        REQUIRE(recordClosestHit.has_value());
        REQUIRE(plainClosestHit.has_value());
        REQUIRE(materialAnyHit.has_value());
        REQUIRE(alphaTestAnyHit.has_value());
        scene.rayGeneration = *rayGeneration;
        scene.miss = *miss;
        scene.recordClosestHit = *recordClosestHit;
        scene.plainClosestHit = *plainClosestHit;
        scene.materialAnyHit = *materialAnyHit;
        scene.alphaTestAnyHit = *alphaTestAnyHit;

        const std::array layoutEntries{
            rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::rayGeneration, 1},
            rachis::BindingLayoutEntry{
                1, rachis::BindingType::accelerationStructure, rachis::ShaderStage::rayGeneration, 1},
        };
        auto bindingLayout = device.createBindingLayout({.entries = layoutEntries, .label = "Shared-hit resources"});
        REQUIRE(bindingLayout.has_value());
        scene.bindingLayout = *bindingLayout;
        const std::array bindingLayouts{*bindingLayout};
        auto pipelineLayout =
            device.createPipelineLayout({.bindingLayouts = bindingLayouts, .label = "Shared-hit pipeline layout"});
        REQUIRE(pipelineLayout.has_value());
        scene.pipelineLayout = *pipelineLayout;

        const std::array vertices{
            std::array{-1.0F, -1.0F, 0.0F},
            std::array{0.0F, 1.0F, 0.0F},
            std::array{1.0F, -1.0F, 0.0F},
        };
        auto vertexBuffer = device.createBuffer({
            .size = sizeof(vertices),
            .usage = rachis::BufferUsage::accelerationStructureBuildInput,
            .memoryType = rachis::MemoryType::upload,
            .label = "Shared-hit triangle vertices",
        });
        REQUIRE(vertexBuffer.has_value());
        scene.vertices = *vertexBuffer;
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
            },
        };
        const rachis::BottomLevelAccelerationStructureBuildInput bottomInput{.geometries = geometry};
        auto bottomSizes = device.queryBottomLevelAccelerationStructureBuildSizes(bottomInput);
        auto topSizes = device.queryTopLevelAccelerationStructureBuildSizes({.instanceCount = 1});
        REQUIRE(bottomSizes.has_value());
        REQUIRE(topSizes.has_value());
        auto bottomStorage = device.createBuffer({
            .size = bottomSizes->accelerationStructureSize,
            .usage = rachis::BufferUsage::accelerationStructureStorage,
            .memoryType = rachis::MemoryType::deviceLocal,
            .label = "Shared-hit BLAS storage",
        });
        auto topStorage = device.createBuffer({
            .size = topSizes->accelerationStructureSize,
            .usage = rachis::BufferUsage::accelerationStructureStorage,
            .memoryType = rachis::MemoryType::deviceLocal,
            .label = "Shared-hit TLAS storage",
        });
        auto scratch = device.createBuffer({
            .size = (std::max)(bottomSizes->buildScratchSize, topSizes->buildScratchSize),
            .usage = rachis::BufferUsage::accelerationStructureScratch,
            .memoryType = rachis::MemoryType::deviceLocal,
            .label = "Shared-hit build scratch",
        });
        REQUIRE(bottomStorage.has_value());
        REQUIRE(topStorage.has_value());
        REQUIRE(scratch.has_value());
        scene.bottomStorage = *bottomStorage;
        scene.topStorage = *topStorage;
        scene.scratch = *scratch;
        auto bottomLevel = device.createAccelerationStructure({
            .type = rachis::AccelerationStructureType::bottomLevel,
            .storage = *bottomStorage,
            .label = "Shared-hit BLAS",
        });
        auto topLevel = device.createAccelerationStructure({
            .type = rachis::AccelerationStructureType::topLevel,
            .storage = *topStorage,
            .label = "Shared-hit TLAS",
        });
        REQUIRE(bottomLevel.has_value());
        REQUIRE(topLevel.has_value());
        scene.bottomLevel = *bottomLevel;
        scene.topLevel = *topLevel;

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
        const std::array instances{
            rachis::AccelerationStructureInstance{
                .bottomLevel = *bottomLevel,
                .flags = rachis::AccelerationStructureInstanceFlag::triangleCullDisable,
            },
        };
        REQUIRE(device
                    .buildTopLevelAccelerationStructure(*command,
                                                        {
                                                            .destination = *topLevel,
                                                            .instances = instances,
                                                            .scratch = *scratch,
                                                        })
                    .has_value());
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());
        return scene;
    }

    void destroySharedHitShaderScene(rachis::Device& device, const SharedHitShaderScene& scene)
    {
        REQUIRE(device.destroyAccelerationStructure(scene.topLevel).has_value());
        REQUIRE(device.destroyAccelerationStructure(scene.bottomLevel).has_value());
        REQUIRE(device.destroyBuffer(scene.scratch).has_value());
        REQUIRE(device.destroyBuffer(scene.topStorage).has_value());
        REQUIRE(device.destroyBuffer(scene.bottomStorage).has_value());
        REQUIRE(device.destroyBuffer(scene.vertices).has_value());
        REQUIRE(device.destroyPipelineLayout(scene.pipelineLayout).has_value());
        REQUIRE(device.destroyBindingLayout(scene.bindingLayout).has_value());
        REQUIRE(device.destroyShaderModule(scene.alphaTestAnyHit).has_value());
        REQUIRE(device.destroyShaderModule(scene.materialAnyHit).has_value());
        REQUIRE(device.destroyShaderModule(scene.plainClosestHit).has_value());
        REQUIRE(device.destroyShaderModule(scene.recordClosestHit).has_value());
        REQUIRE(device.destroyShaderModule(scene.miss).has_value());
        REQUIRE(device.destroyShaderModule(scene.rayGeneration).has_value());
        device.collectGarbage();
    }

    struct HitGroupMaterial final
    {
        rachis::ShaderModuleHandle anyHit = {};
        std::optional<rachis::ShaderRecordConstantRange> constants = {};
    };

    [[nodiscard]] std::array<rachis::RayTracingShaderGroupDesc, 4> sharedHitGroups(
        const SharedHitShaderScene& scene,
        rachis::ShaderModuleHandle closestHit,
        const HitGroupMaterial& first,
        const HitGroupMaterial& second)
    {
        return {
            rachis::RayTracingShaderGroupDesc{
                .type = rachis::RayTracingShaderGroupType::general,
                .generalShader = scene.rayGeneration,
            },
            rachis::RayTracingShaderGroupDesc{
                .type = rachis::RayTracingShaderGroupType::general,
                .generalShader = scene.miss,
            },
            rachis::RayTracingShaderGroupDesc{
                .type = rachis::RayTracingShaderGroupType::trianglesHitGroup,
                .closestHitShader = closestHit,
                .anyHitShader = first.anyHit,
                .shaderRecordConstants = first.constants,
            },
            rachis::RayTracingShaderGroupDesc{
                .type = rachis::RayTracingShaderGroupType::trianglesHitGroup,
                .closestHitShader = closestHit,
                .anyHitShader = second.anyHit,
                .shaderRecordConstants = second.constants,
            },
        };
    }

    [[nodiscard]] std::uint32_t expectedRecordSize(const rachis::RayTracingCapabilities& limits,
                                                   const std::optional<rachis::ShaderRecordConstantRange>& constants)
    {
        const auto localSize = constants.has_value() ? constants->size : 0U;
        const auto unalignedSize = limits.shaderGroupHandleSize + localSize;
        return ((unalignedSize + limits.shaderGroupHandleAlignment - 1U) / limits.shaderGroupHandleAlignment) *
               limits.shaderGroupHandleAlignment;
    }

    void requireSharedHitGroupRecordSizes(rachis::Device& device,
                                          rachis::PipelineHandle pipeline,
                                          std::span<const rachis::RayTracingShaderGroupDesc> groups)
    {
        const auto limits = rachis::test::requireCapabilities(device).rayTracingLimits;
        for (std::uint32_t group = 0; group < groups.size(); ++group)
        {
            INFO("group " << group);
            auto recordSize = device.rayTracingShaderGroupRecordSize(pipeline, group);
            REQUIRE(recordSize.has_value());
            CHECK(*recordSize == expectedRecordSize(limits, groups[group].shaderRecordConstants));
        }
    }

    void requireDistinctHitGroupHandles(rachis::Device& device, rachis::PipelineHandle pipeline)
    {
        const auto handleSize = rachis::test::requireCapabilities(device).rayTracingLimits.shaderGroupHandleSize;
        std::vector<std::byte> handles(2 * static_cast<std::size_t>(handleSize));
        REQUIRE(device.getRayTracingShaderGroupHandles(pipeline, 2, 2, handles).has_value());
        const auto handleBytes = std::span<const std::byte>{handles};
        CHECK_FALSE(std::ranges::equal(handleBytes.first(handleSize), handleBytes.last(handleSize)));
    }

    struct SharedHitPipelines final
    {
        rachis::RayTracingPipelineLibraryHandle library = {};
        rachis::PipelineHandle linked = {};
        rachis::PipelineHandle direct = {};
    };

    [[nodiscard]] SharedHitPipelines createSharedHitPipelines(rachis::Device& device,
                                                              const SharedHitShaderScene& scene,
                                                              std::span<const rachis::RayTracingShaderGroupDesc> groups)
    {
        auto library = device.createRayTracingPipelineLibrary({
            .layout = scene.pipelineLayout,
            .groups = groups,
            .maxPayloadSize = sizeof(std::uint32_t),
            .maxAttributeSize = 8,
            .label = "Shared-hit collection",
        });
        if (!library)
        {
            UNSCOPED_INFO("library creation failed: " << library.error().message);
        }
        REQUIRE(library.has_value());
        const std::array libraries{*library};
        auto linked = device.createRayTracingPipeline({
            .layout = scene.pipelineLayout,
            .libraries = libraries,
            .maxPayloadSize = sizeof(std::uint32_t),
            .maxAttributeSize = 8,
            .maxRecursionDepth = 1,
            .label = "Linked shared-hit pipeline",
        });
        if (!linked)
        {
            UNSCOPED_INFO("library link failed: " << linked.error().message);
        }
        REQUIRE(linked.has_value());
        auto direct = device.createRayTracingPipeline({
            .layout = scene.pipelineLayout,
            .groups = groups,
            .maxPayloadSize = sizeof(std::uint32_t),
            .maxAttributeSize = 8,
            .maxRecursionDepth = 1,
            .label = "Direct shared-hit pipeline",
        });
        if (!direct)
        {
            UNSCOPED_INFO("direct creation failed: " << direct.error().message);
        }
        REQUIRE(direct.has_value());
        requireSharedHitGroupRecordSizes(device, *linked, groups);
        requireSharedHitGroupRecordSizes(device, *direct, groups);
        return {.library = *library, .linked = *linked, .direct = *direct};
    }

    void destroySharedHitPipelines(rachis::Device& device, const SharedHitPipelines& pipelines)
    {
        REQUIRE(device.destroyPipeline(pipelines.direct).has_value());
        REQUIRE(device.destroyPipeline(pipelines.linked).has_value());
        REQUIRE(device.destroyRayTracingPipelineLibrary(pipelines.library).has_value());
    }

    [[nodiscard]] std::array<std::uint32_t, 2> traceSharedHitGroups(
        rachis::Device& device,
        const SharedHitShaderScene& scene,
        rachis::PipelineHandle pipeline,
        std::span<const std::span<const std::byte>> localConstants)
    {
        const auto limits = rachis::test::requireCapabilities(device).rayTracingLimits;
        REQUIRE(localConstants.size() == 4);
        std::array<std::uint32_t, 4> recordSizes{};
        for (std::uint32_t group = 0; group < recordSizes.size(); ++group)
        {
            auto recordSize = device.rayTracingShaderGroupRecordSize(pipeline, group);
            REQUIRE(recordSize.has_value());
            recordSizes[group] = *recordSize;
        }
        const std::uint64_t baseAlignment = limits.shaderGroupBaseAlignment;
        const std::uint64_t hitStride = (std::max)(recordSizes[2], recordSizes[3]);
        const std::array<std::uint64_t, 4> recordOffsets{
            0,
            baseAlignment,
            2 * baseAlignment,
            2 * baseAlignment + hitStride,
        };
        std::vector<std::byte> tableBytes(static_cast<std::size_t>(recordOffsets[3] + hitStride));
        for (std::uint32_t group = 0; group < recordSizes.size(); ++group)
        {
            INFO("group " << group);
            const auto destination = std::span<std::byte>{tableBytes}.subspan(
                static_cast<std::size_t>(recordOffsets[group]), recordSizes[group]);
            REQUIRE(device.writeRayTracingShaderGroupRecord(pipeline, group, localConstants[group], destination)
                        .has_value());
        }

        auto shaderTable = device.createBuffer({
            .size = tableBytes.size(),
            .usage = rachis::BufferUsage::shaderBindingTable,
            .memoryType = rachis::MemoryType::upload,
            .label = "Shared-hit shader table",
        });
        auto output = device.createBuffer({
            .size = 2 * sizeof(std::uint32_t),
            .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::deviceLocal,
            .label = "Shared-hit output",
        });
        auto readback = device.createBuffer({
            .size = 2 * sizeof(std::uint32_t),
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Shared-hit readback",
        });
        REQUIRE(shaderTable.has_value());
        REQUIRE(output.has_value());
        REQUIRE(readback.has_value());
        REQUIRE(device.writeBuffer(*shaderTable, 0, tableBytes).has_value());
        const std::array writes{
            rachis::BindingWrite{
                .binding = 0,
                .resource =
                    rachis::BufferBinding{
                        .buffer = *output,
                        .size = 2 * sizeof(std::uint32_t),
                        .structureByteStride = sizeof(std::uint32_t),
                    },
            },
            rachis::BindingWrite{.binding = 1, .resource = scene.topLevel},
        };
        auto bindingSet = rachis::test::createBindingSet(device,
                                                         {
                                                             .layout = scene.bindingLayout,
                                                             .writes = writes,
                                                             .label = "Shared-hit binding set",
                                                         });
        REQUIRE(bindingSet.has_value());

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .bufferBarrier(*command,
                                   {
                                       .buffer = *output,
                                       .stagesBefore = rachis::PipelineStage::all,
                                       .stagesAfter = rachis::PipelineStage::rayTracingShader,
                                       .accessBefore = rachis::Access::common,
                                       .accessAfter = rachis::Access::unorderedAccess,
                                   })
                    .has_value());
        REQUIRE(device.bindRayTracingPipeline(*command, pipeline).has_value());
        REQUIRE(device.bindRayTracingBindingSet(*command, 0, *bindingSet).has_value());
        REQUIRE(device
                    .dispatchRays(*command,
                                  {
                                      .rayGeneration =
                                          {
                                              .buffer = *shaderTable,
                                              .offset = recordOffsets[0],
                                              .size = recordSizes[0],
                                              .stride = recordSizes[0],
                                          },
                                      .miss =
                                          {
                                              .buffer = *shaderTable,
                                              .offset = recordOffsets[1],
                                              .size = recordSizes[1],
                                              .stride = recordSizes[1],
                                          },
                                      .hitGroup =
                                          {
                                              .buffer = *shaderTable,
                                              .offset = recordOffsets[2],
                                              .size = 2 * hitStride,
                                              .stride = hitStride,
                                          },
                                      .width = 2,
                                  })
                    .has_value());
        REQUIRE(device
                    .bufferBarrier(*command,
                                   {
                                       .buffer = *output,
                                       .stagesBefore = rachis::PipelineStage::rayTracingShader,
                                       .stagesAfter = rachis::PipelineStage::copy,
                                       .accessBefore = rachis::Access::unorderedAccess,
                                       .accessAfter = rachis::Access::copySource,
                                   })
                    .has_value());
        REQUIRE(device
                    .copyBuffer(*command,
                                {
                                    .source = *output,
                                    .destination = *readback,
                                    .size = 2 * sizeof(std::uint32_t),
                                })
                    .has_value());
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());

        std::array<std::uint32_t, 2> result{};
        REQUIRE(device.readBuffer(*readback, 0, std::as_writable_bytes(std::span{result})).has_value());
        REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
        REQUIRE(device.destroyBuffer(*readback).has_value());
        REQUIRE(device.destroyBuffer(*output).has_value());
        REQUIRE(device.destroyBuffer(*shaderTable).has_value());
        device.collectGarbage();
        return result;
    }

    void requireSharedHitShadersUseTheirGroupLayouts(rachis::Device& device)
    {
        if (!rachis::test::requireCapabilities(device).rayTracing)
        {
            SKIP("The selected adapter does not expose DXR pipeline ray tracing.");
        }
        const auto scene = createSharedHitShaderScene(device);

        constexpr std::uint32_t firstMaterialValue = 0x4d415431;
        SECTION("a shared closest hit links beside any-hit shaders with their own record layouts")
        {
            const auto groups =
                sharedHitGroups(scene,
                                scene.plainClosestHit,
                                {.anyHit = scene.materialAnyHit, .constants = materialRecordConstants},
                                {.anyHit = scene.alphaTestAnyHit, .constants = alphaTestRecordConstants});
            const auto pipelines = createSharedHitPipelines(device, scene, groups);
            constexpr std::array<std::uint32_t, 2> rejectingAlphaTest{1, 2};
            const std::array<std::span<const std::byte>, 4> localConstants{
                std::span<const std::byte>{},
                std::span<const std::byte>{},
                std::as_bytes(std::span{&firstMaterialValue, 1}),
                std::as_bytes(std::span{rejectingAlphaTest}),
            };
            for (const auto pipeline : {pipelines.direct, pipelines.linked})
            {
                INFO((pipeline == pipelines.direct ? "direct pipeline" : "linked pipeline"));
                requireDistinctHitGroupHandles(device, pipeline);
                const auto traced = traceSharedHitGroups(device, scene, pipeline, localConstants);
                CHECK(traced[0] == 1);
                CHECK(traced[1] == 0);
            }
            destroySharedHitPipelines(device, pipelines);
        }
        SECTION("a shared closest hit reads the record constants of each group layout")
        {
            const auto groups = sharedHitGroups(scene,
                                                scene.recordClosestHit,
                                                {.constants = materialRecordConstants},
                                                {.constants = extendedMaterialRecordConstants});
            const auto pipelines = createSharedHitPipelines(device, scene, groups);
            constexpr std::array<std::uint32_t, 2> extendedMaterialValues{0x4d415432, 0x414c5048};
            const std::array<std::span<const std::byte>, 4> localConstants{
                std::span<const std::byte>{},
                std::span<const std::byte>{},
                std::as_bytes(std::span{&firstMaterialValue, 1}),
                std::as_bytes(std::span{extendedMaterialValues}),
            };
            for (const auto pipeline : {pipelines.direct, pipelines.linked})
            {
                INFO((pipeline == pipelines.direct ? "direct pipeline" : "linked pipeline"));
                const auto traced = traceSharedHitGroups(device, scene, pipeline, localConstants);
                CHECK(traced[0] == firstMaterialValue);
                CHECK(traced[1] == extendedMaterialValues[0]);
            }
            destroySharedHitPipelines(device, pipelines);
        }
        SECTION("a shared closest hit runs in groups with and without record constants")
        {
            const auto groups =
                sharedHitGroups(scene, scene.plainClosestHit, {.constants = materialRecordConstants}, {});
            const auto pipelines = createSharedHitPipelines(device, scene, groups);
            const std::array<std::span<const std::byte>, 4> localConstants{
                std::span<const std::byte>{},
                std::span<const std::byte>{},
                std::as_bytes(std::span{&firstMaterialValue, 1}),
                std::span<const std::byte>{},
            };
            for (const auto pipeline : {pipelines.direct, pipelines.linked})
            {
                INFO((pipeline == pipelines.direct ? "direct pipeline" : "linked pipeline"));
                const auto traced = traceSharedHitGroups(device, scene, pipeline, localConstants);
                CHECK(traced[0] == 1);
                CHECK(traced[1] == 1);
            }
            destroySharedHitPipelines(device, pipelines);
        }

        destroySharedHitShaderScene(device, scene);
        const auto diagnostics = device.diagnosticCounts();
        CHECK(diagnostics.warnings == 0);
        CHECK(diagnostics.errors == 0);
        CHECK(diagnostics.corruptions == 0);
    }

    void requireRecordFreeGroupHidesSharedShaderRecord(rachis::Device& device)
    {
        if (!rachis::test::requireCapabilities(device).rayTracing)
        {
            SKIP("The selected adapter does not expose DXR pipeline ray tracing.");
        }
        const auto scene = createSharedHitShaderScene(device);
        const auto groups = sharedHitGroups(scene, scene.recordClosestHit, {.constants = materialRecordConstants}, {});
        auto pipeline = device.createRayTracingPipeline({
            .layout = scene.pipelineLayout,
            .groups = groups,
            .maxPayloadSize = sizeof(std::uint32_t),
            .maxAttributeSize = 8,
            .maxRecursionDepth = 1,
        });
        REQUIRE_FALSE(pipeline.has_value());
        CHECK(pipeline.error().code == rachis::ErrorCode::validationError);
        destroySharedHitShaderScene(device, scene);
        CHECK(device.diagnosticCounts().corruptions == 0);
    }

} // namespace

TEST_CASE("ray-generation dispatch retains its complete direct and indirect dependency graph",
          "[size:medium][rhi][ray-tracing-pipeline][hardware]")
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
    const auto limits = rachis::test::requireCapabilities(device).rayTracingLimits;
    if (!rachis::test::requireCapabilities(device).rayTracing)
    {
        SKIP("The selected hardware adapter does not expose DXR pipeline ray tracing.");
    }

    const auto bytecode = loadShader();
    REQUIRE_FALSE(bytecode.empty());
    const std::array shaderBindings{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1},
    };
    constexpr rachis::ShaderRecordConstantRange recordConstants{
        .shaderRegister = 1,
        .registerSpace = rachis::pushConstantRegisterSpace,
        .size = sizeof(std::uint32_t),
    };
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::rayGeneration,
        .bytecode = bytecode,
        .entryPoint = "rayGeneration",
        .label = "Ray-generation module",
        .shaderInterface =
            rachis::ShaderInterfaceDesc{
                .bindings = shaderBindings,
                .shaderRecordConstants = recordConstants,
            },
    });
    REQUIRE(shader.has_value());
    auto missShader = device.createShaderModule({
        .stage = rachis::ShaderStage::miss,
        .bytecode = bytecode,
        .entryPoint = "missShader",
        .label = "Reusable miss module",
        .shaderInterface = {},
    });
    REQUIRE(missShader.has_value());

    const std::array layoutEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::rayGeneration, 1},
    };
    auto bindingLayout = device.createBindingLayout({.entries = layoutEntries, .label = "Ray-generation resources"});
    REQUIRE(bindingLayout.has_value());
    const std::array bindingLayouts{*bindingLayout};
    auto pipelineLayout =
        device.createPipelineLayout({.bindingLayouts = bindingLayouts, .label = "Ray-tracing pipeline layout"});
    REQUIRE(pipelineLayout.has_value());
    const std::array groups{
        rachis::RayTracingShaderGroupDesc{
            .type = rachis::RayTracingShaderGroupType::general,
            .generalShader = *shader,
            .shaderRecordConstants = recordConstants,
        },
    };
    auto missingRecordConstants = groups;
    missingRecordConstants[0].shaderRecordConstants.reset();
    auto missingRecordPipeline = device.createRayTracingPipeline({
        .layout = *pipelineLayout,
        .groups = missingRecordConstants,
        .maxAttributeSize = 8,
        .maxRecursionDepth = 1,
    });
    REQUIRE_FALSE(missingRecordPipeline.has_value());
    CHECK(missingRecordPipeline.error().code == rachis::ErrorCode::validationError);

    auto mismatchedRecordConstants = groups;
    auto mismatchedRange = recordConstants;
    mismatchedRange.shaderRegister += 1;
    mismatchedRecordConstants[0].shaderRecordConstants = mismatchedRange;
    auto mismatchedRecordPipeline = device.createRayTracingPipeline({
        .layout = *pipelineLayout,
        .groups = mismatchedRecordConstants,
        .maxAttributeSize = 8,
        .maxRecursionDepth = 1,
    });
    REQUIRE_FALSE(mismatchedRecordPipeline.has_value());
    CHECK(mismatchedRecordPipeline.error().code == rachis::ErrorCode::validationError);

    auto oversizedRecordConstants = groups;
    auto oversizedRange = recordConstants;
    oversizedRange.size = limits.maxShaderGroupStride;
    oversizedRecordConstants[0].shaderRecordConstants = oversizedRange;
    auto oversizedRecordPipeline = device.createRayTracingPipeline({
        .layout = *pipelineLayout,
        .groups = oversizedRecordConstants,
        .maxAttributeSize = 8,
        .maxRecursionDepth = 1,
    });
    REQUIRE_FALSE(oversizedRecordPipeline.has_value());
    CHECK(oversizedRecordPipeline.error().code == rachis::ErrorCode::invalidArgument);

    auto library = device.createRayTracingPipelineLibrary({
        .layout = *pipelineLayout,
        .groups = groups,
        .maxPayloadSize = sizeof(std::uint32_t),
        .maxAttributeSize = 8,
        .label = "Reusable ray-generation collection",
    });
    REQUIRE(library.has_value());
    const std::array missGroups{
        rachis::RayTracingShaderGroupDesc{
            .type = rachis::RayTracingShaderGroupType::general,
            .generalShader = *missShader,
        },
    };
    auto missLibrary = device.createRayTracingPipelineLibrary({
        .layout = *pipelineLayout,
        .groups = missGroups,
        .maxPayloadSize = sizeof(std::uint32_t),
        .maxAttributeSize = 8,
        .label = "Reusable miss collection",
    });
    REQUIRE(missLibrary.has_value());
    const std::array libraries{*library, *missLibrary};
    auto incompatiblePipeline = device.createRayTracingPipeline({
        .layout = *pipelineLayout,
        .libraries = libraries,
        .maxAttributeSize = 8,
        .maxRecursionDepth = 1,
    });
    REQUIRE_FALSE(incompatiblePipeline.has_value());
    CHECK(incompatiblePipeline.error().code == rachis::ErrorCode::validationError);
    const std::array duplicateLibraries{*library, *library};
    auto duplicatePipeline = device.createRayTracingPipeline({
        .layout = *pipelineLayout,
        .libraries = duplicateLibraries,
        .maxPayloadSize = sizeof(std::uint32_t),
        .maxAttributeSize = 8,
        .maxRecursionDepth = 1,
    });
    REQUIRE_FALSE(duplicatePipeline.has_value());
    CHECK(duplicatePipeline.error().code == rachis::ErrorCode::invalidArgument);

    auto pipeline = device.createRayTracingPipeline({
        .layout = *pipelineLayout,
        .libraries = libraries,
        .groups = missGroups,
        .maxPayloadSize = sizeof(std::uint32_t),
        .maxAttributeSize = 8,
        .maxRecursionDepth = 1,
        .label = "Ray-generation pipeline",
    });
    REQUIRE(pipeline.has_value());
    REQUIRE(device.destroyRayTracingPipelineLibrary(*library).has_value());
    REQUIRE(device.destroyRayTracingPipelineLibrary(*missLibrary).has_value());
    auto staleLibraryPipeline = device.createRayTracingPipeline({
        .layout = *pipelineLayout,
        .libraries = libraries,
        .maxPayloadSize = sizeof(std::uint32_t),
        .maxAttributeSize = 8,
        .maxRecursionDepth = 1,
    });
    REQUIRE_FALSE(staleLibraryPipeline.has_value());
    CHECK(staleLibraryPipeline.error().code == rachis::ErrorCode::invalidArgument);

    std::vector<std::byte> shaderIdentifier(limits.shaderGroupHandleSize);
    auto undersized = device.getRayTracingShaderGroupHandles(
        *pipeline, 0, 1, std::span<std::byte>{shaderIdentifier}.first(shaderIdentifier.size() - 1));
    REQUIRE_FALSE(undersized.has_value());
    CHECK(undersized.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(device.getRayTracingShaderGroupHandles(*pipeline, 0, 1, shaderIdentifier).has_value());

    auto recordSize = device.rayTracingShaderGroupRecordSize(*pipeline, 0);
    REQUIRE(recordSize.has_value());
    CHECK(*recordSize == 2 * limits.shaderGroupHandleAlignment);
    auto missRecordSize = device.rayTracingShaderGroupRecordSize(*pipeline, 1);
    REQUIRE(missRecordSize.has_value());
    CHECK(*missRecordSize == limits.shaderGroupHandleAlignment);
    auto localMissRecordSize = device.rayTracingShaderGroupRecordSize(*pipeline, 2);
    REQUIRE(localMissRecordSize.has_value());
    CHECK(*localMissRecordSize == limits.shaderGroupHandleAlignment);
    auto invalidRecordSize = device.rayTracingShaderGroupRecordSize(*pipeline, 3);
    REQUIRE_FALSE(invalidRecordSize.has_value());
    CHECK(invalidRecordSize.error().code == rachis::ErrorCode::invalidArgument);
    std::vector<std::byte> shaderRecord(*recordSize, std::byte{0x5a});
    auto wrongConstants = device.writeRayTracingShaderGroupRecord(*pipeline, 0, {}, shaderRecord);
    REQUIRE_FALSE(wrongConstants.has_value());
    CHECK(wrongConstants.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(std::ranges::all_of(shaderRecord, [](std::byte value) { return value == std::byte{0x5a}; }));

    constexpr std::uint32_t expectedResult = 0x4c4f434c;
    const auto localConstants = std::as_bytes(std::span{&expectedResult, 1});
    auto wrongDestination = device.writeRayTracingShaderGroupRecord(
        *pipeline, 0, localConstants, std::span<std::byte>{shaderRecord}.first(shaderRecord.size() - 1));
    REQUIRE_FALSE(wrongDestination.has_value());
    CHECK(wrongDestination.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(std::ranges::all_of(shaderRecord, [](std::byte value) { return value == std::byte{0x5a}; }));
    REQUIRE(device.writeRayTracingShaderGroupRecord(*pipeline, 0, localConstants, shaderRecord).has_value());
    CHECK(
        std::ranges::equal(std::span<const std::byte>{shaderRecord}.first(shaderIdentifier.size()), shaderIdentifier));
    std::uint32_t encodedResult = 0;
    std::memcpy(&encodedResult, shaderRecord.data() + limits.shaderGroupHandleSize, sizeof(encodedResult));
    CHECK(encodedResult == expectedResult);
    CHECK(std::ranges::all_of(
        std::span<const std::byte>{shaderRecord}.subspan(limits.shaderGroupHandleSize + sizeof(expectedResult)),
        [](std::byte value) { return value == std::byte{0}; }));

    auto shaderTable = device.createBuffer({
        .size = std::max<std::uint64_t>(limits.shaderGroupBaseAlignment, *recordSize),
        .usage = rachis::BufferUsage::shaderBindingTable,
        .memoryType = rachis::MemoryType::upload,
        .label = "Ray-generation shader table",
    });
    auto output = device.createBuffer({
        .size = sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Ray-generation output",
    });
    auto readback = device.createBuffer({
        .size = sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Ray-generation readback",
    });
    constexpr std::uint64_t indirectArgumentOffset = sizeof(std::uint32_t);
    const rachis::DispatchIndirectArguments indirectDimensions{1, 1, 1};
    std::array<std::byte, indirectArgumentOffset + sizeof(indirectDimensions)> indirectUploadBytes{};
    std::memcpy(indirectUploadBytes.data() + indirectArgumentOffset, &indirectDimensions, sizeof(indirectDimensions));
    auto indirectUpload = device.createBuffer({
        .size = indirectUploadBytes.size(),
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Indirect ray upload",
    });
    auto indirectArguments = device.createBuffer({
        .size = indirectUploadBytes.size(),
        .usage = rachis::BufferUsage::copyDestination | rachis::BufferUsage::indirect,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "GPU-produced indirect ray arguments",
    });
    REQUIRE(shaderTable.has_value());
    REQUIRE(output.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(indirectUpload.has_value());
    REQUIRE(indirectArguments.has_value());
    REQUIRE(device.writeBuffer(*shaderTable, 0, shaderRecord).has_value());
    REQUIRE(device.writeBuffer(*indirectUpload, 0, indirectUploadBytes).has_value());

    const std::array writes{
        rachis::BindingWrite{
            .binding = 0,
            .resource =
                rachis::BufferBinding{
                    .buffer = *output,
                    .size = sizeof(std::uint32_t),
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device,
                                                     {
                                                         .layout = *bindingLayout,
                                                         .writes = writes,
                                                         .label = "Ray-generation binding set",
                                                     });
    REQUIRE(bindingSet.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *output,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::rayTracingShader,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindRayTracingPipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindRayTracingBindingSet(*command, 0, *bindingSet).has_value());
    const auto beforeRejectedPublication = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(beforeRejectedPublication.has_value());
    {
        const FailResourceRetention failInternalOwnerPublication(2);
        auto failedPublication = device.bindRayTracingBindingSet(*command, 0, *bindingSet);
        REQUIRE_FALSE(failedPublication.has_value());
        CHECK(failedPublication.error().code == rachis::ErrorCode::outOfMemory);
    }
    const auto afterRejectedPublication = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(afterRejectedPublication.has_value());
    CHECK(*afterRejectedPublication == *beforeRejectedPublication);
    REQUIRE(device
                .dispatchRays(*command,
                              {
                                  .rayGeneration =
                                      {
                                          .buffer = *shaderTable,
                                          .size = *recordSize,
                                          .stride = *recordSize,
                                      },
                              })
                .has_value());
    if (rachis::test::requireCapabilities(device).indirectRayDispatch)
    {
        REQUIRE(device
                    .bufferBarrier(*command,
                                   {
                                       .buffer = *output,
                                       .stagesBefore = rachis::PipelineStage::rayTracingShader,
                                       .stagesAfter = rachis::PipelineStage::copy,
                                       .accessBefore = rachis::Access::unorderedAccess,
                                       .accessAfter = rachis::Access::copyDestination,
                                   })
                    .has_value());
        REQUIRE(device
                    .bufferBarrier(*command,
                                   {
                                       .buffer = *indirectArguments,
                                       .stagesBefore = rachis::PipelineStage::all,
                                       .stagesAfter = rachis::PipelineStage::copy,
                                       .accessBefore = rachis::Access::common,
                                       .accessAfter = rachis::Access::copyDestination,
                                   })
                    .has_value());
        REQUIRE(device
                    .copyBuffer(*command,
                                {
                                    .source = *indirectUpload,
                                    .destination = *output,
                                    .size = sizeof(std::uint32_t),
                                })
                    .has_value());
        REQUIRE(device
                    .copyBuffer(*command,
                                {
                                    .source = *indirectUpload,
                                    .sourceOffset = indirectArgumentOffset,
                                    .destination = *indirectArguments,
                                    .destinationOffset = indirectArgumentOffset,
                                    .size = sizeof(indirectDimensions),
                                })
                    .has_value());
        const std::array transitions{
            rachis::BufferBarrier{
                .buffer = *output,
                .stagesBefore = rachis::PipelineStage::copy,
                .stagesAfter = rachis::PipelineStage::rayTracingShader,
                .accessBefore = rachis::Access::copyDestination,
                .accessAfter = rachis::Access::unorderedAccess,
            },
            rachis::BufferBarrier{
                .buffer = *indirectArguments,
                .stagesBefore = rachis::PipelineStage::copy,
                .stagesAfter = rachis::PipelineStage::indirect,
                .accessBefore = rachis::Access::copyDestination,
                .accessAfter = rachis::Access::indirectArgument,
            },
        };
        REQUIRE(device.resourceBarriers(*command, {.bufferBarriers = transitions}).has_value());

        const rachis::RayTracingIndirectDispatchDesc indirectDesc{
            .rayGeneration =
                {
                    .buffer = *shaderTable,
                    .size = *recordSize,
                    .stride = *recordSize,
                },
            .argumentBuffer = *indirectArguments,
            .argumentOffset = indirectArgumentOffset,
        };
        auto misaligned = indirectDesc;
        misaligned.argumentOffset = 1;
        auto rejected = device.dispatchRaysIndirect(*command, misaligned);
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);

        auto unretainedPool = device.createCommandPool({
            .queueType = rachis::QueueType::compute,
            .maxCommandBuffers = 1,
            .label = "Unretained indirect ray-dispatch pool",
            .flags = rachis::CommandPoolFlag::unretainedResources,
        });
        REQUIRE(unretainedPool.has_value());
        {
            auto unretainedCommand = device.acquireCommandBuffer(*unretainedPool);
            REQUIRE(unretainedCommand.has_value());
            REQUIRE(device
                        .bufferBarrier(*unretainedCommand,
                                       {
                                           .buffer = *indirectArguments,
                                           .stagesBefore = rachis::PipelineStage::all,
                                           .stagesAfter = rachis::PipelineStage::indirect,
                                           .accessBefore = rachis::Access::common,
                                           .accessAfter = rachis::Access::indirectArgument,
                                       })
                        .has_value());
            REQUIRE(device.bindRayTracingPipeline(*unretainedCommand, *pipeline).has_value());
            REQUIRE(device.bindRayTracingBindingSet(*unretainedCommand, 0, *bindingSet).has_value());
            const auto beforeUnretainedPublication = rachis::d3d12::test::commandDependencySnapshot(*unretainedCommand);
            REQUIRE(beforeUnretainedPublication.has_value());
            const auto beforeUnretainedDependencies =
                beforeUnretainedPublication.value_or(rachis::d3d12::test::CommandDependencySnapshot{});
            REQUIRE(device.dispatchRaysIndirect(*unretainedCommand, indirectDesc).has_value());
            const auto afterUnretainedPublication = rachis::d3d12::test::commandDependencySnapshot(*unretainedCommand);
            REQUIRE(afterUnretainedPublication.has_value());
            const auto afterUnretainedDependencies =
                afterUnretainedPublication.value_or(rachis::d3d12::test::CommandDependencySnapshot{});
            CHECK(afterUnretainedDependencies.resourceOwners == beforeUnretainedDependencies.resourceOwners + 2);
            CHECK(afterUnretainedDependencies.residencyLeases == beforeUnretainedDependencies.residencyLeases);
            CHECK(afterUnretainedDependencies.swapchainReferences == beforeUnretainedDependencies.swapchainReferences);
            CHECK(afterUnretainedDependencies.sparseMappingDependencies ==
                  beforeUnretainedDependencies.sparseMappingDependencies);
        }
        REQUIRE(device.destroyCommandPool(*unretainedPool).has_value());

        const auto beforeIndirectPublication = rachis::d3d12::test::commandDependencySnapshot(*command);
        REQUIRE(beforeIndirectPublication.has_value());
        const auto recordsBeforeIndirectPublication =
            rachis::d3d12::test::indirectExecutionRecordCount().load(std::memory_order_relaxed);
        for (const auto failureStage : std::array{1, 2, 3})
        {
            INFO("failure stage " << failureStage);
            const FailResourceRetention failPublication(failureStage);
            auto failedPublication = device.dispatchRaysIndirect(*command, indirectDesc);
            REQUIRE_FALSE(failedPublication.has_value());
            CHECK(failedPublication.error().code == rachis::ErrorCode::outOfMemory);
            const auto afterFailedPublication = rachis::d3d12::test::commandDependencySnapshot(*command);
            REQUIRE(afterFailedPublication.has_value());
            CHECK(*afterFailedPublication == *beforeIndirectPublication);
            CHECK(rachis::d3d12::test::indirectExecutionRecordCount().load(std::memory_order_relaxed) ==
                  recordsBeforeIndirectPublication);
        }
        REQUIRE(device.dispatchRaysIndirect(*command, indirectDesc).has_value());
        CHECK(rachis::d3d12::test::indirectExecutionRecordCount().load(std::memory_order_relaxed) ==
              recordsBeforeIndirectPublication + 1);
    }
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *output,
                                   .stagesBefore = rachis::PipelineStage::rayTracingShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device.copyBuffer(*command, {.source = *output, .destination = *readback, .size = 4}).has_value());

    REQUIRE(device.destroyBuffer(*shaderTable).has_value());
    REQUIRE(device.destroyBuffer(*output).has_value());
    REQUIRE(device.destroyBuffer(*indirectUpload).has_value());
    REQUIRE(device.destroyBuffer(*indirectArguments).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*missShader).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    std::uint32_t result = 0;
    REQUIRE(device.readBuffer(*readback, 0, std::as_writable_bytes(std::span{&result, 1})).has_value());
    CHECK(result == expectedResult);
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("indirect ray dispatch reads compute-written arguments in every indirect read scope",
          "[size:medium][rhi][ray-tracing-pipeline][hardware]")
{
    const bool legacyBarriers = GENERATE(false, true);
    const auto readScope =
        GENERATE(IndirectArgumentReadScope{rachis::PipelineStage::indirect, rachis::Access::indirectArgument},
                 IndirectArgumentReadScope{rachis::PipelineStage::indirect | rachis::PipelineStage::computeShader,
                                           rachis::Access::indirectArgument | rachis::Access::shaderResource},
                 IndirectArgumentReadScope{rachis::PipelineStage::all, rachis::Access::indirectArgument});
    INFO("legacy barriers: " << legacyBarriers);
    INFO("argument read stages: " << static_cast<std::uint32_t>(readScope.stages)
                                  << ", access: " << static_cast<std::uint32_t>(readScope.access));
    const ForceLegacyBarriers forceLegacyBarriers(legacyBarriers);

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
    if (!capabilities.rayTracing || !capabilities.indirectRayDispatch)
    {
        SKIP("The selected hardware adapter does not expose indirect DXR pipeline dispatch.");
    }
    if (!legacyBarriers && !capabilities.enhancedBarriers)
    {
        SKIP("The selected hardware adapter does not expose enhanced barriers.");
    }
    REQUIRE(capabilities.enhancedBarriers != legacyBarriers);

    const auto rayTracingBytecode = loadShader();
    const auto computeBytecode = loadShader(RACHIS_TEST_COMPUTE_SHADER_PATH);
    REQUIRE_FALSE(rayTracingBytecode.empty());
    REQUIRE_FALSE(computeBytecode.empty());
    const std::array shaderBindings{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1},
    };
    auto counterShader = device.createShaderModule({
        .stage = rachis::ShaderStage::rayGeneration,
        .bytecode = rayTracingBytecode,
        .entryPoint = "rayGenerationCounter",
        .label = "Ray-generation invocation counter",
        .shaderInterface = rachis::ShaderInterfaceDesc{.bindings = shaderBindings},
    });
    REQUIRE(counterShader.has_value());
    auto argumentShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = computeBytecode,
        .entryPoint = "main",
        .label = "Indirect ray argument producer",
        .shaderInterface = rachis::test::computeWriteShaderInterface(),
    });
    REQUIRE(argumentShader.has_value());

    const std::array rayTracingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::rayGeneration, 1},
    };
    auto rayTracingBindingLayout =
        device.createBindingLayout({.entries = rayTracingEntries, .label = "Ray counter resources"});
    REQUIRE(rayTracingBindingLayout.has_value());
    const std::array rayTracingBindingLayouts{*rayTracingBindingLayout};
    auto rayTracingPipelineLayout =
        device.createPipelineLayout({.bindingLayouts = rayTracingBindingLayouts, .label = "Ray counter layout"});
    REQUIRE(rayTracingPipelineLayout.has_value());
    const std::array groups{
        rachis::RayTracingShaderGroupDesc{
            .type = rachis::RayTracingShaderGroupType::general,
            .generalShader = *counterShader,
        },
    };
    auto rayTracingPipeline = device.createRayTracingPipeline({
        .layout = *rayTracingPipelineLayout,
        .groups = groups,
        .maxPayloadSize = sizeof(std::uint32_t),
        .maxAttributeSize = 8,
        .maxRecursionDepth = 1,
        .label = "Ray counter pipeline",
    });
    REQUIRE(rayTracingPipeline.has_value());

    const std::array computeEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto computeBindingLayout =
        device.createBindingLayout({.entries = computeEntries, .label = "Indirect ray argument resources"});
    REQUIRE(computeBindingLayout.has_value());
    const std::array computeBindingLayouts{*computeBindingLayout};
    const std::array pushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)},
    };
    auto computePipelineLayout = device.createPipelineLayout({
        .bindingLayouts = computeBindingLayouts,
        .pushConstantRanges = pushConstantRanges,
        .label = "Indirect ray argument producer layout",
    });
    REQUIRE(computePipelineLayout.has_value());
    auto computePipeline = device.createComputePipeline({
        .layout = *computePipelineLayout,
        .computeShader = *argumentShader,
        .label = "Indirect ray argument producer",
    });
    REQUIRE(computePipeline.has_value());

    auto recordSize = device.rayTracingShaderGroupRecordSize(*rayTracingPipeline, 0);
    REQUIRE(recordSize.has_value());
    std::vector<std::byte> shaderRecord(*recordSize);
    REQUIRE(device.writeRayTracingShaderGroupRecord(*rayTracingPipeline, 0, {}, shaderRecord).has_value());

    constexpr std::uint32_t argumentWordCount = 64;
    constexpr std::uint64_t argumentBufferSize = argumentWordCount * sizeof(std::uint32_t);
    constexpr std::uint32_t recordCount = 8;
    constexpr rachis::DispatchIndirectArguments uploadDimensions{2, 2, 2};
    auto shaderTable = device.createBuffer({
        .size = std::max<std::uint64_t>(capabilities.rayTracingLimits.shaderGroupBaseAlignment, *recordSize),
        .usage = rachis::BufferUsage::shaderBindingTable,
        .memoryType = rachis::MemoryType::upload,
        .label = "Ray counter shader table",
    });
    auto counter = device.createBuffer({
        .size = sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Ray invocation counter",
    });
    auto readback = device.createBuffer({
        .size = sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Ray invocation counter readback",
    });
    auto zeros = device.createBuffer({
        .size = argumentBufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Indirect ray argument clear",
    });
    auto arguments = device.createBuffer({
        .size = argumentBufferSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::indirect | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Compute-written indirect ray arguments",
    });
    auto uploadArguments = device.createBuffer({
        .size = sizeof(uploadDimensions),
        .usage = rachis::BufferUsage::indirect,
        .memoryType = rachis::MemoryType::upload,
        .label = "Upload-heap indirect ray arguments",
    });
    REQUIRE(shaderTable.has_value());
    REQUIRE(counter.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(zeros.has_value());
    REQUIRE(arguments.has_value());
    REQUIRE(uploadArguments.has_value());
    const std::vector<std::byte> zeroBytes(argumentBufferSize);
    REQUIRE(device.writeBuffer(*shaderTable, 0, shaderRecord).has_value());
    REQUIRE(device.writeBuffer(*zeros, 0, zeroBytes).has_value());
    REQUIRE(device.writeBuffer(*uploadArguments, 0, std::as_bytes(std::span{&uploadDimensions, 1})).has_value());

    const std::array counterWrites{
        rachis::BindingWrite{
            .binding = 0,
            .resource =
                rachis::BufferBinding{
                    .buffer = *counter,
                    .size = sizeof(std::uint32_t),
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
    };
    auto counterSet = rachis::test::createBindingSet(device,
                                                     {
                                                         .layout = *rayTracingBindingLayout,
                                                         .writes = counterWrites,
                                                         .label = "Ray counter binding set",
                                                     });
    REQUIRE(counterSet.has_value());
    const std::array argumentWrites{
        rachis::BindingWrite{
            .binding = 0,
            .resource =
                rachis::BufferBinding{
                    .buffer = *arguments,
                    .size = argumentBufferSize,
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
    };
    auto argumentSet = rachis::test::createBindingSet(device,
                                                      {
                                                          .layout = *computeBindingLayout,
                                                          .writes = argumentWrites,
                                                          .label = "Indirect ray argument binding set",
                                                      });
    REQUIRE(argumentSet.has_value());

    const rachis::RayTracingIndirectDispatchDesc dispatchDesc{
        .rayGeneration =
            {
                .buffer = *shaderTable,
                .size = *recordSize,
                .stride = *recordSize,
            },
        .argumentBuffer = *arguments,
    };

    auto producer = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    auto consumer = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(producer.has_value());
    REQUIRE(consumer.has_value());
    const std::array clearTransitions{
        rachis::BufferBarrier{
            .buffer = *counter,
            .stagesBefore = rachis::PipelineStage::all,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::common,
            .accessAfter = rachis::Access::copyDestination,
        },
        rachis::BufferBarrier{
            .buffer = *arguments,
            .stagesBefore = rachis::PipelineStage::all,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::common,
            .accessAfter = rachis::Access::copyDestination,
        },
    };
    REQUIRE(device.resourceBarriers(*producer, {.bufferBarriers = clearTransitions}).has_value());
    REQUIRE(device
                .copyBuffer(*producer,
                            {
                                .source = *zeros,
                                .destination = *counter,
                                .size = sizeof(std::uint32_t),
                            })
                .has_value());
    REQUIRE(device
                .copyBuffer(*producer,
                            {
                                .source = *zeros,
                                .destination = *arguments,
                                .size = argumentBufferSize,
                            })
                .has_value());
    const std::array writeTransitions{
        rachis::BufferBarrier{
            .buffer = *counter,
            .stagesBefore = rachis::PipelineStage::copy,
            .stagesAfter = rachis::PipelineStage::rayTracingShader,
            .accessBefore = rachis::Access::copyDestination,
            .accessAfter = rachis::Access::unorderedAccess,
        },
        rachis::BufferBarrier{
            .buffer = *arguments,
            .stagesBefore = rachis::PipelineStage::copy,
            .stagesAfter = rachis::PipelineStage::computeShader,
            .accessBefore = rachis::Access::copyDestination,
            .accessAfter = rachis::Access::unorderedAccess,
        },
    };
    REQUIRE(device.resourceBarriers(*producer, {.bufferBarriers = writeTransitions}).has_value());
    REQUIRE(device.bindComputePipeline(*producer, *computePipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*producer, 0, *argumentSet).has_value());
    const std::array producerConstants{1U, 1U};
    REQUIRE(device.setComputePushConstants(*producer, 0, std::as_bytes(std::span(producerConstants))).has_value());
    REQUIRE(device.dispatch(*producer, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*producer,
                               {
                                   .buffer = *arguments,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = readScope.stages,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = readScope.access,
                               })
                .has_value());
    REQUIRE(device.bindRayTracingPipeline(*producer, *rayTracingPipeline).has_value());
    REQUIRE(device.bindRayTracingBindingSet(*producer, 0, *counterSet).has_value());
    REQUIRE(device.dispatchRaysIndirect(*producer, dispatchDesc).has_value());
    const auto allocationsBefore = deviceLocalAllocationCount(device);
    for (std::uint32_t record = 1; record < recordCount; ++record)
    {
        auto recordDesc = dispatchDesc;
        recordDesc.argumentOffset = record * sizeof(rachis::DispatchIndirectArguments);
        REQUIRE(device.dispatchRaysIndirect(*producer, recordDesc).has_value());
    }
    const auto allocationsAfter = deviceLocalAllocationCount(device);
    REQUIRE(allocationsBefore.has_value());
    REQUIRE(allocationsAfter.has_value());
    CHECK(*allocationsAfter == *allocationsBefore);

    REQUIRE(device.bindRayTracingPipeline(*consumer, *rayTracingPipeline).has_value());
    REQUIRE(device.bindRayTracingBindingSet(*consumer, 0, *counterSet).has_value());
    REQUIRE(device.dispatchRaysIndirect(*consumer, dispatchDesc).has_value());
    auto uploadDesc = dispatchDesc;
    uploadDesc.argumentBuffer = *uploadArguments;
    REQUIRE(device.dispatchRaysIndirect(*consumer, uploadDesc).has_value());
    REQUIRE(device
                .bufferBarrier(*consumer,
                               {
                                   .buffer = *counter,
                                   .stagesBefore = rachis::PipelineStage::rayTracingShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device
                .copyBuffer(*consumer,
                            {
                                .source = *counter,
                                .destination = *readback,
                                .size = sizeof(std::uint32_t),
                            })
                .has_value());

    std::array commands{std::move(*producer), std::move(*consumer)};
    auto submitted = device.submit(commands);
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    std::uint32_t expectedInvocations =
        uploadDimensions.groupCountX * uploadDimensions.groupCountY * uploadDimensions.groupCountZ;
    for (std::uint32_t record = 0; record < recordCount; ++record)
    {
        const std::uint32_t first = record * 3U + 1U;
        const std::uint32_t recordInvocations = first * (first + 1U) * (first + 2U);
        expectedInvocations += record == 0 ? 2U * recordInvocations : recordInvocations;
    }
    std::uint32_t invocations = 0;
    REQUIRE(device.readBuffer(*readback, 0, std::as_writable_bytes(std::span{&invocations, 1})).has_value());
    CHECK(invocations == expectedInvocations);

    REQUIRE(device.destroyBuffer(*uploadArguments).has_value());
    REQUIRE(device.destroyBuffer(*arguments).has_value());
    REQUIRE(device.destroyBuffer(*zeros).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*counter).has_value());
    REQUIRE(device.destroyBuffer(*shaderTable).has_value());
    REQUIRE(device.destroyBindingSet(*argumentSet).has_value());
    REQUIRE(device.destroyBindingSet(*counterSet).has_value());
    REQUIRE(device.destroyPipeline(*computePipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*computePipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*computeBindingLayout).has_value());
    REQUIRE(device.destroyPipeline(*rayTracingPipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*rayTracingPipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*rayTracingBindingLayout).has_value());
    REQUIRE(device.destroyShaderModule(*argumentShader).has_value());
    REQUIRE(device.destroyShaderModule(*counterShader).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("shader-record constants reject shared global constant registers of every stage visibility",
          "[size:medium][rhi][ray-tracing-pipeline][hardware]")
{
    const auto globalConstantVisibility = GENERATE(rachis::ShaderStage::rayGeneration, rachis::ShaderStage::vertex);
    INFO("global constant-buffer visibility " << static_cast<std::uint32_t>(globalConstantVisibility));
    auto created = rachis::createDevice({
        .adapterPreference = rachis::AdapterPreference::highPerformance,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        skipWithoutHardwareAdapter(created.error());
    }
    auto device = std::move(*created);
    if (!rachis::test::requireCapabilities(device).rayTracing)
    {
        SKIP("The selected hardware adapter does not expose DXR pipeline ray tracing.");
    }

    const auto bytecode = loadShader();
    REQUIRE_FALSE(bytecode.empty());
    const std::array rayGenerationBindings{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1},
    };
    const std::array rayGenerationPushConstants{
        rachis::ShaderPushConstantRequirement{0, sizeof(std::uint32_t)},
    };
    auto rayGeneration = device.createShaderModule({
        .stage = rachis::ShaderStage::rayGeneration,
        .bytecode = bytecode,
        .entryPoint = "rayGenerationWithoutRecord",
        .label = "Ray-generation module with global constants",
        .shaderInterface =
            rachis::ShaderInterfaceDesc{
                .bindings = rayGenerationBindings,
                .pushConstants = rayGenerationPushConstants,
            },
    });
    REQUIRE(rayGeneration.has_value());
    auto miss = device.createShaderModule({
        .stage = rachis::ShaderStage::miss,
        .bytecode = bytecode,
        .entryPoint = "missShader",
        .label = "Miss module with record constants",
    });
    REQUIRE(miss.has_value());
    auto closestHit = device.createShaderModule({
        .stage = rachis::ShaderStage::closestHit,
        .bytecode = bytecode,
        .entryPoint = "closestHitShader",
        .label = "Closest-hit module with record constants",
    });
    REQUIRE(closestHit.has_value());

    constexpr std::uint32_t globalConstantBinding = 1;
    const std::array layoutEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::rayGeneration, 1},
        rachis::BindingLayoutEntry{
            globalConstantBinding, rachis::BindingType::uniformBuffer, globalConstantVisibility, 1},
    };
    auto bindingLayout = device.createBindingLayout({.entries = layoutEntries, .label = "Ray-generation constants"});
    REQUIRE(bindingLayout.has_value());
    const std::array bindingLayouts{*bindingLayout};
    const std::array pushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::rayGeneration, 0, sizeof(std::uint32_t)},
    };
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = pushConstantRanges,
        .label = "Ray-generation-only global constants",
    });
    REQUIRE(pipelineLayout.has_value());

    const rachis::RayTracingShaderGroupDesc rayGenerationGroup{
        .type = rachis::RayTracingShaderGroupType::general,
        .generalShader = *rayGeneration,
    };
    const auto missGroup = [&](const rachis::ShaderRecordConstantRange& constants) {
        return rachis::RayTracingShaderGroupDesc{
            .type = rachis::RayTracingShaderGroupType::general,
            .generalShader = *miss,
            .shaderRecordConstants = constants,
        };
    };
    const auto hitGroup = [&](const rachis::ShaderRecordConstantRange& constants) {
        return rachis::RayTracingShaderGroupDesc{
            .type = rachis::RayTracingShaderGroupType::trianglesHitGroup,
            .closestHitShader = *closestHit,
            .shaderRecordConstants = constants,
        };
    };
    const auto requireRejectedBeforeNativeCreation = [&](const rachis::RayTracingShaderGroupDesc& recordGroup) {
        INFO((recordGroup.type == rachis::RayTracingShaderGroupType::general ? "miss group" : "closest-hit group"));
        const std::array groups{rayGenerationGroup, recordGroup};
        auto library = device.createRayTracingPipelineLibrary({
            .layout = *pipelineLayout,
            .groups = groups,
            .maxPayloadSize = sizeof(std::uint32_t),
            .maxAttributeSize = 8,
        });
        REQUIRE_FALSE(library.has_value());
        CHECK(library.error().code == rachis::ErrorCode::validationError);
        CHECK(library.error().nativeCode == 0);
        auto pipeline = device.createRayTracingPipeline({
            .layout = *pipelineLayout,
            .groups = groups,
            .maxPayloadSize = sizeof(std::uint32_t),
            .maxAttributeSize = 8,
            .maxRecursionDepth = 1,
        });
        REQUIRE_FALSE(pipeline.has_value());
        CHECK(pipeline.error().code == rachis::ErrorCode::validationError);
        CHECK(pipeline.error().nativeCode == 0);
    };

    SECTION("a record range on a global constant-buffer register is rejected")
    {
        constexpr rachis::ShaderRecordConstantRange constants{
            .shaderRegister = globalConstantBinding,
            .registerSpace = 0,
            .size = sizeof(std::uint32_t),
        };
        requireRejectedBeforeNativeCreation(hitGroup(constants));
        requireRejectedBeforeNativeCreation(missGroup(constants));
    }
    SECTION("a record range on the push-constant register is rejected")
    {
        constexpr rachis::ShaderRecordConstantRange constants{
            .shaderRegister = rachis::pushConstantRegister,
            .registerSpace = rachis::pushConstantRegisterSpace,
            .size = sizeof(std::uint32_t),
        };
        requireRejectedBeforeNativeCreation(hitGroup(constants));
        requireRejectedBeforeNativeCreation(missGroup(constants));
    }
    SECTION("record ranges beside the global constant registers link")
    {
        const std::array groups{
            rayGenerationGroup,
            missGroup({
                .shaderRegister = rachis::pushConstantRegister + 1,
                .registerSpace = rachis::pushConstantRegisterSpace,
                .size = sizeof(std::uint32_t),
            }),
            hitGroup({.shaderRegister = 0, .registerSpace = 0, .size = sizeof(std::uint32_t)}),
        };
        auto library = device.createRayTracingPipelineLibrary({
            .layout = *pipelineLayout,
            .groups = groups,
            .maxPayloadSize = sizeof(std::uint32_t),
            .maxAttributeSize = 8,
        });
        REQUIRE(library.has_value());
        const std::array libraries{*library};
        auto linked = device.createRayTracingPipeline({
            .layout = *pipelineLayout,
            .libraries = libraries,
            .maxPayloadSize = sizeof(std::uint32_t),
            .maxAttributeSize = 8,
            .maxRecursionDepth = 1,
        });
        REQUIRE(linked.has_value());
        auto direct = device.createRayTracingPipeline({
            .layout = *pipelineLayout,
            .groups = groups,
            .maxPayloadSize = sizeof(std::uint32_t),
            .maxAttributeSize = 8,
            .maxRecursionDepth = 1,
        });
        REQUIRE(direct.has_value());
        REQUIRE(device.destroyPipeline(*direct).has_value());
        REQUIRE(device.destroyPipeline(*linked).has_value());
        REQUIRE(device.destroyRayTracingPipelineLibrary(*library).has_value());
    }

    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyShaderModule(*closestHit).has_value());
    REQUIRE(device.destroyShaderModule(*miss).has_value());
    REQUIRE(device.destroyShaderModule(*rayGeneration).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("hit groups sharing a shader across shader-record constant layouts read their own records on WARP",
          "[size:medium][rhi][ray-tracing-pipeline][warp]")
{
    auto created = rachis::createDevice({
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    requireSharedHitShadersUseTheirGroupLayouts(device);
}

TEST_CASE("hit groups sharing a shader across shader-record constant layouts read their own records",
          "[size:medium][rhi][ray-tracing-pipeline][hardware]")
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
    requireSharedHitShadersUseTheirGroupLayouts(device);
}

TEST_CASE("a record-free hit group does not inherit the record layout of a group sharing its shader on WARP",
          "[size:medium][rhi][ray-tracing-pipeline][warp]")
{
    auto created = rachis::createDevice({
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .enableDred = true,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    requireRecordFreeGroupHidesSharedShaderRecord(device);
}

TEST_CASE("a record-free hit group does not inherit the record layout of a group sharing its shader",
          "[size:medium][rhi][ray-tracing-pipeline][hardware]")
{
    auto created = rachis::createDevice({
        .adapterPreference = rachis::AdapterPreference::highPerformance,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
    });
    if (!created)
    {
        skipWithoutHardwareAdapter(created.error());
    }
    auto device = std::move(*created);
    requireRecordFreeGroupHidesSharedShaderRecord(device);
}
