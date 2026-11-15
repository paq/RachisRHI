#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include "Vulkan/VulkanTestHooks.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <volk.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
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

TEST_CASE("Vulkan ray-tracing pipeline limits stay coherent",
          "[size:medium][rhi][vulkan][ray-tracing-pipeline][capabilities]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    const auto& limits = capabilities.rayTracingLimits;
    CHECK_FALSE((capabilities.indirectRayDispatch && !capabilities.rayTracing));
    if (capabilities.rayTracing)
    {
        CHECK(limits.shaderGroupHandleSize > 0);
        CHECK(limits.shaderGroupHandleAlignment >= limits.shaderGroupHandleSize);
        CHECK(limits.shaderGroupBaseAlignment >= limits.shaderGroupHandleAlignment);
        CHECK(limits.maxShaderGroupStride >= limits.shaderGroupHandleSize);
        CHECK(limits.maxRecursionDepth > 0);
        CHECK(limits.maxAttributeSize > 0);
        CHECK(limits.maxDispatchInvocations > 0);
    }
    else
    {
        auto table = device.createBuffer({
            .size = 64,
            .usage = rachis::BufferUsage::shaderBindingTable,
            .memoryType = rachis::MemoryType::upload,
        });
        REQUIRE_FALSE(table.has_value());
        CHECK(table.error().code == rachis::ErrorCode::unsupported);
    }
}

#ifdef RACHIS_SPIRV_DIR
TEST_CASE("Vulkan shader libraries reflect a miss entry point as the miss stage",
          "[size:medium][rhi][vulkan][ray-tracing-pipeline]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (!capabilities.rayTracing)
    {
        SKIP("The Vulkan adapter does not advertise ray-tracing pipelines.");
    }

    const auto bytecode = loadSpirv("RayTracingPipeline.miss.spv");
    REQUIRE_FALSE(bytecode.empty());
    const std::array missEntries{rachis::ShaderLibraryEntryDesc{
        .stage = rachis::ShaderStage::miss,
        .bytecode = bytecode,
        .entryPoint = "missShader",
    }};
    auto shaderLibrary = device.createShaderLibrary({.entries = missEntries});
    if (!shaderLibrary)
    {
        UNSCOPED_INFO(shaderLibrary.error().message);
    }
    REQUIRE(shaderLibrary);
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::miss,
        .entryPoint = "missShader",
        .label = "Vulkan miss module",
        .library = *shaderLibrary,
        .shaderInterface = rachis::ShaderInterfaceDesc{},
    });
    if (!shader)
    {
        INFO(shader.error().message);
    }
    REQUIRE(shader.has_value());
    REQUIRE(device.destroyShaderLibrary(*shaderLibrary));

    for (const auto wrongStage : {rachis::ShaderStage::intersection, rachis::ShaderStage::closestHit})
    {
        const std::array wrongEntries{rachis::ShaderLibraryEntryDesc{
            .stage = wrongStage,
            .bytecode = bytecode,
            .entryPoint = "missShader",
        }};
        auto wrongLibrary = device.createShaderLibrary({.entries = wrongEntries});
        CHECK_FALSE(wrongLibrary.has_value());
    }
}

TEST_CASE("Vulkan ray-generation dispatch applies shader-library specialization",
          "[size:medium][rhi][vulkan][ray-tracing-pipeline]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (!capabilities.rayTracing)
    {
        SKIP("The Vulkan adapter does not advertise ray-tracing pipelines.");
    }
    const auto& limits = capabilities.rayTracingLimits;

    const auto bytecode = loadSpirv("RayTracingPipeline.raygen.spv");
    REQUIRE_FALSE(bytecode.empty());
    const std::array shaderBindings{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1},
    };
    const auto expected = GENERATE(0x52545852U, 0x12345678U);
    const std::array constants{rachis::SpecializationConstant::makeUnsignedInt32(7, expected)};
    const std::array shaderEntries{rachis::ShaderLibraryEntryDesc{
        .stage = rachis::ShaderStage::rayGeneration,
        .bytecode = bytecode,
        .entryPoint = "rayGenerationWithoutRecord",
        .specializationConstants = constants,
    }};
    auto shaderLibrary = device.createShaderLibrary({.entries = shaderEntries});
    if (!shaderLibrary)
    {
        UNSCOPED_INFO(shaderLibrary.error().message);
    }
    REQUIRE(shaderLibrary);
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::rayGeneration,
        .entryPoint = "rayGenerationWithoutRecord",
        .label = "Vulkan ray-generation module",
        .library = *shaderLibrary,
        .shaderInterface = rachis::ShaderInterfaceDesc{.bindings = shaderBindings},
        .specializationConstants = constants,
    });
    if (!shader)
    {
        INFO(shader.error().message);
    }
    REQUIRE(shader.has_value());
    REQUIRE(device.destroyShaderLibrary(*shaderLibrary));

    const std::array layoutEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::rayGeneration, 1},
    };
    auto bindingLayout =
        device.createBindingLayout({.entries = layoutEntries, .label = "Vulkan ray-generation resources"});
    REQUIRE(bindingLayout.has_value());
    const std::array bindingLayouts{*bindingLayout};
    auto pipelineLayout =
        device.createPipelineLayout({.bindingLayouts = bindingLayouts, .label = "Vulkan ray-tracing layout"});
    REQUIRE(pipelineLayout.has_value());
    const std::array groups{
        rachis::RayTracingShaderGroupDesc{
            .type = rachis::RayTracingShaderGroupType::general,
            .generalShader = *shader,
        },
    };
    auto library = device.createRayTracingPipelineLibrary({
        .layout = *pipelineLayout,
        .groups = groups,
        .maxAttributeSize = 8,
        .label = "Vulkan ray-generation collection",
    });
    REQUIRE(library.has_value());
    const std::array libraries{*library};
    auto pipeline = device.createRayTracingPipeline({
        .layout = *pipelineLayout,
        .libraries = libraries,
        .maxAttributeSize = 8,
        .maxRecursionDepth = 1,
        .label = "Vulkan ray-generation pipeline",
    });
    if (!pipeline)
    {
        INFO(pipeline.error().message);
    }
    REQUIRE(pipeline.has_value());
    REQUIRE(device.destroyRayTracingPipelineLibrary(*library).has_value());
    auto stale = device.createRayTracingPipeline({
        .layout = *pipelineLayout,
        .libraries = libraries,
        .maxAttributeSize = 8,
        .maxRecursionDepth = 1,
    });
    REQUIRE_FALSE(stale.has_value());
    CHECK(stale.error().code == rachis::ErrorCode::invalidArgument);

    std::vector<std::byte> shaderIdentifier(limits.shaderGroupHandleSize);
    REQUIRE(device.getRayTracingShaderGroupHandles(*pipeline, 0, 1, shaderIdentifier).has_value());
    auto recordSize = device.rayTracingShaderGroupRecordSize(*pipeline, 0);
    REQUIRE(recordSize.has_value());
    CHECK(*recordSize == limits.shaderGroupHandleAlignment);
    std::vector<std::byte> shaderRecord(*recordSize, std::byte{0x5a});
    REQUIRE(device.writeRayTracingShaderGroupRecord(*pipeline, 0, {}, shaderRecord).has_value());
    CHECK(std::equal(shaderIdentifier.begin(), shaderIdentifier.end(), shaderRecord.begin()));

    auto tableLayout = rachis::layoutRayTracingShaderTable({.groups = groups, .rayGenerationGroup = 0}, limits);
    REQUIRE(tableLayout.has_value());
    REQUIRE(tableLayout->records.size() == 1);
    CHECK(tableLayout->records[0].recordSize == *recordSize);
    CHECK(tableLayout->rayGeneration.stride == *recordSize);
    const auto tableSize = std::max<std::uint64_t>(limits.shaderGroupBaseAlignment, tableLayout->size);
    auto shaderTable = device.createBuffer({
        .size = tableSize,
        .usage = rachis::BufferUsage::shaderBindingTable,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan ray-generation shader table",
    });
    REQUIRE(shaderTable.has_value());
    REQUIRE(device.writeBuffer(*shaderTable, tableLayout->records[0].handleOffset, shaderRecord).has_value());
    tableLayout->setBuffer(*shaderTable);

    auto output = device.createBuffer({
        .size = sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan ray-generation output",
    });
    auto readback = device.createBuffer({
        .size = sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan ray-generation readback",
    });
    REQUIRE(output.has_value());
    REQUIRE(readback.has_value());
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
                                                         .label = "Vulkan ray-generation binding set",
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
    const rachis::RayTracingDispatchDesc dispatch{
        .rayGeneration = tableLayout->rayGeneration,
        .miss = tableLayout->miss,
        .hitGroup = tableLayout->hitGroup,
        .callable = tableLayout->callable,
    };
    if (capabilities.indirectRayDispatch)
    {
        const rachis::DispatchIndirectArguments dimensions{1, 1, 1};
        auto arguments = device.createBuffer({
            .size = sizeof(dimensions),
            .usage = rachis::BufferUsage::indirect,
            .memoryType = rachis::MemoryType::upload,
            .label = "Vulkan indirect ray arguments",
        });
        REQUIRE(arguments.has_value());
        REQUIRE(device.writeBuffer(*arguments, 0, std::as_bytes(std::span{&dimensions, 1})).has_value());
        REQUIRE(device
                    .dispatchRaysIndirect(*command,
                                          {
                                              .rayGeneration = dispatch.rayGeneration,
                                              .argumentBuffer = *arguments,
                                          })
                    .has_value());
        REQUIRE(device.destroyBuffer(*arguments).has_value());
    }
    else
    {
        REQUIRE(device.dispatchRays(*command, dispatch).has_value());
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
    REQUIRE(device.copyBuffer(*command, {.source = *output, .destination = *readback, .size = sizeof(std::uint32_t)})
                .has_value());

    REQUIRE(device.destroyBuffer(*shaderTable).has_value());
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

    std::uint32_t result = 0;
    REQUIRE(device.readBuffer(*readback, 0, std::as_writable_bytes(std::span{&result, 1})).has_value());
    CHECK(result == expected);
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan graphics pipeline binding rejects ray-tracing pipelines",
          "[size:medium][rhi][vulkan][ray-tracing-pipeline]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (!capabilities.rayTracing)
    {
        SKIP("The Vulkan adapter does not advertise ray-tracing pipelines.");
    }

    const auto bytecode = loadSpirv("RayTracingPipeline.raygen.spv");
    REQUIRE_FALSE(bytecode.empty());
    const std::array shaderBindings{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1},
    };
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::rayGeneration,
        .bytecode = bytecode,
        .entryPoint = "rayGenerationWithoutRecord",
        .label = "Vulkan misbound ray-generation module",
        .shaderInterface = rachis::ShaderInterfaceDesc{.bindings = shaderBindings},
    });
    if (!shader)
    {
        INFO(shader.error().message);
    }
    REQUIRE(shader.has_value());

    const std::array layoutEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::rayGeneration, 1},
    };
    auto bindingLayout =
        device.createBindingLayout({.entries = layoutEntries, .label = "Vulkan misbound ray-tracing resources"});
    REQUIRE(bindingLayout.has_value());
    const std::array bindingLayouts{*bindingLayout};
    auto pipelineLayout =
        device.createPipelineLayout({.bindingLayouts = bindingLayouts, .label = "Vulkan misbound ray-tracing layout"});
    REQUIRE(pipelineLayout.has_value());
    const std::array groups{
        rachis::RayTracingShaderGroupDesc{
            .type = rachis::RayTracingShaderGroupType::general,
            .generalShader = *shader,
        },
    };
    auto pipeline = device.createRayTracingPipeline({
        .layout = *pipelineLayout,
        .groups = groups,
        .maxAttributeSize = 8,
        .maxRecursionDepth = 1,
        .label = "Vulkan misbound ray-tracing pipeline",
    });
    if (!pipeline)
    {
        INFO(pipeline.error().message);
    }
    REQUIRE(pipeline.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device.beginRendering(*command, rachis::RenderingDesc{}).has_value());
    const auto graphicsBound = device.bindGraphicsPipeline(*command, *pipeline);
    REQUIRE_FALSE(graphicsBound.has_value());
    CHECK(graphicsBound.error().code == rachis::ErrorCode::invalidArgument);
    CHECK_FALSE(device.draw(*command, 3).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    const auto computeBound = device.bindComputePipeline(*command, *pipeline);
    REQUIRE_FALSE(computeBound.has_value());
    CHECK(computeBound.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(device.bindRayTracingPipeline(*command, *pipeline).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    device.collectGarbage();

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan ray-tracing pipelines require layout push constants that exactly match each shader",
          "[size:medium][rhi][vulkan][ray-tracing-pipeline]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (!capabilities.rayTracing)
    {
        SKIP("The Vulkan adapter does not advertise ray-tracing pipelines.");
    }

    const auto bytecode = loadSpirv("RayTracingPushConstants.raygen.spv");
    REQUIRE_FALSE(bytecode.empty());
    const std::array shaderBindings{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1},
    };
    const std::array shaderPushConstants{rachis::ShaderPushConstantRequirement{0, 16}};
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::rayGeneration,
        .bytecode = bytecode,
        .entryPoint = "rayGenerationWithPushConstants",
        .label = "Vulkan push-constant ray-generation module",
        .shaderInterface =
            rachis::ShaderInterfaceDesc{
                .bindings = shaderBindings,
                .pushConstants = shaderPushConstants,
            },
    });
    if (!shader)
    {
        INFO(shader.error().message);
    }
    REQUIRE(shader.has_value());

    const std::array layoutEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::rayGeneration, 1},
    };
    auto bindingLayout = device.createBindingLayout({.entries = layoutEntries});
    REQUIRE(bindingLayout.has_value());
    const std::array bindingLayouts{*bindingLayout};
    const std::array groups{
        rachis::RayTracingShaderGroupDesc{
            .type = rachis::RayTracingShaderGroupType::general,
            .generalShader = *shader,
        },
    };

    const std::array<rachis::PushConstantRange, 0> missing{};
    const std::array computeOnly{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16}};
    const std::array wider{rachis::PushConstantRange{rachis::ShaderStage::rayGeneration, 0, 32}};
    const std::array narrower{rachis::PushConstantRange{rachis::ShaderStage::rayGeneration, 0, 8}};
    for (const auto ranges : {std::span<const rachis::PushConstantRange>(missing),
                              std::span<const rachis::PushConstantRange>(computeOnly),
                              std::span<const rachis::PushConstantRange>(wider),
                              std::span<const rachis::PushConstantRange>(narrower)})
    {
        CAPTURE(ranges.size(), ranges.empty() ? 0U : ranges[0].size);
        auto layout = device.createPipelineLayout({.bindingLayouts = bindingLayouts, .pushConstantRanges = ranges});
        REQUIRE(layout.has_value());
        auto pipeline = device.createRayTracingPipeline({
            .layout = *layout,
            .groups = groups,
            .maxAttributeSize = 8,
            .maxRecursionDepth = 1,
        });
        REQUIRE_FALSE(pipeline.has_value());
        CHECK(pipeline.error().code == rachis::ErrorCode::validationError);
        auto library = device.createRayTracingPipelineLibrary({
            .layout = *layout,
            .groups = groups,
            .maxAttributeSize = 8,
        });
        REQUIRE_FALSE(library.has_value());
        CHECK(library.error().code == rachis::ErrorCode::validationError);
        REQUIRE(device.destroyPipelineLayout(*layout).has_value());
    }

    const std::array exact{rachis::PushConstantRange{rachis::ShaderStage::rayGeneration, 0, 16}};
    auto layout = device.createPipelineLayout({.bindingLayouts = bindingLayouts, .pushConstantRanges = exact});
    REQUIRE(layout.has_value());
    auto pipeline = device.createRayTracingPipeline({
        .layout = *layout,
        .groups = groups,
        .maxAttributeSize = 8,
        .maxRecursionDepth = 1,
    });
    if (!pipeline)
    {
        INFO(pipeline.error().message);
    }
    REQUIRE(pipeline.has_value());

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*layout).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan ray-tracing pipeline creation classifies native create failures by their VkResult",
          "[size:medium][rhi][vulkan][ray-tracing-pipeline]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (!capabilities.rayTracing)
    {
        SKIP("The Vulkan adapter does not advertise ray-tracing pipelines.");
    }

    const auto bytecode = loadSpirv("RayTracingPushConstants.raygen.spv");
    REQUIRE_FALSE(bytecode.empty());
    const std::array shaderBindings{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1},
    };
    const std::array shaderPushConstants{rachis::ShaderPushConstantRequirement{0, 16}};
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::rayGeneration,
        .bytecode = bytecode,
        .entryPoint = "rayGenerationWithPushConstants",
        .label = "Vulkan create-failure ray-generation module",
        .shaderInterface =
            rachis::ShaderInterfaceDesc{
                .bindings = shaderBindings,
                .pushConstants = shaderPushConstants,
            },
    });
    if (!shader)
    {
        INFO(shader.error().message);
    }
    REQUIRE(shader.has_value());
    const std::array layoutEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::rayGeneration, 1},
    };
    auto bindingLayout = device.createBindingLayout({.entries = layoutEntries});
    REQUIRE(bindingLayout.has_value());
    const std::array bindingLayouts{*bindingLayout};
    const std::array pushConstants{rachis::PushConstantRange{rachis::ShaderStage::rayGeneration, 0, 16}};
    auto layout = device.createPipelineLayout({.bindingLayouts = bindingLayouts, .pushConstantRanges = pushConstants});
    REQUIRE(layout.has_value());
    const std::array groups{
        rachis::RayTracingShaderGroupDesc{
            .type = rachis::RayTracingShaderGroupType::general,
            .generalShader = *shader,
        },
    };
    const rachis::RayTracingPipelineDesc pipelineDesc{
        .layout = *layout,
        .groups = groups,
        .maxAttributeSize = 8,
        .maxRecursionDepth = 1,
        .label = "Vulkan create-failure ray-tracing pipeline",
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
        auto pipeline = device.createRayTracingPipeline(pipelineDesc);
        REQUIRE(rachis::vulkan::test::injectCreateResult(VK_SUCCESS));

        REQUIRE_FALSE(pipeline.has_value());
        CHECK(pipeline.error().code == code);
        CHECK(pipeline.error().nativeCode == result);
        CHECK(pipeline.error().message.contains("vkCreateRayTracingPipelinesKHR"));
        CHECK(device.operationalState() == rachis::DeviceOperationalState::active);
        CHECK_FALSE(device.deviceLossError().has_value());
    }

    auto pipeline = device.createRayTracingPipeline(pipelineDesc);
    if (!pipeline)
    {
        INFO(pipeline.error().message);
    }
    REQUIRE(pipeline.has_value());

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*layout).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
}
#endif
