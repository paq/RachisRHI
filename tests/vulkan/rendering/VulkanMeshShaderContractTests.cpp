#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <vector>

namespace
{

    constexpr std::uint32_t targetSize = 64;
    constexpr std::uint32_t rowPitch = 256;
    constexpr std::uint64_t readbackSize = static_cast<std::uint64_t>(rowPitch) * targetSize;

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
TEST_CASE("Vulkan mesh pipelines dispatch directly and indirectly and capture geometry",
          "[size:medium][rhi][vulkan][mesh]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (!capabilities.meshShaders)
    {
        SKIP("The Vulkan adapter does not advertise mesh shaders.");
    }

    const auto meshNoTaskBytecode = loadSpirv("TaskMeshWorkload.no-task.ms.spv");
    const auto pixelBytecode = loadSpirv("TaskMeshWorkload.ps.spv");
    REQUIRE_FALSE(meshNoTaskBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    auto meshNoTaskShader = device.createShaderModule({
        .stage = rachis::ShaderStage::mesh,
        .bytecode = meshNoTaskBytecode,
        .entryPoint = "meshNoTask",
        .shaderInterface = rachis::test::computeShaderInterface(
            {1, 1, 1}, {}, std::array{rachis::ShaderPushConstantRequirement{0, 4 * sizeof(float)}}),
    });
    auto pixelShader = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "meshPixel",
        .shaderInterface = {},
    });
    REQUIRE(meshNoTaskShader.has_value());
    REQUIRE(pixelShader.has_value());

    rachis::ShaderModuleHandle taskShaderHandle{};
    rachis::ShaderModuleHandle meshShaderHandle{};
    rachis::BindingLayoutHandle bindingLayoutHandle{};
    rachis::BindingSetHandle bindingSetHandle{};
    rachis::BufferHandle constantsHandle{};
    if (capabilities.taskShaders)
    {
        const auto taskBytecode = loadSpirv("TaskMeshWorkload.as.spv");
        const auto meshBytecode = loadSpirv("TaskMeshWorkload.ms.spv");
        REQUIRE_FALSE(taskBytecode.empty());
        REQUIRE_FALSE(meshBytecode.empty());
        auto taskShader = device.createShaderModule({
            .stage = rachis::ShaderStage::task,
            .bytecode = taskBytecode,
            .entryPoint = "taskMain",
            .shaderInterface = rachis::test::computeShaderInterface(
                {1, 1, 1},
                std::array{rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1}}),
        });
        auto meshShader = device.createShaderModule({
            .stage = rachis::ShaderStage::mesh,
            .bytecode = meshBytecode,
            .entryPoint = "meshMain",
            .shaderInterface = rachis::test::computeShaderInterface(
                {1, 1, 1}, {}, std::array{rachis::ShaderPushConstantRequirement{0, 4 * sizeof(float)}}),
        });
        REQUIRE(taskShader.has_value());
        REQUIRE(meshShader.has_value());
        taskShaderHandle = *taskShader;
        meshShaderHandle = *meshShader;

        auto constants = device.createBuffer({
            .size = 256,
            .usage = rachis::BufferUsage::constant,
            .memoryType = rachis::MemoryType::upload,
            .label = "Task-stage mesh constants",
        });
        REQUIRE(constants.has_value());
        constantsHandle = *constants;
        auto mapped = device.mapBuffer(*constants);
        REQUIRE(mapped.has_value());
        std::memset(mapped->data, 0, mapped->size);
        const std::array taskColor{0.0F, 1.0F, 0.0F, 1.0F};
        std::memcpy(mapped->data, taskColor.data(), sizeof(taskColor));
        REQUIRE(device.unmapBuffer(*constants).has_value());

        const std::array bindingEntries{
            rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::task, 1},
        };
        auto bindingLayout = device.createBindingLayout({
            .entries = bindingEntries,
            .label = "Task-only binding layout",
        });
        REQUIRE(bindingLayout.has_value());
        bindingLayoutHandle = *bindingLayout;
        const std::array bindingWrites{
            rachis::BindingWrite{
                .binding = 0,
                .resource = rachis::BufferBinding{.buffer = *constants, .size = 256},
            },
        };
        auto bindingSet = rachis::test::createBindingSet(device,
                                                         {
                                                             .layout = *bindingLayout,
                                                             .writes = bindingWrites,
                                                             .label = "Task-only binding set",
                                                         });
        REQUIRE(bindingSet.has_value());
        bindingSetHandle = *bindingSet;
    }

    const std::array pushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::mesh, 0, 4 * sizeof(float)},
    };
    std::vector<rachis::BindingLayoutHandle> bindingLayouts;
    if (bindingLayoutHandle.isValid())
    {
        bindingLayouts.push_back(bindingLayoutHandle);
    }
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = pushConstantRanges,
        .label = "Vulkan mesh contract pipeline layout",
    });
    REQUIRE(pipelineLayout.has_value());

    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    const rachis::GraphicsPipelineDesc noTaskDesc{
        .layout = *pipelineLayout,
        .pixelShader = *pixelShader,
        .meshShader = *meshNoTaskShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .label = "Mesh-only contract pipeline",
    };
    auto invalid = noTaskDesc;
    invalid.meshShader = {};
    CHECK_FALSE(device.createGraphicsPipeline(invalid).has_value());
    invalid = noTaskDesc;
    invalid.vertexShader = *pixelShader;
    CHECK_FALSE(device.createGraphicsPipeline(invalid).has_value());
    invalid = noTaskDesc;
    invalid.topology = rachis::PrimitiveTopology::lineList;
    CHECK_FALSE(device.createGraphicsPipeline(invalid).has_value());

    auto noTaskPipeline = device.createGraphicsPipeline(noTaskDesc);
    REQUIRE(noTaskPipeline.has_value());

    rachis::PipelineHandle taskPipelineHandle{};
    if (capabilities.taskShaders)
    {
        auto taskPipeline = device.createGraphicsPipeline({
            .layout = *pipelineLayout,
            .pixelShader = *pixelShader,
            .taskShader = taskShaderHandle,
            .meshShader = meshShaderHandle,
            .cullMode = rachis::CullMode::none,
            .colorTargets = colorTargets,
            .label = "Task and mesh contract pipeline",
        });
        REQUIRE(taskPipeline.has_value());
        taskPipelineHandle = *taskPipeline;
    }

    constexpr std::uint64_t argumentOffset = sizeof(std::uint32_t);
    constexpr std::uint32_t argumentStride = 16;
    std::array<std::byte, argumentOffset + argumentStride + sizeof(rachis::DispatchIndirectArguments)> argumentBytes{};
    const rachis::DispatchIndirectArguments dispatchArguments{1, 1, 1};
    std::memcpy(argumentBytes.data() + argumentOffset, &dispatchArguments, sizeof(dispatchArguments));
    auto arguments = device.createBuffer({
        .size = argumentBytes.size(),
        .usage = rachis::BufferUsage::indirect,
        .memoryType = rachis::MemoryType::upload,
        .label = "Mesh dispatch arguments",
    });
    auto count = device.createBuffer({
        .size = 2 * sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::indirect,
        .memoryType = rachis::MemoryType::upload,
        .label = "Mesh dispatch count",
    });
    auto wrongUsage = device.createBuffer({
        .size = sizeof(dispatchArguments),
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(arguments.has_value());
    REQUIRE(count.has_value());
    REQUIRE(wrongUsage.has_value());
    auto mappedArguments = device.mapBuffer(*arguments);
    auto mappedCount = device.mapBuffer(*count);
    REQUIRE(mappedArguments.has_value());
    REQUIRE(mappedCount.has_value());
    std::memcpy(mappedArguments->data, argumentBytes.data(), argumentBytes.size());
    const std::uint32_t one = 1;
    std::memcpy(mappedCount->data + sizeof(one), &one, sizeof(one));
    REQUIRE(device.unmapBuffer(*arguments).has_value());
    REQUIRE(device.unmapBuffer(*count).has_value());

    auto target = device.createTexture({
        .width = targetSize,
        .height = targetSize,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
        .label = "Vulkan mesh contract target",
    });
    auto readback = device.createBuffer({
        .size = readbackSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan mesh contract readback",
    });
    REQUIRE(target.has_value());
    REQUIRE(readback.has_value());
    auto targetView = device.createTextureView({
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
        .label = "Vulkan mesh contract target view",
    });
    REQUIRE(targetView.has_value());

    rachis::QueryPoolHandle statisticsHandle{};
    rachis::BufferHandle statisticsReadbackHandle{};
    if (capabilities.meshPipelineStatisticsQueries)
    {
        auto statistics = device.createPipelineStatisticsQueryPool({
            .mode = rachis::PipelineStatisticsQueryMode::meshShader,
            .queryCount = 1,
            .label = "Vulkan mesh pipeline-statistics pool",
        });
        auto statisticsReadback = device.createBuffer({
            .size = rachis::meshPipelineStatisticsQueryResultSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
        });
        REQUIRE(statistics.has_value());
        REQUIRE(statisticsReadback.has_value());
        statisticsHandle = *statistics;
        statisticsReadbackHandle = *statisticsReadback;
    }

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    if (statisticsHandle.isValid())
    {
        REQUIRE(device.beginPipelineStatisticsQuery(*command, statisticsHandle, 0).has_value());
    }
    CHECK_FALSE(device.dispatchMesh(*command, 1).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *target,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array attachments{
        rachis::ColorAttachment{
            .view = *targetView,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.0F, 0.0F, 1.0F, 1.0F},
        },
    };
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
    const auto pipeline = taskPipelineHandle.isValid() ? taskPipelineHandle : *noTaskPipeline;
    REQUIRE(device.bindGraphicsPipeline(*command, pipeline).has_value());
    CHECK_FALSE(device.draw(*command, 3).has_value());
    if (bindingSetHandle.isValid())
    {
        REQUIRE(device.bindGraphicsBindingSet(*command, 0, bindingSetHandle).has_value());
    }
    REQUIRE(device
                .setViewport(*command,
                             {
                                 .width = static_cast<float>(targetSize),
                                 .height = static_cast<float>(targetSize),
                                 .maxDepth = 1.0F,
                             })
                .has_value());
    REQUIRE(device.setScissor(*command, {.width = targetSize, .height = targetSize}).has_value());
    const std::array meshControl{1.0F, 0.0F, 0.0F, 0.0F};
    REQUIRE(
        device.setGraphicsPushConstants(*command, rachis::ShaderStage::mesh, 0, std::as_bytes(std::span(meshControl)))
            .has_value());

    CHECK_FALSE(device.dispatchMesh(*command, 0).has_value());
    CHECK_FALSE(device.dispatchMesh(*command, capabilities.meshShader.maxWorkgroupCount[0] + 1).has_value());
    CHECK_FALSE(device.dispatchMeshIndirect(*command, {.argumentBuffer = *wrongUsage}).has_value());
    CHECK_FALSE(device.dispatchMeshIndirect(*command, {.argumentBuffer = *arguments, .argumentOffset = 2}).has_value());

    const rachis::IndirectMeshDispatchDesc countedDispatch{
        .argumentBuffer = *arguments,
        .argumentOffset = argumentOffset,
        .maxDispatchCount = 2,
        .stride = argumentStride,
        .countBuffer = *count,
        .countOffset = sizeof(std::uint32_t),
    };
    auto zeroDispatch = countedDispatch;
    zeroDispatch.argumentOffset = argumentBytes.size();
    zeroDispatch.maxDispatchCount = 0;
    REQUIRE(device.dispatchMeshIndirect(*command, zeroDispatch).has_value());
    REQUIRE(device.dispatchMesh(*command, 1).has_value());
    REQUIRE(device.dispatchMeshIndirect(*command, countedDispatch).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    if (statisticsHandle.isValid())
    {
        REQUIRE(device.endPipelineStatisticsQuery(*command).has_value());
    }
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *target,
                                    .stagesBefore = rachis::PipelineStage::renderTarget,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::renderTarget,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device
                .copyTextureToBuffer(*command,
                                     {
                                         .buffer = *readback,
                                         .bytesPerRow = rowPitch,
                                         .rowsPerImage = targetSize,
                                         .texture = *target,
                                         .extent = {targetSize, targetSize, 1},
                                     })
                .has_value());
    if (statisticsHandle.isValid())
    {
        REQUIRE(device
                    .resolvePipelineStatisticsQueries(*command,
                                                      {
                                                          .queryPool = statisticsHandle,
                                                          .queryCount = 1,
                                                          .destination = statisticsReadbackHandle,
                                                      })
                    .has_value());
        REQUIRE(device.destroyQueryPool(statisticsHandle).has_value());
    }
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    const auto* corner = mapped->data;
    const auto* center = mapped->data + static_cast<std::uint64_t>(targetSize / 2) * rowPitch +
                         static_cast<std::uint64_t>(targetSize / 2) * 4;
    CHECK(corner[0] == std::byte{0});
    CHECK(corner[1] == std::byte{0});
    CHECK(corner[2] == std::byte{255});
    CHECK(corner[3] == std::byte{255});
    CHECK(center[0] == std::byte{0});
    CHECK(center[1] == std::byte{255});
    CHECK(center[2] == std::byte{0});
    CHECK(center[3] == std::byte{255});
    REQUIRE(device.unmapBuffer(*readback).has_value());

    if (statisticsReadbackHandle.isValid())
    {
        mapped = device.mapBuffer(statisticsReadbackHandle);
        REQUIRE(mapped.has_value());
        rachis::MeshPipelineStatistics meshStatistics{};
        std::memcpy(&meshStatistics, mapped->data, sizeof(meshStatistics));
        CAPTURE(meshStatistics.taskShaderInvocations,
                meshStatistics.meshShaderInvocations,
                meshStatistics.meshShaderPrimitives,
                meshStatistics.base.fragmentShaderInvocations);
        if (capabilities.taskShaders)
        {
            CHECK(meshStatistics.taskShaderInvocations >= 1);
        }
        CHECK(meshStatistics.meshShaderInvocations >= 1);
        CHECK(meshStatistics.meshShaderPrimitives >= 1);
        CHECK(meshStatistics.base.fragmentShaderInvocations > 0);
        REQUIRE(device.unmapBuffer(statisticsReadbackHandle).has_value());
        REQUIRE(device.destroyBuffer(statisticsReadbackHandle).has_value());
    }

    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*arguments).has_value());
    REQUIRE(device.destroyBuffer(*count).has_value());
    REQUIRE(device.destroyBuffer(*wrongUsage).has_value());
    REQUIRE(device.destroyPipeline(*noTaskPipeline).has_value());
    if (taskPipelineHandle.isValid())
    {
        REQUIRE(device.destroyPipeline(taskPipelineHandle).has_value());
    }
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    if (bindingSetHandle.isValid())
    {
        REQUIRE(device.destroyBindingSet(bindingSetHandle).has_value());
        REQUIRE(device.destroyBindingLayout(bindingLayoutHandle).has_value());
        REQUIRE(device.destroyBuffer(constantsHandle).has_value());
        REQUIRE(device.destroyShaderModule(taskShaderHandle).has_value());
        REQUIRE(device.destroyShaderModule(meshShaderHandle).has_value());
    }
    REQUIRE(device.destroyPipeline(*noTaskPipeline).has_value() == false);
    REQUIRE(device.destroyShaderModule(*meshNoTaskShader).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
}
#endif
