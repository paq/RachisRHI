#include "D3D12/D3D12TestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <span>
#include <vector>

namespace
{

    constexpr std::uint32_t targetSize = 64;
    constexpr std::uint32_t rowPitch = 256;
    constexpr std::uint64_t readbackSize = static_cast<std::uint64_t>(rowPitch) * targetSize;
    constexpr rachis::PipelineCacheKey meshCacheKey{0x6d6573682d737461ULL, 0x67652d6361636865ULL};

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
        std::int32_t previous_;
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

    [[nodiscard]] std::vector<std::byte> loadShader(const char* path)
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

} // namespace

TEST_CASE("mesh pipelines retain task and mesh stages and dispatch directly and indirectly",
          "[size:medium][rhi][graphics][mesh][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    const auto capabilities = *device.capabilities();
    REQUIRE(capabilities.meshShaders);
    REQUIRE(capabilities.taskShaders);
    REQUIRE(capabilities.meshPipelineStatisticsQueries);

    const auto taskBytecode = loadShader(RACHIS_TEST_TASK_SHADER_PATH);
    const auto meshBytecode = loadShader(RACHIS_TEST_MESH_SHADER_PATH);
    const auto meshNoTaskBytecode = loadShader(RACHIS_TEST_MESH_NO_TASK_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_MESH_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(taskBytecode.empty());
    REQUIRE_FALSE(meshBytecode.empty());
    REQUIRE_FALSE(meshNoTaskBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());

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
    REQUIRE(taskShader.has_value());
    REQUIRE(meshShader.has_value());
    REQUIRE(meshNoTaskShader.has_value());
    REQUIRE(pixelShader.has_value());

    auto constants = device.createBuffer({
        .size = 256,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Task-stage mesh constants",
    });
    REQUIRE(constants.has_value());
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

    const std::array bindingLayouts{*bindingLayout};
    const std::array pushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::mesh, 0, 4 * sizeof(float)},
    };
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = pushConstantRanges,
        .label = "Mesh contract pipeline layout",
    });
    REQUIRE(pipelineLayout.has_value());
    REQUIRE(capabilities.pipelineCaches);
    auto pipelineCache = device.createPipelineCache({.label = "Mesh pipeline cache"});
    REQUIRE(pipelineCache.has_value());

    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    const rachis::GraphicsPipelineDesc pipelineDesc{
        .layout = *pipelineLayout,
        .pixelShader = *pixelShader,
        .taskShader = *taskShader,
        .meshShader = *meshShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .label = "Task and mesh contract pipeline",
        .cache = *pipelineCache,
        .cacheKey = meshCacheKey,
    };

    auto invalid = pipelineDesc;
    invalid.meshShader = {};
    CHECK_FALSE(device.createGraphicsPipeline(invalid).has_value());
    invalid = pipelineDesc;
    invalid.taskShader = *meshShader;
    CHECK_FALSE(device.createGraphicsPipeline(invalid).has_value());
    invalid = pipelineDesc;
    invalid.meshShader = *pixelShader;
    CHECK_FALSE(device.createGraphicsPipeline(invalid).has_value());
    invalid = pipelineDesc;
    invalid.vertexShader = *pixelShader;
    CHECK_FALSE(device.createGraphicsPipeline(invalid).has_value());
    invalid = pipelineDesc;
    invalid.topology = rachis::PrimitiveTopology::lineList;
    CHECK_FALSE(device.createGraphicsPipeline(invalid).has_value());
    const std::array invalidVertexAttributes{rachis::VertexAttribute{}};
    const std::array invalidVertexBuffers{
        rachis::VertexBufferLayout{.stride = sizeof(float), .attributes = invalidVertexAttributes},
    };
    invalid = pipelineDesc;
    invalid.vertexBuffers = invalidVertexBuffers;
    CHECK_FALSE(device.createGraphicsPipeline(invalid).has_value());

    auto noTaskDesc = pipelineDesc;
    noTaskDesc.taskShader = {};
    noTaskDesc.meshShader = *meshNoTaskShader;
    noTaskDesc.cache = {};
    noTaskDesc.cacheKey = {};
    auto noTaskPipeline = device.createGraphicsPipeline(noTaskDesc);
    REQUIRE(noTaskPipeline.has_value());

    auto pipeline = device.createGraphicsPipeline(pipelineDesc);
    REQUIRE(pipeline.has_value());
    auto cacheData = device.pipelineCacheData(*pipelineCache);
    REQUIRE(cacheData.has_value());
    REQUIRE_FALSE(cacheData->empty());
    REQUIRE(device.destroyPipelineCache(*pipelineCache).has_value());
    auto restoredCache = device.createPipelineCache({
        .initialData = *cacheData,
        .label = "Restored mesh pipeline cache",
    });
    REQUIRE(restoredCache.has_value());
    auto restoredDesc = pipelineDesc;
    restoredDesc.cache = *restoredCache;
    auto restoredPipeline = device.createGraphicsPipeline(restoredDesc);
    REQUIRE(restoredPipeline.has_value());
    REQUIRE(device.destroyPipelineCache(*restoredCache).has_value());
    REQUIRE(device.destroyPipeline(*restoredPipeline).has_value());
    const auto staleTaskShader = *taskShader;
    const auto staleMeshShader = *meshShader;
    const auto staleMeshNoTaskShader = *meshNoTaskShader;
    REQUIRE(device.destroyShaderModule(*taskShader).has_value());
    REQUIRE(device.destroyShaderModule(*meshShader).has_value());
    REQUIRE(device.destroyShaderModule(*meshNoTaskShader).has_value());
    REQUIRE_FALSE(device.createGraphicsPipeline(pipelineDesc).has_value());
    REQUIRE_FALSE(device.createGraphicsPipeline(noTaskDesc).has_value());

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
    auto staleArguments = device.createBuffer({
        .size = sizeof(dispatchArguments),
        .usage = rachis::BufferUsage::indirect,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(arguments.has_value());
    REQUIRE(count.has_value());
    REQUIRE(wrongUsage.has_value());
    REQUIRE(staleArguments.has_value());
    auto mappedArguments = device.mapBuffer(*arguments);
    auto mappedCount = device.mapBuffer(*count);
    REQUIRE(mappedArguments.has_value());
    REQUIRE(mappedCount.has_value());
    std::memcpy(mappedArguments->data, argumentBytes.data(), argumentBytes.size());
    const std::uint32_t one = 1;
    std::memcpy(mappedCount->data + sizeof(one), &one, sizeof(one));
    REQUIRE(device.unmapBuffer(*arguments).has_value());
    REQUIRE(device.unmapBuffer(*count).has_value());
    REQUIRE(device.destroyBuffer(*staleArguments).has_value());

    auto target = device.createTexture({
        .width = targetSize,
        .height = targetSize,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
        .label = "Mesh contract target",
    });
    auto readback = device.createBuffer({
        .size = readbackSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Mesh contract readback",
    });
    auto statistics = device.createPipelineStatisticsQueryPool({
        .mode = rachis::PipelineStatisticsQueryMode::meshShader,
        .queryCount = 1,
        .label = "Mesh pipeline-statistics contract pool",
    });
    auto statisticsReadback = device.createBuffer({
        .size = rachis::meshPipelineStatisticsQueryResultSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Mesh pipeline-statistics contract results",
    });
    REQUIRE(target.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(statistics.has_value());
    REQUIRE(statisticsReadback.has_value());
    auto targetView = device.createTextureView({
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
        .label = "Mesh contract target view",
    });
    REQUIRE(targetView.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device.beginPipelineStatisticsQuery(*command, *statistics, 0).has_value());
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
    REQUIRE(device.bindGraphicsPipeline(*command, *pipeline).has_value());
    CHECK_FALSE(device.draw(*command, 3).has_value());
    REQUIRE(device.bindGraphicsBindingSet(*command, 0, *bindingSet).has_value());
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
    CHECK_FALSE(device.dispatchMesh(*command, 65'535, 65, 1).has_value());
    CHECK_FALSE(device.dispatchMeshIndirect(*command, {.argumentBuffer = *wrongUsage}).has_value());
    CHECK_FALSE(device.dispatchMeshIndirect(*command, {.argumentBuffer = *arguments, .argumentOffset = 2}).has_value());
    CHECK_FALSE(device
                    .dispatchMeshIndirect(*command,
                                          {.argumentBuffer = *arguments, .argumentOffset = argumentOffset, .stride = 8})
                    .has_value());
    CHECK_FALSE(device
                    .dispatchMeshIndirect(*command,
                                          {.argumentBuffer = *arguments,
                                           .argumentOffset = argumentOffset,
                                           .countBuffer = *count,
                                           .countOffset = 2})
                    .has_value());
    CHECK_FALSE(device.dispatchMeshIndirect(*command, {.argumentBuffer = *staleArguments}).has_value());
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

    const auto beforeRejectedPublication = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(beforeRejectedPublication.has_value());
    const auto recordsBeforeRejectedPublication =
        rachis::d3d12::test::indirectExecutionRecordCount().load(std::memory_order_relaxed);
    {
        const FailResourceRetention failCountPublication(2);
        const auto rejected = device.dispatchMeshIndirect(*command, countedDispatch);
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::outOfMemory);
    }
    const auto afterRejectedPublication = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(afterRejectedPublication.has_value());
    CHECK(*afterRejectedPublication == *beforeRejectedPublication);
    CHECK(rachis::d3d12::test::indirectExecutionRecordCount().load(std::memory_order_relaxed) ==
          recordsBeforeRejectedPublication);

    REQUIRE(device.dispatchMesh(*command, 1).has_value());
    REQUIRE(device.dispatchMeshIndirect(*command, countedDispatch).has_value());
    REQUIRE(device.destroyBuffer(*arguments).has_value());
    REQUIRE(device.destroyBuffer(*count).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    REQUIRE(device.endPipelineStatisticsQuery(*command).has_value());
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
    CHECK_FALSE(device
                    .resolvePipelineStatisticsQueries(*command,
                                                      {
                                                          .queryPool = *statistics,
                                                          .queryCount = 1,
                                                          .destination = *statisticsReadback,
                                                          .destinationOffset = sizeof(std::uint64_t),
                                                      })
                    .has_value());
    REQUIRE(device
                .resolvePipelineStatisticsQueries(*command,
                                                  {
                                                      .queryPool = *statistics,
                                                      .queryCount = 1,
                                                      .destination = *statisticsReadback,
                                                  })
                .has_value());
    REQUIRE(device.destroyQueryPool(*statistics).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    mapped = device.mapBuffer(*readback);
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

    mapped = device.mapBuffer(*statisticsReadback);
    REQUIRE(mapped.has_value());
    rachis::MeshPipelineStatistics meshStatistics{};
    std::memcpy(&meshStatistics, mapped->data, sizeof(meshStatistics));
    CAPTURE(meshStatistics.taskShaderInvocations,
            meshStatistics.meshShaderInvocations,
            meshStatistics.meshShaderPrimitives,
            meshStatistics.base.fragmentShaderInvocations);
    CHECK(meshStatistics.taskShaderInvocations >= 2);
    CHECK(meshStatistics.meshShaderInvocations >= 2);
    CHECK(meshStatistics.meshShaderPrimitives >= 2);
    CHECK(meshStatistics.base.fragmentShaderInvocations > 0);
    REQUIRE(device.unmapBuffer(*statisticsReadback).has_value());

    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*statisticsReadback).has_value());
    REQUIRE_FALSE(device.destroyBuffer(*arguments).has_value());
    REQUIRE_FALSE(device.destroyBuffer(*count).has_value());
    REQUIRE(device.destroyBuffer(*wrongUsage).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipeline(*noTaskPipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyBuffer(*constants).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    REQUIRE_FALSE(device.destroyShaderModule(staleTaskShader).has_value());
    REQUIRE_FALSE(device.destroyShaderModule(staleMeshShader).has_value());
    REQUIRE_FALSE(device.destroyShaderModule(staleMeshNoTaskShader).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
