#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
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
    constexpr std::uint64_t statisticsCount = 2;
    constexpr std::uint64_t resultBufferSize =
        statisticsCount * rachis::pipelineStatisticsQueryResultSize + rachis::occlusionQueryResultSize;

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

    [[nodiscard]] bool isZero(const rachis::PipelineStatistics& statistics) noexcept
    {
        return statistics.inputAssemblyVertices == 0 && statistics.inputAssemblyPrimitives == 0 &&
               statistics.vertexShaderInvocations == 0 && statistics.geometryShaderInvocations == 0 &&
               statistics.geometryShaderPrimitives == 0 && statistics.clippingInvocations == 0 &&
               statistics.clippingPrimitives == 0 && statistics.fragmentShaderInvocations == 0 &&
               statistics.tessellationControlShaderPatches == 0 &&
               statistics.tessellationEvaluationShaderInvocations == 0 && statistics.computeShaderInvocations == 0;
    }

} // namespace

TEST_CASE("pipeline-statistics queries span rendering and resolve portable counters",
          "[size:medium][rhi][query][pipeline-statistics][graphics][warp]")
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
    REQUIRE(rachis::test::requireCapabilities(device).pipelineStatisticsQueries);

    CHECK_FALSE(device.createPipelineStatisticsQueryPool({.queryCount = 0}).has_value());
    CHECK_FALSE(device
                    .createPipelineStatisticsQueryPool({
                        .mode = static_cast<rachis::PipelineStatisticsQueryMode>(UINT8_MAX),
                        .queryCount = 1,
                    })
                    .has_value());
    auto pool = device.createPipelineStatisticsQueryPool({
        .queryCount = statisticsCount,
        .label = "Pipeline-statistics contract pool",
    });
    auto transientPool = device.createPipelineStatisticsQueryPool({
        .queryCount = 1,
        .label = "Destroyed active pipeline-statistics pool",
    });
    auto occlusionPool = device.createOcclusionQueryPool({
        .mode = rachis::OcclusionQueryMode::binary,
        .queryCount = 1,
        .label = "Nested occlusion pool",
    });
    auto timestampPool = device.createTimestampQueryPool({
        .queueType = rachis::QueueType::graphics,
        .queryCount = 1,
    });
    REQUIRE(pool.has_value());
    REQUIRE(transientPool.has_value());
    REQUIRE(occlusionPool.has_value());
    REQUIRE(timestampPool.has_value());

    const auto vertexBytecode = loadShader(RACHIS_TEST_GRAPHICS_DIRECT_VERTEX_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_GRAPHICS_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    auto vertexShader = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .entryPoint = "directVertex",
        .shaderInterface = {},
    });
    auto pixelShader = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "pixelMain",
        .shaderInterface = rachis::test::graphicsPixelShaderInterface(),
    });
    REQUIRE(vertexShader.has_value());
    REQUIRE(pixelShader.has_value());

    const std::array pushConstants{
        rachis::PushConstantRange{rachis::ShaderStage::pixel, 0, 4 * sizeof(float)},
    };
    auto pipelineLayout = device.createPipelineLayout({
        .pushConstantRanges = pushConstants,
        .label = "Pipeline-statistics contract layout",
    });
    REQUIRE(pipelineLayout.has_value());
    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    auto pipeline = device.createGraphicsPipeline({
        .layout = *pipelineLayout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .label = "Pipeline-statistics contract pipeline",
    });
    REQUIRE(pipeline.has_value());

    auto target = device.createTexture({
        .width = targetSize,
        .height = targetSize,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .label = "Pipeline-statistics contract target",
    });
    REQUIRE(target.has_value());
    auto targetView = device.createTextureView({
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    auto results = device.createBuffer({
        .size = resultBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Pipeline-statistics contract results",
    });
    REQUIRE(targetView.has_value());
    REQUIRE(results.has_value());

    auto computeCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(computeCommand.has_value());
    CHECK_FALSE(device.beginPipelineStatisticsQuery(*computeCommand, *pool, 0).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    CHECK_FALSE(device.endPipelineStatisticsQuery(*command).has_value());
    CHECK_FALSE(device.beginPipelineStatisticsQuery(*command, *pool, statisticsCount).has_value());
    CHECK_FALSE(device.beginPipelineStatisticsQuery(*command, *timestampPool, 0).has_value());
    CHECK_FALSE(device.beginPipelineStatisticsQuery(*command, *occlusionPool, 0).has_value());

    REQUIRE(device.beginPipelineStatisticsQuery(*command, *transientPool, 0).has_value());
    const auto staleTransientPool = *transientPool;
    REQUIRE(device.destroyQueryPool(*transientPool).has_value());
    REQUIRE(device.endPipelineStatisticsQuery(*command).has_value());
    CHECK_FALSE(device.beginPipelineStatisticsQuery(*command, staleTransientPool, 0).has_value());

    REQUIRE(device.beginPipelineStatisticsQuery(*command, *pool, 0).has_value());
    auto activeStatisticsReset = device.resetQueryPool(*command, *pool, 0, 1);
    REQUIRE_FALSE(activeStatisticsReset.has_value());
    CHECK(activeStatisticsReset.error().code == rachis::ErrorCode::invalidState);
    CHECK_FALSE(device.beginPipelineStatisticsQuery(*command, *pool, 1).has_value());
    CHECK_FALSE(device
                    .resolvePipelineStatisticsQueries(*command,
                                                      {
                                                          .queryPool = *pool,
                                                          .queryCount = 1,
                                                          .destination = *results,
                                                      })
                    .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *target,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array attachments{
        rachis::ColorAttachment{
            .view = *targetView,
            .loadOperation = rachis::LoadOperation::clear,
            .storeOperation = rachis::StoreOperation::discard,
        },
    };
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *pipeline).has_value());
    REQUIRE(device
                .setViewport(*command,
                             {
                                 .width = static_cast<float>(targetSize),
                                 .height = static_cast<float>(targetSize),
                                 .maxDepth = 1.0F,
                             })
                .has_value());
    REQUIRE(device.setScissor(*command, {.width = targetSize, .height = targetSize}).has_value());
    const std::array tint{1.0F, 1.0F, 1.0F, 1.0F};
    REQUIRE(device.setGraphicsPushConstants(*command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(tint)))
                .has_value());
    REQUIRE(device.beginOcclusionQuery(*command, *occlusionPool, 0).has_value());
    REQUIRE(device.draw(*command, 3).has_value());
    REQUIRE(device.endOcclusionQuery(*command).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    REQUIRE(device.endPipelineStatisticsQuery(*command).has_value());
    CHECK_FALSE(device.beginPipelineStatisticsQuery(*command, *pool, 0).has_value());

    REQUIRE(device.beginPipelineStatisticsQuery(*command, *pool, 1).has_value());
    REQUIRE(device.endPipelineStatisticsQuery(*command).has_value());
    CHECK_FALSE(device
                    .resolvePipelineStatisticsQueries(*command,
                                                      {
                                                          .queryPool = *pool,
                                                          .queryCount = statisticsCount,
                                                          .destination = *results,
                                                          .destinationOffset = 1,
                                                      })
                    .has_value());
    CHECK_FALSE(device
                    .resolvePipelineStatisticsQueries(*command,
                                                      {
                                                          .queryPool = *occlusionPool,
                                                          .queryCount = 1,
                                                          .destination = *results,
                                                      })
                    .has_value());
    REQUIRE(device
                .resolvePipelineStatisticsQueries(*command,
                                                  {
                                                      .queryPool = *pool,
                                                      .queryCount = statisticsCount,
                                                      .destination = *results,
                                                  })
                .has_value());
    CHECK_FALSE(device
                    .resolvePipelineStatisticsQueries(*command,
                                                      {
                                                          .queryPool = *pool,
                                                          .queryCount = statisticsCount,
                                                          .destination = *results,
                                                      })
                    .has_value());
    REQUIRE(device
                .resolveOcclusionQueries(
                    *command,
                    {
                        .queryPool = *occlusionPool,
                        .queryCount = 1,
                        .destination = *results,
                        .destinationOffset = statisticsCount * rachis::pipelineStatisticsQueryResultSize,
                    })
                .has_value());

    REQUIRE(device.destroyQueryPool(*pool).has_value());
    REQUIRE(device.destroyQueryPool(*occlusionPool).has_value());
    REQUIRE(device.destroyQueryPool(*timestampPool).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    auto mapped = device.mapBuffer(*results);
    REQUIRE(mapped.has_value());
    std::array<rachis::PipelineStatistics, statisticsCount> statistics{};
    std::memcpy(statistics.data(), mapped->data, sizeof(statistics));
    std::uint64_t occlusion = 0;
    std::memcpy(&occlusion, mapped->data + sizeof(statistics), sizeof(occlusion));
    CAPTURE(statistics[0].inputAssemblyVertices,
            statistics[0].inputAssemblyPrimitives,
            statistics[0].vertexShaderInvocations,
            statistics[0].clippingInvocations,
            statistics[0].clippingPrimitives,
            statistics[0].fragmentShaderInvocations,
            occlusion);
    CHECK(statistics[0].inputAssemblyVertices >= 3);
    CHECK(statistics[0].inputAssemblyPrimitives >= 1);
    CHECK(statistics[0].vertexShaderInvocations >= 3);
    CHECK(statistics[0].clippingInvocations >= 1);
    CHECK(statistics[0].clippingPrimitives >= 1);
    CHECK(statistics[0].fragmentShaderInvocations > 0);
    CHECK(isZero(statistics[1]));
    CHECK(occlusion == 1);
    REQUIRE(device.unmapBuffer(*results).has_value());

    REQUIRE(device.destroyBuffer(*results).has_value());
    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("pipeline-statistics queries begin and end on the same side of a rendering-scope boundary",
          "[size:medium][rhi][query][pipeline-statistics][graphics][warp]")
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
    REQUIRE(rachis::test::requireCapabilities(device).pipelineStatisticsQueries);

    constexpr std::uint32_t queryCount = 3;
    auto pool = device.createPipelineStatisticsQueryPool({
        .queryCount = queryCount,
        .label = "Rendering-scope pipeline-statistics pool",
    });
    auto target = device.createTexture({
        .width = targetSize,
        .height = targetSize,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .label = "Rendering-scope pipeline-statistics target",
    });
    auto results = device.createBuffer({
        .size = queryCount * rachis::pipelineStatisticsQueryResultSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Rendering-scope pipeline-statistics results",
    });
    REQUIRE(pool.has_value());
    REQUIRE(target.has_value());
    REQUIRE(results.has_value());
    auto targetView = device.createTextureView({
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    REQUIRE(targetView.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *target,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array attachments{
        rachis::ColorAttachment{
            .view = *targetView,
            .loadOperation = rachis::LoadOperation::clear,
            .storeOperation = rachis::StoreOperation::store,
        },
    };
    const auto expectInvalidState = [](const rachis::Status& status) {
        REQUIRE_FALSE(status.has_value());
        CHECK(status.error().code == rachis::ErrorCode::invalidState);
    };

    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
    REQUIRE(device.beginPipelineStatisticsQuery(*command, *pool, 0).has_value());
    expectInvalidState(device.endRendering(*command));
    REQUIRE(device.endPipelineStatisticsQuery(*command).has_value());
    REQUIRE(device.endRendering(*command).has_value());

    REQUIRE(device.beginPipelineStatisticsQuery(*command, *pool, 1).has_value());
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
    expectInvalidState(device.endPipelineStatisticsQuery(*command));
    REQUIRE(device.endRendering(*command).has_value());
    REQUIRE(device.endPipelineStatisticsQuery(*command).has_value());

    REQUIRE(device.beginPipelineStatisticsQuery(*command, *pool, 2).has_value());
    REQUIRE(device.endPipelineStatisticsQuery(*command).has_value());
    REQUIRE(device
                .resolvePipelineStatisticsQueries(*command,
                                                  {
                                                      .queryPool = *pool,
                                                      .queryCount = queryCount,
                                                      .destination = *results,
                                                  })
                .has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    REQUIRE(device.destroyBuffer(*results).has_value());
    REQUIRE(device.destroyQueryPool(*pool).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
