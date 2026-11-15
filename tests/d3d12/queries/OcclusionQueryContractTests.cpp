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
    constexpr std::uint64_t resultBufferSize = 3 * rachis::occlusionQueryResultSize;

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

TEST_CASE("occlusion queries enforce render scopes and resolve precise and binary results",
          "[size:medium][rhi][query][occlusion][graphics][warp]")
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
    REQUIRE(rachis::test::requireCapabilities(device).occlusionQueries);

    CHECK_FALSE(device
                    .createOcclusionQueryPool({
                        .mode = static_cast<rachis::OcclusionQueryMode>(255),
                        .queryCount = 1,
                    })
                    .has_value());
    CHECK_FALSE(device.createOcclusionQueryPool({.queryCount = 0}).has_value());

    auto precisePool = device.createOcclusionQueryPool({
        .mode = rachis::OcclusionQueryMode::precise,
        .queryCount = 1,
        .label = "Precise occlusion contract pool",
    });
    auto binaryPool = device.createOcclusionQueryPool({
        .mode = rachis::OcclusionQueryMode::binary,
        .queryCount = 2,
        .label = "Binary occlusion contract pool",
    });
    auto transientPool = device.createOcclusionQueryPool({
        .mode = rachis::OcclusionQueryMode::binary,
        .queryCount = 1,
        .label = "Destroyed active occlusion pool",
    });
    auto timestampPool = device.createTimestampQueryPool({
        .queueType = rachis::QueueType::graphics,
        .queryCount = 1,
    });
    REQUIRE(precisePool.has_value());
    REQUIRE(binaryPool.has_value());
    REQUIRE(transientPool.has_value());
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
        .label = "Occlusion contract pipeline layout",
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
        .label = "Occlusion contract pipeline",
    });
    REQUIRE(pipeline.has_value());

    auto target = device.createTexture({
        .width = targetSize,
        .height = targetSize,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .label = "Occlusion contract target",
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
        .label = "Occlusion contract results",
    });
    REQUIRE(targetView.has_value());
    REQUIRE(results.has_value());

    auto computeCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(computeCommand.has_value());
    auto invalidQueueBegin = device.beginOcclusionQuery(*computeCommand, *precisePool, 0);
    REQUIRE_FALSE(invalidQueueBegin.has_value());
    CHECK(invalidQueueBegin.error().code == rachis::ErrorCode::invalidState);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    CHECK_FALSE(device.beginOcclusionQuery(*command, *precisePool, 0).has_value());
    CHECK_FALSE(device.beginOcclusionQuery(*command, *precisePool, 1).has_value());
    CHECK_FALSE(device.beginOcclusionQuery(*command, *timestampPool, 0).has_value());
    CHECK_FALSE(device.endOcclusionQuery(*command).has_value());
    CHECK_FALSE(device
                    .resolveOcclusionQueries(*command,
                                             {
                                                 .queryPool = *precisePool,
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
    CHECK_FALSE(device
                    .resolveOcclusionQueries(*command,
                                             {
                                                 .queryPool = *precisePool,
                                                 .queryCount = 1,
                                                 .destination = *results,
                                             })
                    .has_value());
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

    REQUIRE(device.beginOcclusionQuery(*command, *precisePool, 0).has_value());
    CHECK_FALSE(device.beginOcclusionQuery(*command, *binaryPool, 0).has_value());
    CHECK_FALSE(device.endRendering(*command).has_value());
    REQUIRE(device.draw(*command, 3).has_value());
    REQUIRE(device.endOcclusionQuery(*command).has_value());
    CHECK_FALSE(device.beginOcclusionQuery(*command, *precisePool, 0).has_value());

    REQUIRE(device.beginOcclusionQuery(*command, *binaryPool, 0).has_value());
    REQUIRE(device.draw(*command, 3).has_value());
    REQUIRE(device.endOcclusionQuery(*command).has_value());
    REQUIRE(device.beginOcclusionQuery(*command, *binaryPool, 1).has_value());
    REQUIRE(device.endOcclusionQuery(*command).has_value());

    REQUIRE(device.beginOcclusionQuery(*command, *transientPool, 0).has_value());
    const auto staleTransientPool = *transientPool;
    REQUIRE(device.destroyQueryPool(*transientPool).has_value());
    REQUIRE(device.endOcclusionQuery(*command).has_value());
    CHECK_FALSE(device.beginOcclusionQuery(*command, staleTransientPool, 0).has_value());
    CHECK_FALSE(device.endOcclusionQuery(*command).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    CHECK_FALSE(device.beginOcclusionQuery(*command, *binaryPool, 0).has_value());

    CHECK_FALSE(device
                    .resolveOcclusionQueries(*command,
                                             {
                                                 .queryPool = *precisePool,
                                                 .queryCount = 1,
                                                 .destination = *results,
                                                 .destinationOffset = 1,
                                             })
                    .has_value());
    CHECK_FALSE(device
                    .resolveOcclusionQueries(*command,
                                             {
                                                 .queryPool = *timestampPool,
                                                 .queryCount = 1,
                                                 .destination = *results,
                                             })
                    .has_value());
    REQUIRE(device
                .resolveOcclusionQueries(*command,
                                         {
                                             .queryPool = *precisePool,
                                             .queryCount = 1,
                                             .destination = *results,
                                         })
                .has_value());
    REQUIRE(device
                .resolveOcclusionQueries(*command,
                                         {
                                             .queryPool = *binaryPool,
                                             .queryCount = 2,
                                             .destination = *results,
                                             .destinationOffset = rachis::occlusionQueryResultSize,
                                         })
                .has_value());
    CHECK_FALSE(device
                    .resolveOcclusionQueries(*command,
                                             {
                                                 .queryPool = *binaryPool,
                                                 .queryCount = 2,
                                                 .destination = *results,
                                                 .destinationOffset = rachis::occlusionQueryResultSize,
                                             })
                    .has_value());

    REQUIRE(device.destroyQueryPool(*precisePool).has_value());
    REQUIRE(device.destroyQueryPool(*binaryPool).has_value());
    REQUIRE(device.destroyQueryPool(*timestampPool).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    auto mapped = device.mapBuffer(*results);
    REQUIRE(mapped.has_value());
    std::array<std::uint64_t, 3> queryResults{};
    std::memcpy(queryResults.data(), mapped->data, sizeof(queryResults));
    CAPTURE(queryResults[0], queryResults[1], queryResults[2]);
    CHECK(queryResults[0] > 0);
    CHECK(queryResults[1] == 1);
    CHECK(queryResults[2] == 0);
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
