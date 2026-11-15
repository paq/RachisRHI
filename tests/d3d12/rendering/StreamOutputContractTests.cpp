#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
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

    constexpr std::uint32_t targetSize = 32;
    constexpr std::uint64_t outputStride = 4 * sizeof(float);
    constexpr std::uint64_t outputSize = 3 * outputStride;
    constexpr std::uint64_t counterOffset = outputSize;
    constexpr std::uint64_t statisticsOffset = (counterOffset + sizeof(std::uint32_t) + 7U) & ~std::uint64_t{7};
    constexpr std::uint64_t readbackSize = statisticsOffset + rachis::streamOutputStatisticsQueryResultSize;

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

TEST_CASE("stream output binds append storage atomically and resolves per-stream pressure",
          "[size:medium][rhi][graphics][stream-output][query][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).streamOutput);
    REQUIRE(rachis::test::requireCapabilities(device).streamOutputStatisticsQueries);

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
    auto layout = device.createPipelineLayout({.pushConstantRanges = pushConstants});
    REQUIRE(layout.has_value());
    const std::array targets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    const std::array declarations{rachis::StreamOutputEntry{
        .stream = 0,
        .shaderLocation = 0,
        .startComponent = 0,
        .componentCount = 4,
        .outputSlot = 0,
    }};
    const std::array<std::uint32_t, 1> strides{outputStride};
    CHECK_FALSE(device
                    .createGraphicsPipeline({
                        .layout = *layout,
                        .vertexShader = *vertexShader,
                        .pixelShader = *pixelShader,
                        .cullMode = rachis::CullMode::none,
                        .colorTargets = targets,
                        .streamOutputEntries = declarations,
                        .streamOutputBufferStrides = std::array<std::uint32_t, 1>{3},
                    })
                    .has_value());
    auto pipeline = device.createGraphicsPipeline({
        .layout = *layout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = targets,
        .streamOutputEntries = declarations,
        .streamOutputBufferStrides = strides,
        .label = "Stream-output contract pipeline",
    });
    REQUIRE(pipeline.has_value());

    CHECK_FALSE(device.createStreamOutputStatisticsQueryPool({.stream = 4, .queryCount = 1}).has_value());
    auto queryPool = device.createStreamOutputStatisticsQueryPool({
        .stream = 0,
        .queryCount = 1,
        .label = "Stream-output contract query",
    });
    REQUIRE(queryPool.has_value());

    auto target = device.createTexture({
        .width = targetSize,
        .height = targetSize,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
    });
    REQUIRE(target.has_value());
    auto targetView = device.createTextureView({
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    auto output = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::streamOutput | rachis::BufferUsage::copySource,
        .label = "Stream-output contract data",
    });
    auto counter = device.createBuffer({
        .size = sizeof(std::uint64_t),
        .usage =
            rachis::BufferUsage::streamOutput | rachis::BufferUsage::copyDestination | rachis::BufferUsage::copySource,
        .label = "Stream-output contract counter",
    });
    auto counterSeed = device.createBuffer({
        .size = sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto readback = device.createBuffer({
        .size = readbackSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(targetView.has_value());
    REQUIRE(output.has_value());
    REQUIRE(counter.has_value());
    REQUIRE(counterSeed.has_value());
    REQUIRE(readback.has_value());
    const std::uint32_t zero = 0;
    REQUIRE(device.writeBuffer(*counterSeed, 0, std::as_bytes(std::span{&zero, 1})).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *counter,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::copyDestination,
                               })
                .has_value());
    REQUIRE(device.copyBuffer(*command, {*counterSeed, 0, *counter, 0, sizeof(zero)}).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *counter,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::streamOutput,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::streamOutputWrite,
                               })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *output,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::streamOutput,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::streamOutputWrite,
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
    REQUIRE(device.beginStreamOutputStatisticsQuery(*command, *queryPool, 0).has_value());
    const std::array attachments{rachis::ColorAttachment{.view = *targetView}};
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *pipeline).has_value());
    REQUIRE(
        device
            .setViewport(
                *command,
                {.width = static_cast<float>(targetSize), .height = static_cast<float>(targetSize), .maxDepth = 1.0F})
            .has_value());
    REQUIRE(device.setScissor(*command, {.width = targetSize, .height = targetSize}).has_value());
    const std::array tint{1.0F, 1.0F, 1.0F, 1.0F};
    REQUIRE(device.setGraphicsPushConstants(*command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(tint)))
                .has_value());
    CHECK_FALSE(device
                    .bindStreamOutputBuffers(*command,
                                             0,
                                             std::array{rachis::StreamOutputBufferBinding{
                                                 .buffer = *output,
                                                 .size = outputSize,
                                                 .filledSizeBuffer = *output,
                                             }})
                    .has_value());
    CHECK_FALSE(device.draw(*command, 3).has_value());
    REQUIRE(device
                .bindStreamOutputBuffers(*command,
                                         0,
                                         std::array{rachis::StreamOutputBufferBinding{
                                             .buffer = *output,
                                             .size = outputSize,
                                             .filledSizeBuffer = *counter,
                                         }})
                .has_value());
    REQUIRE(device.draw(*command, 3).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    REQUIRE(device.endStreamOutputStatisticsQuery(*command, 0).has_value());

    for (const auto buffer : {*output, *counter})
    {
        REQUIRE(device
                    .bufferBarrier(*command,
                                   {
                                       .buffer = buffer,
                                       .stagesBefore = rachis::PipelineStage::streamOutput,
                                       .stagesAfter = rachis::PipelineStage::copy,
                                       .accessBefore = rachis::Access::streamOutputWrite,
                                       .accessAfter = rachis::Access::copySource,
                                   })
                    .has_value());
    }
    REQUIRE(device.copyBuffer(*command, {*output, 0, *readback, 0, outputSize}).has_value());
    REQUIRE(device.copyBuffer(*command, {*counter, 0, *readback, counterOffset, sizeof(zero)}).has_value());
    REQUIRE(device
                .resolveStreamOutputStatisticsQueries(*command,
                                                      {
                                                          .queryPool = *queryPool,
                                                          .queryCount = 1,
                                                          .destination = *readback,
                                                          .destinationOffset = statisticsOffset,
                                                      })
                .has_value());

    REQUIRE(device.destroyBuffer(*output).has_value());
    REQUIRE(device.destroyBuffer(*counter).has_value());
    REQUIRE(device.destroyQueryPool(*queryPool).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    std::array<std::array<float, 4>, 3> captured{};
    std::memcpy(captured.data(), mapped->data, outputSize);
    std::uint32_t bytesWritten = 0;
    std::memcpy(&bytesWritten, mapped->data + counterOffset, sizeof(bytesWritten));
    rachis::StreamOutputStatistics statistics{};
    std::memcpy(&statistics, mapped->data + statisticsOffset, sizeof(statistics));
    CHECK((captured[0] == std::array{0.0F, 0.75F, 0.0F, 1.0F}));
    CHECK((captured[1] == std::array{0.75F, -0.75F, 0.0F, 1.0F}));
    CHECK((captured[2] == std::array{-0.75F, -0.75F, 0.0F, 1.0F}));
    CHECK(bytesWritten == outputSize);
    CHECK(statistics.primitivesWritten == 1);
    CHECK(statistics.primitivesStorageNeeded == 1);
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*counterSeed).has_value());
    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*layout).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("capture-only stream output omits raster targets and viewport state",
          "[size:medium][rhi][graphics][stream-output][rasterization-disabled][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).streamOutput);
    REQUIRE(rachis::test::requireCapabilities(device).streamOutputRasterizationDisable);

    const auto vertexBytecode = loadShader(RACHIS_TEST_GRAPHICS_DIRECT_VERTEX_SHADER_PATH);
    REQUIRE_FALSE(vertexBytecode.empty());
    auto vertexShader = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .entryPoint = "directVertex",
        .shaderInterface = {},
    });
    REQUIRE(vertexShader.has_value());
    auto layout = device.createPipelineLayout({});
    REQUIRE(layout.has_value());

    const std::array declarations{rachis::StreamOutputEntry{
        .stream = 0,
        .shaderLocation = 0,
        .startComponent = 0,
        .componentCount = 4,
        .outputSlot = 0,
    }};
    const std::array<std::uint32_t, 1> strides{outputStride};
    CHECK_FALSE(device
                    .createGraphicsPipeline({
                        .layout = *layout,
                        .vertexShader = *vertexShader,
                        .rasterizedStream = std::nullopt,
                    })
                    .has_value());
    auto pipeline = device.createGraphicsPipeline({
        .layout = *layout,
        .vertexShader = *vertexShader,
        .cullMode = rachis::CullMode::none,
        .streamOutputEntries = declarations,
        .streamOutputBufferStrides = strides,
        .rasterizedStream = std::nullopt,
        .label = "Capture-only stream-output contract pipeline",
    });
    REQUIRE(pipeline.has_value());

    auto output = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::streamOutput | rachis::BufferUsage::copySource,
        .label = "Capture-only stream-output data",
    });
    auto counter = device.createBuffer({
        .size = sizeof(std::uint64_t),
        .usage =
            rachis::BufferUsage::streamOutput | rachis::BufferUsage::copyDestination | rachis::BufferUsage::copySource,
        .label = "Capture-only stream-output counter",
    });
    auto counterSeed = device.createBuffer({
        .size = sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto readback = device.createBuffer({
        .size = counterOffset + sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(output.has_value());
    REQUIRE(counter.has_value());
    REQUIRE(counterSeed.has_value());
    REQUIRE(readback.has_value());
    const std::uint32_t zero = 0;
    REQUIRE(device.writeBuffer(*counterSeed, 0, std::as_bytes(std::span{&zero, 1})).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *counter,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::copyDestination,
                               })
                .has_value());
    REQUIRE(device.copyBuffer(*command, {*counterSeed, 0, *counter, 0, sizeof(zero)}).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *counter,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::streamOutput,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::streamOutputWrite,
                               })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *output,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::streamOutput,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::streamOutputWrite,
                               })
                .has_value());
    REQUIRE(device.beginRendering(*command, {}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *pipeline).has_value());
    REQUIRE(device
                .bindStreamOutputBuffers(*command,
                                         0,
                                         std::array{rachis::StreamOutputBufferBinding{
                                             .buffer = *output,
                                             .size = outputSize,
                                             .filledSizeBuffer = *counter,
                                         }})
                .has_value());
    REQUIRE(device.draw(*command, 3).has_value());
    REQUIRE(device.endRendering(*command).has_value());

    for (const auto buffer : {*output, *counter})
    {
        REQUIRE(device
                    .bufferBarrier(*command,
                                   {
                                       .buffer = buffer,
                                       .stagesBefore = rachis::PipelineStage::streamOutput,
                                       .stagesAfter = rachis::PipelineStage::copy,
                                       .accessBefore = rachis::Access::streamOutputWrite,
                                       .accessAfter = rachis::Access::copySource,
                                   })
                    .has_value());
    }
    REQUIRE(device.copyBuffer(*command, {*output, 0, *readback, 0, outputSize}).has_value());
    REQUIRE(device.copyBuffer(*command, {*counter, 0, *readback, counterOffset, sizeof(zero)}).has_value());
    REQUIRE(device.destroyBuffer(*output).has_value());
    REQUIRE(device.destroyBuffer(*counter).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    std::array<std::array<float, 4>, 3> captured{};
    std::memcpy(captured.data(), mapped->data, outputSize);
    std::uint32_t bytesWritten = 0;
    std::memcpy(&bytesWritten, mapped->data + counterOffset, sizeof(bytesWritten));
    CHECK((captured[0] == std::array{0.0F, 0.75F, 0.0F, 1.0F}));
    CHECK((captured[1] == std::array{0.75F, -0.75F, 0.0F, 1.0F}));
    CHECK((captured[2] == std::array{-0.75F, -0.75F, 0.0F, 1.0F}));
    CHECK(bytesWritten == outputSize);
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*counterSeed).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*layout).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("stream-output-statistics queries begin and end on the same side of a rendering-scope boundary",
          "[size:medium][rhi][graphics][stream-output][query][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).streamOutputStatisticsQueries);

    constexpr std::uint32_t queryCount = 2;
    auto streamZero = device.createStreamOutputStatisticsQueryPool({
        .stream = 0,
        .queryCount = queryCount,
        .label = "Rendering-scope stream 0 query",
    });
    auto streamOne = device.createStreamOutputStatisticsQueryPool({
        .stream = 1,
        .queryCount = queryCount,
        .label = "Rendering-scope stream 1 query",
    });
    auto target = device.createTexture({
        .width = targetSize,
        .height = targetSize,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
    });
    auto results = device.createBuffer({
        .size = 2 * queryCount * rachis::streamOutputStatisticsQueryResultSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(streamZero.has_value());
    REQUIRE(streamOne.has_value());
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
    const std::array attachments{rachis::ColorAttachment{.view = *targetView}};
    const auto expectInvalidState = [](const rachis::Status& status) {
        REQUIRE_FALSE(status.has_value());
        CHECK(status.error().code == rachis::ErrorCode::invalidState);
    };

    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
    REQUIRE(device.beginStreamOutputStatisticsQuery(*command, *streamZero, 0).has_value());
    expectInvalidState(device.endRendering(*command));
    REQUIRE(device.endStreamOutputStatisticsQuery(*command, 0).has_value());
    REQUIRE(device.endRendering(*command).has_value());

    REQUIRE(device.beginStreamOutputStatisticsQuery(*command, *streamZero, 1).has_value());
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
    expectInvalidState(device.endStreamOutputStatisticsQuery(*command, 0));
    REQUIRE(device.beginStreamOutputStatisticsQuery(*command, *streamOne, 0).has_value());
    expectInvalidState(device.endRendering(*command));
    REQUIRE(device.endStreamOutputStatisticsQuery(*command, 1).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    REQUIRE(device.endStreamOutputStatisticsQuery(*command, 0).has_value());

    REQUIRE(device.beginStreamOutputStatisticsQuery(*command, *streamOne, 1).has_value());
    REQUIRE(device.endStreamOutputStatisticsQuery(*command, 1).has_value());
    REQUIRE(device
                .resolveStreamOutputStatisticsQueries(*command,
                                                      {
                                                          .queryPool = *streamZero,
                                                          .queryCount = queryCount,
                                                          .destination = *results,
                                                      })
                .has_value());
    REQUIRE(device
                .resolveStreamOutputStatisticsQueries(
                    *command,
                    {
                        .queryPool = *streamOne,
                        .queryCount = queryCount,
                        .destination = *results,
                        .destinationOffset = queryCount * rachis::streamOutputStatisticsQueryResultSize,
                    })
                .has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    REQUIRE(device.destroyBuffer(*results).has_value());
    REQUIRE(device.destroyQueryPool(*streamOne).has_value());
    REQUIRE(device.destroyQueryPool(*streamZero).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
