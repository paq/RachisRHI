#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

#if RACHIS_ENABLE_METAL
    constexpr std::uint64_t bufferSize = 256;
    constexpr std::uint32_t textureWidth = 4;
    constexpr std::uint32_t textureHeight = 4;

    [[nodiscard]] rachis::DeviceDesc metalDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
    }

    [[nodiscard]] std::byte pattern(std::uint64_t index) noexcept
    {
        return static_cast<std::byte>((index * 29U + 7U) & 0xFFU);
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

    [[nodiscard]] std::filesystem::path pnmsPath(std::string_view metallibName)
    {
        return std::filesystem::path(RACHIS_METALLIB_DIR) /
               std::filesystem::path(metallibName).replace_extension(".pnms");
    }
#endif

} // namespace

#if RACHIS_ENABLE_METAL

TEST_CASE("Metal barrier batches validate atomically and reject illegal scopes",
          "[size:large][rhi][metal][barrier][batch]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    const auto checkedCapabilities = device.capabilities();
    REQUIRE(checkedCapabilities.has_value());
    REQUIRE(checkedCapabilities->enhancedBarriers);
    REQUIRE(checkedCapabilities->bufferBarrierRanges);

    auto first = device.createBuffer(rachis::BufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .label = "Metal barrier first",
    });
    auto second = device.createBuffer(rachis::BufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .label = "Metal barrier second",
    });
    auto texture = device.createTexture(rachis::TextureDesc{
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage =
            rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination | rachis::TextureUsage::sampled,
        .label = "Metal barrier texture",
    });
    auto stale = device.createBuffer(rachis::BufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
    });
    REQUIRE(first);
    REQUIRE(second);
    REQUIRE(texture);
    REQUIRE(stale);
    const auto staleHandle = *stale;
    REQUIRE(device.destroyBuffer(*stale).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(command);
    CHECK_FALSE(device.resourceBarriers(*command, {}).has_value());

    const std::array initialBuffers{
        rachis::BufferBarrier{
            .buffer = *first,
            .stagesBefore = rachis::PipelineStage::all,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::common,
            .accessAfter = rachis::Access::copyDestination,
        },
        rachis::BufferBarrier{
            .buffer = *second,
            .stagesBefore = rachis::PipelineStage::all,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::common,
            .accessAfter = rachis::Access::copyDestination,
        },
    };
    const std::array initialTextures{
        rachis::TextureBarrier{
            .texture = *texture,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessAfter = rachis::Access::copyDestination,
            .layoutBefore = rachis::TextureLayout::undefined,
            .layoutAfter = rachis::TextureLayout::copyDestination,
        },
    };

    const std::array duplicate{initialBuffers[0], initialBuffers[0]};
    CHECK_FALSE(device.resourceBarriers(*command, {.bufferBarriers = duplicate}).has_value());
    const std::array overlapping{initialTextures[0], initialTextures[0]};
    CHECK_FALSE(device.resourceBarriers(*command, {.textureBarriers = overlapping}).has_value());

    auto mismatched = initialTextures[0];
    mismatched.accessAfter = rachis::Access::copySource;
    CHECK_FALSE(device.resourceBarriers(*command, {.textureBarriers = std::array{mismatched}}).has_value());

    auto unknownStage = initialBuffers[0];
    unknownStage.stagesAfter = static_cast<rachis::PipelineStage>(1U << 31U);
    CHECK_FALSE(device.resourceBarriers(*command, {.bufferBarriers = std::array{unknownStage}}).has_value());

    auto invalidCommon = rachis::GlobalBarrier{
        .stagesBefore = rachis::PipelineStage::all,
        .stagesAfter = rachis::PipelineStage::copy,
        .accessBefore = rachis::Access::common,
        .accessAfter = rachis::Access::copyDestination,
    };
    CHECK_FALSE(device.resourceBarriers(*command, {.globalBarriers = std::array{invalidCommon}}).has_value());

    auto unknownMemory = rachis::GlobalBarrier{
        .stagesBefore = rachis::PipelineStage::copy,
        .stagesAfter = static_cast<rachis::PipelineStage>(1U << 31U),
        .accessBefore = rachis::Access::copyDestination,
        .accessAfter = rachis::Access::copyDestination,
    };
    CHECK_FALSE(device
                    .resourceBarriers(*command,
                                      {
                                          .globalBarriers = std::array{unknownMemory},
                                          .bufferBarriers = initialBuffers,
                                          .textureBarriers = initialTextures,
                                      })
                    .has_value());

    const std::array staleBatch{
        initialBuffers[0],
        rachis::BufferBarrier{
            .buffer = staleHandle,
            .stagesBefore = rachis::PipelineStage::all,
            .stagesAfter = rachis::PipelineStage::copy,
            .accessBefore = rachis::Access::common,
            .accessAfter = rachis::Access::copyDestination,
        },
    };
    CHECK_FALSE(device.resourceBarriers(*command, {.bufferBarriers = staleBatch}).has_value());

    auto samplerFeedback =
        device.resourceBarriers(*command, {.samplerFeedbackBarriers = std::array{rachis::SamplerFeedbackBarrier{}}});
    REQUIRE_FALSE(samplerFeedback);
    CHECK(samplerFeedback.error().code == rachis::ErrorCode::unsupported);

    REQUIRE(device
                .resourceBarriers(*command,
                                  {
                                      .bufferBarriers = initialBuffers,
                                      .textureBarriers = initialTextures,
                                  })
                .has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal buffer barrier ranges are accepted and do not publish on failure",
          "[size:large][rhi][metal][barrier][range]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    const auto checkedCapabilities = device.capabilities();
    REQUIRE(checkedCapabilities.has_value());
    CHECK(checkedCapabilities->bufferBarrierRanges);

    auto buffer = device.createBuffer(rachis::BufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .label = "Metal ranged buffer",
    });
    REQUIRE(buffer);
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(command);

    auto emptyRange = device.bufferBarrier(*command,
                                           rachis::BufferBarrier{
                                               .buffer = *buffer,
                                               .stagesAfter = rachis::PipelineStage::copy,
                                               .accessAfter = rachis::Access::copyDestination,
                                               .offset = 16,
                                               .size = 0,
                                           });
    REQUIRE_FALSE(emptyRange);
    CHECK(emptyRange.error().code == rachis::ErrorCode::invalidArgument);

    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *buffer,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessAfter = rachis::Access::copyDestination,
                                   .offset = 16,
                                   .size = 64,
                               })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *buffer,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::copySource,
                                   .offset = 16,
                                   .size = 64,
                               })
                .has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal producer-consumer copies complete after explicit stage barriers",
          "[size:large][rhi][metal][barrier][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    auto upload = device.createBuffer(rachis::BufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal barrier upload",
    });
    auto intermediate = device.createBuffer(rachis::BufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Metal barrier intermediate",
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal barrier readback",
    });
    REQUIRE(upload);
    REQUIRE(intermediate);
    REQUIRE(readback);

    auto mappedUpload = device.mapBuffer(*upload);
    REQUIRE(mappedUpload);
    for (std::uint64_t index = 0; index < bufferSize; ++index)
    {
        mappedUpload->data[index] = pattern(index);
    }
    REQUIRE(device.unmapBuffer(*upload).has_value());

    auto producer = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(producer);
    REQUIRE(device
                .copyBuffer(*producer,
                            rachis::BufferCopy{
                                .source = *upload,
                                .destination = *intermediate,
                                .size = bufferSize,
                            })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*producer,
                               rachis::BufferBarrier{
                                   .buffer = *intermediate,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    auto copyPoint = device.submit(std::move(*producer));
    REQUIRE(copyPoint);

    auto consumer = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(consumer);
    REQUIRE(device
                .bufferBarrier(*consumer,
                               rachis::BufferBarrier{
                                   .buffer = *intermediate,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::copySource,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device
                .copyBuffer(*consumer,
                            rachis::BufferCopy{
                                .source = *intermediate,
                                .destination = *readback,
                                .size = bufferSize,
                            })
                .has_value());
    auto graphicsPoint = device.submit(std::move(*consumer), std::array{*copyPoint});
    REQUIRE(graphicsPoint);
    REQUIRE(device.wait(*graphicsPoint).has_value());
    device.collectGarbage();

    auto mappedReadback = device.mapBuffer(*readback);
    REQUIRE(mappedReadback);
    std::uint64_t firstMismatch = bufferSize;
    for (std::uint64_t index = 0; index < bufferSize; ++index)
    {
        if (mappedReadback->data[index] != pattern(index))
        {
            firstMismatch = index;
            break;
        }
    }
    CAPTURE(firstMismatch);
    CHECK(firstMismatch == bufferSize);
    REQUIRE(device.unmapBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal inter-pass buffer and texture barriers flush at render and compute encoder open",
          "[size:large][rhi][metal][barrier][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    struct MeshVertex final
    {
        float position[2];
        float color[3];
    };
    const std::array vertices{
        MeshVertex{{0.0F, 0.75F}, {1.0F, 0.0F, 0.0F}},
        MeshVertex{{-0.75F, -0.75F}, {0.0F, 1.0F, 0.0F}},
        MeshVertex{{0.75F, -0.75F}, {0.0F, 0.0F, 1.0F}},
    };

    auto uploadVertices = device.createBuffer({
        .size = sizeof(vertices),
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Inter-pass upload vertices",
    });
    auto deviceVertices = device.createBuffer({
        .size = sizeof(vertices),
        .usage = rachis::BufferUsage::vertex | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Inter-pass device vertices",
    });
    REQUIRE(uploadVertices);
    REQUIRE(deviceVertices);
    REQUIRE(device.writeBuffer(*uploadVertices, 0, std::as_bytes(std::span{vertices})).has_value());

    constexpr std::uint32_t renderSize = 32;
    auto target = device.createTexture({
        .width = renderSize,
        .height = renderSize,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource,
        .label = "Inter-pass render target",
    });
    REQUIRE(target);
    auto targetView = device.createTextureView({
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
        .label = "Inter-pass render view",
    });
    REQUIRE(targetView);

    const rachis::BufferTextureCopyRegion region{
        .extent = {.width = renderSize, .height = renderSize, .depth = 1},
    };
    const auto layoutInfo = device.queryTextureTransferLayout(*target, region);
    REQUIRE(layoutInfo);
    const rachis::BufferTextureCopyRegion copyRegion{
        .bytesPerRow = layoutInfo->bytesPerRow,
        .rowsPerImage = layoutInfo->rowsPerImage,
        .extent = {.width = renderSize, .height = renderSize, .depth = 1},
    };
    auto colorReadback = device.createBuffer({
        .size = layoutInfo->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Inter-pass color readback",
    });
    REQUIRE(colorReadback);

    const auto vertexVs = loadBytes(pnmsPath("MetalRasterVertex.vs.metallib"));
    const auto vertexPs = loadBytes(pnmsPath("MetalRasterVertex.ps.metallib"));
    REQUIRE_FALSE(vertexVs.empty());
    REQUIRE_FALSE(vertexPs.empty());
    const std::array vertexLocations{0U, 1U};
    auto vsModule = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexVs,
        .entryPoint = "vertexMain",
        .shaderInterface = rachis::test::shaderInterface({}, {}, vertexLocations),
    });
    auto psModule = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = vertexPs,
        .entryPoint = "pixelMain",
    });
    REQUIRE(vsModule);
    REQUIRE(psModule);

    auto pipeLayout = device.createPipelineLayout({});
    REQUIRE(pipeLayout);
    const std::array meshAttributes{
        rachis::VertexAttribute{rachis::VertexFormat::float32x2, 0, 0},
        rachis::VertexAttribute{rachis::VertexFormat::float32x3, 8, 1},
    };
    const std::array meshLayouts{rachis::VertexBufferLayout{
        .stride = sizeof(MeshVertex),
        .attributes = meshAttributes,
    }};
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    auto pipeline = device.createGraphicsPipeline({
        .layout = *pipeLayout,
        .vertexShader = *vsModule,
        .pixelShader = *psModule,
        .vertexBuffers = meshLayouts,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .label = "Inter-pass raster pipeline",
    });
    REQUIRE(pipeline);

    auto queryPool = device.createTimestampQueryPool({
        .queueType = rachis::QueueType::graphics,
        .queryCount = 1,
        .label = "Inter-pass encoder break pool",
    });
    REQUIRE(queryPool);

    auto cmd = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(cmd);
    REQUIRE(
        device.copyBuffer(*cmd, {.source = *uploadVertices, .destination = *deviceVertices, .size = sizeof(vertices)})
            .has_value());

    REQUIRE(device.writeTimestamp(*cmd, *queryPool, 0).has_value());

    const std::array preRenderBuffers{
        rachis::BufferBarrier{
            .buffer = *deviceVertices,
            .stagesBefore = rachis::PipelineStage::copy,
            .stagesAfter = rachis::PipelineStage::vertexInput | rachis::PipelineStage::vertexShader,
            .accessBefore = rachis::Access::copyDestination,
            .accessAfter = rachis::Access::vertexBuffer,
        },
    };
    const std::array preRenderTextures{
        rachis::TextureBarrier{
            .texture = *target,
            .stagesAfter = rachis::PipelineStage::renderTarget,
            .accessAfter = rachis::Access::renderTarget,
            .layoutAfter = rachis::TextureLayout::renderTarget,
        },
    };
    REQUIRE(device.resourceBarriers(*cmd, {.bufferBarriers = preRenderBuffers, .textureBarriers = preRenderTextures})
                .has_value());

    const std::array attachments{
        rachis::ColorAttachment{
            .view = *targetView,
            .loadOperation = rachis::LoadOperation::clear,
            .storeOperation = rachis::StoreOperation::store,
        },
    };
    REQUIRE(device.beginRendering(*cmd, {.colorAttachments = attachments}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*cmd, *pipeline).has_value());
    REQUIRE(
        device.setViewport(*cmd, {.width = static_cast<float>(renderSize), .height = static_cast<float>(renderSize)})
            .has_value());
    REQUIRE(device.setScissor(*cmd, {.width = renderSize, .height = renderSize}).has_value());
    REQUIRE(device.bindVertexBuffer(*cmd, 0, *deviceVertices, 0).has_value());
    REQUIRE(device.draw(*cmd, 3, 1, 0, 0).has_value());
    REQUIRE(device.endRendering(*cmd).has_value());

    REQUIRE(device
                .textureBarrier(*cmd,
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
    REQUIRE(device.copyTextureToBuffer(*cmd, *target, *colorReadback, std::array{copyRegion}).has_value());

    auto submitted = device.submit(std::move(*cmd));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());

    auto mapped = device.mapBuffer(*colorReadback);
    REQUIRE(mapped);
    std::uint32_t colored = 0;
    for (std::uint32_t y = 0; y < renderSize; ++y)
    {
        for (std::uint32_t x = 0; x < renderSize; ++x)
        {
            const auto* pixel =
                mapped->data + static_cast<std::size_t>(y) * layoutInfo->bytesPerRow + static_cast<std::size_t>(x) * 4;
            if (std::to_integer<std::uint8_t>(pixel[0]) != 0 || std::to_integer<std::uint8_t>(pixel[1]) != 0 ||
                std::to_integer<std::uint8_t>(pixel[2]) != 0)
            {
                ++colored;
            }
        }
    }
    REQUIRE(device.unmapBuffer(*colorReadback).has_value());
    CHECK(colored > 100);

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Metal deferred barriers flush across batched command buffers and intervening copy encoders",
          "[size:large][rhi][metal][barrier][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    constexpr std::uint32_t renderSize = 16;
    auto target = device.createTexture({
        .width = renderSize,
        .height = renderSize,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::sampled | rachis::TextureUsage::copySource,
        .label = "Batch/compute producer render target",
    });
    REQUIRE(target);
    auto targetRenderView = device.createTextureView({
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
        .label = "Batch/compute render target view",
    });
    auto targetSampledView = device.createTextureView({
        .texture = *target,
        .usage = rachis::TextureViewUsage::sampled,
        .label = "Batch/compute sampled view",
    });
    REQUIRE(targetRenderView);
    REQUIRE(targetSampledView);

    const rachis::BufferTextureCopyRegion region{
        .extent = {.width = renderSize, .height = renderSize, .depth = 1},
    };
    const auto layoutInfo = device.queryTextureTransferLayout(*target, region);
    REQUIRE(layoutInfo);
    const rachis::BufferTextureCopyRegion copyRegion{
        .bytesPerRow = layoutInfo->bytesPerRow,
        .rowsPerImage = layoutInfo->rowsPerImage,
        .extent = {.width = renderSize, .height = renderSize, .depth = 1},
    };
    auto colorReadback = device.createBuffer({
        .size = layoutInfo->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Batch deferred barrier readback",
    });
    REQUIRE(colorReadback);

    auto cmdA = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    auto cmdB = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(cmdA);
    REQUIRE(cmdB);

    REQUIRE(device
                .textureBarrier(*cmdA,
                                {
                                    .texture = *target,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array batchAttachments{
        rachis::ColorAttachment{
            .view = *targetRenderView,
            .loadOperation = rachis::LoadOperation::clear,
            .storeOperation = rachis::StoreOperation::store,
            .clearValue = {0.2F, 0.4F, 0.8F, 1.0F},
        },
    };
    REQUIRE(device.beginRendering(*cmdA, {.colorAttachments = batchAttachments}).has_value());
    REQUIRE(device.endRendering(*cmdA).has_value());
    REQUIRE(device
                .textureBarrier(*cmdA,
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
    REQUIRE(device.copyTextureToBuffer(*cmdB, *target, *colorReadback, std::array{copyRegion}).has_value());

    std::array batch{std::move(*cmdA), std::move(*cmdB)};
    auto batchSubmitted = device.submit(batch);
    REQUIRE(batchSubmitted);
    REQUIRE(device.wait(*batchSubmitted).has_value());

    {
        auto mapped = device.mapBuffer(*colorReadback);
        REQUIRE(mapped);
        CHECK(std::to_integer<std::uint8_t>(mapped->data[0]) == 51);
        CHECK(std::to_integer<std::uint8_t>(mapped->data[1]) == 102);
        CHECK(std::to_integer<std::uint8_t>(mapped->data[2]) == 204);
        CHECK(std::to_integer<std::uint8_t>(mapped->data[3]) == 255);
        REQUIRE(device.unmapBuffer(*colorReadback).has_value());
    }

    const auto computePnms = loadBytes(pnmsPath("MetalRemapContract.single.cs.metallib"));
    REQUIRE_FALSE(computePnms.empty());

    const std::array computeBindings{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{1, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{2, rachis::BindingType::sampler, rachis::ShaderStage::compute, 1},
        rachis::BindingLayoutEntry{3, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    auto setLayout = device.createBindingLayout({.entries = computeBindings, .label = "Barrier compute layout"});
    REQUIRE(setLayout);
    const std::array layouts{*setLayout};
    const std::array pushRanges{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16}};
    auto computePipeLayout = device.createPipelineLayout({
        .bindingLayouts = layouts,
        .pushConstantRanges = pushRanges,
        .label = "Barrier compute pipe layout",
    });
    REQUIRE(computePipeLayout);

    const std::array shaderBindings{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 1},
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::sampledTexture, 1},
        rachis::ShaderBindingRequirement{0, 2, rachis::ShaderResourceType::sampler, 1},
        rachis::ShaderBindingRequirement{0, 3, rachis::ShaderResourceType::storageBuffer, 1},
    };
    const std::array shaderPush{rachis::ShaderPushConstantRequirement{0, 16}};
    auto computeModule = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = computePnms,
        .entryPoint = "pushAndBindingZeroCompute",
        .shaderInterface =
            {
                .bindings = shaderBindings,
                .pushConstants = shaderPush,
                .workgroupSize = rachis::ShaderWorkgroupSize{1, 1, 1},
            },
    });
    REQUIRE(computeModule);
    auto computePipeline = device.createComputePipeline({
        .layout = *computePipeLayout,
        .computeShader = *computeModule,
        .label = "Barrier compute consumer pipeline",
    });
    REQUIRE(computePipeline);

    const std::array<std::uint32_t, 6> uboWords{1, 2, 3, 4, 5, 6};
    auto ubo = device.createBuffer({
        .size = sizeof(uboWords),
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::upload,
        .label = "Barrier compute UBO",
    });
    auto scratchUpload = device.createBuffer({
        .size = 64,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Barrier intervening upload",
    });
    auto scratchDevice = device.createBuffer({
        .size = 64,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Barrier intervening device",
    });
    auto computeOut = device.createBuffer({
        .size = 8 * sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Barrier compute output",
    });
    auto computeReadback = device.createBuffer({
        .size = 8 * sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Barrier compute readback",
    });
    auto sampler = device.createSampler({.label = "Barrier compute sampler"});
    REQUIRE(ubo);
    REQUIRE(scratchUpload);
    REQUIRE(scratchDevice);
    REQUIRE(computeOut);
    REQUIRE(computeReadback);
    REQUIRE(sampler);
    REQUIRE(device.writeBuffer(*ubo, 0, std::as_bytes(std::span{uboWords})).has_value());

    const std::array writes{
        rachis::BindingWrite{0, 0, rachis::BufferBinding{*ubo, 0, sizeof(uboWords)}},
        rachis::BindingWrite{1, 0, *targetSampledView},
        rachis::BindingWrite{2, 0, *sampler},
        rachis::BindingWrite{3, 0, rachis::BufferBinding{*computeOut, 0, 8 * sizeof(std::uint32_t)}},
    };
    auto set = rachis::test::createBindingSet(device,
                                              {
                                                  .layout = *setLayout,
                                                  .writes = writes,
                                                  .label = "Barrier compute set",
                                              });
    REQUIRE(set);

    auto cmd2 = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(cmd2);
    REQUIRE(device
                .textureBarrier(*cmd2,
                                {
                                    .texture = *target,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessBefore = rachis::Access::copySource,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutBefore = rachis::TextureLayout::copySource,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array computeClearAttachments{
        rachis::ColorAttachment{
            .view = *targetRenderView,
            .loadOperation = rachis::LoadOperation::clear,
            .storeOperation = rachis::StoreOperation::store,
            .clearValue = {1.0F, 0.5F, 0.25F, 1.0F},
        },
    };
    REQUIRE(device.beginRendering(*cmd2, {.colorAttachments = computeClearAttachments}).has_value());
    REQUIRE(device.endRendering(*cmd2).has_value());

    REQUIRE(
        device.copyBuffer(*cmd2, {.source = *scratchUpload, .destination = *scratchDevice, .size = 64}).has_value());
    REQUIRE(device
                .textureBarrier(*cmd2,
                                {
                                    .texture = *target,
                                    .stagesBefore = rachis::PipelineStage::renderTarget,
                                    .stagesAfter = rachis::PipelineStage::computeShader,
                                    .accessBefore = rachis::Access::renderTarget,
                                    .accessAfter = rachis::Access::shaderResource,
                                    .layoutBefore = rachis::TextureLayout::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::shaderResource,
                                })
                .has_value());
    const std::array<std::uint32_t, 4> pushData{0, 0, 0, 0};
    REQUIRE(device.bindComputePipeline(*cmd2, *computePipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*cmd2, 0, *set).has_value());
    REQUIRE(device.setComputePushConstants(*cmd2, 0, std::as_bytes(std::span{pushData})).has_value());
    REQUIRE(device.dispatch(*cmd2, 1, 1, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*cmd2,
                               {
                                   .buffer = *computeOut,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device
                .copyBuffer(*cmd2,
                            {.source = *computeOut, .destination = *computeReadback, .size = 8 * sizeof(std::uint32_t)})
                .has_value());

    auto computeSubmitted = device.submit(std::move(*cmd2));
    REQUIRE(computeSubmitted);
    REQUIRE(device.wait(*computeSubmitted).has_value());

    {
        auto mapped = device.mapBuffer(*computeReadback);
        REQUIRE(mapped);
        std::array<std::uint32_t, 8> outWords{};
        std::memcpy(outWords.data(), mapped->data, sizeof(outWords));
        REQUIRE(device.unmapBuffer(*computeReadback).has_value());
        CHECK(outWords[4] == 255U);
        CHECK((outWords[5] >= 127U && outWords[5] <= 128U));
        CHECK((outWords[6] >= 63U && outWords[6] <= 64U));
    }

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

#endif
