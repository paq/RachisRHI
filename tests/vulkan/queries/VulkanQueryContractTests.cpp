#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include "Internal/VulkanDevice.h"
#include "Rachis/TestSupport/VulkanHostClock.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <thread>
#include <vector>

namespace
{

    constexpr std::uint32_t targetSize = 64;
    constexpr double calibrationToleranceSeconds = 0.05;
    constexpr std::uint64_t timestampResultBytes = 4 * rachis::timestampQueryResultSize;
    constexpr std::uint64_t occlusionResultBytes = 3 * rachis::occlusionQueryResultSize;
    constexpr std::uint64_t statisticsCount = 2;
    constexpr std::uint64_t statisticsResultBytes =
        statisticsCount * rachis::pipelineStatisticsQueryResultSize + rachis::occlusionQueryResultSize;

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

    [[nodiscard]] bool isZero(const rachis::PipelineStatistics& statistics) noexcept
    {
        return statistics.inputAssemblyVertices == 0 && statistics.inputAssemblyPrimitives == 0 &&
               statistics.vertexShaderInvocations == 0 && statistics.geometryShaderInvocations == 0 &&
               statistics.geometryShaderPrimitives == 0 && statistics.clippingInvocations == 0 &&
               statistics.clippingPrimitives == 0 && statistics.fragmentShaderInvocations == 0 &&
               statistics.tessellationControlShaderPatches == 0 &&
               statistics.tessellationEvaluationShaderInvocations == 0 && statistics.computeShaderInvocations == 0;
    }

    [[nodiscard]] double tickDeltaSeconds(std::uint64_t later, std::uint64_t earlier, std::uint64_t frequency) noexcept
    {
        return static_cast<double>(static_cast<std::int64_t>(later - earlier)) / static_cast<double>(frequency);
    }

} // namespace

TEST_CASE("Vulkan advertises complete GPU query families and keeps incomplete ones unsupported",
          "[size:medium][rhi][vulkan][query][capabilities]")
{
#ifdef __APPLE__
    SUCCEED("Complete query-family advertisement is a Windows/Linux desktop contract.");
    return;
#endif
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = device.capabilities();
    REQUIRE(capabilities.has_value());

    const auto graphics = device.queueProperties(rachis::QueueType::graphics);
    REQUIRE(graphics.timestampQueries);
    REQUIRE(graphics.inlineTimestampWrites);
    REQUIRE(graphics.timestampFrequency > 0);

    REQUIRE(capabilities->occlusionQueries);
    REQUIRE(capabilities->pipelineStatisticsQueries);
    CHECK_FALSE((capabilities->meshPipelineStatisticsQueries && !capabilities->meshShaders));

    auto meshStatistics = device.createPipelineStatisticsQueryPool({
        .mode = rachis::PipelineStatisticsQueryMode::meshShader,
        .queryCount = 1,
    });
    REQUIRE(meshStatistics.has_value() == capabilities->meshPipelineStatisticsQueries);
    if (meshStatistics)
    {
        REQUIRE(device.destroyQueryPool(*meshStatistics));
    }
    else
    {
        CHECK(meshStatistics.error().code == rachis::ErrorCode::unsupported);
    }
}

TEST_CASE("Vulkan timestamps write, reset, resolve, and stamp render-pass bounds",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][query][timestamp]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    auto pool = device.createTimestampQueryPool({
        .queueType = rachis::QueueType::graphics,
        .queryCount = 4,
        .label = "Vulkan timestamp contract pool",
    });
    auto transientPool = device.createTimestampQueryPool({
        .queueType = rachis::QueueType::graphics,
        .queryCount = 2,
        .label = "Vulkan transient timestamp pool",
    });
    auto results = device.createBuffer({
        .size = timestampResultBytes,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan timestamp results",
    });
    auto tooSmall = device.createBuffer({
        .size = rachis::timestampQueryResultSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan timestamp too small",
    });
    auto wrongUsage = device.createBuffer({
        .size = timestampResultBytes,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan timestamp wrong usage",
    });
    auto target = device.createTexture({
        .width = targetSize,
        .height = targetSize,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .label = "Vulkan timestamp target",
    });
    REQUIRE(pool);
    REQUIRE(transientPool);
    REQUIRE(results);
    REQUIRE(tooSmall);
    REQUIRE(wrongUsage);
    REQUIRE(target);
    auto targetView = device.createTextureView({
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    REQUIRE(targetView);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    CHECK_FALSE(device.resetQueryPool(*command, {}, 0, 1).has_value());
    CHECK_FALSE(device.resetQueryPool(*command, *pool, 0, 0).has_value());
    CHECK_FALSE(device.writeTimestamp(*command, *pool, 4).has_value());
    CHECK_FALSE(device
                    .resolveTimestamps(*command,
                                       {
                                           .queryPool = *pool,
                                           .queryCount = 1,
                                           .destination = *results,
                                       })
                    .has_value());

    REQUIRE(device.writeTimestamp(*command, *pool, 0).has_value());
    auto duplicateWrite = device.writeTimestamp(*command, *pool, 0);
    REQUIRE_FALSE(duplicateWrite.has_value());
    CHECK(duplicateWrite.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.resetQueryPool(*command, *pool, 0, 1).has_value());
    REQUIRE(device.writeTimestamp(*command, *pool, 0).has_value());
    REQUIRE(device.writeTimestamp(*command, *pool, 1).has_value());

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
    CHECK_FALSE(device
                    .beginRendering(*command,
                                    {
                                        .timestampWrites =
                                            rachis::PassTimestampWrites{
                                                .queryPool = *pool,
                                                .beginningOfPassWriteIndex = 2,
                                                .endOfPassWriteIndex = 3,
                                            },
                                    })
                    .has_value());
    CHECK_FALSE(device
                    .beginRendering(*command,
                                    {
                                        .colorAttachments = attachments,
                                        .timestampWrites =
                                            rachis::PassTimestampWrites{
                                                .queryPool = *pool,
                                                .beginningOfPassWriteIndex = 2,
                                                .endOfPassWriteIndex = 2,
                                            },
                                    })
                    .has_value());
    REQUIRE(device
                .beginRendering(*command,
                                {
                                    .colorAttachments = attachments,
                                    .timestampWrites =
                                        rachis::PassTimestampWrites{
                                            .queryPool = *pool,
                                            .beginningOfPassWriteIndex = 2,
                                            .endOfPassWriteIndex = 3,
                                        },
                                })
                .has_value());
    auto activeRenderingReset = device.resetQueryPool(*command, *pool, 0, 1);
    REQUIRE_FALSE(activeRenderingReset.has_value());
    CHECK(activeRenderingReset.error().code == rachis::ErrorCode::invalidState);
    CHECK_FALSE(device.writeTimestamp(*command, *pool, 3).has_value());
    REQUIRE(device.endRendering(*command).has_value());

    CHECK_FALSE(device
                    .resolveTimestamps(*command,
                                       {
                                           .queryPool = *pool,
                                           .queryCount = 4,
                                           .destination = *results,
                                           .destinationOffset = 1,
                                       })
                    .has_value());
    CHECK_FALSE(device
                    .resolveTimestamps(*command,
                                       {
                                           .queryPool = *pool,
                                           .queryCount = 4,
                                           .destination = *tooSmall,
                                       })
                    .has_value());
    CHECK_FALSE(device
                    .resolveTimestamps(*command,
                                       {
                                           .queryPool = *pool,
                                           .queryCount = 4,
                                           .destination = *wrongUsage,
                                       })
                    .has_value());
    REQUIRE(device
                .resolveTimestamps(*command,
                                   {
                                       .queryPool = *pool,
                                       .queryCount = 4,
                                       .destination = *results,
                                   })
                .has_value());
    CHECK_FALSE(device
                    .resolveTimestamps(*command,
                                       {
                                           .queryPool = *pool,
                                           .queryCount = 4,
                                           .destination = *results,
                                       })
                    .has_value());

    REQUIRE(device
                .beginRendering(*command,
                                {
                                    .colorAttachments = attachments,
                                    .timestampWrites =
                                        rachis::PassTimestampWrites{
                                            .queryPool = *transientPool,
                                            .endOfPassWriteIndex = 1,
                                        },
                                })
                .has_value());
    REQUIRE(device.destroyQueryPool(*transientPool).has_value());
    REQUIRE(device.endRendering(*command).has_value());

    const auto stalePool = *pool;
    REQUIRE(device.destroyQueryPool(*pool).has_value());
    CHECK_FALSE(device.destroyQueryPool(stalePool).has_value());
    CHECK_FALSE(device.writeTimestamp(*command, stalePool, 0).has_value());
    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted));
    device.collectGarbage();

    auto mapped = device.mapBuffer(*results);
    REQUIRE(mapped);
    std::array<std::uint64_t, 4> timestamps{};
    std::memcpy(timestamps.data(), mapped->data, sizeof(timestamps));
    CAPTURE(timestamps[0], timestamps[1], timestamps[2], timestamps[3]);
#ifdef __APPLE__
    if (timestamps[1] != 0 && timestamps[3] != 0)
    {
        CHECK(timestamps[1] >= timestamps[0]);
        CHECK(timestamps[3] >= timestamps[2]);
    }
#else
    CHECK(timestamps[1] >= timestamps[0]);
    CHECK(timestamps[3] >= timestamps[2]);
#endif
    REQUIRE(device.unmapBuffer(*results));
    REQUIRE(device.destroyBuffer(*wrongUsage));
    REQUIRE(device.destroyBuffer(*tooSmall));
    REQUIRE(device.destroyBuffer(*results));

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan GPU profile scopes pair timestamps with an owned debug hierarchy",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][query][gpu-profile]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    REQUIRE(device.queueProperties(rachis::QueueType::graphics).inlineTimestampWrites);

    auto pool = device.createTimestampQueryPool({
        .queueType = rachis::QueueType::graphics,
        .queryCount = 4,
        .label = "Vulkan GPU profile timestamps",
    });
    REQUIRE(pool);
    auto results = device.createBuffer({
        .size = timestampResultBytes,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan GPU profile results",
    });
    REQUIRE(results);
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device.beginDebugGroup(*command, {.label = "CPU frame annotation"}));

    CHECK_FALSE(device
                    .beginGpuProfileScope(*command,
                                          {
                                              .queryPool = *pool,
                                              .beginningQueryIndex = 0,
                                              .endQueryIndex = 0,
                                              .label = "invalid duplicate",
                                          })
                    .has_value());
    CHECK_FALSE(device
                    .beginGpuProfileScope(*command,
                                          {
                                              .queryPool = *pool,
                                              .beginningQueryIndex = 0,
                                              .endQueryIndex = 3,
                                          })
                    .has_value());

    REQUIRE(device.beginGpuProfileScope(*command,
                                        {
                                            .queryPool = *pool,
                                            .beginningQueryIndex = 0,
                                            .endQueryIndex = 3,
                                            .label = "Frame",
                                            .color =
                                                rachis::DebugColor{
                                                    .red = 0.2F,
                                                    .green = 0.7F,
                                                    .blue = 1.0F,
                                                    .alpha = 1.0F,
                                                },
                                        }));
    auto activeProfileReset = device.resetQueryPool(*command, *pool, 0, 1);
    REQUIRE_FALSE(activeProfileReset.has_value());
    CHECK(activeProfileReset.error().code == rachis::ErrorCode::invalidState);
    CHECK_FALSE(device.endDebugGroup(*command).has_value());
    CHECK_FALSE(device.writeTimestamp(*command, *pool, 3).has_value());

    REQUIRE(device.beginGpuProfileScope(*command,
                                        {
                                            .queryPool = *pool,
                                            .beginningQueryIndex = 1,
                                            .endQueryIndex = 2,
                                            .label = "Lighting",
                                        }));
    REQUIRE(device.endGpuProfileScope(*command));
    REQUIRE(device.beginDebugGroup(*command, {.label = "profile annotation"}));
    CHECK_FALSE(device.endGpuProfileScope(*command).has_value());
    REQUIRE(device.endDebugGroup(*command));

    rachis::Status crossThreadEnd;
    std::jthread otherThread([&] { crossThreadEnd = device.endGpuProfileScope(*command); });
    otherThread.join();
    REQUIRE_FALSE(crossThreadEnd.has_value());
    CHECK(crossThreadEnd.error().code == rachis::ErrorCode::invalidState);

    auto rejectedSubmit = device.submit(std::move(*command));
    REQUIRE_FALSE(rejectedSubmit.has_value());
    CHECK(rejectedSubmit.error().code == rachis::ErrorCode::invalidState);
    CHECK(command->isValid());

    REQUIRE(device.endGpuProfileScope(*command));
    REQUIRE(device.endDebugGroup(*command));
    CHECK_FALSE(device.endGpuProfileScope(*command).has_value());
    REQUIRE(device.resolveTimestamps(*command,
                                     {
                                         .queryPool = *pool,
                                         .queryCount = 4,
                                         .destination = *results,
                                     }));

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted));
    device.collectGarbage();

    auto mapped = device.mapBuffer(*results);
    REQUIRE(mapped);
    std::array<std::uint64_t, 4> timestamps{};
    std::memcpy(timestamps.data(), mapped->data, sizeof(timestamps));
    CAPTURE(timestamps[0], timestamps[1], timestamps[2], timestamps[3]);
#ifdef __APPLE__
    if (timestamps[1] != 0 && timestamps[2] != 0 && timestamps[3] != 0)
    {
        CHECK(timestamps[1] >= timestamps[0]);
        CHECK(timestamps[2] >= timestamps[1]);
        CHECK(timestamps[3] >= timestamps[2]);
    }
#else
    CHECK(timestamps[1] >= timestamps[0]);
    CHECK(timestamps[2] >= timestamps[1]);
    CHECK(timestamps[3] >= timestamps[2]);
#endif
    REQUIRE(device.unmapBuffer(*results));
    REQUIRE(device.destroyQueryPool(*pool));
    REQUIRE(device.destroyBuffer(*results));

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

#ifdef RACHIS_SPIRV_DIR
TEST_CASE("Vulkan occlusion queries resolve precise and binary GPU results",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][query][occlusion]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).occlusionQueries);

    auto precisePool = device.createOcclusionQueryPool({
        .mode = rachis::OcclusionQueryMode::precise,
        .queryCount = 1,
        .label = "Vulkan precise occlusion",
    });
    auto binaryPool = device.createOcclusionQueryPool({
        .mode = rachis::OcclusionQueryMode::binary,
        .queryCount = 2,
        .label = "Vulkan binary occlusion",
    });
    auto timestampPool = device.createTimestampQueryPool({
        .queueType = rachis::QueueType::graphics,
        .queryCount = 1,
    });
    REQUIRE(precisePool);
    REQUIRE(binaryPool);
    REQUIRE(timestampPool);

    const auto vertexBytecode = loadSpirv("VulkanGraphicsTriangle.direct.vs.spv");
    const auto pixelBytecode = loadSpirv("VulkanGraphicsTriangle.ps.spv");
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
    REQUIRE(vertexShader);
    REQUIRE(pixelShader);
    const std::array pushConstants{
        rachis::PushConstantRange{rachis::ShaderStage::pixel, 0, 4 * sizeof(float)},
    };
    auto pipelineLayout = device.createPipelineLayout({
        .pushConstantRanges = pushConstants,
        .label = "Vulkan occlusion layout",
    });
    REQUIRE(pipelineLayout);
    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    auto pipeline = device.createGraphicsPipeline({
        .layout = *pipelineLayout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .label = "Vulkan occlusion pipeline",
    });
    REQUIRE(pipeline);
    auto target = device.createTexture({
        .width = targetSize,
        .height = targetSize,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .label = "Vulkan occlusion target",
    });
    REQUIRE(target);
    auto targetView = device.createTextureView({
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    auto results = device.createBuffer({
        .size = occlusionResultBytes,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan occlusion results",
    });
    REQUIRE(targetView);
    REQUIRE(results);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    CHECK_FALSE(device.beginOcclusionQuery(*command, *precisePool, 0).has_value());
    CHECK_FALSE(device.beginOcclusionQuery(*command, *timestampPool, 0).has_value());

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
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}));
    REQUIRE(device.bindGraphicsPipeline(*command, *pipeline));
    REQUIRE(device.setViewport(*command,
                               {
                                   .width = static_cast<float>(targetSize),
                                   .height = static_cast<float>(targetSize),
                                   .maxDepth = 1.0F,
                               }));
    REQUIRE(device.setScissor(*command, {.width = targetSize, .height = targetSize}));
    const std::array tint{1.0F, 1.0F, 1.0F, 1.0F};
    REQUIRE(device.setGraphicsPushConstants(*command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(tint))));

    REQUIRE(device.beginOcclusionQuery(*command, *precisePool, 0));
    CHECK_FALSE(device.beginOcclusionQuery(*command, *binaryPool, 0).has_value());
    CHECK_FALSE(device.endRendering(*command).has_value());
    REQUIRE(device.draw(*command, 3));
    REQUIRE(device.endOcclusionQuery(*command));
    CHECK_FALSE(device.beginOcclusionQuery(*command, *precisePool, 0).has_value());

    REQUIRE(device.beginOcclusionQuery(*command, *binaryPool, 0));
    REQUIRE(device.draw(*command, 3));
    REQUIRE(device.endOcclusionQuery(*command));
    REQUIRE(device.beginOcclusionQuery(*command, *binaryPool, 1));
    REQUIRE(device.endOcclusionQuery(*command));
    REQUIRE(device.endRendering(*command));

    REQUIRE(device.resolveOcclusionQueries(*command,
                                           {
                                               .queryPool = *precisePool,
                                               .queryCount = 1,
                                               .destination = *results,
                                           }));
    REQUIRE(device.resolveOcclusionQueries(*command,
                                           {
                                               .queryPool = *binaryPool,
                                               .queryCount = 2,
                                               .destination = *results,
                                               .destinationOffset = rachis::occlusionQueryResultSize,
                                           }));

    REQUIRE(device.destroyQueryPool(*precisePool));
    REQUIRE(device.destroyQueryPool(*binaryPool));
    REQUIRE(device.destroyQueryPool(*timestampPool));
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted));
    device.collectGarbage();

    auto mapped = device.mapBuffer(*results);
    REQUIRE(mapped);
    std::array<std::uint64_t, 3> queryResults{};
    std::memcpy(queryResults.data(), mapped->data, sizeof(queryResults));
    CAPTURE(queryResults[0], queryResults[1], queryResults[2]);
    CHECK(queryResults[0] > 0);
    CHECK(queryResults[1] != 0);
    CHECK(queryResults[2] == 0);
    REQUIRE(device.unmapBuffer(*results));
    REQUIRE(device.destroyBuffer(*results));
    REQUIRE(device.destroyTextureView(*targetView));
    REQUIRE(device.destroyTexture(*target));
    REQUIRE(device.destroyPipeline(*pipeline));
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout));
    REQUIRE(device.destroyShaderModule(*pixelShader));
    REQUIRE(device.destroyShaderModule(*vertexShader));

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan pipeline-statistics queries resolve portable GPU counters",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][query][pipeline-statistics]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    if (rachis::test::skipWithoutCapability(rachis::test::requireCapabilities(device).pipelineStatisticsQueries,
                                            "pipelineStatisticsQueries"))
    {
        return;
    }

    auto pool = device.createPipelineStatisticsQueryPool({
        .queryCount = statisticsCount,
        .label = "Vulkan pipeline-statistics pool",
    });
    auto occlusionPool = device.createOcclusionQueryPool({
        .mode = rachis::OcclusionQueryMode::binary,
        .queryCount = 1,
        .label = "Vulkan nested occlusion pool",
    });
    REQUIRE(pool);
    REQUIRE(occlusionPool);

    const auto vertexBytecode = loadSpirv("VulkanGraphicsTriangle.direct.vs.spv");
    const auto pixelBytecode = loadSpirv("VulkanGraphicsTriangle.ps.spv");
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
    REQUIRE(vertexShader);
    REQUIRE(pixelShader);
    const std::array pushConstants{
        rachis::PushConstantRange{rachis::ShaderStage::pixel, 0, 4 * sizeof(float)},
    };
    auto pipelineLayout = device.createPipelineLayout({
        .pushConstantRanges = pushConstants,
        .label = "Vulkan pipeline-statistics layout",
    });
    REQUIRE(pipelineLayout);
    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    auto pipeline = device.createGraphicsPipeline({
        .layout = *pipelineLayout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .label = "Vulkan pipeline-statistics pipeline",
    });
    REQUIRE(pipeline);
    auto target = device.createTexture({
        .width = targetSize,
        .height = targetSize,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .label = "Vulkan pipeline-statistics target",
    });
    REQUIRE(target);
    auto targetView = device.createTextureView({
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    auto results = device.createBuffer({
        .size = statisticsResultBytes,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan pipeline-statistics results",
    });
    REQUIRE(targetView);
    REQUIRE(results);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device.beginPipelineStatisticsQuery(*command, *pool, 0));
    auto activeStatisticsReset = device.resetQueryPool(*command, *pool, 0, 1);
    REQUIRE_FALSE(activeStatisticsReset.has_value());
    CHECK(activeStatisticsReset.error().code == rachis::ErrorCode::invalidState);
    CHECK_FALSE(device.beginPipelineStatisticsQuery(*command, *pool, 1).has_value());

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
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}));
    REQUIRE(device.bindGraphicsPipeline(*command, *pipeline));
    REQUIRE(device.setViewport(*command,
                               {
                                   .width = static_cast<float>(targetSize),
                                   .height = static_cast<float>(targetSize),
                                   .maxDepth = 1.0F,
                               }));
    REQUIRE(device.setScissor(*command, {.width = targetSize, .height = targetSize}));
    const std::array tint{1.0F, 1.0F, 1.0F, 1.0F};
    REQUIRE(device.setGraphicsPushConstants(*command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(tint))));
    REQUIRE(device.beginOcclusionQuery(*command, *occlusionPool, 0));
    REQUIRE(device.draw(*command, 3));
    REQUIRE(device.endOcclusionQuery(*command));
    REQUIRE(device.endRendering(*command));
    REQUIRE(device.endPipelineStatisticsQuery(*command));
    CHECK_FALSE(device.beginPipelineStatisticsQuery(*command, *pool, 0).has_value());

    REQUIRE(device.beginPipelineStatisticsQuery(*command, *pool, 1));
    REQUIRE(device.endPipelineStatisticsQuery(*command));
    REQUIRE(device.resolvePipelineStatisticsQueries(*command,
                                                    {
                                                        .queryPool = *pool,
                                                        .queryCount = statisticsCount,
                                                        .destination = *results,
                                                    }));
    REQUIRE(device.resolveOcclusionQueries(
        *command,
        {
            .queryPool = *occlusionPool,
            .queryCount = 1,
            .destination = *results,
            .destinationOffset = statisticsCount * rachis::pipelineStatisticsQueryResultSize,
        }));

    REQUIRE(device.destroyQueryPool(*pool));
    REQUIRE(device.destroyQueryPool(*occlusionPool));
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted));
    device.collectGarbage();

    auto mapped = device.mapBuffer(*results);
    REQUIRE(mapped);
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
    CHECK(occlusion != 0);
    REQUIRE(device.unmapBuffer(*results));
    REQUIRE(device.destroyBuffer(*results));
    REQUIRE(device.destroyTextureView(*targetView));
    REQUIRE(device.destroyTexture(*target));
    REQUIRE(device.destroyPipeline(*pipeline));
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout));
    REQUIRE(device.destroyShaderModule(*pixelShader));
    REQUIRE(device.destroyShaderModule(*vertexShader));

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
#endif

TEST_CASE("Vulkan pipeline-statistics queries begin and end on the same side of a rendering-scope boundary",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][query][pipeline-statistics]")
{
    auto desc = vulkanDesc();
    desc.requireValidation = true;
    auto created = rachis::createDevice(desc);
    if (!created && created.error().code == rachis::ErrorCode::validationError)
    {
        SKIP("The rendering-scope query contract needs the Khronos validation layer.");
    }
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    if (rachis::test::skipWithoutCapability(rachis::test::requireCapabilities(device).pipelineStatisticsQueries,
                                            "pipelineStatisticsQueries"))
    {
        return;
    }

    constexpr std::uint32_t queryCount = 3;
    auto pool = device.createPipelineStatisticsQueryPool({
        .queryCount = queryCount,
        .label = "Vulkan rendering-scope statistics pool",
    });
    auto target = device.createTexture({
        .width = targetSize,
        .height = targetSize,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .label = "Vulkan rendering-scope statistics target",
    });
    auto results = device.createBuffer({
        .size = queryCount * rachis::pipelineStatisticsQueryResultSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan rendering-scope statistics results",
    });
    REQUIRE(pool);
    REQUIRE(target);
    REQUIRE(results);
    auto targetView = device.createTextureView({
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    REQUIRE(targetView);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device.textureBarrier(*command,
                                  {
                                      .texture = *target,
                                      .stagesBefore = rachis::PipelineStage::all,
                                      .stagesAfter = rachis::PipelineStage::renderTarget,
                                      .accessAfter = rachis::Access::renderTarget,
                                      .layoutBefore = rachis::TextureLayout::undefined,
                                      .layoutAfter = rachis::TextureLayout::renderTarget,
                                  }));
    const rachis::TextureBarrier betweenPasses{
        .texture = *target,
        .stagesBefore = rachis::PipelineStage::renderTarget,
        .stagesAfter = rachis::PipelineStage::renderTarget,
        .accessBefore = rachis::Access::renderTarget,
        .accessAfter = rachis::Access::renderTarget,
        .layoutBefore = rachis::TextureLayout::renderTarget,
        .layoutAfter = rachis::TextureLayout::renderTarget,
    };
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

    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}));
    REQUIRE(device.beginPipelineStatisticsQuery(*command, *pool, 0));
    expectInvalidState(device.endRendering(*command));
    REQUIRE(device.endPipelineStatisticsQuery(*command));
    REQUIRE(device.endRendering(*command));

    REQUIRE(device.textureBarrier(*command, betweenPasses));
    REQUIRE(device.beginPipelineStatisticsQuery(*command, *pool, 1));
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}));
    expectInvalidState(device.endPipelineStatisticsQuery(*command));
    REQUIRE(device.endRendering(*command));
    REQUIRE(device.endPipelineStatisticsQuery(*command));

    REQUIRE(device.beginPipelineStatisticsQuery(*command, *pool, 2));
    REQUIRE(device.endPipelineStatisticsQuery(*command));
    REQUIRE(device.resolvePipelineStatisticsQueries(*command,
                                                    {
                                                        .queryPool = *pool,
                                                        .queryCount = queryCount,
                                                        .destination = *results,
                                                    }));
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted));
    device.collectGarbage();

    REQUIRE(device.destroyTextureView(*targetView));
    REQUIRE(device.destroyTexture(*target));
    REQUIRE(device.destroyBuffer(*results));
    REQUIRE(device.destroyQueryPool(*pool));

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan rejects queries inside multiview rendering and writes pass timestamps outside it",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][query][multiview]")
{
    auto desc = vulkanDesc();
    desc.requireValidation = true;
    auto created = rachis::createDevice(desc);
    if (!created && created.error().code == rachis::ErrorCode::validationError)
    {
        SKIP("The multiview query contract needs the Khronos validation layer.");
    }
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (rachis::test::skipWithoutCapability(capabilities.multiviewRendering && capabilities.maxMultiviewViewCount >= 2,
                                            "multiviewRendering with two views"))
    {
        return;
    }

    constexpr std::uint32_t twoViewMask = 0b11;
    constexpr std::uint64_t timestampOffset = 0;
    constexpr std::uint64_t passOffset = timestampOffset + rachis::timestampQueryResultSize;
    constexpr std::uint64_t scopeOffset = passOffset + 2 * rachis::timestampQueryResultSize;
    constexpr std::uint64_t occlusionOffset = scopeOffset + 4 * rachis::timestampQueryResultSize;
    constexpr std::uint64_t statisticsOffset = occlusionOffset + rachis::occlusionQueryResultSize;
    constexpr std::uint64_t multiviewResultBytes = statisticsOffset + rachis::pipelineStatisticsQueryResultSize;

    auto timestampPool = device.createTimestampQueryPool({
        .queueType = rachis::QueueType::graphics,
        .queryCount = 1,
        .label = "Vulkan multiview timestamp pool",
    });
    auto passPool = device.createTimestampQueryPool({
        .queueType = rachis::QueueType::graphics,
        .queryCount = 2,
        .label = "Vulkan multiview pass timestamp pool",
    });
    auto scopePool = device.createTimestampQueryPool({
        .queueType = rachis::QueueType::graphics,
        .queryCount = 4,
        .label = "Vulkan multiview profile pool",
    });
    auto results = device.createBuffer({
        .size = multiviewResultBytes,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan multiview query results",
    });
    auto target = device.createTexture({
        .width = targetSize,
        .height = targetSize,
        .depthOrArrayLayers = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .label = "Vulkan multiview query target",
    });
    REQUIRE(timestampPool);
    REQUIRE(passPool);
    REQUIRE(scopePool);
    REQUIRE(results);
    REQUIRE(target);
    auto targetView = device.createTextureView({
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
        .dimension = rachis::TextureViewDimension::texture2DArray,
        .arrayLayerCount = 2,
    });
    REQUIRE(targetView);
    std::optional<rachis::QueryPoolHandle> occlusionPool;
    if (capabilities.occlusionQueries)
    {
        auto pool = device.createOcclusionQueryPool({
            .mode = rachis::OcclusionQueryMode::precise,
            .queryCount = 1,
            .label = "Vulkan multiview occlusion pool",
        });
        REQUIRE(pool);
        occlusionPool = *pool;
    }
    std::optional<rachis::QueryPoolHandle> statisticsPool;
    if (capabilities.pipelineStatisticsQueries)
    {
        auto pool = device.createPipelineStatisticsQueryPool({
            .queryCount = 1,
            .label = "Vulkan multiview pipeline-statistics pool",
        });
        REQUIRE(pool);
        statisticsPool = *pool;
    }

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device.textureBarrier(*command,
                                  {
                                      .texture = *target,
                                      .stagesBefore = rachis::PipelineStage::all,
                                      .stagesAfter = rachis::PipelineStage::renderTarget,
                                      .accessAfter = rachis::Access::renderTarget,
                                      .layoutBefore = rachis::TextureLayout::undefined,
                                      .layoutAfter = rachis::TextureLayout::renderTarget,
                                  }));
    const std::array attachments{
        rachis::ColorAttachment{
            .view = *targetView,
            .loadOperation = rachis::LoadOperation::clear,
            .storeOperation = rachis::StoreOperation::store,
        },
    };
    const auto expectUnsupported = [](const rachis::Status& status) {
        REQUIRE_FALSE(status.has_value());
        CHECK(status.error().code == rachis::ErrorCode::unsupported);
    };

    REQUIRE(device.beginGpuProfileScope(*command,
                                        {
                                            .queryPool = *scopePool,
                                            .beginningQueryIndex = 0,
                                            .endQueryIndex = 1,
                                            .label = "Outer multiview frame",
                                        }));
    REQUIRE(device.beginRendering(*command,
                                  {
                                      .colorAttachments = attachments,
                                      .timestampWrites =
                                          rachis::PassTimestampWrites{
                                              .queryPool = *passPool,
                                              .beginningOfPassWriteIndex = 1,
                                              .endOfPassWriteIndex = 0,
                                          },
                                      .viewMask = twoViewMask,
                                  }));
    REQUIRE(device.diagnosticCounts().errors == 0);
    expectUnsupported(device.writeTimestamp(*command, *timestampPool, 0));
    expectUnsupported(device.beginGpuProfileScope(*command,
                                                  {
                                                      .queryPool = *scopePool,
                                                      .beginningQueryIndex = 2,
                                                      .endQueryIndex = 3,
                                                      .label = "Inner multiview pass",
                                                  }));
    expectUnsupported(device.endGpuProfileScope(*command));
    if (occlusionPool)
    {
        expectUnsupported(device.beginOcclusionQuery(*command, *occlusionPool, 0));
        CHECK_FALSE(device.endOcclusionQuery(*command).has_value());
    }
    if (statisticsPool)
    {
        expectUnsupported(device.beginPipelineStatisticsQuery(*command, *statisticsPool, 0));
        CHECK_FALSE(device.endPipelineStatisticsQuery(*command).has_value());
    }
    REQUIRE(device.endRendering(*command));
    REQUIRE(device.endGpuProfileScope(*command));
    REQUIRE(device.writeTimestamp(*command, *timestampPool, 0));

    REQUIRE(device.textureBarrier(*command,
                                  {
                                      .texture = *target,
                                      .stagesBefore = rachis::PipelineStage::renderTarget,
                                      .stagesAfter = rachis::PipelineStage::renderTarget,
                                      .accessBefore = rachis::Access::renderTarget,
                                      .accessAfter = rachis::Access::renderTarget,
                                      .layoutBefore = rachis::TextureLayout::renderTarget,
                                      .layoutAfter = rachis::TextureLayout::renderTarget,
                                  }));
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments, .viewMask = 0b01}));
    REQUIRE(device.beginGpuProfileScope(*command,
                                        {
                                            .queryPool = *scopePool,
                                            .beginningQueryIndex = 2,
                                            .endQueryIndex = 3,
                                            .label = "Single-view pass",
                                        }));
    if (occlusionPool)
    {
        REQUIRE(device.beginOcclusionQuery(*command, *occlusionPool, 0));
        REQUIRE(device.endOcclusionQuery(*command));
    }
    if (statisticsPool)
    {
        REQUIRE(device.beginPipelineStatisticsQuery(*command, *statisticsPool, 0));
        REQUIRE(device.endPipelineStatisticsQuery(*command));
    }
    REQUIRE(device.endGpuProfileScope(*command));
    REQUIRE(device.endRendering(*command));

    REQUIRE(device.resolveTimestamps(*command,
                                     {
                                         .queryPool = *timestampPool,
                                         .queryCount = 1,
                                         .destination = *results,
                                         .destinationOffset = timestampOffset,
                                     }));
    REQUIRE(device.resolveTimestamps(*command,
                                     {
                                         .queryPool = *passPool,
                                         .queryCount = 2,
                                         .destination = *results,
                                         .destinationOffset = passOffset,
                                     }));
    REQUIRE(device.resolveTimestamps(*command,
                                     {
                                         .queryPool = *scopePool,
                                         .queryCount = 4,
                                         .destination = *results,
                                         .destinationOffset = scopeOffset,
                                     }));
    if (occlusionPool)
    {
        REQUIRE(device.resolveOcclusionQueries(*command,
                                               {
                                                   .queryPool = *occlusionPool,
                                                   .queryCount = 1,
                                                   .destination = *results,
                                                   .destinationOffset = occlusionOffset,
                                               }));
    }
    if (statisticsPool)
    {
        REQUIRE(device.resolvePipelineStatisticsQueries(*command,
                                                        {
                                                            .queryPool = *statisticsPool,
                                                            .queryCount = 1,
                                                            .destination = *results,
                                                            .destinationOffset = statisticsOffset,
                                                        }));
    }
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted));
    device.collectGarbage();

    auto mapped = device.mapBuffer(*results);
    REQUIRE(mapped);
    std::array<std::uint64_t, 2> passStamps{};
    std::memcpy(passStamps.data(), mapped->data + passOffset, sizeof(passStamps));
    std::array<std::uint64_t, 4> scopeStamps{};
    std::memcpy(scopeStamps.data(), mapped->data + scopeOffset, sizeof(scopeStamps));
    CAPTURE(passStamps[0], passStamps[1], scopeStamps[0], scopeStamps[1], scopeStamps[2], scopeStamps[3]);
#ifdef __APPLE__
    if (passStamps[0] != 0 && passStamps[1] != 0 && scopeStamps[1] != 0 && scopeStamps[3] != 0)
    {
        CHECK(passStamps[0] >= passStamps[1]);
        CHECK(scopeStamps[1] >= scopeStamps[0]);
        CHECK(scopeStamps[3] >= scopeStamps[2]);
    }
#else
    CHECK(passStamps[0] >= passStamps[1]);
    CHECK(scopeStamps[1] >= scopeStamps[0]);
    CHECK(scopeStamps[3] >= scopeStamps[2]);
#endif
    REQUIRE(device.unmapBuffer(*results));

    if (statisticsPool)
    {
        REQUIRE(device.destroyQueryPool(*statisticsPool));
    }
    if (occlusionPool)
    {
        REQUIRE(device.destroyQueryPool(*occlusionPool));
    }
    REQUIRE(device.destroyTextureView(*targetView));
    REQUIRE(device.destroyTexture(*target));
    REQUIRE(device.destroyBuffer(*results));
    REQUIRE(device.destroyQueryPool(*scopePool));
    REQUIRE(device.destroyQueryPool(*passPool));
    REQUIRE(device.destroyQueryPool(*timestampPool));

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan queue clock calibration pairs the device clock with the platform host clock on every queue",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][query][calibration]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    REQUIRE(device.queueProperties(rachis::QueueType::graphics).timestampQueries);

    const auto domains = rachis::vulkan::queryVulkanCalibrateableTimeDomains();
    const bool calibrateable = std::ranges::find(domains, VK_TIME_DOMAIN_DEVICE_EXT) != domains.end() &&
                               std::ranges::find(domains, rachis::test::vulkanHostTimeDomain()) != domains.end();
    CAPTURE(domains.size(), calibrateable);

    constexpr std::array queueTypes{
        rachis::QueueType::graphics,
        rachis::QueueType::compute,
        rachis::QueueType::copy,
    };
    for (const auto queueType : queueTypes)
    {
        const auto properties = device.queueProperties(queueType);
        const bool calibrated = calibrateable && properties.timestampQueries;
        CAPTURE(static_cast<int>(queueType), properties.timestampQueries, calibrated);

        const auto hostBefore = rachis::test::readVulkanHostClock();
        auto before = device.calibrateQueueClock(queueType);
        const auto hostAfterBefore = rachis::test::readVulkanHostClock();
        REQUIRE(before);
        REQUIRE(hostBefore.frequency > 0);
        CAPTURE(hostBefore.timestamp, hostAfterBefore.timestamp, before->gpuTimestamp, before->cpuTimestamp);
        CHECK(before->gpuFrequency == properties.timestampFrequency);
        CHECK(before->cpuFrequency == hostBefore.frequency);
        CHECK(before->cpuTimestamp >= hostBefore.timestamp);
        CHECK(before->cpuTimestamp <= hostAfterBefore.timestamp);
        if (!calibrated)
        {
            CHECK(before->gpuTimestamp == 0);
            continue;
        }
        REQUIRE(before->gpuTimestamp != 0);

        auto pool = device.createTimestampQueryPool({
            .queueType = queueType,
            .queryCount = 1,
            .label = "Vulkan calibrated timestamp pool",
        });
        REQUIRE(pool);
        auto results = device.createBuffer({
            .size = rachis::timestampQueryResultSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Vulkan calibrated timestamp result",
        });
        REQUIRE(results);
        auto command = rachis::test::acquireCommandBuffer(device, queueType);
        REQUIRE(command);
        REQUIRE(device.writeTimestamp(*command, *pool, 0));
        REQUIRE(device.resolveTimestamps(*command,
                                         {
                                             .queryPool = *pool,
                                             .queryCount = 1,
                                             .destination = *results,
                                         }));
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted);
        REQUIRE(device.wait(*submitted));

        auto mapped = device.mapBuffer(*results);
        REQUIRE(mapped);
        std::uint64_t resolvedTimestamp = 0;
        std::memcpy(&resolvedTimestamp, mapped->data, sizeof(resolvedTimestamp));
        REQUIRE(device.unmapBuffer(*results));

        const auto hostBeforeAfter = rachis::test::readVulkanHostClock();
        auto after = device.calibrateQueueClock(queueType);
        const auto hostAfter = rachis::test::readVulkanHostClock();
        REQUIRE(after);
        CAPTURE(resolvedTimestamp, after->gpuTimestamp, after->cpuTimestamp);
        REQUIRE(after->gpuTimestamp != 0);
        CHECK(after->cpuTimestamp >= hostBeforeAfter.timestamp);
        CHECK(after->cpuTimestamp <= hostAfter.timestamp);
        CHECK(after->cpuTimestamp >= before->cpuTimestamp);
        CHECK(after->gpuFrequency == before->gpuFrequency);
        CHECK(after->cpuFrequency == before->cpuFrequency);
#ifndef __APPLE__
        CHECK(after->gpuTimestamp >= before->gpuTimestamp);
        CHECK(resolvedTimestamp >= before->gpuTimestamp);
        CHECK(resolvedTimestamp <= after->gpuTimestamp);
        const double gpuElapsed = tickDeltaSeconds(after->gpuTimestamp, before->gpuTimestamp, before->gpuFrequency);
        const double cpuElapsed = tickDeltaSeconds(after->cpuTimestamp, before->cpuTimestamp, before->cpuFrequency);
        const double resolvedOffset = tickDeltaSeconds(resolvedTimestamp, before->gpuTimestamp, before->gpuFrequency);
        CAPTURE(gpuElapsed, cpuElapsed, resolvedOffset);
        CHECK(std::abs(gpuElapsed - cpuElapsed) <= calibrationToleranceSeconds);
        CHECK(resolvedOffset >= -calibrationToleranceSeconds);
        CHECK(resolvedOffset <= cpuElapsed + calibrationToleranceSeconds);
#endif

        REQUIRE(device.destroyQueryPool(*pool));
        REQUIRE(device.destroyBuffer(*results));
    }

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
