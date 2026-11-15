#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#if RACHIS_ENABLE_METAL && defined(RACHIS_TESTING)
#include "Metal/MetalTestHooks.h"
#endif

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string_view>
#include <thread>
#include <vector>

namespace
{

    constexpr std::uint32_t targetSize = 64;
    constexpr std::uint32_t marker = 0x5150U;

    [[nodiscard]] rachis::DeviceDesc metalDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
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

    [[nodiscard]] std::filesystem::path pnmsPath(std::string_view metallibName)
    {
        return std::filesystem::path(RACHIS_METALLIB_DIR) /
               std::filesystem::path(metallibName).replace_extension(".pnms");
    }

    [[nodiscard]] rachis::PipelineHandle makePushCompute(rachis::Device& device,
                                                         rachis::BindingSetHandle& set,
                                                         rachis::BufferHandle storage)
    {
        const auto bytecode = loadBytes(pnmsPath("MetalPushConstant.cs.metallib"));
        REQUIRE_FALSE(bytecode.empty());
        const std::array shaderBindings{
            rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
        };
        const std::array shaderPush{rachis::ShaderPushConstantRequirement{0, 16}};
        auto shader = device.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = bytecode,
            .entryPoint = "computeMain",
            .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, shaderBindings, shaderPush),
        });
        REQUIRE(shader);
        const std::array layoutEntries{
            rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
        };
        auto bindingLayout = device.createBindingLayout({.entries = layoutEntries});
        REQUIRE(bindingLayout);
        auto createdSet = rachis::test::createBindingSet(
            device,
            {.layout = *bindingLayout,
             .writes = std::array{rachis::BindingWrite{
                 1, 0, rachis::BufferBinding{storage, 0, sizeof(std::uint32_t), sizeof(std::uint32_t)}}}});
        REQUIRE(createdSet);
        set = *createdSet;
        auto pipelineLayout = device.createPipelineLayout({
            .bindingLayouts = std::array{*bindingLayout},
            .pushConstantRanges = std::array{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16}},
        });
        REQUIRE(pipelineLayout);
        auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *shader});
        REQUIRE(pipeline);
        return *pipeline;
    }

    [[nodiscard]] std::uint64_t readU64(rachis::Device& device, rachis::BufferHandle buffer, std::uint64_t offset)
    {
        std::uint64_t value = 0;
        REQUIRE(device.readBuffer(buffer, offset, std::as_writable_bytes(std::span{&value, 1})).has_value());
        return value;
    }

    [[nodiscard]] std::uint32_t readU32(rachis::Device& device, rachis::BufferHandle buffer)
    {
        std::uint32_t value = 0;
        REQUIRE(device.readBuffer(buffer, 0, std::as_writable_bytes(std::span{&value, 1})).has_value());
        return value;
    }

} // namespace

TEST_CASE("Metal advertises timestamp clocks and single-record indirect only as complete GPU contracts",
          "[size:large][rhi][metal][query]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    for (const auto queueType : {rachis::QueueType::graphics, rachis::QueueType::compute, rachis::QueueType::copy})
    {
        const auto properties = device.queueProperties(queueType);
        REQUIRE(properties.timestampQueries);
        REQUIRE(properties.inlineTimestampWrites);
        REQUIRE(properties.timestampFrequency > 0);
        auto calibration = device.calibrateQueueClock(queueType);
        REQUIRE(calibration.has_value());
        CHECK(calibration->gpuFrequency == properties.timestampFrequency);
        CHECK(calibration->cpuFrequency > 0);
    }
}

TEST_CASE("Metal queue clock calibration tracks host time and brackets resolved GPU timestamps",
          "[size:large][rhi][metal][query][gpu][calibration]")
{
    using namespace std::chrono_literals;
    constexpr double clockToleranceSeconds = 0.002;

    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto properties = device.queueProperties(rachis::QueueType::graphics);
    REQUIRE(properties.timestampQueries);
    REQUIRE(properties.timestampFrequency > 0);

    auto pool = device.createTimestampQueryPool({
        .queueType = rachis::QueueType::graphics,
        .queryCount = 1,
        .label = "Metal calibrated timestamp pool",
    });
    REQUIRE(pool);
    auto results = device.createBuffer({
        .size = rachis::timestampQueryResultSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal calibrated timestamp result",
    });
    REQUIRE(results);

    const auto hostBeforeStart = std::chrono::steady_clock::now();
    auto before = device.calibrateQueueClock(rachis::QueueType::graphics);
    const auto hostBeforeEnd = std::chrono::steady_clock::now();
    REQUIRE(before);
    REQUIRE(before->gpuFrequency == properties.timestampFrequency);
    REQUIRE(before->cpuFrequency > 0);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device.resetQueryPool(*command, *pool, 0, 1).has_value());
    REQUIRE(device.writeTimestamp(*command, *pool, 0).has_value());
    REQUIRE(device
                .resolveTimestamps(*command,
                                   {
                                       .queryPool = *pool,
                                       .queryCount = 1,
                                       .destination = *results,
                                   })
                .has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    std::this_thread::sleep_for(200ms);

    const auto hostAfterStart = std::chrono::steady_clock::now();
    auto after = device.calibrateQueueClock(rachis::QueueType::graphics);
    const auto hostAfterEnd = std::chrono::steady_clock::now();
    REQUIRE(after);
    REQUIRE(after->gpuFrequency == before->gpuFrequency);
    REQUIRE(after->cpuFrequency == before->cpuFrequency);
    REQUIRE(after->cpuTimestamp >= before->cpuTimestamp);
    REQUIRE(after->gpuTimestamp >= before->gpuTimestamp);

    const auto resolvedTimestamp = readU64(device, *results, 0);
    const auto shortestHostSeconds = std::chrono::duration<double>(hostAfterStart - hostBeforeEnd).count();
    const auto longestHostSeconds = std::chrono::duration<double>(hostAfterEnd - hostBeforeStart).count();
    const auto cpuSeconds =
        static_cast<double>(after->cpuTimestamp - before->cpuTimestamp) / static_cast<double>(after->cpuFrequency);
    const auto gpuSeconds =
        static_cast<double>(after->gpuTimestamp - before->gpuTimestamp) / static_cast<double>(after->gpuFrequency);
    CAPTURE(before->gpuTimestamp,
            resolvedTimestamp,
            after->gpuTimestamp,
            before->cpuTimestamp,
            after->cpuTimestamp,
            after->gpuFrequency,
            after->cpuFrequency,
            shortestHostSeconds,
            longestHostSeconds,
            cpuSeconds,
            gpuSeconds);
    CHECK(cpuSeconds >= shortestHostSeconds - clockToleranceSeconds);
    CHECK(cpuSeconds <= longestHostSeconds + clockToleranceSeconds);
    CHECK(gpuSeconds >= shortestHostSeconds - clockToleranceSeconds);
    CHECK(gpuSeconds <= longestHostSeconds + clockToleranceSeconds);
    REQUIRE(resolvedTimestamp != 0);
    CHECK(resolvedTimestamp >= before->gpuTimestamp);
    CHECK(resolvedTimestamp <= after->gpuTimestamp);

    REQUIRE(device.destroyQueryPool(*pool).has_value());
    REQUIRE(device.destroyBuffer(*results).has_value());
}

TEST_CASE("Metal timestamp writes, GPU profile scopes, and single-record indirect dispatch match GPU readback",
          "[size:large][rhi][metal][query][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    auto storage = device.createBuffer({
        .size = sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Metal query storage",
    });
    auto readback = device.createBuffer({
        .size = sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal query storage readback",
    });
    REQUIRE(storage);
    REQUIRE(readback);
    rachis::BindingSetHandle set{};
    auto pipeline = makePushCompute(device, set, *storage);

    auto pool = device.createTimestampQueryPool({
        .queueType = rachis::QueueType::compute,
        .queryCount = 4,
        .label = "Metal timestamp pool",
    });
    REQUIRE(pool);
    auto timestamps = device.createBuffer({
        .size = 4 * rachis::timestampQueryResultSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal timestamp readback",
    });
    REQUIRE(timestamps);

    const rachis::DispatchIndirectArguments groups{1, 1, 1};
    auto arguments = device.createBuffer({
        .size = sizeof(groups),
        .usage = rachis::BufferUsage::indirect | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal dispatch arguments",
    });
    REQUIRE(arguments);
    REQUIRE(device.writeBuffer(*arguments, 0, std::as_bytes(std::span{&groups, 1})).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command);
    REQUIRE(device.resetQueryPool(*command, *pool, 0, 4).has_value());
    REQUIRE(device.writeTimestamp(*command, *pool, 0).has_value());
    REQUIRE(device.bindComputePipeline(*command, pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, set).has_value());
    const std::array<std::uint32_t, 4> constants{marker, 0, 0, 0};
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span{constants})).has_value());
    REQUIRE(device
                .dispatchIndirect(*command,
                                  {
                                      .argumentBuffer = *arguments,
                                  })
                .has_value());
    REQUIRE(device.writeTimestamp(*command, *pool, 1).has_value());
    REQUIRE(device
                .beginGpuProfileScope(*command,
                                      {
                                          .queryPool = *pool,
                                          .beginningQueryIndex = 2,
                                          .endQueryIndex = 3,
                                          .label = "Metal profile",
                                      })
                .has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
    REQUIRE(device.endGpuProfileScope(*command).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *storage,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device.copyBuffer(*command, {.source = *storage, .destination = *readback, .size = sizeof(std::uint32_t)})
                .has_value());
    REQUIRE(device
                .resolveTimestamps(*command,
                                   {
                                       .queryPool = *pool,
                                       .queryCount = 4,
                                       .destination = *timestamps,
                                   })
                .has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());

    CHECK(readU32(device, *readback) == marker);
    const auto t0 = readU64(device, *timestamps, 0);
    const auto t1 = readU64(device, *timestamps, rachis::timestampQueryResultSize);
    const auto t2 = readU64(device, *timestamps, 2 * rachis::timestampQueryResultSize);
    const auto t3 = readU64(device, *timestamps, 3 * rachis::timestampQueryResultSize);
    INFO("t0=" << t0 << " t1=" << t1 << " t2=" << t2 << " t3=" << t3);
    REQUIRE(t0 != 0);
    REQUIRE(t1 != 0);
    REQUIRE(t2 != 0);
    REQUIRE(t3 != 0);
    CHECK(t1 >= t0);
    CHECK(t3 >= t2);
}

TEST_CASE("Metal single-record indirect draws write Apple GPU color and fail closed for counted batches",
          "[size:large][rhi][metal][query][gpu][draw]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    auto texture = device.createTexture({
        .width = targetSize,
        .height = targetSize,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource,
        .label = "Metal indirect color",
    });
    REQUIRE(texture);
    auto view = device.createTextureView({
        .texture = *texture,
        .usage = rachis::TextureViewUsage::renderTarget,
        .dimension = rachis::TextureViewDimension::texture2D,
    });
    REQUIRE(view);
    const rachis::BufferTextureCopyRegion region{
        .extent = {.width = targetSize, .height = targetSize, .depth = 1},
    };
    auto layoutInfo = device.queryTextureTransferLayout(*texture, region);
    REQUIRE(layoutInfo);
    auto colorReadback = device.createBuffer({
        .size = layoutInfo->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal indirect color readback",
    });
    REQUIRE(colorReadback);

    auto pipelineLayout = device.createPipelineLayout({});
    REQUIRE(pipelineLayout);
    const auto triangleVs = loadBytes(pnmsPath("BootstrapTriangle.vs.metallib"));
    const auto trianglePs = loadBytes(pnmsPath("BootstrapTriangle.ps.metallib"));
    REQUIRE_FALSE(triangleVs.empty());
    REQUIRE_FALSE(trianglePs.empty());
    auto vertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = triangleVs,
        .entryPoint = "vertexMain",
    });
    auto pixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = trianglePs,
        .entryPoint = "pixelMain",
    });
    REQUIRE(vertex);
    REQUIRE(pixel);
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    auto pipeline = device.createGraphicsPipeline({
        .layout = *pipelineLayout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .cullMode = rachis::CullMode::back,
        .colorTargets = colorTargets,
        .label = "Metal indirect triangle",
    });
    REQUIRE(pipeline);

    const rachis::DrawIndirectArguments draw{3, 1, 0, 0};
    auto drawArgs = device.createBuffer({
        .size = sizeof(draw),
        .usage = rachis::BufferUsage::indirect | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal draw arguments",
    });
    REQUIRE(drawArgs);
    REQUIRE(device.writeBuffer(*drawArgs, 0, std::as_bytes(std::span{&draw, 1})).has_value());

    auto pool = device.createTimestampQueryPool({
        .queueType = rachis::QueueType::graphics,
        .queryCount = 2,
        .label = "Metal pass timestamps",
    });
    REQUIRE(pool);
    auto timestamps = device.createBuffer({
        .size = 2 * rachis::timestampQueryResultSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal pass timestamp readback",
    });
    REQUIRE(timestamps);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *texture,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array colorAttachments{rachis::ColorAttachment{
        .view = *view,
        .loadOperation = rachis::LoadOperation::clear,
        .storeOperation = rachis::StoreOperation::store,
    }};
    REQUIRE(device
                .beginRendering(*command,
                                {
                                    .colorAttachments = colorAttachments,
                                    .timestampWrites =
                                        rachis::PassTimestampWrites{
                                            .queryPool = *pool,
                                            .beginningOfPassWriteIndex = 0,
                                            .endOfPassWriteIndex = 1,
                                        },
                                })
                .has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *pipeline).has_value());
    REQUIRE(device
                .setViewports(*command,
                              std::array{rachis::Viewport{
                                  .width = static_cast<float>(targetSize),
                                  .height = static_cast<float>(targetSize),
                              }})
                .has_value());
    REQUIRE(device
                .setScissors(*command,
                             std::array{rachis::ScissorRect{
                                 .width = targetSize,
                                 .height = targetSize,
                             }})
                .has_value());
    REQUIRE(device.drawIndirect(*command, {.argumentBuffer = *drawArgs}).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::renderTarget,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::renderTarget,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    const rachis::BufferTextureCopyRegion copyRegion{
        .bytesPerRow = layoutInfo->bytesPerRow,
        .rowsPerImage = layoutInfo->rowsPerImage,
        .extent = {.width = targetSize, .height = targetSize, .depth = 1},
    };
    REQUIRE(device.copyTextureToBuffer(*command, *texture, *colorReadback, std::array{copyRegion}).has_value());
    REQUIRE(device
                .resolveTimestamps(*command,
                                   {
                                       .queryPool = *pool,
                                       .queryCount = 2,
                                       .destination = *timestamps,
                                   })
                .has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());

    auto mapped = device.mapBuffer(*colorReadback);
    REQUIRE(mapped);
    std::size_t colored = 0;
    for (std::uint32_t y = 0; y < targetSize; ++y)
    {
        for (std::uint32_t x = 0; x < targetSize; ++x)
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
    CHECK(colored > 200);

    const auto beginTick = readU64(device, *timestamps, 0);
    const auto endTick = readU64(device, *timestamps, rachis::timestampQueryResultSize);
    INFO("pass begin=" << beginTick << " end=" << endTick);
    REQUIRE(beginTick != 0);
    REQUIRE(endTick != 0);
    CHECK(endTick >= beginTick);

    const rachis::DrawIndexedIndirectArguments indexed{3, 1, 0, 0, 0};
    auto indexedArgs = device.createBuffer({
        .size = sizeof(indexed),
        .usage = rachis::BufferUsage::indirect | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal indexed arguments",
    });
    REQUIRE(indexedArgs);
    REQUIRE(device.writeBuffer(*indexedArgs, 0, std::as_bytes(std::span{&indexed, 1})).has_value());
    struct MeshVertex final
    {
        float position[2];
        float color[3];
    };
    const std::array vertices{
        MeshVertex{{0.0F, 0.72F}, {1.0F, 0.0F, 0.0F}},
        MeshVertex{{-0.72F, -0.72F}, {0.0F, 1.0F, 0.0F}},
        MeshVertex{{0.72F, -0.72F}, {0.0F, 0.0F, 1.0F}},
    };
    const std::array<std::uint16_t, 3> indices{0, 1, 2};
    auto vertexBuffer = device.createBuffer({
        .size = sizeof(vertices),
        .usage = rachis::BufferUsage::vertex | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal vertices",
    });
    auto indexBuffer = device.createBuffer({
        .size = sizeof(indices),
        .usage = rachis::BufferUsage::index | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal indices",
    });
    REQUIRE(vertexBuffer);
    REQUIRE(indexBuffer);
    REQUIRE(device.writeBuffer(*vertexBuffer, 0, std::as_bytes(std::span{vertices})).has_value());
    REQUIRE(device.writeBuffer(*indexBuffer, 0, std::as_bytes(std::span{indices})).has_value());

    const auto vertexVs = loadBytes(pnmsPath("MetalRasterVertex.vs.metallib"));
    const auto vertexPs = loadBytes(pnmsPath("MetalRasterVertex.ps.metallib"));
    REQUIRE_FALSE(vertexVs.empty());
    REQUIRE_FALSE(vertexPs.empty());
    const std::array vertexLocations{0U, 1U};
    auto meshVertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexVs,
        .entryPoint = "vertexMain",
        .shaderInterface = rachis::test::shaderInterface({}, {}, vertexLocations),
    });
    auto meshPixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = vertexPs,
        .entryPoint = "pixelMain",
    });
    REQUIRE(meshVertex);
    REQUIRE(meshPixel);
    const std::array meshAttributes{
        rachis::VertexAttribute{rachis::VertexFormat::float32x2, 0, 0},
        rachis::VertexAttribute{rachis::VertexFormat::float32x3, 8, 1},
    };
    const std::array meshLayouts{rachis::VertexBufferLayout{
        .stride = 20,
        .attributes = meshAttributes,
    }};
    auto meshPipeline = device.createGraphicsPipeline({
        .layout = *pipelineLayout,
        .vertexShader = *meshVertex,
        .pixelShader = *meshPixel,
        .vertexBuffers = meshLayouts,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .label = "Metal indexed pipeline",
    });
    REQUIRE(meshPipeline);

    command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *texture,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    REQUIRE(device.beginRendering(*command, {.colorAttachments = colorAttachments}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *meshPipeline).has_value());
    REQUIRE(device
                .setViewports(*command,
                              std::array{rachis::Viewport{
                                  .width = static_cast<float>(targetSize),
                                  .height = static_cast<float>(targetSize),
                              }})
                .has_value());
    REQUIRE(device
                .setScissors(*command,
                             std::array{rachis::ScissorRect{
                                 .width = targetSize,
                                 .height = targetSize,
                             }})
                .has_value());
    REQUIRE(device.bindVertexBuffers(*command, 0, std::array{rachis::VertexBufferBinding{*vertexBuffer}}).has_value());
    REQUIRE(device.bindIndexBuffer(*command, *indexBuffer, rachis::IndexFormat::uint16, 0).has_value());
    REQUIRE(device.drawIndexedIndirect(*command, {.argumentBuffer = *indexedArgs}).has_value());
    auto counted = device.drawIndexedIndirect(*command, {.argumentBuffer = *indexedArgs, .maxDrawCount = 2});
    REQUIRE_FALSE(counted);
    CHECK(counted.error().code == rachis::ErrorCode::unsupported);
    REQUIRE(device.endRendering(*command).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::renderTarget,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::renderTarget,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device.copyTextureToBuffer(*command, *texture, *colorReadback, std::array{copyRegion}).has_value());
    submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    mapped = device.mapBuffer(*colorReadback);
    REQUIRE(mapped);
    colored = 0;
    for (std::uint32_t y = 0; y < targetSize; ++y)
    {
        for (std::uint32_t x = 0; x < targetSize; ++x)
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
    CHECK(colored > 200);
}

TEST_CASE("Metal rejects occlusion, pipeline statistics, predicates, and counted indirect execution",
          "[size:large][rhi][metal][query][unsupported]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    auto occlusion = device.createOcclusionQueryPool({.queryCount = 1});
    REQUIRE_FALSE(occlusion);
    CHECK(occlusion.error().code == rachis::ErrorCode::unsupported);

    auto stats = device.createPipelineStatisticsQueryPool({.queryCount = 1});
    REQUIRE_FALSE(stats);
    CHECK(stats.error().code == rachis::ErrorCode::unsupported);

    auto stream = device.createStreamOutputStatisticsQueryPool({.queryCount = 1});
    REQUIRE_FALSE(stream);
    CHECK(stream.error().code == rachis::ErrorCode::unsupported);

    auto predicate = device.createBuffer({
        .size = rachis::predicateValueSize,
        .usage = rachis::BufferUsage::predicate,
        .label = "Metal predicate",
    });
    REQUIRE_FALSE(predicate);
    CHECK(predicate.error().code == rachis::ErrorCode::unsupported);

    auto countBuffer = device.createBuffer({
        .size = sizeof(std::uint32_t),
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal count buffer",
    });
    REQUIRE(countBuffer);
    auto arguments = device.createBuffer({
        .size = sizeof(rachis::DispatchIndirectArguments),
        .usage = rachis::BufferUsage::indirect | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal rejected arguments",
    });
    REQUIRE(arguments);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command);
    auto countedDispatch = device.dispatchIndirect(*command,
                                                   {
                                                       .argumentBuffer = *arguments,
                                                       .maxDispatchCount = 2,
                                                   });
    REQUIRE_FALSE(countedDispatch);
    CHECK(countedDispatch.error().code == rachis::ErrorCode::unsupported);
    auto countedWithBuffer = device.dispatchIndirect(*command,
                                                     {
                                                         .argumentBuffer = *arguments,
                                                         .countBuffer = *countBuffer,
                                                     });
    REQUIRE_FALSE(countedWithBuffer);
    CHECK(countedWithBuffer.error().code == rachis::ErrorCode::unsupported);

    auto graphics = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(graphics);
    auto countedDraw = device.drawIndirect(*graphics,
                                           {
                                               .argumentBuffer = *arguments,
                                               .maxDrawCount = 2,
                                           });
    REQUIRE_FALSE(countedDraw);
    CHECK(countedDraw.error().code == rachis::ErrorCode::unsupported);
    auto conditional = device.beginConditionalExecution(*graphics, {.buffer = *countBuffer});
    REQUIRE_FALSE(conditional);
    CHECK(conditional.error().code == rachis::ErrorCode::unsupported);
}

TEST_CASE("Metal timestamp query pools enforce cross-command-buffer reset, rollback, and scope rules",
          "[size:large][rhi][metal][query][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    auto pool = device.createTimestampQueryPool({
        .queueType = rachis::QueueType::compute,
        .queryCount = 4,
        .label = "Metal cross-CB timestamp pool",
    });
    REQUIRE(pool);

    auto timestamps = device.createBuffer({
        .size = 4 * rachis::timestampQueryResultSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal cross-CB timestamp readback",
    });
    REQUIRE(timestamps);

    {
        auto scoped = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(scoped);
        REQUIRE(device
                    .beginGpuProfileScope(*scoped,
                                          {
                                              .queryPool = *pool,
                                              .beginningQueryIndex = 0,
                                              .endQueryIndex = 1,
                                              .label = "Open scope guard",
                                          })
                    .has_value());
        auto resetInsideScope = device.resetQueryPool(*scoped, *pool, 2, 1);
        auto resolveInsideScope = device.resolveTimestamps(
            *scoped, {.queryPool = *pool, .firstQuery = 0, .queryCount = 1, .destination = *timestamps});
        REQUIRE_FALSE(resetInsideScope);
        REQUIRE_FALSE(resolveInsideScope);
        CHECK(resetInsideScope.error().code == rachis::ErrorCode::invalidState);
        CHECK(resolveInsideScope.error().code == rachis::ErrorCode::invalidState);
        REQUIRE(device.endGpuProfileScope(*scoped).has_value());
    }

    {
        auto resetCmd = rachis::test::acquireCommandBuffer(
            device, rachis::QueueType::compute, rachis::CommandPoolFlag::allowIndividualReset);
        REQUIRE(resetCmd);
        REQUIRE(device.writeTimestamp(*resetCmd, *pool, 2).has_value());
        REQUIRE(device.resetCommandBuffer(*resetCmd).has_value());
        REQUIRE(device.writeTimestamp(*resetCmd, *pool, 2).has_value());
        REQUIRE(device.resetCommandBuffer(*resetCmd).has_value());
    }

    auto cbWrite1 = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(cbWrite1);
    REQUIRE(device.writeTimestamp(*cbWrite1, *pool, 0).has_value());
    REQUIRE(device.writeTimestamp(*cbWrite1, *pool, 1).has_value());
    REQUIRE(device.writeTimestamp(*cbWrite1, *pool, 2).has_value());
    auto ptWrite1 = device.submit(std::move(*cbWrite1));
    REQUIRE(ptWrite1);
    REQUIRE(device.wait(*ptWrite1).has_value());

    {
        auto cbChecks = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(cbChecks);
        auto dupWrite = device.writeTimestamp(*cbChecks, *pool, 0);
        auto unwrittenResolve = device.resolveTimestamps(
            *cbChecks, {.queryPool = *pool, .firstQuery = 3, .queryCount = 1, .destination = *timestamps});
        REQUIRE_FALSE(dupWrite);
        REQUIRE_FALSE(unwrittenResolve);
        CHECK(dupWrite.error().code == rachis::ErrorCode::invalidState);
        CHECK(unwrittenResolve.error().code == rachis::ErrorCode::invalidState);
    }

    auto cbResolve1 = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(cbResolve1);
    REQUIRE(device
                .resolveTimestamps(*cbResolve1,
                                   {.queryPool = *pool, .firstQuery = 0, .queryCount = 2, .destination = *timestamps})
                .has_value());

    REQUIRE(device.resetQueryPool(*cbResolve1, *pool, 2, 1).has_value());
    auto ptResolve1 = device.submit(std::move(*cbResolve1));
    REQUIRE(ptResolve1);
    REQUIRE(device.wait(*ptResolve1).has_value());

    const auto t0 = readU64(device, *timestamps, 0);
    const auto t1 = readU64(device, *timestamps, rachis::timestampQueryResultSize);
    CHECK(t0 > 0);
    CHECK(t1 >= t0);

    {
        auto cbReResolve = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(cbReResolve);
        auto dupResolve = device.resolveTimestamps(
            *cbReResolve, {.queryPool = *pool, .firstQuery = 0, .queryCount = 1, .destination = *timestamps});
        REQUIRE_FALSE(dupResolve);
        CHECK(dupResolve.error().code == rachis::ErrorCode::invalidState);
    }

    auto cbWrite2 = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(cbWrite2);
    REQUIRE(device.writeTimestamp(*cbWrite2, *pool, 2).has_value());
    auto ptWrite2 = device.submit(std::move(*cbWrite2));
    REQUIRE(ptWrite2);
    REQUIRE(device.wait(*ptWrite2).has_value());

    auto cbResolve2 = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(cbResolve2);
    REQUIRE(device
                .resolveTimestamps(*cbResolve2,
                                   {
                                       .queryPool = *pool,
                                       .firstQuery = 2,
                                       .queryCount = 1,
                                       .destination = *timestamps,
                                       .destinationOffset = 2 * rachis::timestampQueryResultSize,
                                   })
                .has_value());
    auto ptResolve2 = device.submit(std::move(*cbResolve2));
    REQUIRE(ptResolve2);
    REQUIRE(device.wait(*ptResolve2).has_value());

    const auto t2 = readU64(device, *timestamps, 2 * rachis::timestampQueryResultSize);
    CHECK(t2 >= t1);
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal GPU profile scopes reject nested and closing calls from another thread",
          "[size:large][rhi][metal][query][gpu][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    auto pool = device.createTimestampQueryPool({
        .queueType = rachis::QueueType::compute,
        .queryCount = 6,
        .label = "Metal cross-thread profile pool",
    });
    REQUIRE(pool);
    auto timestamps = device.createBuffer({
        .size = 6 * rachis::timestampQueryResultSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal cross-thread profile readback",
    });
    REQUIRE(timestamps);
    const rachis::GpuProfileScopeDesc outer{
        .queryPool = *pool,
        .beginningQueryIndex = 0,
        .endQueryIndex = 1,
        .label = "Outer scope",
    };
    const rachis::GpuProfileScopeDesc nested{
        .queryPool = *pool,
        .beginningQueryIndex = 2,
        .endQueryIndex = 3,
        .label = "Nested scope",
    };
    const rachis::GpuProfileScopeDesc moved{
        .queryPool = *pool,
        .beginningQueryIndex = 4,
        .endQueryIndex = 5,
        .label = "Moved scope",
    };

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command);
    REQUIRE(device.beginGpuProfileScope(*command, outer).has_value());

    rachis::Status crossThreadBegin;
    rachis::Status crossThreadEnd;
    std::jthread otherThread([&] {
        crossThreadBegin = device.beginGpuProfileScope(*command, nested);
        crossThreadEnd = device.endGpuProfileScope(*command);
    });
    otherThread.join();
    REQUIRE_FALSE(crossThreadBegin);
    CHECK(crossThreadBegin.error().code == rachis::ErrorCode::invalidState);
    REQUIRE_FALSE(crossThreadEnd);
    CHECK(crossThreadEnd.error().code == rachis::ErrorCode::invalidState);

    REQUIRE(device.beginGpuProfileScope(*command, nested).has_value());
    REQUIRE(device.endGpuProfileScope(*command).has_value());
    REQUIRE(device.endGpuProfileScope(*command).has_value());

    rachis::Status movedBegin;
    rachis::Status movedEnd;
    std::jthread movedThread([&] {
        movedBegin = device.beginGpuProfileScope(*command, moved);
        movedEnd = device.endGpuProfileScope(*command);
    });
    movedThread.join();
    REQUIRE(movedBegin.has_value());
    REQUIRE(movedEnd.has_value());

    REQUIRE(device
                .resolveTimestamps(*command,
                                   {
                                       .queryPool = *pool,
                                       .queryCount = 6,
                                       .destination = *timestamps,
                                   })
                .has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());

    std::array<std::uint64_t, 6> ticks{};
    for (std::size_t index = 0; index < ticks.size(); ++index)
    {
        ticks[index] = readU64(device, *timestamps, index * rachis::timestampQueryResultSize);
    }
    CAPTURE(ticks);
    for (const auto tick : ticks)
    {
        REQUIRE(tick != 0);
    }
    CHECK(ticks[1] >= ticks[0]);
    CHECK(ticks[3] >= ticks[2]);
    CHECK(ticks[5] >= ticks[4]);
    CHECK(device.diagnosticCounts().errors == 0);
}

#if RACHIS_ENABLE_METAL && defined(RACHIS_TESTING)
TEST_CASE("Metal render-pass and profile-scope timestamps open at the vertex stage and close after fragments",
          "[size:large][rhi][metal][query][gpu][draw][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    auto texture = device.createTexture({
        .width = targetSize,
        .height = targetSize,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .label = "Metal timestamp stage color",
    });
    REQUIRE(texture);
    auto view = device.createTextureView({
        .texture = *texture,
        .usage = rachis::TextureViewUsage::renderTarget,
        .dimension = rachis::TextureViewDimension::texture2D,
    });
    REQUIRE(view);

    auto pipelineLayout = device.createPipelineLayout({});
    REQUIRE(pipelineLayout);
    const auto triangleVs = loadBytes(pnmsPath("BootstrapTriangle.vs.metallib"));
    const auto trianglePs = loadBytes(pnmsPath("BootstrapTriangle.ps.metallib"));
    REQUIRE_FALSE(triangleVs.empty());
    REQUIRE_FALSE(trianglePs.empty());
    auto vertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = triangleVs,
        .entryPoint = "vertexMain",
    });
    auto pixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = trianglePs,
        .entryPoint = "pixelMain",
    });
    REQUIRE(vertex);
    REQUIRE(pixel);
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    auto pipeline = device.createGraphicsPipeline({
        .layout = *pipelineLayout,
        .vertexShader = *vertex,
        .pixelShader = *pixel,
        .cullMode = rachis::CullMode::back,
        .colorTargets = colorTargets,
        .label = "Metal timestamp stage triangle",
    });
    REQUIRE(pipeline);

    auto pool = device.createTimestampQueryPool({
        .queueType = rachis::QueueType::graphics,
        .queryCount = 5,
        .label = "Metal timestamp stage pool",
    });
    REQUIRE(pool);
    auto timestamps = device.createBuffer({
        .size = 5 * rachis::timestampQueryResultSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal timestamp stage readback",
    });
    REQUIRE(timestamps);

    rachis::metal::resetRecordingTelemetry();
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *texture,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array colorAttachments{rachis::ColorAttachment{
        .view = *view,
        .loadOperation = rachis::LoadOperation::clear,
        .storeOperation = rachis::StoreOperation::store,
    }};
    REQUIRE(device
                .beginRendering(*command,
                                {
                                    .colorAttachments = colorAttachments,
                                    .timestampWrites =
                                        rachis::PassTimestampWrites{
                                            .queryPool = *pool,
                                            .beginningOfPassWriteIndex = 0,
                                            .endOfPassWriteIndex = 4,
                                        },
                                })
                .has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *pipeline).has_value());
    REQUIRE(device
                .setViewports(*command,
                              std::array{rachis::Viewport{
                                  .width = static_cast<float>(targetSize),
                                  .height = static_cast<float>(targetSize),
                              }})
                .has_value());
    REQUIRE(device
                .setScissors(*command,
                             std::array{rachis::ScissorRect{
                                 .width = targetSize,
                                 .height = targetSize,
                             }})
                .has_value());
    REQUIRE(device.draw(*command, 3).has_value());
    REQUIRE(device
                .beginGpuProfileScope(*command,
                                      {
                                          .queryPool = *pool,
                                          .beginningQueryIndex = 1,
                                          .endQueryIndex = 3,
                                          .label = "Metal render-pass scope",
                                      })
                .has_value());
    REQUIRE(device.draw(*command, 3).has_value());
    REQUIRE(device.writeTimestamp(*command, *pool, 2).has_value());
    REQUIRE(device.endGpuProfileScope(*command).has_value());
    REQUIRE(device.endRendering(*command).has_value());

    const auto telemetry = rachis::metal::recordingTelemetry();
    CHECK(telemetry.renderTimestampVertexStageCount == 2);
    CHECK(telemetry.renderTimestampFragmentStageCount == 3);

    REQUIRE(device
                .resolveTimestamps(*command,
                                   {
                                       .queryPool = *pool,
                                       .queryCount = 5,
                                       .destination = *timestamps,
                                   })
                .has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());

    std::array<std::uint64_t, 5> ticks{};
    for (std::size_t index = 0; index < ticks.size(); ++index)
    {
        ticks[index] = readU64(device, *timestamps, index * rachis::timestampQueryResultSize);
    }
    CAPTURE(ticks);
    for (const auto tick : ticks)
    {
        REQUIRE(tick != 0);
    }
    for (std::size_t index = 1; index < ticks.size(); ++index)
    {
        CHECK(ticks[index] >= ticks[index - 1]);
    }
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal submit keeps destroyed-handle query pools alive until query locks unlock",
          "[size:large][rhi][metal][query][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    {
        auto transientPool = device.createTimestampQueryPool({
            .queueType = rachis::QueueType::compute,
            .queryCount = 2,
            .label = "Metal close-failure transient query pool",
        });
        REQUIRE(transientPool);

        auto cb = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(cb);
        REQUIRE(device.writeTimestamp(*cb, *transientPool, 0).has_value());
        REQUIRE(device.destroyQueryPool(*transientPool).has_value());

        struct ScopedCloseFailure final
        {
            ScopedCloseFailure() noexcept
            {
                rachis::metal::test::nextSubmitCloseFailure().store(true, std::memory_order_relaxed);
            }
            ~ScopedCloseFailure()
            {
                rachis::metal::test::nextSubmitCloseFailure().store(false, std::memory_order_relaxed);
            }
        } const guard;
        auto failed = device.submit(std::move(*cb));
        REQUIRE_FALSE(failed);
        CHECK(failed.error().code == rachis::ErrorCode::validationError);
    }

    {
        auto transientPool = device.createTimestampQueryPool({
            .queueType = rachis::QueueType::compute,
            .queryCount = 2,
            .label = "Metal normal-submit transient query pool",
        });
        REQUIRE(transientPool);

        auto cb = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(cb);
        REQUIRE(device.writeTimestamp(*cb, *transientPool, 0).has_value());
        REQUIRE(device.destroyQueryPool(*transientPool).has_value());

        auto submitted = device.submit(std::move(*cb));
        REQUIRE(submitted);
        REQUIRE(device.wait(*submitted).has_value());
        device.collectGarbage();
    }
}
#endif
