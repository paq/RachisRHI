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

    constexpr std::uint32_t targetSize = 128;

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

    template<typename ResultLike>
    void checkError(const ResultLike& status, rachis::ErrorCode code)
    {
        REQUIRE_FALSE(status.has_value());
        CHECK(status.error().code == code);
    }

    [[nodiscard]] rachis::Device createDevice(rachis::AdapterPreference preference)
    {
        auto created = rachis::createDevice(rachis::DeviceDesc{
            .adapterPreference = preference,
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
        return std::move(*created);
    }

} // namespace

TEST_CASE("per-draw shading rates validate scope, capability, and sample count",
          "[size:medium][rhi][graphics][variable-rate-shading][warp]")
{
    auto device = createDevice(rachis::AdapterPreference::warp);
    const auto capabilities = *device.capabilities();
    STATIC_REQUIRE_FALSE(rachis::shadingRateImageTexel(static_cast<rachis::ShadingRate>(255)).has_value());

    auto target = device.createTexture({
        .width = targetSize,
        .height = targetSize,
        .sampleCount = 4,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
    });
    if (!target)
    {
        SKIP("WARP does not expose four-sample RGBA8 render targets.");
    }
    auto targetView = device.createTextureView({
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    REQUIRE(targetView.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    checkError(device.setShadingRate(*command, rachis::ShadingRate::oneByOne), rachis::ErrorCode::invalidArgument);
    checkError(device.setShadingRateImage(*command), rachis::ErrorCode::invalidArgument);

    const rachis::TextureDesc rateImageDesc{
        .width = 1,
        .height = 1,
        .format = rachis::Format::r8Uint,
        .usage = rachis::TextureUsage::shadingRateSource,
    };
    auto rateImage = device.createTexture(rateImageDesc);
    if (capabilities.variableRateShadingTier2)
    {
        REQUIRE(rateImage.has_value());
        auto invalidFormat = rateImageDesc;
        invalidFormat.format = rachis::Format::r8Unorm;
        checkError(device.createTexture(invalidFormat), rachis::ErrorCode::invalidArgument);
        auto invalidArray = rateImageDesc;
        invalidArray.depthOrArrayLayers = 2;
        checkError(device.createTexture(invalidArray), rachis::ErrorCode::invalidArgument);
    }
    else
    {
        checkError(rateImage, rachis::ErrorCode::unsupported);
    }
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
            .storeOperation = rachis::StoreOperation::discard,
        },
    };
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
    checkError(device.setShadingRate(*command, static_cast<rachis::ShadingRate>(255)),
               rachis::ErrorCode::invalidArgument);
    checkError(device.setShadingRate(*command,
                                     rachis::ShadingRate::oneByOne,
                                     {.primitive = static_cast<rachis::ShadingRateCombiner>(255)}),
               rachis::ErrorCode::invalidArgument);

    if (!capabilities.variableRateShading)
    {
        checkError(device.setShadingRate(*command, rachis::ShadingRate::oneByOne), rachis::ErrorCode::unsupported);
    }
    else
    {
        REQUIRE(device.setShadingRate(*command, rachis::ShadingRate::oneByOne).has_value());
        REQUIRE(device.setShadingRate(*command, rachis::ShadingRate::twoByTwo).has_value());
        checkError(device.setShadingRate(*command, rachis::ShadingRate::twoByFour), rachis::ErrorCode::unsupported);
        checkError(device.setShadingRate(*command, rachis::ShadingRate::fourByTwo), rachis::ErrorCode::unsupported);
        checkError(device.setShadingRate(*command, rachis::ShadingRate::fourByFour), rachis::ErrorCode::unsupported);
    }
    const auto tier2CombinerStatus = device.setShadingRate(
        *command, rachis::ShadingRate::oneByOne, {.primitive = rachis::ShadingRateCombiner::override});
    const auto tier2ImageStatus = device.setShadingRateImage(*command);
    if (capabilities.variableRateShadingTier2)
    {
        REQUIRE(tier2CombinerStatus.has_value());
        REQUIRE(tier2ImageStatus.has_value());
    }
    else
    {
        checkError(tier2CombinerStatus, rachis::ErrorCode::unsupported);
        checkError(tier2ImageStatus, rachis::ErrorCode::unsupported);
    }
    REQUIRE(device.endRendering(*command).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    if (rateImage)
    {
        REQUIRE(device.destroyTexture(*rateImage).has_value());
    }

    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
    CHECK(device.diagnosticCounts().corruptions == 0);
}

TEST_CASE("Tier 2 primitive and image rates reduce measured fragment invocations",
          "[size:medium][rhi][graphics][variable-rate-shading][tier2][hardware][pipeline-statistics]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::highPerformance,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        SKIP("No hardware D3D12 device is available.");
    }
    auto device = std::move(*created);
    const auto capabilities = *device.capabilities();
    if (!capabilities.variableRateShadingTier2)
    {
        SKIP("The selected hardware adapter does not support Tier 2 variable-rate shading.");
    }
    REQUIRE(capabilities.pipelineStatisticsQueries);

    const auto directVertexBytecode = loadShader(RACHIS_TEST_GRAPHICS_DIRECT_VERTEX_SHADER_PATH);
    const auto rateVertexBytecode = loadShader(RACHIS_TEST_GRAPHICS_SHADING_RATE_VERTEX_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_GRAPHICS_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(directVertexBytecode.empty());
    REQUIRE_FALSE(rateVertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    auto directVertexShader = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = directVertexBytecode,
        .entryPoint = "directVertex",
        .shaderInterface = {},
    });
    auto rateVertexShader = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = rateVertexBytecode,
        .entryPoint = "shadingRateVertex",
        .shaderInterface = {},
    });
    auto pixelShader = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "pixelMain",
        .shaderInterface = rachis::test::graphicsPixelShaderInterface(),
    });
    REQUIRE(directVertexShader.has_value());
    REQUIRE(rateVertexShader.has_value());
    REQUIRE(pixelShader.has_value());

    const std::array pushConstants{
        rachis::PushConstantRange{rachis::ShaderStage::pixel, 0, 4 * sizeof(float)},
    };
    auto layout = device.createPipelineLayout({.pushConstantRanges = pushConstants});
    REQUIRE(layout.has_value());
    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    const auto createPipeline = [&](rachis::ShaderModuleHandle vertexShader) {
        return device.createGraphicsPipeline({
            .layout = *layout,
            .vertexShader = vertexShader,
            .pixelShader = *pixelShader,
            .cullMode = rachis::CullMode::none,
            .colorTargets = colorTargets,
        });
    };
    auto directPipeline = createPipeline(*directVertexShader);
    auto ratePipeline = createPipeline(*rateVertexShader);
    REQUIRE(directPipeline.has_value());
    REQUIRE(ratePipeline.has_value());

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
    REQUIRE(targetView.has_value());

    const std::uint32_t tileSize = capabilities.shadingRateImageTileSize;
    const std::uint32_t imageWidth = (targetSize + tileSize - 1) / tileSize;
    const std::uint32_t imageHeight = (targetSize + tileSize - 1) / tileSize;
    auto invalidRateImageDesc = rachis::TextureDesc{
        .width = imageWidth,
        .height = imageHeight,
        .format = rachis::Format::r8Unorm,
        .usage = rachis::TextureUsage::shadingRateSource,
    };
    checkError(device.createTexture(invalidRateImageDesc), rachis::ErrorCode::invalidArgument);
    constexpr std::uint32_t rowPitch = 256;
    std::vector<std::byte> imageData(static_cast<std::size_t>(rowPitch) * imageHeight);
    constexpr auto coarseImageTexel = rachis::shadingRateImageTexel(rachis::ShadingRate::twoByTwo);
    for (std::uint32_t row = 0; row < imageHeight; ++row)
    {
        std::fill_n(imageData.begin() + static_cast<std::ptrdiff_t>(row) * rowPitch,
                    imageWidth,
                    static_cast<std::byte>(*coarseImageTexel));
    }
    auto upload = device.createBuffer({
        .size = imageData.size(),
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(upload.has_value());
    REQUIRE(device.writeBuffer(*upload, 0, imageData).has_value());
    auto rateImage = device.createTexture({
        .width = imageWidth,
        .height = imageHeight,
        .format = rachis::Format::r8Uint,
        .usage = rachis::TextureUsage::copyDestination | rachis::TextureUsage::shadingRateSource,
    });
    REQUIRE(rateImage.has_value());

    constexpr std::uint32_t queryCount = 4;
    auto queryPool = device.createPipelineStatisticsQueryPool({.queryCount = queryCount});
    auto results = device.createBuffer({
        .size = queryCount * rachis::pipelineStatisticsQueryResultSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(queryPool.has_value());
    REQUIRE(results.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *rateImage,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    REQUIRE(device
                .copyBufferToTexture(*command,
                                     {
                                         .buffer = *upload,
                                         .bytesPerRow = rowPitch,
                                         .rowsPerImage = imageHeight,
                                         .texture = *rateImage,
                                         .extent = {.width = imageWidth, .height = imageHeight},
                                     })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *rateImage,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::pixelShader,
                                    .accessBefore = rachis::Access::copyDestination,
                                    .accessAfter = rachis::Access::shadingRateSource,
                                    .layoutBefore = rachis::TextureLayout::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::shadingRateSource,
                                })
                .has_value());
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
            .storeOperation = rachis::StoreOperation::discard,
        },
    };
    const std::array tint{1.0F, 1.0F, 1.0F, 1.0F};
    for (std::uint32_t query = 0; query < queryCount; ++query)
    {
        REQUIRE(device.beginPipelineStatisticsQuery(*command, *queryPool, query).has_value());
        REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
        if (query == 1)
        {
            REQUIRE(device
                        .setShadingRate(*command,
                                        rachis::ShadingRate::oneByOne,
                                        {.primitive = rachis::ShadingRateCombiner::override})
                        .has_value());
        }
        else if (query == 2)
        {
            REQUIRE(device
                        .setShadingRate(
                            *command, rachis::ShadingRate::oneByOne, {.image = rachis::ShadingRateCombiner::override})
                        .has_value());
            REQUIRE(device.setShadingRateImage(*command, *rateImage).has_value());
        }
        REQUIRE(device.bindGraphicsPipeline(*command, query == 1 ? *ratePipeline : *directPipeline).has_value());
        REQUIRE(device
                    .setViewport(*command,
                                 {
                                     .width = static_cast<float>(targetSize),
                                     .height = static_cast<float>(targetSize),
                                 })
                    .has_value());
        REQUIRE(device.setScissor(*command, {.width = targetSize, .height = targetSize}).has_value());
        REQUIRE(device.setGraphicsPushConstants(*command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(tint)))
                    .has_value());
        REQUIRE(device.draw(*command, 3).has_value());
        REQUIRE(device.endRendering(*command).has_value());
        REQUIRE(device.endPipelineStatisticsQuery(*command).has_value());
    }
    REQUIRE(device
                .resolvePipelineStatisticsQueries(*command,
                                                  {
                                                      .queryPool = *queryPool,
                                                      .queryCount = queryCount,
                                                      .destination = *results,
                                                  })
                .has_value());
    REQUIRE(device.destroyTexture(*rateImage).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto mapped = device.mapBuffer(*results);
    REQUIRE(mapped.has_value());
    std::array<rachis::PipelineStatistics, queryCount> statistics{};
    std::memcpy(statistics.data(), mapped->data, sizeof(statistics));
    CAPTURE(rachis::test::requireAdapterInfo(device).name,
            tileSize,
            statistics[0].fragmentShaderInvocations,
            statistics[1].fragmentShaderInvocations,
            statistics[2].fragmentShaderInvocations,
            statistics[3].fragmentShaderInvocations);
    CHECK(statistics[0].fragmentShaderInvocations > 0);
    CHECK(statistics[1].fragmentShaderInvocations > 0);
    CHECK(statistics[2].fragmentShaderInvocations > 0);
    CHECK(statistics[3].fragmentShaderInvocations > 0);
    CHECK(statistics[1].fragmentShaderInvocations < statistics[0].fragmentShaderInvocations);
    CHECK(statistics[2].fragmentShaderInvocations < statistics[0].fragmentShaderInvocations);
    CHECK(statistics[3].fragmentShaderInvocations == statistics[0].fragmentShaderInvocations);
    REQUIRE(device.unmapBuffer(*results).has_value());

    REQUIRE(device.destroyBuffer(*results).has_value());
    REQUIRE(device.destroyQueryPool(*queryPool).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    REQUIRE(device.destroyPipeline(*ratePipeline).has_value());
    REQUIRE(device.destroyPipeline(*directPipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*layout).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    REQUIRE(device.destroyShaderModule(*rateVertexShader).has_value());
    REQUIRE(device.destroyShaderModule(*directVertexShader).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
    CHECK(device.diagnosticCounts().corruptions == 0);
}

TEST_CASE("coarse per-draw shading reduces measured fragment invocations",
          "[size:medium][rhi][graphics][variable-rate-shading][hardware][pipeline-statistics]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::highPerformance,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        SKIP("No hardware D3D12 device is available.");
    }
    auto device = std::move(*created);
    if (!rachis::test::requireCapabilities(device).variableRateShading)
    {
        SKIP("The selected hardware adapter does not support per-draw variable-rate shading.");
    }
    REQUIRE(rachis::test::requireCapabilities(device).pipelineStatisticsQueries);

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
    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    auto pipeline = device.createGraphicsPipeline({
        .layout = *layout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
    });
    REQUIRE(pipeline.has_value());

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
    auto queryPool = device.createPipelineStatisticsQueryPool({.queryCount = 2});
    auto results = device.createBuffer({
        .size = 2 * rachis::pipelineStatisticsQueryResultSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(targetView.has_value());
    REQUIRE(queryPool.has_value());
    REQUIRE(results.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
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
            .storeOperation = rachis::StoreOperation::discard,
        },
    };
    const std::array tint{1.0F, 1.0F, 1.0F, 1.0F};
    for (std::uint32_t query = 0; query < 2; ++query)
    {
        REQUIRE(device.beginPipelineStatisticsQuery(*command, *queryPool, query).has_value());
        REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
        if (query == 1)
        {
            REQUIRE(device.setShadingRate(*command, rachis::ShadingRate::twoByTwo).has_value());
        }
        REQUIRE(device.bindGraphicsPipeline(*command, *pipeline).has_value());
        REQUIRE(device
                    .setViewport(*command,
                                 {
                                     .width = static_cast<float>(targetSize),
                                     .height = static_cast<float>(targetSize),
                                 })
                    .has_value());
        REQUIRE(device.setScissor(*command, {.width = targetSize, .height = targetSize}).has_value());
        REQUIRE(device.setGraphicsPushConstants(*command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(tint)))
                    .has_value());
        REQUIRE(device.draw(*command, 3).has_value());
        REQUIRE(device.endRendering(*command).has_value());
        REQUIRE(device.endPipelineStatisticsQuery(*command).has_value());
    }
    REQUIRE(device
                .resolvePipelineStatisticsQueries(*command,
                                                  {
                                                      .queryPool = *queryPool,
                                                      .queryCount = 2,
                                                      .destination = *results,
                                                  })
                .has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto mapped = device.mapBuffer(*results);
    REQUIRE(mapped.has_value());
    std::array<rachis::PipelineStatistics, 2> statistics{};
    std::memcpy(statistics.data(), mapped->data, sizeof(statistics));
    CAPTURE(rachis::test::requireAdapterInfo(device).name,
            statistics[0].fragmentShaderInvocations,
            statistics[1].fragmentShaderInvocations);
    CHECK(statistics[0].fragmentShaderInvocations > 0);
    CHECK(statistics[1].fragmentShaderInvocations > 0);
    CHECK(statistics[1].fragmentShaderInvocations < statistics[0].fragmentShaderInvocations);
    REQUIRE(device.unmapBuffer(*results).has_value());

    REQUIRE(device.destroyBuffer(*results).has_value());
    REQUIRE(device.destroyQueryPool(*queryPool).has_value());
    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*layout).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
    CHECK(device.diagnosticCounts().corruptions == 0);
}
