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

#include <algorithm>
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

    constexpr std::uint32_t targetSize = 128;

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

    template<typename ResultLike>
    void checkError(const ResultLike& status, rachis::ErrorCode code)
    {
        REQUIRE_FALSE(status.has_value());
        CHECK(status.error().code == code);
    }

} // namespace

TEST_CASE("Vulkan per-draw shading rates validate scope, capability, and sample count",
          "[size:medium][rhi][vulkan][variable-rate-shading]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);

    auto target = device.createTexture({
        .width = targetSize,
        .height = targetSize,
        .sampleCount = 4,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
    });
    if (!target)
    {
        SKIP("The Vulkan adapter does not expose four-sample RGBA8 render targets.");
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
    checkError(
        device.setShadingRate(*command, rachis::ShadingRate::oneByOne, {.image = rachis::ShadingRateCombiner::sum}),
        rachis::ErrorCode::unsupported);

    if (!capabilities.variableRateShading)
    {
        checkError(device.setShadingRate(*command, rachis::ShadingRate::oneByOne), rachis::ErrorCode::unsupported);
    }
    else
    {
        REQUIRE(device.setShadingRate(*command, rachis::ShadingRate::oneByOne).has_value());
        REQUIRE(device.setShadingRate(*command, rachis::ShadingRate::twoByTwo).has_value());
        if (!capabilities.additionalShadingRates)
        {
            checkError(device.setShadingRate(*command, rachis::ShadingRate::twoByFour), rachis::ErrorCode::unsupported);
            checkError(device.setShadingRate(*command, rachis::ShadingRate::fourByTwo), rachis::ErrorCode::unsupported);
            checkError(device.setShadingRate(*command, rachis::ShadingRate::fourByFour),
                       rachis::ErrorCode::unsupported);
        }
    }
    const auto tier2CombinerStatus = device.setShadingRate(
        *command, rachis::ShadingRate::oneByOne, {.primitive = rachis::ShadingRateCombiner::override});
    const auto tier2ImageStatus = device.setShadingRateImage(*command);
    if (capabilities.variableRateShadingTier2)
    {
        REQUIRE(tier2CombinerStatus.has_value());
        REQUIRE(tier2ImageStatus.has_value());
        checkError(device.setShadingRateImage(*command, *rateImage), rachis::ErrorCode::invalidArgument);
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
}

#ifdef RACHIS_SPIRV_DIR
TEST_CASE("Vulkan coarse per-draw shading reduces measured fragment invocations",
          "[size:medium][rhi][vulkan][variable-rate-shading][pipeline-statistics]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    if (!rachis::test::requireCapabilities(device).variableRateShading)
    {
        SKIP("The Vulkan adapter does not advertise per-draw variable-rate shading.");
    }
    REQUIRE(rachis::test::requireCapabilities(device).pipelineStatisticsQueries);

    const auto vertexBytecode = loadSpirv("VariableRateShadingWorkload.direct.vs.spv");
    const auto pixelBytecode = loadSpirv("VariableRateShadingWorkload.ps.spv");
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
    REQUIRE(target.has_value());
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
    CAPTURE(statistics[0].fragmentShaderInvocations, statistics[1].fragmentShaderInvocations);
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
}

TEST_CASE("Vulkan primitive and image rates reduce measured fragment invocations",
          "[size:medium][rhi][vulkan][variable-rate-shading][tier2][pipeline-statistics]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (!capabilities.variableRateShadingTier2)
    {
        SKIP("The Vulkan adapter does not advertise primitive and attachment fragment shading rates.");
    }
    REQUIRE(capabilities.pipelineStatisticsQueries);
    REQUIRE(capabilities.shadingRateImageTileSize != 0);

    const auto directVertexBytecode = loadSpirv("VariableRateShadingWorkload.direct.vs.spv");
    const auto rateVertexBytecode = loadSpirv("VariableRateShadingWorkload.rate.vs.spv");
    const auto pixelBytecode = loadSpirv("VariableRateShadingWorkload.ps.spv");
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
    constexpr std::uint32_t rowPitch = 256;
    std::vector<std::byte> imageData(static_cast<std::size_t>(rowPitch) * imageHeight);
    constexpr auto coarseImageTexel = rachis::shadingRateImageTexel(rachis::ShadingRate::twoByTwo);
    REQUIRE(coarseImageTexel.has_value());
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
    CAPTURE(tileSize,
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
}

TEST_CASE("Vulkan command buffers keep backend shading-rate views alive until retirement in every pool mode",
          "[size:medium][rhi][vulkan][variable-rate-shading][tier2][command-pool]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (!capabilities.variableRateShadingTier2)
    {
        SKIP("The Vulkan adapter does not advertise primitive and attachment fragment shading rates.");
    }
    REQUIRE(capabilities.shadingRateImageTileSize != 0);

    const auto vertexBytecode = loadSpirv("VariableRateShadingWorkload.direct.vs.spv");
    const auto pixelBytecode = loadSpirv("VariableRateShadingWorkload.ps.spv");
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
    REQUIRE(targetView.has_value());

    const std::uint32_t tileSize = capabilities.shadingRateImageTileSize;
    const std::uint32_t imageWidth = (targetSize + tileSize - 1) / tileSize;
    const std::uint32_t imageHeight = (targetSize + tileSize - 1) / tileSize;
    constexpr std::uint32_t rowPitch = 256;
    std::vector<std::byte> imageData(static_cast<std::size_t>(rowPitch) * imageHeight);
    constexpr auto coarseImageTexel = rachis::shadingRateImageTexel(rachis::ShadingRate::twoByTwo);
    REQUIRE(coarseImageTexel.has_value());
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

    const auto flags = GENERATE(rachis::CommandPoolFlag::none, rachis::CommandPoolFlag::unretainedResources);
    const bool unretained = flags == rachis::CommandPoolFlag::unretainedResources;
    CAPTURE(unretained);
    auto& liveViews = rachis::vulkan::test::liveShadingRateViews();
    const auto baseline = liveViews.load();
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics, flags);
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
    const auto drawOnce = [&]() {
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
    };
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
    REQUIRE(
        device.setShadingRate(*command, rachis::ShadingRate::oneByOne, {.image = rachis::ShadingRateCombiner::override})
            .has_value());
    REQUIRE(device.setShadingRateImage(*command, *rateImage).has_value());
    CHECK(liveViews.load() == baseline + 1);
    drawOnce();
    REQUIRE(device.setShadingRateImage(*command, *rateImage).has_value());
    CHECK(liveViews.load() == baseline + 2);
    drawOnce();
    REQUIRE(device.endRendering(*command).has_value());
    CHECK(liveViews.load() == baseline + 2);

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    CHECK(liveViews.load() == baseline + 2);
    REQUIRE(device.wait(*submitted).has_value());
    CHECK(liveViews.load() == baseline);
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);

    REQUIRE(device.destroyTexture(*rateImage).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*layout).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
}
#endif
