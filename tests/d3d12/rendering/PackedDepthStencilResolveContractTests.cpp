#include "D3D12/D3D12TestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <span>
#include <vector>

namespace
{

    constexpr std::uint32_t targetWidth = 32;
    constexpr std::uint32_t targetHeight = 32;
    constexpr std::uint32_t rowPitch = 256;
    constexpr std::uint64_t readbackSize = static_cast<std::uint64_t>(rowPitch) * targetHeight;

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
        std::int32_t previous_ = 0;
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

    [[nodiscard]] constexpr std::uint32_t preferredSampleCount(rachis::SampleCountMask mask) noexcept
    {
        for (const auto count : {4U, 2U, 8U})
        {
            if (rachis::supportsSampleCount(mask, count))
            {
                return count;
            }
        }
        return 0;
    }

    [[nodiscard]] rachis::TextureBarrier textureBarrier(rachis::TextureHandle texture,
                                                        rachis::TextureAspect aspect,
                                                        rachis::PipelineStage stagesBefore,
                                                        rachis::PipelineStage stagesAfter,
                                                        rachis::Access accessBefore,
                                                        rachis::Access accessAfter,
                                                        rachis::TextureLayout layoutBefore,
                                                        rachis::TextureLayout layoutAfter) noexcept
    {
        return {
            .texture = texture,
            .stagesBefore = stagesBefore,
            .stagesAfter = stagesAfter,
            .accessBefore = accessBefore,
            .accessAfter = accessAfter,
            .layoutBefore = layoutBefore,
            .layoutAfter = layoutAfter,
            .subresources = {.aspect = aspect},
        };
    }

} // namespace

TEST_CASE("packed depth and stencil planes resolve independently on WARP",
          "[size:medium][rhi][graphics][resolve][warp]")
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

    const auto properties = device.queryFormatProperties(rachis::Format::d24UnormS8Uint);
    const auto sampleCount = preferredSampleCount(properties.sampleCounts);
    REQUIRE(rachis::hasAny(properties.features, rachis::FormatFeature::multisampleResolve));
    REQUIRE(sampleCount != 0);
    CHECK(rachis::supportsResolveMode(properties.depthResolveModes, rachis::ResolveMode::minimum));
    CHECK(rachis::supportsResolveMode(properties.depthResolveModes, rachis::ResolveMode::maximum));
    CHECK_FALSE(rachis::supportsResolveMode(properties.depthResolveModes, rachis::ResolveMode::average));
    CHECK(rachis::supportsResolveMode(properties.stencilResolveModes, rachis::ResolveMode::minimum));
    CHECK(rachis::supportsResolveMode(properties.stencilResolveModes, rachis::ResolveMode::maximum));
    CHECK_FALSE(rachis::supportsResolveMode(properties.stencilResolveModes, rachis::ResolveMode::average));

    auto source = device.createTexture({
        .width = targetWidth,
        .height = targetHeight,
        .sampleCount = sampleCount,
        .format = rachis::Format::d24UnormS8Uint,
        .usage = rachis::TextureUsage::depthStencil,
        .label = "Packed multisample resolve source",
    });
    auto destination = device.createTexture({
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::d24UnormS8Uint,
        .usage = rachis::TextureUsage::depthStencil,
        .label = "Packed multisample resolve destination",
    });
    auto color = device.createTexture({
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
        .label = "Packed resolve verification color",
    });
    REQUIRE(source.has_value());
    REQUIRE(destination.has_value());
    REQUIRE(color.has_value());

    auto sourceView = device.createTextureView({
        .texture = *source,
        .usage = rachis::TextureViewUsage::depthStencil,
        .aspect = rachis::TextureAspect::depth,
    });
    auto destinationView = device.createTextureView({
        .texture = *destination,
        .usage = rachis::TextureViewUsage::depthStencil,
        .aspect = rachis::TextureAspect::depth,
    });
    auto colorView = device.createTextureView({
        .texture = *color,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    REQUIRE(sourceView.has_value());
    REQUIRE(destinationView.has_value());
    REQUIRE(colorView.has_value());

    auto readback = device.createBuffer({
        .size = readbackSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Packed resolve verification readback",
    });
    REQUIRE(readback.has_value());

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
    const std::array pushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::pixel, 0, 4 * sizeof(float)},
    };
    auto pipelineLayout = device.createPipelineLayout({.pushConstantRanges = pushConstantRanges});
    REQUIRE(pipelineLayout.has_value());
    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    const rachis::StencilFaceState stencilEqual{.compare = rachis::CompareOperation::equal};
    auto pipeline = device.createGraphicsPipeline({
        .layout = *pipelineLayout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .depthStencil =
            {
                .format = rachis::Format::d24UnormS8Uint,
                .depthTestEnabled = true,
                .depthWriteEnabled = false,
                .depthCompare = rachis::CompareOperation::equal,
                .stencilTestEnabled = true,
                .frontFace = stencilEqual,
                .backFace = stencilEqual,
            },
    });
    REQUIRE(pipeline.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    for (const auto aspect : {rachis::TextureAspect::depth, rachis::TextureAspect::stencil})
    {
        REQUIRE(device
                    .textureBarrier(*command,
                                    textureBarrier(*source,
                                                   aspect,
                                                   rachis::PipelineStage::all,
                                                   rachis::PipelineStage::depthStencil,
                                                   rachis::Access::none,
                                                   rachis::Access::depthStencilWrite,
                                                   rachis::TextureLayout::undefined,
                                                   rachis::TextureLayout::depthStencilWrite))
                    .has_value());
    }
    const rachis::DepthStencilAttachment sourceAttachment{
        .view = *sourceView,
        .depthLoadOperation = rachis::LoadOperation::clear,
        .depthClearValue = 0.0F,
        .stencilLoadOperation = rachis::LoadOperation::clear,
        .stencilClearValue = 37,
    };
    REQUIRE(device.beginRendering(*command, {.depthStencilAttachment = sourceAttachment}).has_value());
    REQUIRE(device.endRendering(*command).has_value());

    for (const auto aspect : {rachis::TextureAspect::depth, rachis::TextureAspect::stencil})
    {
        REQUIRE(device
                    .textureBarrier(*command,
                                    textureBarrier(*source,
                                                   aspect,
                                                   rachis::PipelineStage::depthStencil,
                                                   rachis::PipelineStage::resolve,
                                                   rachis::Access::depthStencilWrite,
                                                   rachis::Access::resolveSource,
                                                   rachis::TextureLayout::depthStencilWrite,
                                                   rachis::TextureLayout::resolveSource))
                    .has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    textureBarrier(*destination,
                                                   aspect,
                                                   rachis::PipelineStage::all,
                                                   rachis::PipelineStage::resolve,
                                                   rachis::Access::none,
                                                   rachis::Access::resolveDestination,
                                                   rachis::TextureLayout::undefined,
                                                   rachis::TextureLayout::resolveDestination))
                    .has_value());
    }

    CHECK_FALSE(device
                    .resolveTexture(*command,
                                    {
                                        .source = *source,
                                        .sourceSubresource = {.aspect = rachis::TextureAspect::stencil},
                                        .destination = *destination,
                                        .destinationSubresource = {.aspect = rachis::TextureAspect::stencil},
                                        .mode = rachis::ResolveMode::average,
                                    })
                    .has_value());
    CHECK_FALSE(device
                    .resolveTexture(*command,
                                    {
                                        .source = *source,
                                        .sourceSubresource = {.aspect = rachis::TextureAspect::depth},
                                        .destination = *destination,
                                        .destinationSubresource = {.aspect = rachis::TextureAspect::stencil},
                                        .mode = rachis::ResolveMode::minimum,
                                    })
                    .has_value());
    const auto beforeRejectedPackedResolve = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(beforeRejectedPackedResolve.has_value());
    {
        const FailResourceRetention failDestinationPublication(2);
        auto failedResolve = device.resolveTexture(*command,
                                                   {
                                                       .source = *source,
                                                       .sourceSubresource =
                                                           {
                                                               .aspect = rachis::TextureAspect::depth,
                                                           },
                                                       .destination = *destination,
                                                       .destinationSubresource =
                                                           {
                                                               .aspect = rachis::TextureAspect::depth,
                                                           },
                                                       .mode = rachis::ResolveMode::minimum,
                                                   });
        REQUIRE_FALSE(failedResolve.has_value());
        CHECK(failedResolve.error().code == rachis::ErrorCode::outOfMemory);
    }
    const auto afterRejectedPackedResolve = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(afterRejectedPackedResolve.has_value());
    CHECK(*afterRejectedPackedResolve == *beforeRejectedPackedResolve);
    REQUIRE(device
                .resolveTexture(*command,
                                {
                                    .source = *source,
                                    .sourceSubresource = {.aspect = rachis::TextureAspect::depth},
                                    .destination = *destination,
                                    .destinationSubresource = {.aspect = rachis::TextureAspect::depth},
                                    .mode = rachis::ResolveMode::minimum,
                                })
                .has_value());
    REQUIRE(device
                .resolveTexture(*command,
                                {
                                    .source = *source,
                                    .sourceSubresource = {.aspect = rachis::TextureAspect::stencil},
                                    .destination = *destination,
                                    .destinationSubresource = {.aspect = rachis::TextureAspect::stencil},
                                    .mode = rachis::ResolveMode::maximum,
                                })
                .has_value());

    for (const auto aspect : {rachis::TextureAspect::depth, rachis::TextureAspect::stencil})
    {
        REQUIRE(device
                    .textureBarrier(*command,
                                    textureBarrier(*destination,
                                                   aspect,
                                                   rachis::PipelineStage::resolve,
                                                   rachis::PipelineStage::depthStencil,
                                                   rachis::Access::resolveDestination,
                                                   rachis::Access::depthStencilWrite,
                                                   rachis::TextureLayout::resolveDestination,
                                                   rachis::TextureLayout::depthStencilWrite))
                    .has_value());
    }
    REQUIRE(device
                .textureBarrier(*command,
                                textureBarrier(*color,
                                               rachis::TextureAspect::color,
                                               rachis::PipelineStage::all,
                                               rachis::PipelineStage::renderTarget,
                                               rachis::Access::none,
                                               rachis::Access::renderTarget,
                                               rachis::TextureLayout::undefined,
                                               rachis::TextureLayout::renderTarget))
                .has_value());

    const std::array colorAttachments{
        rachis::ColorAttachment{
            .view = *colorView,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.0F, 0.0F, 0.0F, 1.0F},
        },
    };
    const rachis::DepthStencilAttachment destinationAttachment{
        .view = *destinationView,
        .depthLoadOperation = rachis::LoadOperation::load,
        .stencilLoadOperation = rachis::LoadOperation::load,
    };
    REQUIRE(device
                .beginRendering(*command,
                                {
                                    .colorAttachments = colorAttachments,
                                    .depthStencilAttachment = destinationAttachment,
                                })
                .has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *pipeline).has_value());
    REQUIRE(device
                .setViewport(*command,
                             {.width = static_cast<float>(targetWidth), .height = static_cast<float>(targetHeight)})
                .has_value());
    REQUIRE(device.setScissor(*command, {.width = targetWidth, .height = targetHeight}).has_value());
    REQUIRE(device.setStencilReference(*command, 37).has_value());
    const std::array green{0.0F, 1.0F, 0.0F, 1.0F};
    REQUIRE(device.setGraphicsPushConstants(*command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(green)))
                .has_value());
    REQUIRE(device.draw(*command, 3).has_value());
    REQUIRE(device.endRendering(*command).has_value());

    REQUIRE(device
                .textureBarrier(*command,
                                textureBarrier(*color,
                                               rachis::TextureAspect::color,
                                               rachis::PipelineStage::renderTarget,
                                               rachis::PipelineStage::copy,
                                               rachis::Access::renderTarget,
                                               rachis::Access::copySource,
                                               rachis::TextureLayout::renderTarget,
                                               rachis::TextureLayout::copySource))
                .has_value());
    REQUIRE(device
                .copyTextureToBuffer(*command,
                                     {
                                         .buffer = *readback,
                                         .bytesPerRow = rowPitch,
                                         .rowsPerImage = targetHeight,
                                         .texture = *color,
                                         .extent = {targetWidth, targetHeight, 1},
                                     })
                .has_value());

    REQUIRE(device.destroyTexture(*source).has_value());
    REQUIRE(device.destroyTexture(*destination).has_value());
    REQUIRE(device.destroyTexture(*color).has_value());
    REQUIRE(device.destroyTextureView(*sourceView).has_value());
    REQUIRE(device.destroyTextureView(*destinationView).has_value());
    REQUIRE(device.destroyTextureView(*colorView).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    auto waited = device.wait(*submitted);
    if (!waited)
    {
        INFO(waited.error().message);
    }
    REQUIRE(waited.has_value());
    device.collectGarbage();

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    const auto* corner = mapped->data;
    const auto* center = mapped->data + static_cast<std::uint64_t>(targetHeight / 2) * rowPitch +
                         static_cast<std::uint64_t>(targetWidth / 2) * 4;
    CHECK(corner[0] == std::byte{0});
    CHECK(corner[1] == std::byte{0});
    CHECK(corner[2] == std::byte{0});
    CHECK(corner[3] == std::byte{255});
    CHECK(center[0] == std::byte{0});
    CHECK(center[1] == std::byte{255});
    CHECK(center[2] == std::byte{0});
    CHECK(center[3] == std::byte{255});
    REQUIRE(device.unmapBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
