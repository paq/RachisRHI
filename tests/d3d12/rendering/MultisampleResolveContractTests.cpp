#include "D3D12/D3D12TestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace
{

    constexpr std::uint32_t targetWidth = 64;
    constexpr std::uint32_t targetHeight = 64;
    constexpr std::uint32_t rowPitch = 256;
    constexpr std::uint64_t readbackSize = static_cast<std::uint64_t>(rowPitch) * targetHeight;
    constexpr std::uint32_t computeValueCount = 64;
    constexpr std::uint64_t computeOutputSize = computeValueCount * sizeof(std::uint32_t);

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

    [[nodiscard]] bool nearByte(std::byte actual, std::uint8_t expected) noexcept
    {
        const auto value = std::to_integer<std::uint8_t>(actual);
        const auto lower = expected == 0 ? 0U : static_cast<std::uint32_t>(expected) - 1U;
        const auto upper = (std::min)(static_cast<std::uint32_t>(expected) + 1U, 255U);
        return value >= lower && value <= upper;
    }

} // namespace

TEST_CASE("multisampled color and depth attachments resolve on WARP", "[size:medium][rhi][graphics][resolve][warp]")
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

    const auto colorProperties = device.queryFormatProperties(rachis::Format::rgba8Unorm);
    const auto depthProperties = device.queryFormatProperties(rachis::Format::d32Float);
    const auto colorSampleCount = preferredSampleCount(colorProperties.sampleCounts);
    const auto depthSampleCount = preferredSampleCount(depthProperties.sampleCounts);
    REQUIRE(rachis::hasAny(colorProperties.features, rachis::FormatFeature::multisampleResolve));
    REQUIRE(rachis::hasAny(depthProperties.features, rachis::FormatFeature::multisampleResolve));
    CHECK(rachis::supportsResolveMode(colorProperties.colorResolveModes, rachis::ResolveMode::average));
    CHECK_FALSE(rachis::supportsResolveMode(colorProperties.colorResolveModes, rachis::ResolveMode::minimum));
    CHECK(rachis::supportsResolveMode(depthProperties.depthResolveModes, rachis::ResolveMode::minimum));
    CHECK(rachis::supportsResolveMode(depthProperties.depthResolveModes, rachis::ResolveMode::maximum));
    CHECK_FALSE(rachis::supportsResolveMode(depthProperties.depthResolveModes, rachis::ResolveMode::average));
    REQUIRE(colorSampleCount != 0);
    REQUIRE(depthSampleCount != 0);

    CHECK_FALSE(device
                    .createTexture(rachis::TextureDesc{
                        .width = targetWidth,
                        .height = targetHeight,
                        .sampleCount = 3,
                        .format = rachis::Format::rgba8Unorm,
                        .usage = rachis::TextureUsage::renderTarget,
                    })
                    .has_value());
    CHECK_FALSE(device
                    .createTexture(rachis::TextureDesc{
                        .width = targetWidth,
                        .height = targetHeight,
                        .mipLevels = 2,
                        .sampleCount = colorSampleCount,
                        .format = rachis::Format::rgba8Unorm,
                        .usage = rachis::TextureUsage::renderTarget,
                    })
                    .has_value());
    CHECK_FALSE(device
                    .createTexture(rachis::TextureDesc{
                        .width = targetWidth,
                        .height = targetHeight,
                        .sampleCount = colorSampleCount,
                        .format = rachis::Format::rgba8Unorm,
                        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::renderTarget,
                    })
                    .has_value());

    auto colorSource = device.createTexture(rachis::TextureDesc{
        .width = targetWidth,
        .height = targetHeight,
        .sampleCount = colorSampleCount,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .label = "MSAA color source",
    });
    auto colorDestination = device.createTexture(rachis::TextureDesc{
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
        .label = "Resolved color destination",
    });
    REQUIRE(colorSource.has_value());
    REQUIRE(colorDestination.has_value());
    auto colorSourceView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *colorSource,
        .usage = rachis::TextureViewUsage::renderTarget,
        .label = "MSAA color source view",
    });
    REQUIRE(colorSourceView.has_value());

    auto depthSource = device.createTexture(rachis::TextureDesc{
        .width = targetWidth,
        .height = targetHeight,
        .sampleCount = depthSampleCount,
        .format = rachis::Format::d32Float,
        .usage = rachis::TextureUsage::depthStencil,
        .label = "MSAA depth source",
    });
    auto depthDestination = device.createTexture(rachis::TextureDesc{
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::d32Float,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::depthStencil,
        .label = "Resolved depth destination",
    });
    REQUIRE(depthSource.has_value());
    REQUIRE(depthDestination.has_value());
    auto depthSourceView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *depthSource,
        .usage = rachis::TextureViewUsage::depthStencil,
        .aspect = rachis::TextureAspect::depth,
        .label = "MSAA depth source view",
    });
    REQUIRE(depthSourceView.has_value());

    auto colorReadback = device.createBuffer(rachis::BufferDesc{
        .size = readbackSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Resolved color readback",
    });
    REQUIRE(colorReadback.has_value());
    auto depthReadback = device.createBuffer(rachis::BufferDesc{
        .size = readbackSize * 2,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Resolved depth readback",
    });
    REQUIRE(depthReadback.has_value());

    const auto computeBytecode = loadShader(RACHIS_TEST_COMPUTE_SHADER_PATH);
    REQUIRE_FALSE(computeBytecode.empty());
    auto computeShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::compute,
        .bytecode = computeBytecode,
        .entryPoint = "main",
        .label = "Resolve continuation compute shader",
        .shaderInterface = rachis::test::computeWriteShaderInterface(),
    });
    REQUIRE(computeShader.has_value());
    auto computeOutput = device.createBuffer(rachis::BufferDesc{
        .size = computeOutputSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Resolve continuation compute output",
    });
    auto computeReadback = device.createBuffer(rachis::BufferDesc{
        .size = computeOutputSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Resolve continuation compute readback",
    });
    REQUIRE(computeOutput.has_value());
    REQUIRE(computeReadback.has_value());
    const std::array computeBindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto computeBindingLayout = device.createBindingLayout({.entries = computeBindingEntries});
    REQUIRE(computeBindingLayout.has_value());
    const std::array computeWrites{
        rachis::BindingWrite{
            .binding = 0,
            .resource =
                rachis::BufferBinding{
                    .buffer = *computeOutput,
                    .size = computeOutputSize,
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
    };
    auto computeBindingSet =
        rachis::test::createBindingSet(device, {.layout = *computeBindingLayout, .writes = computeWrites});
    REQUIRE(computeBindingSet.has_value());
    const std::array computeBindingLayouts{*computeBindingLayout};
    const std::array computePushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)},
    };
    auto computePipelineLayout = device.createPipelineLayout({
        .bindingLayouts = computeBindingLayouts,
        .pushConstantRanges = computePushConstantRanges,
    });
    REQUIRE(computePipelineLayout.has_value());
    auto computePipeline = device.createComputePipeline({
        .layout = *computePipelineLayout,
        .computeShader = *computeShader,
        .label = "Resolve continuation compute pipeline",
    });
    REQUIRE(computePipeline.has_value());

    const auto vertexBytecode = loadShader(RACHIS_TEST_GRAPHICS_DIRECT_VERTEX_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_GRAPHICS_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());
    auto vertexShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .entryPoint = "directVertex",
        .shaderInterface = {},
    });
    auto pixelShader = device.createShaderModule(rachis::ShaderModuleDesc{
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
    auto pipelineLayout = device.createPipelineLayout(rachis::PipelineLayoutDesc{
        .pushConstantRanges = pushConstantRanges,
    });
    REQUIRE(pipelineLayout.has_value());
    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    auto pipeline = device.createGraphicsPipeline(rachis::GraphicsPipelineDesc{
        .layout = *pipelineLayout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .sampleCount = colorSampleCount,
        .sampleMask = 1,
        .alphaToCoverageEnabled = true,
    });
    REQUIRE(pipeline.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *colorSource,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array colorAttachments{
        rachis::ColorAttachment{
            .view = *colorSourceView,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.0F, 0.0F, 0.0F, 0.0F},
        },
    };
    REQUIRE(device.beginRendering(*command, rachis::RenderingDesc{.colorAttachments = colorAttachments}).has_value());
    CHECK_FALSE(device
                    .resolveTexture(*command,
                                    rachis::TextureResolve{
                                        .source = *colorSource,
                                        .destination = *colorDestination,
                                    })
                    .has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *pipeline).has_value());
    REQUIRE(device
                .setViewport(*command,
                             rachis::Viewport{
                                 .width = static_cast<float>(targetWidth),
                                 .height = static_cast<float>(targetHeight),
                             })
                .has_value());
    REQUIRE(device
                .setScissor(*command,
                            rachis::ScissorRect{
                                .width = targetWidth,
                                .height = targetHeight,
                            })
                .has_value());
    const std::array tint{0.125F, 0.5F, 0.75F, 1.0F};
    REQUIRE(device.setGraphicsPushConstants(*command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(tint)))
                .has_value());
    REQUIRE(device.draw(*command, 3).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *colorSource,
                                    .stagesBefore = rachis::PipelineStage::renderTarget,
                                    .stagesAfter = rachis::PipelineStage::resolve,
                                    .accessBefore = rachis::Access::renderTarget,
                                    .accessAfter = rachis::Access::resolveSource,
                                    .layoutBefore = rachis::TextureLayout::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::resolveSource,
                                })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *colorDestination,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::resolve,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::resolveDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::resolveDestination,
                                })
                .has_value());
    CHECK_FALSE(device
                    .resolveTexture(*command,
                                    rachis::TextureResolve{
                                        .source = *colorSource,
                                        .destination = *colorDestination,
                                        .mode = rachis::ResolveMode::minimum,
                                    })
                    .has_value());
    const auto beforeRejectedColorResolve = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(beforeRejectedColorResolve.has_value());
    {
        const FailResourceRetention failDestinationPublication(2);
        auto failedResolve = device.resolveTexture(*command,
                                                   rachis::TextureResolve{
                                                       .source = *colorSource,
                                                       .destination = *colorDestination,
                                                   });
        REQUIRE_FALSE(failedResolve.has_value());
        CHECK(failedResolve.error().code == rachis::ErrorCode::outOfMemory);
    }
    const auto afterRejectedColorResolve = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(afterRejectedColorResolve.has_value());
    CHECK(*afterRejectedColorResolve == *beforeRejectedColorResolve);
    REQUIRE(device
                .resolveTexture(*command,
                                rachis::TextureResolve{
                                    .source = *colorSource,
                                    .destination = *colorDestination,
                                })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *colorDestination,
                                    .stagesBefore = rachis::PipelineStage::resolve,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::resolveDestination,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::resolveDestination,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device
                .copyTextureToBuffer(*command,
                                     rachis::BufferTextureCopy{
                                         .buffer = *colorReadback,
                                         .bytesPerRow = rowPitch,
                                         .rowsPerImage = targetHeight,
                                         .texture = *colorDestination,
                                         .extent = {targetWidth, targetHeight, 1},
                                     })
                .has_value());
    auto colorSubmitted = device.submit(std::move(*command));
    REQUIRE(colorSubmitted.has_value());
    auto colorWaited = device.wait(*colorSubmitted);
    if (!colorWaited)
    {
        INFO(colorWaited.error().message);
    }
    REQUIRE(colorWaited.has_value());
    auto unretainedPool = device.createCommandPool({
        .queueType = rachis::QueueType::graphics,
        .maxCommandBuffers = 1,
        .label = "Resolve private-retention command pool",
        .flags = rachis::CommandPoolFlag::unretainedResources,
    });
    REQUIRE(unretainedPool.has_value());
    command = device.acquireCommandBuffer(*unretainedPool);
    REQUIRE(command.has_value());

    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *depthSource,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::depthStencil,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::depthStencilWrite,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::depthStencilWrite,
                                    .subresources = {.aspect = rachis::TextureAspect::depth},
                                })
                .has_value());
    const rachis::DepthStencilAttachment depthAttachment{
        .view = *depthSourceView,
        .depthLoadOperation = rachis::LoadOperation::clear,
        .depthClearValue = 0.625F,
    };
    REQUIRE(
        device.beginRendering(*command, rachis::RenderingDesc{.depthStencilAttachment = depthAttachment}).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    CHECK_FALSE(device
                    .resolveTexture(*command,
                                    rachis::TextureResolve{
                                        .source = *depthSource,
                                        .sourceSubresource = {.aspect = rachis::TextureAspect::depth},
                                        .destination = *depthDestination,
                                        .destinationSubresource = {.aspect = rachis::TextureAspect::depth},
                                        .mode = rachis::ResolveMode::average,
                                    })
                    .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *depthSource,
                                    .stagesBefore = rachis::PipelineStage::depthStencil,
                                    .stagesAfter = rachis::PipelineStage::resolve,
                                    .accessBefore = rachis::Access::depthStencilWrite,
                                    .accessAfter = rachis::Access::resolveSource,
                                    .layoutBefore = rachis::TextureLayout::depthStencilWrite,
                                    .layoutAfter = rachis::TextureLayout::resolveSource,
                                    .subresources = {.aspect = rachis::TextureAspect::depth},
                                })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *depthDestination,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::resolve,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::resolveDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::resolveDestination,
                                    .subresources = {.aspect = rachis::TextureAspect::depth},
                                })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *computeOutput,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *computePipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *computeBindingSet).has_value());
    const std::array computeConstants{3U, 7U};
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(computeConstants))).has_value());
    const auto beforeRejectedDepthResolve = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(beforeRejectedDepthResolve.has_value());
    {
        const FailResourceRetention failDestinationDescriptorPublication(4);
        auto failedResolve = device.resolveTexture(*command,
                                                   rachis::TextureResolve{
                                                       .source = *depthSource,
                                                       .sourceSubresource = {.aspect = rachis::TextureAspect::depth},
                                                       .destination = *depthDestination,
                                                       .destinationSubresource =
                                                           {
                                                               .aspect = rachis::TextureAspect::depth,
                                                           },
                                                       .mode = rachis::ResolveMode::minimum,
                                                   });
        REQUIRE_FALSE(failedResolve.has_value());
        CHECK(failedResolve.error().code == rachis::ErrorCode::outOfMemory);
    }
    const auto afterRejectedDepthResolve = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(afterRejectedDepthResolve.has_value());
    CHECK(*afterRejectedDepthResolve == *beforeRejectedDepthResolve);
    REQUIRE(device
                .resolveTexture(*command,
                                rachis::TextureResolve{
                                    .source = *depthSource,
                                    .sourceSubresource = {.aspect = rachis::TextureAspect::depth},
                                    .destination = *depthDestination,
                                    .destinationSubresource = {.aspect = rachis::TextureAspect::depth},
                                    .mode = rachis::ResolveMode::minimum,
                                })
                .has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *computeOutput,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device.copyBuffer(*command, {*computeOutput, 0, *computeReadback, 0, computeOutputSize}).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *depthDestination,
                                    .stagesBefore = rachis::PipelineStage::resolve,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::resolveDestination,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::resolveDestination,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                    .subresources = {.aspect = rachis::TextureAspect::depth},
                                })
                .has_value());
    REQUIRE(device
                .copyTextureToBuffer(*command,
                                     rachis::BufferTextureCopy{
                                         .buffer = *depthReadback,
                                         .bytesPerRow = rowPitch,
                                         .rowsPerImage = targetHeight,
                                         .texture = *depthDestination,
                                         .textureSubresource = {.aspect = rachis::TextureAspect::depth},
                                         .extent = {targetWidth, targetHeight, 1},
                                     })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *depthDestination,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::resolve,
                                    .accessBefore = rachis::Access::copySource,
                                    .accessAfter = rachis::Access::resolveDestination,
                                    .layoutBefore = rachis::TextureLayout::copySource,
                                    .layoutAfter = rachis::TextureLayout::resolveDestination,
                                    .subresources = {.aspect = rachis::TextureAspect::depth},
                                })
                .has_value());
    REQUIRE(device
                .resolveTexture(*command,
                                rachis::TextureResolve{
                                    .source = *depthSource,
                                    .sourceSubresource = {.aspect = rachis::TextureAspect::depth},
                                    .destination = *depthDestination,
                                    .destinationSubresource = {.aspect = rachis::TextureAspect::depth},
                                    .mode = rachis::ResolveMode::maximum,
                                })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *depthDestination,
                                    .stagesBefore = rachis::PipelineStage::resolve,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::resolveDestination,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::resolveDestination,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                    .subresources = {.aspect = rachis::TextureAspect::depth},
                                })
                .has_value());
    REQUIRE(device
                .copyTextureToBuffer(*command,
                                     rachis::BufferTextureCopy{
                                         .buffer = *depthReadback,
                                         .bufferOffset = readbackSize,
                                         .bytesPerRow = rowPitch,
                                         .rowsPerImage = targetHeight,
                                         .texture = *depthDestination,
                                         .textureSubresource = {.aspect = rachis::TextureAspect::depth},
                                         .extent = {targetWidth, targetHeight, 1},
                                     })
                .has_value());

    auto copyCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(copyCommand.has_value());
    CHECK_FALSE(device
                    .resolveTexture(*copyCommand,
                                    rachis::TextureResolve{
                                        .source = *colorSource,
                                        .destination = *colorDestination,
                                    })
                    .has_value());

    REQUIRE(device.destroyTexture(*colorSource).has_value());
    REQUIRE(device.destroyTexture(*colorDestination).has_value());
    REQUIRE(device.destroyTextureView(*colorSourceView).has_value());
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

    REQUIRE(device.destroyTexture(*depthSource).has_value());
    REQUIRE(device.destroyTexture(*depthDestination).has_value());
    REQUIRE(device.destroyTextureView(*depthSourceView).has_value());
    REQUIRE(device.destroyPipeline(*computePipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*computePipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*computeShader).has_value());
    REQUIRE(device.destroyBindingSet(*computeBindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*computeBindingLayout).has_value());
    REQUIRE(device.destroyBuffer(*computeOutput).has_value());
    REQUIRE(device.destroyCommandPool(*unretainedPool).has_value());

    auto mappedColor = device.mapBuffer(*colorReadback);
    REQUIRE(mappedColor.has_value());
    const auto* corner = mappedColor->data;
    const auto* center = mappedColor->data + static_cast<std::uint64_t>(targetHeight / 2) * rowPitch +
                         static_cast<std::uint64_t>(targetWidth / 2) * 4;
    CHECK(corner[0] == std::byte{0});
    CHECK(corner[1] == std::byte{0});
    CHECK(corner[2] == std::byte{0});
    CHECK(corner[3] == std::byte{0});
    const auto expectedMaskedByte = [colorSampleCount](std::uint32_t value) {
        return static_cast<std::uint8_t>((value + colorSampleCount / 2U) / colorSampleCount);
    };
    CHECK(nearByte(center[0], expectedMaskedByte(32)));
    CHECK(nearByte(center[1], expectedMaskedByte(128)));
    CHECK(nearByte(center[2], expectedMaskedByte(191)));
    CHECK(nearByte(center[3], expectedMaskedByte(255)));
    REQUIRE(device.unmapBuffer(*colorReadback).has_value());

    auto mappedDepth = device.mapBuffer(*depthReadback);
    REQUIRE(mappedDepth.has_value());
    float minimumDepth = 0.0F;
    float maximumDepth = 0.0F;
    std::memcpy(&minimumDepth, mappedDepth->data, sizeof(minimumDepth));
    std::memcpy(&maximumDepth, mappedDepth->data + readbackSize, sizeof(maximumDepth));
    CHECK(minimumDepth == 0.625F);
    CHECK(maximumDepth == 0.625F);
    REQUIRE(device.unmapBuffer(*depthReadback).has_value());

    auto mappedCompute = device.mapBuffer(*computeReadback);
    REQUIRE(mappedCompute.has_value());
    std::uint32_t firstComputeValue = 0;
    std::uint32_t seventeenthComputeValue = 0;
    std::memcpy(&firstComputeValue, mappedCompute->data, sizeof(firstComputeValue));
    std::memcpy(&seventeenthComputeValue,
                mappedCompute->data + 17 * sizeof(seventeenthComputeValue),
                sizeof(seventeenthComputeValue));
    CHECK(firstComputeValue == 7);
    CHECK(seventeenthComputeValue == 58);
    REQUIRE(device.unmapBuffer(*computeReadback).has_value());

    REQUIRE(device.destroyBuffer(*colorReadback).has_value());
    REQUIRE(device.destroyBuffer(*depthReadback).has_value());
    REQUIRE(device.destroyBuffer(*computeReadback).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("multisampled integer color targets clear and reject average resolves on WARP",
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

    constexpr std::uint32_t sampleCount = 4;
    const auto integerProperties = device.queryFormatProperties(rachis::Format::rgba8Uint);
    const auto unormProperties = device.queryFormatProperties(rachis::Format::rgba8Unorm);
    REQUIRE(rachis::hasAny(integerProperties.features, rachis::FormatFeature::colorAttachment));
    REQUIRE(rachis::supportsSampleCount(integerProperties.sampleCounts, sampleCount));
    REQUIRE(rachis::supportsSampleCount(unormProperties.sampleCounts, sampleCount));
    CHECK_FALSE(rachis::hasAny(integerProperties.features, rachis::FormatFeature::multisampleResolve));
    CHECK(integerProperties.colorResolveModes == rachis::ResolveModeMask::none);
    CHECK(rachis::supportsResolveMode(unormProperties.colorResolveModes, rachis::ResolveMode::average));

    struct ResolveTargets final
    {
        rachis::TextureHandle source;
        rachis::TextureHandle destination;
        rachis::TextureViewHandle sourceView;
    };
    const auto createTargets = [&device](rachis::Format format) {
        auto source = device.createTexture(rachis::TextureDesc{
            .width = targetWidth,
            .height = targetHeight,
            .sampleCount = sampleCount,
            .format = format,
            .usage = rachis::TextureUsage::renderTarget,
            .label = "MSAA resolve format source",
        });
        INFO("createTexture: " << (source ? std::string{} : source.error().message));
        REQUIRE(source.has_value());
        auto destination = device.createTexture(rachis::TextureDesc{
            .width = targetWidth,
            .height = targetHeight,
            .format = format,
            .usage = rachis::TextureUsage::renderTarget,
            .label = "MSAA resolve format destination",
        });
        REQUIRE(destination.has_value());
        auto sourceView = device.createTextureView(rachis::TextureViewDesc{
            .texture = *source,
            .usage = rachis::TextureViewUsage::renderTarget,
            .label = "MSAA resolve format source view",
        });
        REQUIRE(sourceView.has_value());
        return ResolveTargets{
            .source = *source,
            .destination = *destination,
            .sourceView = *sourceView,
        };
    };
    const auto integerTargets = createTargets(rachis::Format::rgba8Uint);
    const auto unormTargets = createTargets(rachis::Format::rgba8Unorm);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    const auto clearAndPrepareResolve = [&device, &command](const ResolveTargets& targets) {
        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = targets.source,
                                        .stagesBefore = rachis::PipelineStage::all,
                                        .stagesAfter = rachis::PipelineStage::renderTarget,
                                        .accessBefore = rachis::Access::none,
                                        .accessAfter = rachis::Access::renderTarget,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::renderTarget,
                                    })
                    .has_value());
        const std::array colorAttachments{
            rachis::ColorAttachment{
                .view = targets.sourceView,
                .loadOperation = rachis::LoadOperation::clear,
                .clearValue = {0.0F, 0.0F, 0.0F, 0.0F},
            },
        };
        REQUIRE(
            device.beginRendering(*command, rachis::RenderingDesc{.colorAttachments = colorAttachments}).has_value());
        REQUIRE(device.endRendering(*command).has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = targets.source,
                                        .stagesBefore = rachis::PipelineStage::renderTarget,
                                        .stagesAfter = rachis::PipelineStage::resolve,
                                        .accessBefore = rachis::Access::renderTarget,
                                        .accessAfter = rachis::Access::resolveSource,
                                        .layoutBefore = rachis::TextureLayout::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::resolveSource,
                                    })
                    .has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = targets.destination,
                                        .stagesBefore = rachis::PipelineStage::all,
                                        .stagesAfter = rachis::PipelineStage::resolve,
                                        .accessBefore = rachis::Access::none,
                                        .accessAfter = rachis::Access::resolveDestination,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::resolveDestination,
                                    })
                    .has_value());
    };
    clearAndPrepareResolve(integerTargets);
    clearAndPrepareResolve(unormTargets);

    const auto beforeRejectedResolve = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(beforeRejectedResolve.has_value());
    auto integerResolve = device.resolveTexture(*command,
                                                rachis::TextureResolve{
                                                    .source = integerTargets.source,
                                                    .destination = integerTargets.destination,
                                                });
    REQUIRE_FALSE(integerResolve.has_value());
    CHECK(integerResolve.error().code == rachis::ErrorCode::unsupported);
    const auto afterRejectedResolve = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(afterRejectedResolve.has_value());
    CHECK(*afterRejectedResolve == *beforeRejectedResolve);
    REQUIRE(device
                .resolveTexture(*command,
                                rachis::TextureResolve{
                                    .source = unormTargets.source,
                                    .destination = unormTargets.destination,
                                })
                .has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    auto waited = device.wait(*submitted);
    if (!waited)
    {
        INFO(waited.error().message);
    }
    REQUIRE(waited.has_value());

    for (const auto& targets : {integerTargets, unormTargets})
    {
        REQUIRE(device.destroyTextureView(targets.sourceView).has_value());
        REQUIRE(device.destroyTexture(targets.source).has_value());
        REQUIRE(device.destroyTexture(targets.destination).has_value());
    }
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("multisampled integer color textures report no resolve and never resolve on WARP",
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

    for (const auto format :
         {rachis::Format::rgba8Uint, rachis::Format::r32Uint, rachis::Format::rgba16Sint, rachis::Format::rg32Uint})
    {
        INFO("Format " << static_cast<int>(format));
        const auto properties = device.queryFormatProperties(format);
        CHECK(properties.colorResolveModes == rachis::ResolveModeMask::none);
        CHECK_FALSE(rachis::hasAny(properties.features, rachis::FormatFeature::multisampleResolve));
        const auto sampleCount = preferredSampleCount(properties.sampleCounts);
        if (!rachis::hasAny(properties.features, rachis::FormatFeature::colorAttachment) || sampleCount == 0)
        {
            continue;
        }

        auto source = device.createTexture(rachis::TextureDesc{
            .width = targetWidth,
            .height = targetHeight,
            .sampleCount = sampleCount,
            .format = format,
            .usage = rachis::TextureUsage::renderTarget,
            .label = "Integer MSAA source",
        });
        INFO("createTexture: " << (source ? std::string{} : source.error().message));
        REQUIRE(source.has_value());
        auto destination = device.createTexture(rachis::TextureDesc{
            .width = targetWidth,
            .height = targetHeight,
            .format = format,
            .usage = rachis::TextureUsage::renderTarget,
            .label = "Integer resolve destination",
        });
        REQUIRE(destination.has_value());

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = *source,
                                        .stagesBefore = rachis::PipelineStage::all,
                                        .stagesAfter = rachis::PipelineStage::resolve,
                                        .accessBefore = rachis::Access::none,
                                        .accessAfter = rachis::Access::resolveSource,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::resolveSource,
                                    })
                    .has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = *destination,
                                        .stagesBefore = rachis::PipelineStage::all,
                                        .stagesAfter = rachis::PipelineStage::resolve,
                                        .accessBefore = rachis::Access::none,
                                        .accessAfter = rachis::Access::resolveDestination,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::resolveDestination,
                                    })
                    .has_value());
        auto resolved = device.resolveTexture(*command,
                                              rachis::TextureResolve{
                                                  .source = *source,
                                                  .destination = *destination,
                                              });
        REQUIRE_FALSE(resolved.has_value());
        CHECK(resolved.error().code == rachis::ErrorCode::unsupported);

        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());
        REQUIRE(device.destroyTexture(*source).has_value());
        REQUIRE(device.destroyTexture(*destination).has_value());
    }
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
