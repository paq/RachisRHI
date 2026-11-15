#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <vector>

namespace
{

    constexpr std::uint32_t targetWidth = 64;
    constexpr std::uint32_t targetHeight = 64;
    constexpr std::uint32_t rowPitch = 256;
    constexpr std::uint64_t readbackSize = static_cast<std::uint64_t>(rowPitch) * targetHeight;
    constexpr std::uint32_t valueCount = 64;
    constexpr std::uint64_t outputSize = valueCount * sizeof(std::uint32_t);

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

    [[nodiscard]] bool nearByte(std::byte actual, std::uint8_t expected) noexcept
    {
        const auto value = std::to_integer<std::uint8_t>(actual);
        const auto lower = expected == 0 ? 0U : static_cast<std::uint32_t>(expected) - 1U;
        const auto upper = (std::min)(static_cast<std::uint32_t>(expected) + 1U, 255U);
        return value >= lower && value <= upper;
    }

    [[nodiscard]] const std::byte* pixelAt(const std::byte* data, std::uint32_t x, std::uint32_t y) noexcept
    {
        return data + static_cast<std::uint64_t>(y) * rowPitch + static_cast<std::uint64_t>(x) * 4;
    }

    [[nodiscard]] rachis::ShaderInterfaceDesc argumentWriterInterface() noexcept
    {
        static constexpr std::array bindings{
            rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1},
        };
        return rachis::test::computeShaderInterface({1, 1, 1}, bindings);
    }

} // namespace

TEST_CASE("Vulkan indirect draws reject malformed ranges and execute GPU-written arguments",
          "[size:medium][rhi][vulkan][indirect][draw]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    const auto writerBytecode =
        loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanIndirectArguments.draw.spv");
    const auto indexedWriterBytecode =
        loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanIndirectArguments.indexed.spv");
    const auto vertexBytecode =
        loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanGraphicsTriangle.direct.vs.spv");
    const auto inputVertexBytecode =
        loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanGraphicsTriangle.input.vs.spv");
    const auto pixelBytecode =
        loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanGraphicsTriangle.ps.spv");
    REQUIRE_FALSE(writerBytecode.empty());
    REQUIRE_FALSE(indexedWriterBytecode.empty());
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(inputVertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());

    auto writerShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = writerBytecode,
        .entryPoint = "writeDraw",
        .shaderInterface = argumentWriterInterface(),
    });
    auto indexedWriterShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = indexedWriterBytecode,
        .entryPoint = "writeIndexedDraw",
        .shaderInterface = argumentWriterInterface(),
    });
    auto vertexShader = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .entryPoint = "directVertex",
        .shaderInterface = {},
    });
    auto inputVertexShader = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = inputVertexBytecode,
        .entryPoint = "inputVertex",
        .shaderInterface = rachis::test::graphicsInputVertexShaderInterface(),
    });
    auto pixelShader = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "pixelMain",
        .shaderInterface = rachis::test::graphicsPixelShaderInterface(),
    });
    REQUIRE(writerShader);
    REQUIRE(indexedWriterShader);
    REQUIRE(vertexShader);
    REQUIRE(inputVertexShader);
    REQUIRE(pixelShader);

    const std::array computeBindings{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto computeLayout = device.createBindingLayout({.entries = computeBindings});
    REQUIRE(computeLayout);
    const std::array computeLayouts{*computeLayout};
    auto writerPipelineLayout = device.createPipelineLayout({.bindingLayouts = computeLayouts});
    REQUIRE(writerPipelineLayout);
    auto writerPipeline = device.createComputePipeline({
        .layout = *writerPipelineLayout,
        .computeShader = *writerShader,
    });
    auto indexedWriterPipeline = device.createComputePipeline({
        .layout = *writerPipelineLayout,
        .computeShader = *indexedWriterShader,
    });
    REQUIRE(writerPipeline);
    REQUIRE(indexedWriterPipeline);

    const std::array pushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::pixel, 0, 4 * sizeof(float)},
    };
    auto graphicsLayout = device.createPipelineLayout({.pushConstantRanges = pushConstantRanges});
    REQUIRE(graphicsLayout);
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    auto directPipeline = device.createGraphicsPipeline({
        .layout = *graphicsLayout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
    });
    const std::array vertexAttributes{
        rachis::VertexAttribute{.format = rachis::VertexFormat::float32x2, .shaderLocation = 0},
    };
    const std::array vertexLayouts{
        rachis::VertexBufferLayout{.stride = 2 * sizeof(float), .attributes = vertexAttributes},
    };
    auto indexedPipeline = device.createGraphicsPipeline({
        .layout = *graphicsLayout,
        .vertexShader = *inputVertexShader,
        .pixelShader = *pixelShader,
        .vertexBuffers = vertexLayouts,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
    });
    REQUIRE(directPipeline);
    REQUIRE(indexedPipeline);

    auto target = device.createTexture({
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
    });
    REQUIRE(target);
    auto targetView = device.createTextureView({
        .texture = *target,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    REQUIRE(targetView);
    auto readback = device.createBuffer({
        .size = readbackSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(readback);

    const std::array vertices{std::array{0.0F, 0.75F}, std::array{0.75F, -0.75F}, std::array{-0.75F, -0.75F}};
    const std::array<std::uint16_t, 3> indices{0, 1, 2};
    auto vertexBuffer = device.createBuffer({
        .size = sizeof(vertices),
        .usage = rachis::BufferUsage::vertex,
        .memoryType = rachis::MemoryType::upload,
    });
    auto indexBuffer = device.createBuffer({
        .size = sizeof(indices),
        .usage = rachis::BufferUsage::index,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(vertexBuffer);
    REQUIRE(indexBuffer);
    auto mappedVertices = device.mapBuffer(*vertexBuffer);
    auto mappedIndices = device.mapBuffer(*indexBuffer);
    REQUIRE(mappedVertices);
    REQUIRE(mappedIndices);
    std::memcpy(mappedVertices->data, vertices.data(), sizeof(vertices));
    std::memcpy(mappedIndices->data, indices.data(), sizeof(indices));
    REQUIRE(device.unmapBuffer(*vertexBuffer).has_value());
    REQUIRE(device.unmapBuffer(*indexBuffer).has_value());

    auto arguments = device.createBuffer({
        .size = 32,
        .usage = rachis::BufferUsage::copyDestination | rachis::BufferUsage::storage | rachis::BufferUsage::indirect,
        .memoryType = rachis::MemoryType::deviceLocal,
    });
    auto zeroArgumentsUpload = device.createBuffer({
        .size = 32,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto wrongUsage = device.createBuffer({
        .size = 32,
        .usage = rachis::BufferUsage::storage,
    });
    auto count = device.createBuffer({
        .size = 4,
        .usage = rachis::BufferUsage::indirect,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(arguments);
    REQUIRE(zeroArgumentsUpload);
    REQUIRE(wrongUsage);
    REQUIRE(count);
    {
        auto mappedZero = device.mapBuffer(*zeroArgumentsUpload);
        auto mappedCount = device.mapBuffer(*count);
        REQUIRE(mappedZero);
        REQUIRE(mappedCount);
        std::memset(mappedZero->data, 0, 32);
        const std::uint32_t countValue = 1U;
        std::memcpy(mappedCount->data, &countValue, sizeof(countValue));
        REQUIRE(device.unmapBuffer(*zeroArgumentsUpload).has_value());
        REQUIRE(device.unmapBuffer(*count).has_value());
    }
    const std::array argumentWrites{
        rachis::BindingWrite{
            .binding = 0,
            .resource =
                rachis::BufferBinding{
                    .buffer = *arguments,
                    .size = 32,
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
    };
    auto argumentSet = rachis::test::createBindingSet(device,
                                                      {
                                                          .layout = *computeLayout,
                                                          .writes = argumentWrites,
                                                      });
    REQUIRE(argumentSet);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *arguments,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessAfter = rachis::Access::copyDestination,
                               })
                .has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            {
                                .source = *zeroArgumentsUpload,
                                .destination = *arguments,
                                .size = 32,
                            })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *arguments,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *writerPipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *argumentSet).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *arguments,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::indirect,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::indirectArgument,
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

    const std::array colorAttachments{
        rachis::ColorAttachment{
            .view = *targetView,
            .loadOperation = rachis::LoadOperation::clear,
        },
    };
    REQUIRE(device.beginRendering(*command, {.colorAttachments = colorAttachments}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *directPipeline).has_value());
    CHECK_FALSE(device.drawIndirect(*command, {.argumentBuffer = *arguments}).has_value());
    REQUIRE(device
                .setViewport(*command,
                             {
                                 .width = static_cast<float>(targetWidth),
                                 .height = static_cast<float>(targetHeight),
                             })
                .has_value());
    REQUIRE(device.setScissor(*command, {.width = targetWidth, .height = targetHeight}).has_value());
    const std::array tint{0.125F, 0.5F, 0.75F, 1.0F};
    REQUIRE(device.setGraphicsPushConstants(*command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(tint)))
                .has_value());

    CHECK_FALSE(device.drawIndirect(*command, {.argumentBuffer = *wrongUsage}).has_value());
    CHECK_FALSE(device.drawIndirect(*command, {.argumentBuffer = *arguments, .argumentOffset = 2}).has_value());
    CHECK_FALSE(
        device.drawIndirect(*command, {.argumentBuffer = *arguments, .argumentOffset = 4, .stride = 12}).has_value());
    CHECK_FALSE(device.drawIndirect(*command, {.argumentBuffer = *arguments, .countOffset = 4}).has_value());
    const auto& capabilities = rachis::test::requireCapabilities(device);
    auto counted = device.drawIndirect(*command,
                                       {
                                           .argumentBuffer = *arguments,
                                           .maxDrawCount = 1,
                                           .countBuffer = *count,
                                       });
    if (capabilities.drawIndirectCount)
    {
        REQUIRE(counted.has_value());
    }
    else
    {
        CHECK_FALSE(counted.has_value());
        CHECK(counted.error().code == rachis::ErrorCode::unsupported);
    }
    auto multi = device.drawIndirect(*command, {.argumentBuffer = *arguments, .maxDrawCount = 2});
    if (capabilities.multiDrawIndirect)
    {
        REQUIRE(multi.has_value());
    }
    else
    {
        CHECK_FALSE(multi.has_value());
        CHECK(multi.error().code == rachis::ErrorCode::unsupported);
    }
    REQUIRE(device.drawIndirect(*command, {.argumentBuffer = *arguments, .maxDrawCount = 0}).has_value());
    REQUIRE(device.drawIndirect(*command, {.argumentBuffer = *arguments}).has_value());

    REQUIRE(device.bindGraphicsPipeline(*command, *indexedPipeline).has_value());
    CHECK_FALSE(device.drawIndexedIndirect(*command, {.argumentBuffer = *arguments}).has_value());
    REQUIRE(device.endRendering(*command).has_value());

    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *arguments,
                                   .stagesBefore = rachis::PipelineStage::indirect,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::indirectArgument,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *indexedWriterPipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *argumentSet).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *arguments,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::indirect,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::indirectArgument,
                               })
                .has_value());
    REQUIRE(device
                .beginRendering(*command,
                                {
                                    .colorAttachments =
                                        std::array{
                                            rachis::ColorAttachment{
                                                .view = *targetView,
                                                .loadOperation = rachis::LoadOperation::load,
                                                .storeOperation = rachis::StoreOperation::store,
                                            },
                                        },
                                })
                .has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *indexedPipeline).has_value());
    REQUIRE(device.bindVertexBuffer(*command, 0, *vertexBuffer).has_value());
    REQUIRE(device.bindIndexBuffer(*command, *indexBuffer, rachis::IndexFormat::uint16).has_value());
    REQUIRE(device.setGraphicsPushConstants(*command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(tint)))
                .has_value());
    REQUIRE(device
                .setViewport(*command,
                             {
                                 .width = static_cast<float>(targetWidth),
                                 .height = static_cast<float>(targetHeight),
                             })
                .has_value());
    REQUIRE(device.setScissor(*command, {.width = targetWidth, .height = targetHeight}).has_value());
    REQUIRE(device.drawIndexedIndirect(*command, {.argumentBuffer = *arguments}).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    REQUIRE(device
                .textureBarrier(*command,
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
    REQUIRE(device
                .copyTextureToBuffer(*command,
                                     {
                                         .buffer = *readback,
                                         .bytesPerRow = rowPitch,
                                         .rowsPerImage = targetHeight,
                                         .texture = *target,
                                         .extent = {targetWidth, targetHeight, 1},
                                     })
                .has_value());

    REQUIRE(device.destroyTextureView(*targetView).has_value());
    REQUIRE(device.destroyTexture(*target).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped);
    const auto* center = pixelAt(mapped->data, targetWidth / 2, targetHeight / 2);
    CHECK(nearByte(center[0], 32));
    CHECK(nearByte(center[1], 128));
    CHECK(nearByte(center[2], 191));
    CHECK(nearByte(center[3], 255));
    REQUIRE(device.unmapBuffer(*readback).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan indirect dispatch executes GPU-written group counts",
          "[size:medium][rhi][vulkan][indirect][dispatch]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    const auto writerBytecode =
        loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanIndirectArguments.dispatch.spv");
    const auto computeBytecode = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanComputeWrite.spv");
    REQUIRE_FALSE(writerBytecode.empty());
    REQUIRE_FALSE(computeBytecode.empty());
    auto writerShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = writerBytecode,
        .entryPoint = "writeDispatch",
        .shaderInterface = argumentWriterInterface(),
    });
    auto computeShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = computeBytecode,
        .entryPoint = "main",
        .shaderInterface = rachis::test::computeWriteShaderInterface(),
    });
    REQUIRE(writerShader);
    REQUIRE(computeShader);

    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = bindingEntries});
    REQUIRE(bindingLayout);
    const std::array layouts{*bindingLayout};
    auto writerLayout = device.createPipelineLayout({.bindingLayouts = layouts});
    const std::array pushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)},
    };
    auto computeLayout = device.createPipelineLayout({
        .bindingLayouts = layouts,
        .pushConstantRanges = pushConstantRanges,
    });
    REQUIRE(writerLayout);
    REQUIRE(computeLayout);
    auto writerPipeline = device.createComputePipeline({.layout = *writerLayout, .computeShader = *writerShader});
    auto computePipeline = device.createComputePipeline({.layout = *computeLayout, .computeShader = *computeShader});
    REQUIRE(writerPipeline);
    REQUIRE(computePipeline);

    auto arguments = device.createBuffer({
        .size = 16,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::indirect,
        .memoryType = rachis::MemoryType::deviceLocal,
    });
    auto storage = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
    });
    auto readback = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    auto count = device.createBuffer({.size = 4, .usage = rachis::BufferUsage::indirect});
    REQUIRE(arguments);
    REQUIRE(storage);
    REQUIRE(readback);
    REQUIRE(count);

    const std::array argumentWrites{
        rachis::BindingWrite{
            .binding = 0,
            .resource =
                rachis::BufferBinding{
                    .buffer = *arguments,
                    .size = 16,
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
    };
    const std::array storageWrites{
        rachis::BindingWrite{
            .binding = 0,
            .resource =
                rachis::BufferBinding{
                    .buffer = *storage,
                    .size = outputSize,
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
    };
    auto argumentSet = rachis::test::createBindingSet(device, {.layout = *bindingLayout, .writes = argumentWrites});
    auto storageSet = rachis::test::createBindingSet(device, {.layout = *bindingLayout, .writes = storageWrites});
    REQUIRE(argumentSet);
    REQUIRE(storageSet);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command);
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *arguments,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *writerPipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *argumentSet).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *arguments,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::indirect,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::indirectArgument,
                               })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *storage,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *computePipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *storageSet).has_value());
    const std::array constants{3U, 7U};
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(constants))).has_value());
    auto counted = device.dispatchIndirect(*command,
                                           {
                                               .argumentBuffer = *arguments,
                                               .countBuffer = *count,
                                           });
    CHECK_FALSE(counted.has_value());
    CHECK(counted.error().code == rachis::ErrorCode::unsupported);
    CHECK_FALSE(device.dispatchIndirect(*command, {.argumentBuffer = *arguments, .maxDispatchCount = 2}).has_value());
    REQUIRE(device.dispatchIndirect(*command, {.argumentBuffer = *arguments, .maxDispatchCount = 0}).has_value());
    REQUIRE(device.dispatchIndirect(*command, *arguments, 0).has_value());
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
    REQUIRE(device
                .copyBuffer(*command,
                            {
                                .source = *storage,
                                .destination = *readback,
                                .size = outputSize,
                            })
                .has_value());

    REQUIRE(device.destroyBuffer(*arguments).has_value());
    REQUIRE(device.destroyBuffer(*storage).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped);
    const auto* values = reinterpret_cast<const std::uint32_t*>(mapped->data);
    CHECK(values[0] == 7);
    CHECK(values[1] == 10);
    CHECK(values[63] == 196);
    REQUIRE(device.unmapBuffer(*readback).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
