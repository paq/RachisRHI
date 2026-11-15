#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <vector>

namespace
{

    constexpr std::uint32_t valueCount = 64;
    constexpr std::uint64_t outputSize = valueCount * sizeof(std::uint32_t);
    constexpr std::uint32_t imageWidth = 8;
    constexpr std::uint32_t imageHeight = 8;
    constexpr std::uint32_t imageBytesPerPixel = 16;
    constexpr std::uint32_t imageRowPitch = imageWidth * imageBytesPerPixel;
    constexpr std::uint64_t imageReadbackSize = static_cast<std::uint64_t>(imageRowPitch) * imageHeight;

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

} // namespace

TEST_CASE("Vulkan compute dispatch writes storage buffers and survives logical destruction",
          "[size:medium][rhi][vulkan][compute]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    const auto bytecode = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanComputeWrite.spv");
    const auto scaleBytecode =
        loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanComputeWrite.scale.spv");
    REQUIRE_FALSE(bytecode.empty());
    REQUIRE_FALSE(scaleBytecode.empty());
    auto shader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Vulkan compute write",
        .shaderInterface = rachis::test::computeWriteShaderInterface(),
    });
    if (!shader)
    {
        INFO(shader.error().message);
    }
    REQUIRE(shader.has_value());
    auto scaleShader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::compute,
        .bytecode = scaleBytecode,
        .entryPoint = "scale",
        .label = "Vulkan compute scale",
        .shaderInterface = rachis::test::computeWriteShaderInterface(),
    });
    REQUIRE(scaleShader.has_value());

    auto storage = device.createBuffer(rachis::BufferDesc{
        .size = outputSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan compute output",
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = outputSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan compute readback",
    });
    REQUIRE(storage.has_value());
    REQUIRE(readback.has_value());

    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout(rachis::BindingLayoutDesc{
        .entries = bindingEntries,
        .label = "Vulkan compute output layout",
    });
    REQUIRE(bindingLayout.has_value());
    const std::array writes{
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
    auto bindingSet = rachis::test::createBindingSet(device,
                                                     rachis::BindingSetDesc{
                                                         .layout = *bindingLayout,
                                                         .writes = writes,
                                                         .label = "Vulkan compute output set",
                                                     });
    REQUIRE(bindingSet.has_value());

    const std::array bindingLayouts{*bindingLayout};
    const std::array pushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)},
    };
    auto pipelineLayout = device.createPipelineLayout(rachis::PipelineLayoutDesc{
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = pushConstantRanges,
        .label = "Vulkan compute write pipeline layout",
    });
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline(rachis::ComputePipelineDesc{
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Vulkan compute write pipeline",
    });
    REQUIRE(pipeline.has_value());
    auto scalePipeline = device.createComputePipeline(rachis::ComputePipelineDesc{
        .layout = *pipelineLayout,
        .computeShader = *scaleShader,
        .label = "Vulkan compute scale pipeline",
    });
    REQUIRE(scalePipeline.has_value());

    auto emptyLayout = device.createBindingLayout({});
    REQUIRE(emptyLayout.has_value());
    auto emptySet = rachis::test::createBindingSet(device, rachis::BindingSetDesc{.layout = *emptyLayout});
    REQUIRE(emptySet.has_value());

    auto copyCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(copyCommand.has_value());
    CHECK_FALSE(device.bindComputePipeline(*copyCommand, *pipeline).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    CHECK_FALSE(device.dispatch(*command, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *storage,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    CHECK_FALSE(device.dispatch(*command, 1).has_value());
    CHECK_FALSE(device.bindComputeBindingSet(*command, 0, *emptySet).has_value());
    CHECK_FALSE(device.bindComputeBindingSet(*command, 1, *bindingSet).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());

    const std::array firstConstants{3U, 7U};
    CHECK_FALSE(device.setComputePushConstants(*command, 4, std::as_bytes(std::span(firstConstants))).has_value());
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(firstConstants))).has_value());
    CHECK_FALSE(device.dispatch(*command, 0).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *storage,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *scalePipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
    const std::array scaleConstants{5U, 0U};
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(scaleConstants))).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *storage,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            rachis::BufferCopy{
                                .source = *storage,
                                .destination = *readback,
                                .size = outputSize,
                            })
                .has_value());

    const auto stalePipeline = *pipeline;
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipeline(*scalePipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyShaderModule(*scaleShader).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyBuffer(*storage).has_value());
    CHECK_FALSE(device.destroyPipeline(stalePipeline).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    REQUIRE(mapped->size == outputSize);
    std::uint32_t firstMismatch = valueCount;
    for (std::uint32_t index = 0; index < valueCount; ++index)
    {
        std::uint32_t value = 0;
        std::memcpy(&value, mapped->data + index * sizeof(value), sizeof(value));
        if (value != (index * firstConstants[0] + firstConstants[1]) * scaleConstants[0])
        {
            firstMismatch = index;
            break;
        }
    }
    CAPTURE(firstMismatch);
    CHECK(firstMismatch == valueCount);
    REQUIRE(device.unmapBuffer(*readback).has_value());
    REQUIRE(device.destroyBindingSet(*emptySet).has_value());
    REQUIRE(device.destroyBindingLayout(*emptyLayout).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan compute storage-image writes become visible to copies", "[size:medium][rhi][vulkan][compute][image]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    const auto bytecode = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanComputeImage.spv");
    REQUIRE_FALSE(bytecode.empty());
    static constexpr std::array bindings{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageTexture, 1},
    };
    auto shader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Vulkan compute image",
        .shaderInterface = rachis::test::computeShaderInterface({8, 8, 1}, bindings),
    });
    REQUIRE(shader.has_value());

    auto image = device.createTexture(rachis::TextureDesc{
        .width = imageWidth,
        .height = imageHeight,
        .format = rachis::Format::rgba32Float,
        .usage = rachis::TextureUsage::storage | rachis::TextureUsage::copySource,
        .label = "Vulkan compute storage image",
    });
    REQUIRE(image.has_value());
    auto view = device.createTextureView(rachis::TextureViewDesc{
        .texture = *image,
        .usage = rachis::TextureViewUsage::storage,
    });
    REQUIRE(view.has_value());
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = imageReadbackSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(readback.has_value());

    const std::array layoutEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageTexture, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = layoutEntries});
    REQUIRE(bindingLayout.has_value());
    const std::array writes{
        rachis::BindingWrite{
            .binding = 0,
            .resource = *view,
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device,
                                                     rachis::BindingSetDesc{
                                                         .layout = *bindingLayout,
                                                         .writes = writes,
                                                     });
    REQUIRE(bindingSet.has_value());
    const std::array layouts{*bindingLayout};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
    });
    REQUIRE(pipeline.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *image,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::computeShader,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::unorderedAccess,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::unorderedAccess,
                                })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
    REQUIRE(device.dispatch(*command, 1, 1, 1).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *image,
                                    .stagesBefore = rachis::PipelineStage::computeShader,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::unorderedAccess,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::unorderedAccess,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device
                .copyTextureToBuffer(*command,
                                     rachis::BufferTextureCopy{
                                         .buffer = *readback,
                                         .bytesPerRow = imageRowPitch,
                                         .rowsPerImage = imageHeight,
                                         .texture = *image,
                                         .extent = {imageWidth, imageHeight, 1},
                                     })
                .has_value());

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyTextureView(*view).has_value());
    REQUIRE(device.destroyTexture(*image).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    std::array<float, 4> pixel{};
    std::memcpy(pixel.data(), mapped->data, sizeof(pixel));
    CHECK(pixel[0] == 0.25F);
    CHECK(pixel[1] == 0.5F);
    CHECK(pixel[2] == 0.75F);
    CHECK(pixel[3] == 1.0F);
    REQUIRE(device.unmapBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan storage-image bindings keep the texture alive after its view and texture are destroyed",
          "[size:medium][rhi][vulkan][compute][image][lifetime]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    const auto bytecode = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanComputeImage.spv");
    REQUIRE_FALSE(bytecode.empty());
    static constexpr std::array bindings{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageTexture, 1},
    };
    auto shader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Vulkan bound-only storage image",
        .shaderInterface = rachis::test::computeShaderInterface({8, 8, 1}, bindings),
    });
    REQUIRE(shader.has_value());

    auto image = device.createTexture(rachis::TextureDesc{
        .width = imageWidth,
        .height = imageHeight,
        .format = rachis::Format::rgba32Float,
        .usage = rachis::TextureUsage::storage,
        .label = "Vulkan bound-only storage image",
    });
    REQUIRE(image.has_value());
    auto view = device.createTextureView(rachis::TextureViewDesc{
        .texture = *image,
        .usage = rachis::TextureViewUsage::storage,
    });
    REQUIRE(view.has_value());

    const std::array layoutEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageTexture, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = layoutEntries});
    REQUIRE(bindingLayout.has_value());
    const std::array writes{
        rachis::BindingWrite{
            .binding = 0,
            .resource = *view,
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device,
                                                     rachis::BindingSetDesc{
                                                         .layout = *bindingLayout,
                                                         .writes = writes,
                                                     });
    REQUIRE(bindingSet.has_value());
    const std::array layouts{*bindingLayout};
    auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = layouts});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
    });
    REQUIRE(pipeline.has_value());

    auto transition = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(transition.has_value());
    REQUIRE(device
                .textureBarrier(*transition,
                                rachis::TextureBarrier{
                                    .texture = *image,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::computeShader,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::unorderedAccess,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::unorderedAccess,
                                })
                .has_value());
    auto transitioned = device.submit(std::move(*transition));
    REQUIRE(transitioned.has_value());
    REQUIRE(device.wait(*transitioned).has_value());
    device.collectGarbage();

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
    REQUIRE(device.dispatch(*command, 1, 1, 1).has_value());

    const auto destroyBoundObjects = [&]() {
        REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
        REQUIRE(device.destroyTextureView(*view).has_value());
        REQUIRE(device.destroyTexture(*image).has_value());
    };

    rachis::TimelinePoint submitted{};
    SECTION("destroyed after recording")
    {
        destroyBoundObjects();
        auto point = device.submit(std::move(*command));
        REQUIRE(point.has_value());
        submitted = *point;
    }
    SECTION("destroyed after submit")
    {
        auto point = device.submit(std::move(*command));
        REQUIRE(point.has_value());
        submitted = *point;
        destroyBoundObjects();
    }
    REQUIRE(submitted.isValid());
    REQUIRE(device.wait(submitted).has_value());
    device.collectGarbage();

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
