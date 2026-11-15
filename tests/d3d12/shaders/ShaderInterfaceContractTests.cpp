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

    constexpr std::uint32_t valueCount = 64;
    constexpr std::uint64_t outputSize = valueCount * sizeof(std::uint32_t);

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

TEST_CASE("shader interfaces validate portable layouts before native pipeline creation",
          "[size:medium][rhi][pipeline][shader-interface][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    const auto computeBytecode = loadShader(RACHIS_TEST_COMPUTE_SHADER_PATH);
    const auto vertexBytecode = loadShader(RACHIS_TEST_GRAPHICS_INPUT_VERTEX_SHADER_PATH);
    const auto pixelBytecode = loadShader(RACHIS_TEST_GRAPHICS_PIXEL_SHADER_PATH);
    REQUIRE_FALSE(computeBytecode.empty());
    REQUIRE_FALSE(vertexBytecode.empty());
    REQUIRE_FALSE(pixelBytecode.empty());

    std::array computeBindings{
        rachis::ShaderBindingRequirement{
            .set = 0,
            .binding = 0,
            .type = rachis::ShaderResourceType::storageBuffer,
            .count = 1,
        },
    };
    std::array computePushConstants{
        rachis::ShaderPushConstantRequirement{.offset = 0, .size = 2 * sizeof(std::uint32_t)},
    };
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = computeBytecode,
        .entryPoint = "main",
        .label = "Reflected compute shader",
        .shaderInterface =
            rachis::ShaderInterfaceDesc{
                .bindings = computeBindings,
                .pushConstants = computePushConstants,
                .workgroupSize = rachis::ShaderWorkgroupSize{64, 1, 1},
            },
    });
    REQUIRE(shader.has_value());

    computeBindings[0].type = rachis::ShaderResourceType::sampledTexture;
    computePushConstants[0] = {.offset = 4, .size = sizeof(std::uint32_t)};

    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::compute,
                        .bytecode = computeBytecode,
                        .shaderInterface = rachis::ShaderInterfaceDesc{},
                    })
                    .has_value());
    const std::array duplicateBindings{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::storageBuffer, 1},
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::sampler, 1},
    };
    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::compute,
                        .bytecode = computeBytecode,
                        .shaderInterface =
                            rachis::ShaderInterfaceDesc{
                                .bindings = duplicateBindings,
                                .workgroupSize = rachis::ShaderWorkgroupSize{64, 1, 1},
                            },
                    })
                    .has_value());
    const std::array overlappingPushConstants{
        rachis::ShaderPushConstantRequirement{0, 8},
        rachis::ShaderPushConstantRequirement{4, 4},
    };
    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::compute,
                        .bytecode = computeBytecode,
                        .shaderInterface =
                            rachis::ShaderInterfaceDesc{
                                .pushConstants = overlappingPushConstants,
                                .workgroupSize = rachis::ShaderWorkgroupSize{64, 1, 1},
                            },
                    })
                    .has_value());
    const std::array invalidBinding{
        rachis::ShaderBindingRequirement{
            .type = static_cast<rachis::ShaderResourceType>(255),
            .count = 1,
        },
    };
    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::compute,
                        .bytecode = computeBytecode,
                        .shaderInterface =
                            rachis::ShaderInterfaceDesc{
                                .bindings = invalidBinding,
                                .workgroupSize = rachis::ShaderWorkgroupSize{64, 1, 1},
                            },
                    })
                    .has_value());
    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::compute,
                        .bytecode = computeBytecode,
                        .shaderInterface =
                            rachis::ShaderInterfaceDesc{
                                .workgroupSize =
                                    rachis::ShaderWorkgroupSize{
                                        rachis::test::requireCapabilities(device).limits.maxComputeWorkgroupSize[0] + 1,
                                        1,
                                        1,
                                    },
                            },
                    })
                    .has_value());
    const std::array nonVertexLocation{0U};
    CHECK_FALSE(device
                    .createShaderModule({
                        .stage = rachis::ShaderStage::pixel,
                        .bytecode = pixelBytecode,
                        .entryPoint = "pixelMain",
                        .shaderInterface =
                            rachis::ShaderInterfaceDesc{
                                .vertexInputLocations = nonVertexLocation,
                            },
                    })
                    .has_value());

    const std::array correctBindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = correctBindingEntries});
    REQUIRE(bindingLayout.has_value());
    const std::array bindingLayouts{*bindingLayout};
    const std::array correctPushConstants{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)},
    };
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = correctPushConstants,
    });
    REQUIRE(pipelineLayout.has_value());

    auto emptyPipelineLayout = device.createPipelineLayout({});
    REQUIRE(emptyPipelineLayout.has_value());
    CHECK_FALSE(device
                    .createComputePipeline({
                        .layout = *emptyPipelineLayout,
                        .computeShader = *shader,
                    })
                    .has_value());

    const std::array wrongTypeEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute},
    };
    auto wrongTypeLayout = device.createBindingLayout({.entries = wrongTypeEntries});
    REQUIRE(wrongTypeLayout.has_value());
    const std::array wrongTypeLayouts{*wrongTypeLayout};
    auto wrongTypePipelineLayout = device.createPipelineLayout({
        .bindingLayouts = wrongTypeLayouts,
        .pushConstantRanges = correctPushConstants,
    });
    REQUIRE(wrongTypePipelineLayout.has_value());
    CHECK_FALSE(device
                    .createComputePipeline({
                        .layout = *wrongTypePipelineLayout,
                        .computeShader = *shader,
                    })
                    .has_value());

    const std::array wrongCountEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 2},
    };
    auto wrongCountLayout = device.createBindingLayout({.entries = wrongCountEntries});
    REQUIRE(wrongCountLayout.has_value());
    const std::array wrongCountLayouts{*wrongCountLayout};
    auto wrongCountPipelineLayout = device.createPipelineLayout({
        .bindingLayouts = wrongCountLayouts,
        .pushConstantRanges = correctPushConstants,
    });
    REQUIRE(wrongCountPipelineLayout.has_value());
    CHECK_FALSE(device
                    .createComputePipeline({
                        .layout = *wrongCountPipelineLayout,
                        .computeShader = *shader,
                    })
                    .has_value());

    const std::array wrongVisibilityEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::vertex},
    };
    auto wrongVisibilityLayout = device.createBindingLayout({.entries = wrongVisibilityEntries});
    REQUIRE(wrongVisibilityLayout.has_value());
    const std::array wrongVisibilityLayouts{*wrongVisibilityLayout};
    auto wrongVisibilityPipelineLayout = device.createPipelineLayout({
        .bindingLayouts = wrongVisibilityLayouts,
        .pushConstantRanges = correctPushConstants,
    });
    REQUIRE(wrongVisibilityPipelineLayout.has_value());
    CHECK_FALSE(device
                    .createComputePipeline({
                        .layout = *wrongVisibilityPipelineLayout,
                        .computeShader = *shader,
                    })
                    .has_value());

    const std::array extraPushConstants{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 3 * sizeof(std::uint32_t)},
    };
    auto extraPushPipelineLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = extraPushConstants,
    });
    REQUIRE(extraPushPipelineLayout.has_value());
    CHECK_FALSE(device
                    .createComputePipeline({
                        .layout = *extraPushPipelineLayout,
                        .computeShader = *shader,
                    })
                    .has_value());

    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
    });
    REQUIRE(pipeline.has_value());

    const std::array vertexLocations{0U};
    auto vertexShader = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = vertexBytecode,
        .entryPoint = "inputVertex",
        .shaderInterface =
            rachis::ShaderInterfaceDesc{
                .vertexInputLocations = vertexLocations,
            },
    });
    const std::array pixelPushConstants{
        rachis::ShaderPushConstantRequirement{0, 4 * sizeof(float)},
    };
    auto pixelShader = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = pixelBytecode,
        .entryPoint = "pixelMain",
        .shaderInterface =
            rachis::ShaderInterfaceDesc{
                .pushConstants = pixelPushConstants,
            },
    });
    REQUIRE(vertexShader.has_value());
    REQUIRE(pixelShader.has_value());
    const std::array graphicsPushConstants{
        rachis::PushConstantRange{rachis::ShaderStage::pixel, 0, 4 * sizeof(float)},
    };
    auto graphicsLayout = device.createPipelineLayout({.pushConstantRanges = graphicsPushConstants});
    REQUIRE(graphicsLayout.has_value());
    const std::array vertexAttributes{
        rachis::VertexAttribute{.format = rachis::VertexFormat::float32x2, .shaderLocation = 0},
    };
    const std::array vertexBuffers{
        rachis::VertexBufferLayout{.stride = 2 * sizeof(float), .attributes = vertexAttributes},
    };
    const std::array colorTargets{
        rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm},
    };
    const rachis::GraphicsPipelineDesc graphicsDesc{
        .layout = *graphicsLayout,
        .vertexShader = *vertexShader,
        .pixelShader = *pixelShader,
        .vertexBuffers = vertexBuffers,
        .colorTargets = colorTargets,
    };
    auto graphicsPipeline = device.createGraphicsPipeline(graphicsDesc);
    REQUIRE(graphicsPipeline.has_value());
    auto missingVertexInputDesc = graphicsDesc;
    missingVertexInputDesc.vertexBuffers = {};
    CHECK_FALSE(device.createGraphicsPipeline(missingVertexInputDesc).has_value());

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
    REQUIRE(storage.has_value());
    REQUIRE(readback.has_value());
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
    auto bindingSet = rachis::test::createBindingSet(device, {.layout = *bindingLayout, .writes = writes});
    REQUIRE(bindingSet.has_value());

    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *storage,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
    const std::array constants{11U, 23U};
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(constants))).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
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
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBuffer(*storage).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    for (std::uint32_t index = 0; index < valueCount; ++index)
    {
        std::uint32_t value = 0;
        std::memcpy(&value, mapped->data + index * sizeof(value), sizeof(value));
        CAPTURE(index, value);
        REQUIRE(value == index * constants[0] + constants[1]);
    }
    REQUIRE(device.unmapBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    REQUIRE(device.destroyPipeline(*graphicsPipeline).has_value());
    REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
    REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
    REQUIRE(device.destroyPipelineLayout(*graphicsLayout).has_value());
    REQUIRE(device.destroyPipelineLayout(*emptyPipelineLayout).has_value());
    REQUIRE(device.destroyPipelineLayout(*wrongTypePipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*wrongTypeLayout).has_value());
    REQUIRE(device.destroyPipelineLayout(*wrongCountPipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*wrongCountLayout).has_value());
    REQUIRE(device.destroyPipelineLayout(*wrongVisibilityPipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*wrongVisibilityLayout).has_value());
    REQUIRE(device.destroyPipelineLayout(*extraPushPipelineLayout).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
