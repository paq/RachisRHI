#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_approx.hpp>
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

    constexpr std::uint32_t textureRowPitch = 256;
    constexpr std::uint64_t outputSize = 4 * sizeof(float);

    void printDiagnostic(const rachis::DiagnosticMessage& message, void*) noexcept
    {
        static_cast<void>(std::fprintf(stderr, "D3D12 [%u]: ", static_cast<unsigned int>(message.nativeId)));
        if (!message.description.empty())
        {
            static_cast<void>(std::fwrite(message.description.data(), 1, message.description.size(), stderr));
        }
        static_cast<void>(std::fputc('\n', stderr));
    }

    [[nodiscard]] std::vector<std::byte> loadShader()
    {
        std::ifstream stream(RACHIS_TEST_TEXTURE_VIEW_COMPONENT_SWIZZLE_SHADER_PATH, std::ios::binary | std::ios::ate);
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

TEST_CASE("sampled texture views route components without shader-side repacking",
          "[size:medium][rhi][texture][view][swizzle][compute][warp]")
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
    REQUIRE(rachis::test::requireCapabilities(device).textureViewComponentSwizzle);

    const auto bytecode = loadShader();
    REQUIRE_FALSE(bytecode.empty());
    auto shader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Texture-view component swizzle shader",
        .shaderInterface = rachis::test::computeShaderInterface(
            {1, 1, 1},
            std::array{
                rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::sampledTexture, 1},
                rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
            }),
    });
    auto upload = device.createBuffer(rachis::BufferDesc{
        .size = textureRowPitch,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Texture-view component swizzle upload",
    });
    auto output = device.createBuffer(rachis::BufferDesc{
        .size = outputSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .label = "Texture-view component swizzle output",
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = outputSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Texture-view component swizzle readback",
    });
    auto texture = device.createTexture(rachis::TextureDesc{
        .width = 1,
        .height = 1,
        .format = rachis::Format::rgba8Unorm,
        .usage =
            rachis::TextureUsage::copyDestination | rachis::TextureUsage::sampled | rachis::TextureUsage::renderTarget,
        .label = "Texture-view component swizzle source",
    });
    REQUIRE(shader.has_value());
    REQUIRE(upload.has_value());
    REQUIRE(output.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(texture.has_value());

    constexpr std::array sourceTexel{
        std::byte{16},
        std::byte{64},
        std::byte{128},
        std::byte{255},
    };
    REQUIRE(device.writeBuffer(*upload, 0, sourceTexel).has_value());

    constexpr rachis::ComponentMapping swizzle{
        .red = rachis::ComponentSwizzle::blue,
        .green = rachis::ComponentSwizzle::red,
        .blue = rachis::ComponentSwizzle::one,
        .alpha = rachis::ComponentSwizzle::zero,
    };
    auto sampledView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *texture,
        .usage = rachis::TextureViewUsage::sampled,
        .components = swizzle,
        .label = "Swizzled sampled view",
    });
    REQUIRE(sampledView.has_value());

    CHECK_FALSE(device
                    .createTextureView(rachis::TextureViewDesc{
                        .texture = *texture,
                        .usage = rachis::TextureViewUsage::renderTarget,
                        .components = swizzle,
                    })
                    .has_value());
    auto invalidMapping = swizzle;
    invalidMapping.alpha = static_cast<rachis::ComponentSwizzle>(255);
    CHECK_FALSE(device
                    .createTextureView(rachis::TextureViewDesc{
                        .texture = *texture,
                        .usage = rachis::TextureViewUsage::sampled,
                        .components = invalidMapping,
                    })
                    .has_value());

    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = bindingEntries, .label = "Swizzle binding layout"});
    REQUIRE(bindingLayout.has_value());
    const std::array writes{
        rachis::BindingWrite{.binding = 0, .resource = *sampledView},
        rachis::BindingWrite{
            .binding = 1,
            .resource =
                rachis::BufferBinding{
                    .buffer = *output,
                    .size = outputSize,
                    .structureByteStride = outputSize,
                },
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device,
                                                     {
                                                         .layout = *bindingLayout,
                                                         .writes = writes,
                                                         .label = "Swizzle binding set",
                                                     });
    REQUIRE(bindingSet.has_value());
    const std::array bindingLayouts{*bindingLayout};
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .label = "Swizzle pipeline layout",
    });
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Swizzle compute pipeline",
    });
    REQUIRE(pipeline.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    const rachis::TextureSubresourceRange colorRange{
        .mipLevelCount = 1,
        .arrayLayerCount = 1,
        .aspect = rachis::TextureAspect::color,
    };
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                    .subresources = colorRange,
                                })
                .has_value());
    REQUIRE(device
                .copyBufferToTexture(*command,
                                     rachis::BufferTextureCopy{
                                         .buffer = *upload,
                                         .bytesPerRow = textureRowPitch,
                                         .rowsPerImage = 1,
                                         .texture = *texture,
                                         .extent = {1, 1, 1},
                                     })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::computeShader,
                                    .accessBefore = rachis::Access::copyDestination,
                                    .accessAfter = rachis::Access::shaderResource,
                                    .layoutBefore = rachis::TextureLayout::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::shaderResource,
                                    .subresources = colorRange,
                                })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *output,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*command, 0, *bindingSet).has_value());
    REQUIRE(device.dispatch(*command, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *output,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device.copyBuffer(*command, {.source = *output, .destination = *readback, .size = outputSize}).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    std::array<float, 4> actual{};
    std::memcpy(actual.data(), mapped->data, sizeof(actual));
    CHECK(actual[0] == Catch::Approx(128.0F / 255.0F));
    CHECK(actual[1] == Catch::Approx(16.0F / 255.0F));
    CHECK(actual[2] == Catch::Approx(1.0F));
    CHECK(actual[3] == Catch::Approx(0.0F));
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyTextureView(*sampledView).has_value());
    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*output).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
