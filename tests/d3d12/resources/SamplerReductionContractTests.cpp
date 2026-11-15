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
    constexpr std::uint64_t textureUploadSize = static_cast<std::uint64_t>(textureRowPitch) * 2;
    constexpr std::uint64_t outputSize = 2 * sizeof(float);

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
        std::ifstream stream(RACHIS_TEST_SAMPLER_REDUCTION_SHADER_PATH, std::ios::binary | std::ios::ate);
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

TEST_CASE("sampler min/max reduction selects component-wise footprint extrema",
          "[size:medium][rhi][sampler][reduction][compute][warp]")
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
    REQUIRE(rachis::test::requireCapabilities(device).samplerMinMaxReduction);

    const auto bytecode = loadShader();
    REQUIRE_FALSE(bytecode.empty());
    auto shader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Sampler reduction shader",
        .shaderInterface = rachis::test::computeShaderInterface(
            {1, 1, 1},
            std::array{
                rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::sampledTexture, 1},
                rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::sampler, 2},
                rachis::ShaderBindingRequirement{0, 3, rachis::ShaderResourceType::storageBuffer, 1},
            }),
    });
    auto upload = device.createBuffer(rachis::BufferDesc{
        .size = textureUploadSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Sampler reduction upload",
    });
    auto output = device.createBuffer(rachis::BufferDesc{
        .size = outputSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .label = "Sampler reduction output",
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = outputSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Sampler reduction readback",
    });
    auto texture = device.createTexture(rachis::TextureDesc{
        .width = 2,
        .height = 2,
        .format = rachis::Format::r32Float,
        .usage = rachis::TextureUsage::copyDestination | rachis::TextureUsage::sampled,
        .label = "Sampler reduction source",
    });
    REQUIRE(shader.has_value());
    REQUIRE(upload.has_value());
    REQUIRE(output.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(texture.has_value());

    std::array<std::byte, textureUploadSize> uploadBytes{};
    constexpr std::array sourceValues{0.125F, 0.375F, 0.625F, 0.875F};
    std::memcpy(uploadBytes.data(), sourceValues.data(), 2 * sizeof(float));
    std::memcpy(uploadBytes.data() + textureRowPitch, sourceValues.data() + 2, 2 * sizeof(float));
    REQUIRE(device.writeBuffer(*upload, 0, uploadBytes).has_value());

    auto sampledView = device.createTextureView(rachis::TextureViewDesc{
        .texture = *texture,
        .usage = rachis::TextureViewUsage::sampled,
        .label = "Sampler reduction view",
    });
    auto minimumSampler = device.createSampler(rachis::SamplerDesc{
        .minFilter = rachis::FilterMode::linear,
        .magFilter = rachis::FilterMode::linear,
        .reductionMode = rachis::SamplerReductionMode::minimum,
        .label = "Minimum reduction sampler",
    });
    auto maximumSampler = device.createSampler(rachis::SamplerDesc{
        .minFilter = rachis::FilterMode::linear,
        .magFilter = rachis::FilterMode::linear,
        .reductionMode = rachis::SamplerReductionMode::maximum,
        .label = "Maximum reduction sampler",
    });
    auto anisotropicMinimumSampler = device.createSampler(rachis::SamplerDesc{
        .minFilter = rachis::FilterMode::linear,
        .magFilter = rachis::FilterMode::linear,
        .mipFilter = rachis::MipmapFilterMode::linear,
        .maxAnisotropy = 4,
        .reductionMode = rachis::SamplerReductionMode::minimum,
        .label = "Anisotropic minimum reduction sampler",
    });
    REQUIRE(sampledView.has_value());
    REQUIRE(minimumSampler.has_value());
    REQUIRE(maximumSampler.has_value());
    REQUIRE(anisotropicMinimumSampler.has_value());
    CHECK_FALSE(device
                    .createSampler(rachis::SamplerDesc{
                        .compareEnabled = true,
                        .reductionMode = rachis::SamplerReductionMode::minimum,
                    })
                    .has_value());
    CHECK_FALSE(device
                    .createSampler(rachis::SamplerDesc{
                        .reductionMode = static_cast<rachis::SamplerReductionMode>(255),
                    })
                    .has_value());

    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute},
        rachis::BindingLayoutEntry{1, rachis::BindingType::sampler, rachis::ShaderStage::compute, 2},
        rachis::BindingLayoutEntry{3, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout({.entries = bindingEntries, .label = "Sampler reduction bindings"});
    REQUIRE(bindingLayout.has_value());
    const std::array writes{
        rachis::BindingWrite{.binding = 0, .resource = *sampledView},
        rachis::BindingWrite{.binding = 1, .arrayElement = 0, .resource = *minimumSampler},
        rachis::BindingWrite{.binding = 1, .arrayElement = 1, .resource = *maximumSampler},
        rachis::BindingWrite{
            .binding = 3,
            .resource =
                rachis::BufferBinding{
                    .buffer = *output,
                    .size = outputSize,
                    .structureByteStride = sizeof(float),
                },
        },
    };
    auto bindingSet = rachis::test::createBindingSet(device,
                                                     {
                                                         .layout = *bindingLayout,
                                                         .writes = writes,
                                                         .label = "Sampler reduction set",
                                                     });
    REQUIRE(bindingSet.has_value());
    const std::array bindingLayouts{*bindingLayout};
    auto pipelineLayout = device.createPipelineLayout({
        .bindingLayouts = bindingLayouts,
        .label = "Sampler reduction pipeline layout",
    });
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Sampler reduction pipeline",
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
                                         .rowsPerImage = 2,
                                         .texture = *texture,
                                         .extent = {2, 2, 1},
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
    std::array<float, 2> actual{};
    std::memcpy(actual.data(), mapped->data, sizeof(actual));
    CHECK(actual[0] == Catch::Approx(sourceValues.front()));
    CHECK(actual[1] == Catch::Approx(sourceValues.back()));
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroySampler(*minimumSampler).has_value());
    REQUIRE(device.destroySampler(*maximumSampler).has_value());
    REQUIRE(device.destroySampler(*anisotropicMinimumSampler).has_value());
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
