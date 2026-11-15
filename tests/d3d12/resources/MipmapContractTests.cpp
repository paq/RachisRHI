#include "D3D12/D3D12TestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

    constexpr std::uint32_t baseWidth = 4;
    constexpr std::uint32_t baseHeight = 4;
    constexpr std::uint32_t mipLevels = 3;
    constexpr std::uint32_t bytesPerPixel = 4;
    constexpr std::uint32_t rowPitch = 256;
    constexpr std::uint64_t mip2Offset = 512;
    constexpr std::uint64_t readbackSize = mip2Offset + rowPitch;

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

    class ForceLegacyBarriers final
    {
    public:
        ForceLegacyBarriers()
            : previous_(rachis::d3d12::test::forceLegacyBarriers().exchange(true, std::memory_order_relaxed))
        {
        }

        ~ForceLegacyBarriers()
        {
            rachis::d3d12::test::forceLegacyBarriers().store(previous_, std::memory_order_relaxed);
        }

        ForceLegacyBarriers(const ForceLegacyBarriers&) = delete;
        ForceLegacyBarriers& operator=(const ForceLegacyBarriers&) = delete;
        ForceLegacyBarriers(ForceLegacyBarriers&&) = delete;
        ForceLegacyBarriers& operator=(ForceLegacyBarriers&&) = delete;

    private:
        bool previous_ = false;
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

    [[nodiscard]] constexpr std::array<std::uint8_t, 4> baseTexel(std::uint32_t x, std::uint32_t y) noexcept
    {
        return {
            static_cast<std::uint8_t>(x * 40U),
            static_cast<std::uint8_t>(y * 40U),
            static_cast<std::uint8_t>((x + y) * 20U),
            255,
        };
    }

    [[nodiscard]] constexpr std::array<std::uint8_t, 4> mip1Texel(std::uint32_t x, std::uint32_t y) noexcept
    {
        return {
            static_cast<std::uint8_t>(20U + x * 80U),
            static_cast<std::uint8_t>(20U + y * 80U),
            static_cast<std::uint8_t>(20U + (x + y) * 40U),
            255,
        };
    }

    [[nodiscard]] bool nearByte(std::byte actual, std::uint8_t expected) noexcept
    {
        const auto value = std::to_integer<std::uint8_t>(actual);
        const auto lower = expected == 0 ? 0U : static_cast<std::uint32_t>(expected) - 1U;
        const auto upper = (std::min)(static_cast<std::uint32_t>(expected) + 1U, 255U);
        return value >= lower && value <= upper;
    }

    [[nodiscard]] std::vector<std::byte> loadShader(const char* path = RACHIS_TEST_COMPUTE_SHADER_PATH)
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

    constexpr std::uint32_t renderedSize = 2048;
    constexpr std::uint32_t renderedMipLevels = 12;
    constexpr std::uint32_t renderedLastMip = renderedMipLevels - 1;
    constexpr std::uint32_t renderedIterations = 4;

    [[nodiscard]] rachis::PipelineHandle createRenderedMipPipeline(rachis::Device& device,
                                                                   const char* pixelPath,
                                                                   std::string_view pixelEntry,
                                                                   const rachis::BlendState& blend)
    {
        const auto vertexBytes = loadShader(RACHIS_TEST_MIPMAP_AFTER_RENDERING_VERTEX_SHADER_PATH);
        const auto pixelBytes = loadShader(pixelPath);
        REQUIRE_FALSE(vertexBytes.empty());
        REQUIRE_FALSE(pixelBytes.empty());
        auto vertex = device.createShaderModule({
            .stage = rachis::ShaderStage::vertex,
            .bytecode = vertexBytes,
            .entryPoint = "fullscreenVertex",
        });
        auto pixel = device.createShaderModule({
            .stage = rachis::ShaderStage::pixel,
            .bytecode = pixelBytes,
            .entryPoint = pixelEntry,
        });
        auto layout = device.createPipelineLayout({});
        REQUIRE(vertex);
        REQUIRE(pixel);
        REQUIRE(layout);
        const std::array targets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm, .blend = blend}};
        auto pipeline = device.createGraphicsPipeline({
            .layout = *layout,
            .vertexShader = *vertex,
            .pixelShader = *pixel,
            .cullMode = rachis::CullMode::none,
            .colorTargets = targets,
            .label = "D3D12 rendered mip pipeline",
        });
        REQUIRE(pipeline);
        return *pipeline;
    }

    [[nodiscard]] rachis::TextureDesc renderedMipDesc() noexcept
    {
        return rachis::TextureDesc{
            .width = renderedSize,
            .height = renderedSize,
            .mipLevels = renderedMipLevels,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::sampled |
                     rachis::TextureUsage::storage | rachis::TextureUsage::copySource,
            .label = "D3D12 rendered mip chain",
        };
    }

    [[nodiscard]] rachis::TextureViewHandle createRenderedMipView(rachis::Device& device,
                                                                  rachis::TextureHandle texture,
                                                                  std::uint32_t mipLevel)
    {
        auto view = device.createTextureView({
            .texture = texture,
            .usage = rachis::TextureViewUsage::renderTarget,
            .format = rachis::Format::rgba8Unorm,
            .firstMipLevel = mipLevel,
            .mipLevelCount = 1,
        });
        REQUIRE(view);
        return *view;
    }

    void recordRenderedMip(rachis::Device& device,
                           rachis::CommandBuffer& command,
                           rachis::TextureViewHandle view,
                           rachis::PipelineHandle pipeline,
                           std::uint32_t size,
                           std::uint32_t instanceCount,
                           rachis::LoadOperation loadOperation)
    {
        const std::array attachments{rachis::ColorAttachment{
            .view = view,
            .loadOperation = loadOperation,
            .clearValue = {.red = 0.0F, .green = 0.0F, .blue = 1.0F, .alpha = 1.0F},
        }};
        REQUIRE(device.beginRendering(command, {.colorAttachments = attachments}).has_value());
        REQUIRE(device.bindGraphicsPipeline(command, pipeline).has_value());
        REQUIRE(device.setViewport(command, {.width = static_cast<float>(size), .height = static_cast<float>(size)})
                    .has_value());
        REQUIRE(device.setScissor(command, {.width = size, .height = size}).has_value());
        REQUIRE(device.draw(command, 3, instanceCount).has_value());
        REQUIRE(device.endRendering(command).has_value());
    }

    void recordRenderedBase(rachis::Device& device,
                            rachis::CommandBuffer& command,
                            rachis::TextureHandle texture,
                            rachis::TextureViewHandle baseView,
                            rachis::PipelineHandle seed)
    {
        REQUIRE(device
                    .textureBarrier(command,
                                    {
                                        .texture = texture,
                                        .stagesAfter = rachis::PipelineStage::renderTarget,
                                        .accessAfter = rachis::Access::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::renderTarget,
                                        .subresources = {.firstMipLevel = 0, .mipLevelCount = 1},
                                    })
                    .has_value());
        recordRenderedMip(device, command, baseView, seed, renderedSize, 2, rachis::LoadOperation::clear);
    }

    [[nodiscard]] std::array<std::uint32_t, 4> readRenderedLastMip(rachis::Device& device,
                                                                   rachis::CommandBuffer command,
                                                                   rachis::TextureHandle texture)
    {
        const rachis::BufferTextureCopyRegion lastMip{
            .textureSubresource = {.mipLevel = renderedLastMip},
            .extent = {.width = 1, .height = 1},
        };
        auto layout = device.queryTextureTransferLayout(texture, lastMip);
        REQUIRE(layout);
        auto readback = device.createBuffer({
            .size = layout->requiredBufferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
        });
        REQUIRE(readback);
        REQUIRE(device.copyTextureToBuffer(command, texture, *readback, std::array{lastMip}).has_value());
        auto submitted = device.submit(std::move(command));
        REQUIRE(submitted);
        REQUIRE(device.wait(*submitted).has_value());
        auto mapped = device.mapBuffer(*readback);
        REQUIRE(mapped);
        std::array<std::uint32_t, 4> texel{};
        for (std::size_t channel = 0; channel < texel.size(); ++channel)
        {
            texel[channel] = std::to_integer<std::uint32_t>(mapped->data[channel]);
        }
        REQUIRE(device.unmapBuffer(*readback).has_value());
        REQUIRE(device.destroyBuffer(*readback).has_value());
        return texel;
    }

    [[nodiscard]] bool nearTexel(const std::array<std::uint32_t, 4>& actual,
                                 const std::array<std::uint32_t, 4>& expected,
                                 std::uint32_t slack) noexcept
    {
        for (std::size_t channel = 0; channel < actual.size(); ++channel)
        {
            if (actual[channel] + slack < expected[channel] || actual[channel] > expected[channel] + slack)
            {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] rachis::Result<rachis::Device> createHardwareDevice()
    {
        return rachis::createDevice(rachis::DeviceDesc{
            .adapterPreference = rachis::AdapterPreference::highPerformance,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = true,
            .diagnosticCallback = &printDiagnostic,
        });
    }

} // namespace

TEST_CASE("mipmap generation filters a selected 2D array slice and owns its transitions",
          "[size:medium][rhi][texture][mipmap][warp]")
{
    const ForceLegacyBarriers forceLegacyBarriers;
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

    REQUIRE(rachis::test::requireCapabilities(device).mipmapGeneration);
    REQUIRE_FALSE(rachis::test::requireCapabilities(device).enhancedBarriers);
    const auto format = device.queryFormatProperties(rachis::Format::rgba8Unorm);
    REQUIRE(rachis::hasAll(format.features, rachis::FormatFeature::filterable | rachis::FormatFeature::storageWrite));

    auto upload = device.createBuffer(rachis::BufferDesc{
        .size = static_cast<std::uint64_t>(rowPitch) * baseHeight,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Mipmap seed upload",
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = readbackSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Generated mip readback",
    });
    auto texture = device.createTexture(rachis::TextureDesc{
        .dimension = rachis::TextureDimension::texture2D,
        .width = baseWidth,
        .height = baseHeight,
        .depthOrArrayLayers = 2,
        .mipLevels = mipLevels,
        .sampleCount = 1,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
        .label = "Generated mip texture array",
    });
    auto sampledOnly = device.createTexture(rachis::TextureDesc{
        .width = baseWidth,
        .height = baseHeight,
        .mipLevels = mipLevels,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled,
        .label = "Invalid sampled-only mip texture",
    });
    REQUIRE(upload.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(texture.has_value());
    REQUIRE(sampledOnly.has_value());

    const auto bytecode = loadShader();
    REQUIRE_FALSE(bytecode.empty());
    auto shader = device.createShaderModule(rachis::ShaderModuleDesc{
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Mipmap state invalidation shader",
        .shaderInterface = rachis::test::computeWriteShaderInterface(),
    });
    const std::array bindingEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto bindingLayout = device.createBindingLayout(rachis::BindingLayoutDesc{
        .entries = bindingEntries,
        .label = "Mipmap state invalidation binding layout",
    });
    REQUIRE(shader.has_value());
    REQUIRE(bindingLayout.has_value());
    const std::array bindingLayouts{*bindingLayout};
    const std::array pushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)},
    };
    auto pipelineLayout = device.createPipelineLayout(rachis::PipelineLayoutDesc{
        .bindingLayouts = bindingLayouts,
        .pushConstantRanges = pushConstantRanges,
        .label = "Mipmap state invalidation pipeline layout",
    });
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline(rachis::ComputePipelineDesc{
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Mipmap state invalidation pipeline",
    });
    REQUIRE(pipeline.has_value());

    auto mapped = device.mapBuffer(*upload);
    REQUIRE(mapped.has_value());
    std::fill_n(mapped->data, mapped->size, std::byte{0});
    for (std::uint32_t y = 0; y < baseHeight; ++y)
    {
        for (std::uint32_t x = 0; x < baseWidth; ++x)
        {
            const auto expected = baseTexel(x, y);
            for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
            {
                mapped->data[static_cast<std::uint64_t>(y) * rowPitch + static_cast<std::uint64_t>(x) * bytesPerPixel +
                             channel] = static_cast<std::byte>(expected[channel]);
            }
        }
    }
    REQUIRE(device.unmapBuffer(*upload).has_value());

    const rachis::TextureSubresourceRange generatedRange{
        .firstMipLevel = 0,
        .mipLevelCount = mipLevels,
        .firstArrayLayer = 1,
        .arrayLayerCount = 1,
        .aspect = rachis::TextureAspect::color,
    };
    const rachis::TextureMipmapGeneration generation{
        .texture = *texture,
        .subresources = generatedRange,
        .baseLayout = rachis::TextureLayout::copyDestination,
        .finalLayout = rachis::TextureLayout::copySource,
    };

    auto copyCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(copyCommand.has_value());
    CHECK_FALSE(device.generateMipmaps(*copyCommand, generation).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    const std::array pushConstants{3U, 7U};
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(pushConstants))).has_value());
    CHECK_FALSE(device
                    .generateMipmaps(*command,
                                     rachis::TextureMipmapGeneration{
                                         .texture = *sampledOnly,
                                         .subresources = generatedRange,
                                         .baseLayout = rachis::TextureLayout::shaderResource,
                                     })
                    .has_value());
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(pushConstants))).has_value());
    CHECK_FALSE(device
                    .generateMipmaps(*command,
                                     rachis::TextureMipmapGeneration{
                                         .texture = *texture,
                                         .subresources =
                                             {
                                                 .firstMipLevel = 0,
                                                 .mipLevelCount = 1,
                                                 .firstArrayLayer = 1,
                                                 .arrayLayerCount = 1,
                                                 .aspect = rachis::TextureAspect::color,
                                             },
                                         .baseLayout = rachis::TextureLayout::copyDestination,
                                     })
                    .has_value());
    CHECK_FALSE(device
                    .generateMipmaps(*command,
                                     rachis::TextureMipmapGeneration{
                                         .texture = *texture,
                                         .subresources = generatedRange,
                                     })
                    .has_value());
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(pushConstants))).has_value());

    const rachis::TextureSubresource seed{
        .mipLevel = 0,
        .arrayLayer = 1,
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
                                    .subresources =
                                        {
                                            .firstMipLevel = 0,
                                            .mipLevelCount = 1,
                                            .firstArrayLayer = 1,
                                            .arrayLayerCount = 1,
                                            .aspect = rachis::TextureAspect::color,
                                        },
                                })
                .has_value());
    REQUIRE(device
                .copyBufferToTexture(*command,
                                     rachis::BufferTextureCopy{
                                         .buffer = *upload,
                                         .bytesPerRow = rowPitch,
                                         .rowsPerImage = baseHeight,
                                         .texture = *texture,
                                         .textureSubresource = seed,
                                         .extent = {baseWidth, baseHeight, 1},
                                     })
                .has_value());
    const auto beforeRejectedGeneration = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(beforeRejectedGeneration.has_value());
    {
        const FailResourceRetention failDescriptorPublication(2);
        auto failedGeneration = device.generateMipmaps(*command, generation);
        REQUIRE_FALSE(failedGeneration.has_value());
        CHECK(failedGeneration.error().code == rachis::ErrorCode::outOfMemory);
    }
    const auto afterRejectedGeneration = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(afterRejectedGeneration.has_value());
    CHECK(*afterRejectedGeneration == *beforeRejectedGeneration);
    REQUIRE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(pushConstants))).has_value());

    REQUIRE(device.generateMipmaps(*command, generation).has_value());
    const auto afterAcceptedGeneration = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(afterAcceptedGeneration.has_value());
    const auto acceptedDependencies =
        afterAcceptedGeneration.value_or(rachis::d3d12::test::CommandDependencySnapshot{});
    const auto priorDependencies = beforeRejectedGeneration.value_or(rachis::d3d12::test::CommandDependencySnapshot{});
    CHECK(acceptedDependencies.resourceOwners == priorDependencies.resourceOwners + 2);
    CHECK(acceptedDependencies.residencyLeases == priorDependencies.residencyLeases);
    CHECK(acceptedDependencies.swapchainReferences == priorDependencies.swapchainReferences);
    CHECK(acceptedDependencies.sparseMappingDependencies == priorDependencies.sparseMappingDependencies);
    CHECK_FALSE(device.setComputePushConstants(*command, 0, std::as_bytes(std::span(pushConstants))).has_value());
    REQUIRE(device
                .copyTextureToBuffer(*command,
                                     rachis::BufferTextureCopy{
                                         .buffer = *readback,
                                         .bufferOffset = 0,
                                         .bytesPerRow = rowPitch,
                                         .rowsPerImage = 2,
                                         .texture = *texture,
                                         .textureSubresource =
                                             {
                                                 .mipLevel = 1,
                                                 .arrayLayer = 1,
                                                 .aspect = rachis::TextureAspect::color,
                                             },
                                         .extent = {2, 2, 1},
                                     })
                .has_value());
    REQUIRE(device
                .copyTextureToBuffer(*command,
                                     rachis::BufferTextureCopy{
                                         .buffer = *readback,
                                         .bufferOffset = mip2Offset,
                                         .bytesPerRow = rowPitch,
                                         .rowsPerImage = 1,
                                         .texture = *texture,
                                         .textureSubresource =
                                             {
                                                 .mipLevel = 2,
                                                 .arrayLayer = 1,
                                                 .aspect = rachis::TextureAspect::color,
                                             },
                                         .extent = {1, 1, 1},
                                     })
                .has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    const auto staleTexture = *texture;
    REQUIRE(device.destroyTexture(*texture).has_value());
    CHECK_FALSE(device.destroyTexture(staleTexture).has_value());
    REQUIRE(device.wait(*submitted).has_value());

    mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    for (std::uint32_t y = 0; y < 2; ++y)
    {
        for (std::uint32_t x = 0; x < 2; ++x)
        {
            const auto expected = mip1Texel(x, y);
            for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
            {
                const auto offset =
                    static_cast<std::uint64_t>(y) * rowPitch + static_cast<std::uint64_t>(x) * bytesPerPixel + channel;
                CAPTURE(x, y, channel, expected[channel], std::to_integer<std::uint8_t>(mapped->data[offset]));
                CHECK(nearByte(mapped->data[offset], expected[channel]));
            }
        }
    }
    constexpr std::array<std::uint8_t, 4> mip2Expected{60, 60, 60, 255};
    for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
    {
        CAPTURE(channel, mip2Expected[channel], std::to_integer<std::uint8_t>(mapped->data[mip2Offset + channel]));
        CHECK(nearByte(mapped->data[mip2Offset + channel], mip2Expected[channel]));
    }
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyTexture(*sampledOnly).has_value());
    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("legacy mipmap generation repeats on one texture from the generated mips' current state",
          "[size:medium][rhi][texture][mipmap][warp]")
{
    const ForceLegacyBarriers forceLegacyBarriers;
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
    REQUIRE(rachis::test::requireCapabilities(device).mipmapGeneration);
    REQUIRE_FALSE(rachis::test::requireCapabilities(device).enhancedBarriers);

    auto upload = device.createBuffer(rachis::BufferDesc{
        .size = static_cast<std::uint64_t>(rowPitch) * baseHeight,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Repeated mipmap seed upload",
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = readbackSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Repeated mipmap readback",
    });
    auto texture = device.createTexture(rachis::TextureDesc{
        .width = baseWidth,
        .height = baseHeight,
        .mipLevels = mipLevels,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
        .label = "Repeated mipmap texture",
    });
    REQUIRE(upload.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(texture.has_value());

    auto mapped = device.mapBuffer(*upload);
    REQUIRE(mapped.has_value());
    std::fill_n(mapped->data, mapped->size, std::byte{0});
    for (std::uint32_t y = 0; y < baseHeight; ++y)
    {
        for (std::uint32_t x = 0; x < baseWidth; ++x)
        {
            const auto expected = baseTexel(x, y);
            for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
            {
                mapped->data[static_cast<std::uint64_t>(y) * rowPitch + static_cast<std::uint64_t>(x) * bytesPerPixel +
                             channel] = static_cast<std::byte>(expected[channel]);
            }
        }
    }
    REQUIRE(device.unmapBuffer(*upload).has_value());

    auto first = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(first.has_value());
    REQUIRE(device
                .textureBarrier(*first,
                                rachis::TextureBarrier{
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                    .subresources = {.firstMipLevel = 0, .mipLevelCount = 1},
                                })
                .has_value());
    REQUIRE(device
                .copyBufferToTexture(*first,
                                     rachis::BufferTextureCopy{
                                         .buffer = *upload,
                                         .bytesPerRow = rowPitch,
                                         .rowsPerImage = baseHeight,
                                         .texture = *texture,
                                         .extent = {baseWidth, baseHeight, 1},
                                     })
                .has_value());
    REQUIRE(device
                .generateMipmaps(*first,
                                 rachis::TextureMipmapGeneration{
                                     .texture = *texture,
                                     .baseLayout = rachis::TextureLayout::copyDestination,
                                 })
                .has_value());
    REQUIRE(device
                .generateMipmaps(*first,
                                 rachis::TextureMipmapGeneration{
                                     .texture = *texture,
                                     .baseLayout = rachis::TextureLayout::shaderResource,
                                 })
                .has_value());
    auto firstSubmitted = device.submit(std::move(*first));
    REQUIRE(firstSubmitted.has_value());
    REQUIRE(device.wait(*firstSubmitted).has_value());
    CHECK(device.diagnosticCounts().errors == 0);

    auto second = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(second.has_value());
    REQUIRE(device
                .generateMipmaps(*second,
                                 rachis::TextureMipmapGeneration{
                                     .texture = *texture,
                                     .baseLayout = rachis::TextureLayout::shaderResource,
                                     .finalLayout = rachis::TextureLayout::copySource,
                                 })
                .has_value());
    REQUIRE(device
                .copyTextureToBuffer(*second,
                                     rachis::BufferTextureCopy{
                                         .buffer = *readback,
                                         .bytesPerRow = rowPitch,
                                         .rowsPerImage = 2,
                                         .texture = *texture,
                                         .textureSubresource = {.mipLevel = 1},
                                         .extent = {2, 2, 1},
                                     })
                .has_value());
    REQUIRE(device
                .copyTextureToBuffer(*second,
                                     rachis::BufferTextureCopy{
                                         .buffer = *readback,
                                         .bufferOffset = mip2Offset,
                                         .bytesPerRow = rowPitch,
                                         .rowsPerImage = 1,
                                         .texture = *texture,
                                         .textureSubresource = {.mipLevel = 2},
                                         .extent = {1, 1, 1},
                                     })
                .has_value());
    auto secondSubmitted = device.submit(std::move(*second));
    REQUIRE(secondSubmitted.has_value());
    REQUIRE(device.wait(*secondSubmitted).has_value());

    mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    for (std::uint32_t y = 0; y < 2; ++y)
    {
        for (std::uint32_t x = 0; x < 2; ++x)
        {
            const auto expected = mip1Texel(x, y);
            for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
            {
                const auto offset =
                    static_cast<std::uint64_t>(y) * rowPitch + static_cast<std::uint64_t>(x) * bytesPerPixel + channel;
                CAPTURE(x, y, channel, expected[channel], std::to_integer<std::uint8_t>(mapped->data[offset]));
                CHECK(nearByte(mapped->data[offset], expected[channel]));
            }
        }
    }
    constexpr std::array<std::uint8_t, 4> mip2Expected{60, 60, 60, 255};
    for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
    {
        CAPTURE(channel, mip2Expected[channel], std::to_integer<std::uint8_t>(mapped->data[mip2Offset + channel]));
        CHECK(nearByte(mapped->data[mip2Offset + channel], mip2Expected[channel]));
    }
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("enhanced mipmap generation discards destination mips that earlier work in the batch accessed",
          "[size:medium][rhi][texture][mipmap][warp]")
{
    const bool splitBatch = GENERATE(false, true);
    CAPTURE(splitBatch);
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
    REQUIRE(rachis::test::requireCapabilities(device).mipmapGeneration);
    REQUIRE(rachis::test::requireCapabilities(device).enhancedBarriers);

    constexpr std::uint64_t sentinelOffset = static_cast<std::uint64_t>(rowPitch) * baseHeight;
    constexpr std::array<std::uint64_t, mipLevels> mipOffsets{
        0,
        sentinelOffset,
        sentinelOffset + static_cast<std::uint64_t>(rowPitch) * (baseHeight / 2U),
    };
    constexpr std::array<std::uint8_t, 4> sentinel{0x11, 0x22, 0x33, 0x44};
    auto upload = device.createBuffer(rachis::BufferDesc{
        .size = sentinelOffset + rowPitch,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Discarded mipmap seed upload",
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = mipOffsets[mipLevels - 1U] + rowPitch,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Discarded mipmap readback",
    });
    auto texture = device.createTexture(rachis::TextureDesc{
        .width = baseWidth,
        .height = baseHeight,
        .mipLevels = mipLevels,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
        .label = "Discarded mipmap texture",
    });
    REQUIRE(upload.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(texture.has_value());

    auto mapped = device.mapBuffer(*upload);
    REQUIRE(mapped.has_value());
    std::fill_n(mapped->data, mapped->size, std::byte{0});
    for (std::uint32_t y = 0; y < baseHeight; ++y)
    {
        for (std::uint32_t x = 0; x < baseWidth; ++x)
        {
            const auto expected = baseTexel(x, y);
            for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
            {
                mapped->data[static_cast<std::uint64_t>(y) * rowPitch + static_cast<std::uint64_t>(x) * bytesPerPixel +
                             channel] = static_cast<std::byte>(expected[channel]);
            }
        }
    }
    for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
    {
        mapped->data[sentinelOffset + channel] = static_cast<std::byte>(sentinel[channel]);
    }
    REQUIRE(device.unmapBuffer(*upload).has_value());

    std::array<rachis::CommandBuffer, 2> batch{};
    auto first = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(first.has_value());
    batch[0] = std::move(*first);
    REQUIRE(device
                .textureBarrier(batch[0],
                                rachis::TextureBarrier{
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    REQUIRE(device
                .copyBufferToTexture(batch[0],
                                     rachis::BufferTextureCopy{
                                         .buffer = *upload,
                                         .bytesPerRow = rowPitch,
                                         .rowsPerImage = baseHeight,
                                         .texture = *texture,
                                         .extent = {baseWidth, baseHeight, 1},
                                     })
                .has_value());
    REQUIRE(device
                .copyBufferToTexture(batch[0],
                                     rachis::BufferTextureCopy{
                                         .buffer = *upload,
                                         .bufferOffset = sentinelOffset,
                                         .bytesPerRow = rowPitch,
                                         .rowsPerImage = 1,
                                         .texture = *texture,
                                         .textureSubresource = {.mipLevel = mipLevels - 1U},
                                         .extent = {1, 1, 1},
                                     })
                .has_value());
    REQUIRE(device
                .generateMipmaps(batch[0],
                                 rachis::TextureMipmapGeneration{
                                     .texture = *texture,
                                     .baseLayout = rachis::TextureLayout::copyDestination,
                                 })
                .has_value());
    if (splitBatch)
    {
        auto second = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(second.has_value());
        batch[1] = std::move(*second);
    }
    auto& tail = splitBatch ? batch[1] : batch[0];
    REQUIRE(device
                .generateMipmaps(tail,
                                 rachis::TextureMipmapGeneration{
                                     .texture = *texture,
                                     .baseLayout = rachis::TextureLayout::shaderResource,
                                     .finalLayout = rachis::TextureLayout::copySource,
                                 })
                .has_value());
    for (std::uint32_t mipLevel = 0; mipLevel < mipLevels; ++mipLevel)
    {
        const auto width = (std::max)(baseWidth >> mipLevel, 1U);
        const auto height = (std::max)(baseHeight >> mipLevel, 1U);
        REQUIRE(device
                    .copyTextureToBuffer(tail,
                                         rachis::BufferTextureCopy{
                                             .buffer = *readback,
                                             .bufferOffset = mipOffsets[mipLevel],
                                             .bytesPerRow = rowPitch,
                                             .rowsPerImage = height,
                                             .texture = *texture,
                                             .textureSubresource = {.mipLevel = mipLevel},
                                             .extent = {width, height, 1},
                                         })
                    .has_value());
    }
    auto submitted = device.submit(std::span<rachis::CommandBuffer>(batch.data(), splitBatch ? 2U : 1U));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    for (std::uint32_t y = 0; y < baseHeight; ++y)
    {
        for (std::uint32_t x = 0; x < baseWidth; ++x)
        {
            const auto expected = baseTexel(x, y);
            for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
            {
                const auto offset =
                    static_cast<std::uint64_t>(y) * rowPitch + static_cast<std::uint64_t>(x) * bytesPerPixel + channel;
                CAPTURE(x, y, channel, expected[channel], std::to_integer<std::uint8_t>(mapped->data[offset]));
                CHECK(mapped->data[offset] == static_cast<std::byte>(expected[channel]));
            }
        }
    }
    for (std::uint32_t y = 0; y < 2; ++y)
    {
        for (std::uint32_t x = 0; x < 2; ++x)
        {
            const auto expected = mip1Texel(x, y);
            for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
            {
                const auto offset = mipOffsets[1] + static_cast<std::uint64_t>(y) * rowPitch +
                                    static_cast<std::uint64_t>(x) * bytesPerPixel + channel;
                CAPTURE(x, y, channel, expected[channel], std::to_integer<std::uint8_t>(mapped->data[offset]));
                CHECK(nearByte(mapped->data[offset], expected[channel]));
            }
        }
    }
    constexpr std::array<std::uint8_t, 4> mip2Expected{60, 60, 60, 255};
    for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
    {
        const auto offset = mipOffsets[2] + channel;
        CAPTURE(channel, mip2Expected[channel], std::to_integer<std::uint8_t>(mapped->data[offset]));
        CHECK(nearByte(mapped->data[offset], mip2Expected[channel]));
    }
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("mipmap publication captures one immutable sparse generation and its timeline",
          "[size:medium][rhi][texture][mipmap][publication][sparse][warp]")
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

    REQUIRE(rachis::test::requireCapabilities(device).mipmapGeneration);
    REQUIRE(rachis::test::requireCapabilities(device).enhancedBarriers);
    if (!rachis::test::requireCapabilities(device).sparseTextures)
    {
        return;
    }

    constexpr std::uint32_t sparseWidth = 256;
    constexpr std::uint32_t sparseHeight = 256;
    constexpr std::uint32_t sparseMipLevels = 9;
    const rachis::TextureDesc sparseDesc{
        .width = sparseWidth,
        .height = sparseHeight,
        .mipLevels = sparseMipLevels,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
        .label = "Sparse mipmap publication texture",
    };
    auto texture = device.createSparseTexture(sparseDesc);
    REQUIRE(texture.has_value());
    const auto properties = device.querySparseTextureProperties(*texture);
    REQUIRE(properties.has_value());
    auto heap = device.createResourceHeap(rachis::ResourceHeapDesc{
        .size = properties->tileSize * properties->totalTileCount,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::nonAttachmentTexture,
        .flags = rachis::ResourceHeapFlag::sparseBacking,
        .label = "Sparse mipmap publication heap",
    });
    REQUIRE(heap.has_value());

    std::vector<rachis::SparseTextureTileMapping> standardMappings;
    standardMappings.reserve(properties->standardMipTileGrids.size());
    std::uint32_t heapTileOffset = 0;
    for (std::uint32_t mipLevel = 0; mipLevel < properties->standardMipCount; ++mipLevel)
    {
        const auto grid = properties->standardMipTileGrids[mipLevel];
        standardMappings.push_back(rachis::SparseTextureTileMapping{
            .texture = *texture,
            .subresource = {.mipLevel = mipLevel},
            .textureTileExtent = grid,
            .heap = *heap,
            .heapTileOffset = heapTileOffset,
        });
        heapTileOffset += grid.width * grid.height * grid.depth;
    }
    std::vector<rachis::SparseTextureMipTailMapping> tailMappings;
    if (properties->packedMipCount != 0)
    {
        tailMappings.push_back(rachis::SparseTextureMipTailMapping{
            .texture = *texture,
            .heap = *heap,
            .heapTileOffset = heapTileOffset,
        });
        heapTileOffset += properties->mipTailTileCount;
    }
    REQUIRE(heapTileOffset == properties->totalTileCount);
    const auto mappingPoint = device.enqueueSparseTextureMappings(
        rachis::QueueType::graphics,
        rachis::SparseTextureMappingBatch{.standardTiles = standardMappings, .mipTails = tailMappings});
    REQUIRE(mappingPoint.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    const rachis::TextureMipmapGeneration generation{
        .texture = *texture,
        .subresources =
            {
                .firstMipLevel = 0,
                .mipLevelCount = sparseMipLevels,
                .firstArrayLayer = 0,
                .arrayLayerCount = 1,
                .aspect = rachis::TextureAspect::color,
            },
        .baseLayout = rachis::TextureLayout::common,
        .finalLayout = rachis::TextureLayout::common,
    };
    const auto beforeRejectedGeneration = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(beforeRejectedGeneration.has_value());
    {
        const FailResourceRetention failSparseGenerationPublication(3);
        const auto rejected = device.generateMipmaps(*command, generation);
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::outOfMemory);
    }
    const auto afterRejectedGeneration = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(afterRejectedGeneration.has_value());
    CHECK(*afterRejectedGeneration == *beforeRejectedGeneration);

    REQUIRE(device.generateMipmaps(*command, generation).has_value());
    const auto afterAcceptedGeneration = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(afterAcceptedGeneration.has_value());
    const auto acceptedDependencies =
        afterAcceptedGeneration.value_or(rachis::d3d12::test::CommandDependencySnapshot{});
    const auto priorDependencies = beforeRejectedGeneration.value_or(rachis::d3d12::test::CommandDependencySnapshot{});
    CHECK(acceptedDependencies.resourceOwners == priorDependencies.resourceOwners + 3);
    CHECK(acceptedDependencies.residencyLeases == priorDependencies.residencyLeases);
    CHECK(acceptedDependencies.swapchainReferences == priorDependencies.swapchainReferences);
    CHECK(acceptedDependencies.sparseMappingDependencies == priorDependencies.sparseMappingDependencies + 1);

    const auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyResourceHeap(*heap).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("mipmap publication rejects an unpublished sparse mapping generation",
          "[size:medium][rhi][texture][mipmap][publication][sparse][failure][warp]")
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

    REQUIRE(rachis::test::requireCapabilities(device).mipmapGeneration);
    REQUIRE(rachis::test::requireCapabilities(device).enhancedBarriers);
    if (!rachis::test::requireCapabilities(device).sparseTextures)
    {
        return;
    }

    constexpr std::uint32_t sparseMipLevels = 9;
    auto texture = device.createSparseTexture({
        .width = 256,
        .height = 256,
        .mipLevels = sparseMipLevels,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
        .label = "Unpublished sparse mipmap texture",
    });
    REQUIRE(texture.has_value());
    const auto properties = device.querySparseTextureProperties(*texture);
    REQUIRE(properties.has_value());
    auto heap = device.createResourceHeap(rachis::ResourceHeapDesc{
        .size = properties->tileSize,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::nonAttachmentTexture,
        .flags = rachis::ResourceHeapFlag::sparseBacking,
        .label = "Unpublished sparse mipmap heap",
    });
    REQUIRE(heap.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());

    const std::array mapping{
        rachis::SparseTextureTileMapping{
            .texture = *texture,
            .heap = *heap,
        },
    };
    rachis::d3d12::test::nextSparseMappingSignalFailure().store(true, std::memory_order_relaxed);
    const auto unpublished =
        device.enqueueSparseTextureMappings(rachis::QueueType::graphics, {.standardTiles = mapping});
    REQUIRE_FALSE(unpublished.has_value());
    CHECK(unpublished.error().code == rachis::ErrorCode::deviceLost);

    const auto beforeRejectedGeneration = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(beforeRejectedGeneration.has_value());
    const auto rejected = device.generateMipmaps(*command,
                                                 {
                                                     .texture = *texture,
                                                     .subresources =
                                                         {
                                                             .firstMipLevel = 0,
                                                             .mipLevelCount = sparseMipLevels,
                                                             .firstArrayLayer = 0,
                                                             .arrayLayerCount = 1,
                                                             .aspect = rachis::TextureAspect::color,
                                                         },
                                                     .baseLayout = rachis::TextureLayout::common,
                                                     .finalLayout = rachis::TextureLayout::common,
                                                 });
    REQUIRE_FALSE(rejected.has_value());
    CHECK(rejected.error().code == rachis::ErrorCode::invalidState);
    const auto afterRejectedGeneration = rachis::d3d12::test::commandDependencySnapshot(*command);
    REQUIRE(afterRejectedGeneration.has_value());
    CHECK(*afterRejectedGeneration == *beforeRejectedGeneration);

    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyResourceHeap(*heap).has_value());
}

TEST_CASE("mipmap generation waits for the render pass that wrote its base level",
          "[size:medium][requires:HardwareD3D12][rhi][texture][mipmap][hardware]")
{
    auto created = createHardwareDevice();
    if (!created)
    {
        SKIP("No D3D12 hardware device is available.");
    }
    auto device = std::move(*created);
    if (rachis::test::requireAdapterInfo(device).type == rachis::AdapterType::software)
    {
        SKIP("The selected D3D12 adapter is a software adapter.");
    }
    const auto seed =
        createRenderedMipPipeline(device, RACHIS_TEST_MIPMAP_AFTER_RENDERING_SLOW_PIXEL_SHADER_PATH, "slowPixel", {});

    for (std::uint32_t iteration = 0; iteration < renderedIterations; ++iteration)
    {
        CAPTURE(iteration);
        auto texture = device.createTexture(renderedMipDesc());
        REQUIRE(texture);
        const auto baseView = createRenderedMipView(device, *texture, 0);
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command);
        recordRenderedBase(device, *command, *texture, baseView, seed);
        REQUIRE(device
                    .generateMipmaps(*command,
                                     {
                                         .texture = *texture,
                                         .baseLayout = rachis::TextureLayout::renderTarget,
                                         .finalLayout = rachis::TextureLayout::copySource,
                                     })
                    .has_value());
        const auto texel = readRenderedLastMip(device, std::move(*command), *texture);
        CAPTURE(texel[0], texel[1], texel[2], texel[3]);
        CHECK(nearTexel(texel, {255, 128, 64, 255}, 1));
        REQUIRE(device.destroyTextureView(baseView).has_value());
        REQUIRE(device.destroyTexture(*texture).has_value());
    }
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("mipmap generation finishes before a later render pass blends into its last mip",
          "[size:medium][requires:HardwareD3D12][rhi][texture][mipmap][hardware]")
{
    auto created = createHardwareDevice();
    if (!created)
    {
        SKIP("No D3D12 hardware device is available.");
    }
    auto device = std::move(*created);
    if (rachis::test::requireAdapterInfo(device).type == rachis::AdapterType::software)
    {
        SKIP("The selected D3D12 adapter is a software adapter.");
    }
    const auto seed =
        createRenderedMipPipeline(device, RACHIS_TEST_MIPMAP_AFTER_RENDERING_SLOW_PIXEL_SHADER_PATH, "slowPixel", {});
    constexpr rachis::BlendComponent additive{
        .sourceFactor = rachis::BlendFactor::one,
        .destinationFactor = rachis::BlendFactor::one,
    };
    const auto blend = createRenderedMipPipeline(device,
                                                 RACHIS_TEST_MIPMAP_AFTER_RENDERING_BLEND_PIXEL_SHADER_PATH,
                                                 "blendPixel",
                                                 {.enabled = true, .color = additive, .alpha = additive});

    for (std::uint32_t iteration = 0; iteration < renderedIterations; ++iteration)
    {
        CAPTURE(iteration);
        auto texture = device.createTexture(renderedMipDesc());
        REQUIRE(texture);
        const auto baseView = createRenderedMipView(device, *texture, 0);
        const auto lastView = createRenderedMipView(device, *texture, renderedLastMip);
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command);
        recordRenderedBase(device, *command, *texture, baseView, seed);
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *texture,
                                        .stagesBefore = rachis::PipelineStage::renderTarget,
                                        .stagesAfter = rachis::PipelineStage::computeShader,
                                        .accessBefore = rachis::Access::renderTarget,
                                        .accessAfter = rachis::Access::shaderResource,
                                        .layoutBefore = rachis::TextureLayout::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::shaderResource,
                                        .subresources = {.firstMipLevel = 0, .mipLevelCount = 1},
                                    })
                    .has_value());
        REQUIRE(device
                    .generateMipmaps(*command,
                                     {
                                         .texture = *texture,
                                         .baseLayout = rachis::TextureLayout::shaderResource,
                                         .finalLayout = rachis::TextureLayout::renderTarget,
                                     })
                    .has_value());
        recordRenderedMip(device, *command, lastView, blend, 1, 1, rachis::LoadOperation::load);
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
                                        .subresources = {.firstMipLevel = renderedLastMip, .mipLevelCount = 1},
                                    })
                    .has_value());
        const auto texel = readRenderedLastMip(device, std::move(*command), *texture);
        CAPTURE(texel[0], texel[1], texel[2], texel[3]);
        CHECK(nearTexel(texel, {255, 192, 192, 255}, 2));
        REQUIRE(device.destroyTextureView(lastView).has_value());
        REQUIRE(device.destroyTextureView(baseView).has_value());
        REQUIRE(device.destroyTexture(*texture).has_value());
    }
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
