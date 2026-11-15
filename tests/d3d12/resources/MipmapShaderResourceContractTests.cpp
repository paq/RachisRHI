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
#include <optional>
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
    constexpr std::uint64_t placementAlignment = 512;
    constexpr std::array<std::uint8_t, 4> mip2Expected{60, 60, 60, 255};

    constexpr auto graphicsShaderReads =
        rachis::PipelineStage::vertexShader | rachis::PipelineStage::pixelShader | rachis::PipelineStage::computeShader;

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

    [[nodiscard]] bool nearUnorm(float actual, std::uint8_t expected) noexcept
    {
        const auto scaled = actual * 255.0F;
        const auto target = static_cast<float>(expected);
        return scaled >= target - 1.0F && scaled <= target + 1.0F;
    }

    [[nodiscard]] rachis::Device createWarpDevice(bool legacyBarriers)
    {
        std::optional<ForceLegacyBarriers> forceLegacyBarriers;
        if (legacyBarriers)
        {
            forceLegacyBarriers.emplace();
        }
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
        const auto capabilities = rachis::test::requireCapabilities(device);
        REQUIRE(capabilities.mipmapGeneration);
        REQUIRE(capabilities.enhancedBarriers == !legacyBarriers);
        return device;
    }

    void writeSeed(rachis::Device& device, rachis::BufferHandle upload)
    {
        auto mapped = device.mapBuffer(upload);
        REQUIRE(mapped.has_value());
        std::fill_n(mapped->data, mapped->size, std::byte{0});
        for (std::uint32_t y = 0; y < baseHeight; ++y)
        {
            for (std::uint32_t x = 0; x < baseWidth; ++x)
            {
                const auto texel = baseTexel(x, y);
                for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
                {
                    mapped->data[static_cast<std::uint64_t>(y) * rowPitch +
                                 static_cast<std::uint64_t>(x) * bytesPerPixel + channel] =
                        static_cast<std::byte>(texel[channel]);
                }
            }
        }
        REQUIRE(device.unmapBuffer(upload).has_value());
    }

    [[nodiscard]] constexpr rachis::TextureSubresourceRange mipRange(std::uint32_t firstMipLevel,
                                                                     std::uint32_t mipLevelCount) noexcept
    {
        return rachis::TextureSubresourceRange{
            .firstMipLevel = firstMipLevel,
            .mipLevelCount = mipLevelCount,
            .firstArrayLayer = 0,
            .arrayLayerCount = 1,
            .aspect = rachis::TextureAspect::color,
        };
    }

    [[nodiscard]] rachis::Result<rachis::TextureHandle> createChainTexture(rachis::Device& device,
                                                                           std::string_view label)
    {
        return device.createTexture(rachis::TextureDesc{
            .width = baseWidth,
            .height = baseHeight,
            .mipLevels = mipLevels,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                     rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
            .label = label,
        });
    }

    void seedShaderResourceBase(rachis::Device& device,
                                rachis::CommandBuffer& command,
                                rachis::TextureHandle texture,
                                rachis::BufferHandle upload,
                                rachis::PipelineStage shaderReads)
    {
        REQUIRE(device
                    .textureBarrier(command,
                                    rachis::TextureBarrier{
                                        .texture = texture,
                                        .stagesBefore = rachis::PipelineStage::all,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::none,
                                        .accessAfter = rachis::Access::copyDestination,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::copyDestination,
                                        .subresources = mipRange(0, 1),
                                    })
                    .has_value());
        REQUIRE(device
                    .copyBufferToTexture(command,
                                         rachis::BufferTextureCopy{
                                             .buffer = upload,
                                             .bytesPerRow = rowPitch,
                                             .rowsPerImage = baseHeight,
                                             .texture = texture,
                                             .extent = {baseWidth, baseHeight, 1},
                                         })
                    .has_value());
        REQUIRE(device
                    .textureBarrier(command,
                                    rachis::TextureBarrier{
                                        .texture = texture,
                                        .stagesBefore = rachis::PipelineStage::copy,
                                        .stagesAfter = shaderReads,
                                        .accessBefore = rachis::Access::copyDestination,
                                        .accessAfter = rachis::Access::shaderResource,
                                        .layoutBefore = rachis::TextureLayout::copyDestination,
                                        .layoutAfter = rachis::TextureLayout::shaderResource,
                                        .subresources = mipRange(0, 1),
                                    })
                    .has_value());
    }

    void copyMipToReadback(rachis::Device& device,
                           rachis::CommandBuffer& command,
                           rachis::TextureHandle texture,
                           rachis::BufferHandle readback,
                           std::uint64_t offset,
                           std::uint32_t mipLevel)
    {
        const auto extent = (std::max)(1U, baseWidth >> mipLevel);
        REQUIRE(device
                    .copyTextureToBuffer(command,
                                         rachis::BufferTextureCopy{
                                             .buffer = readback,
                                             .bufferOffset = offset,
                                             .bytesPerRow = rowPitch,
                                             .rowsPerImage = extent,
                                             .texture = texture,
                                             .textureSubresource = {.mipLevel = mipLevel},
                                             .extent = {extent, extent, 1},
                                         })
                    .has_value());
    }

    void checkMip1(const std::byte* data)
    {
        for (std::uint32_t y = 0; y < 2; ++y)
        {
            for (std::uint32_t x = 0; x < 2; ++x)
            {
                const auto expected = mip1Texel(x, y);
                for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
                {
                    const auto offset = static_cast<std::uint64_t>(y) * rowPitch +
                                        static_cast<std::uint64_t>(x) * bytesPerPixel + channel;
                    CAPTURE(x, y, channel, expected[channel], std::to_integer<std::uint8_t>(data[offset]));
                    CHECK(nearByte(data[offset], expected[channel]));
                }
            }
        }
    }

    void checkMip2(const std::byte* data)
    {
        for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
        {
            CAPTURE(channel, mip2Expected[channel], std::to_integer<std::uint8_t>(data[channel]));
            CHECK(nearByte(data[channel], mip2Expected[channel]));
        }
    }

    void checkCleanDiagnostics(const rachis::Device& device)
    {
        const auto diagnostics = device.diagnosticCounts();
        CHECK(diagnostics.warnings == 0);
        CHECK(diagnostics.errors == 0);
        CHECK(diagnostics.corruptions == 0);
    }

    void verifyPixelShaderReadsGeneratedChain(bool legacyBarriers)
    {
        auto device = createWarpDevice(legacyBarriers);

        constexpr std::uint64_t sampledMip1Offset = 0;
        constexpr std::uint64_t copiedMip1Offset = placementAlignment;
        constexpr std::uint64_t copiedMip2Offset = placementAlignment * 2;
        constexpr std::uint64_t targetOffset = placementAlignment * 3;
        auto upload = device.createBuffer(rachis::BufferDesc{
            .size = static_cast<std::uint64_t>(rowPitch) * baseHeight,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
            .label = "Pixel-read mip seed upload",
        });
        auto readback = device.createBuffer(rachis::BufferDesc{
            .size = targetOffset + rowPitch,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Pixel-read mip readback",
        });
        auto sampledTexture = createChainTexture(device, "Pixel-read generated mip texture");
        auto copiedTexture = createChainTexture(device, "Pixel-read copied mip texture");
        auto target = device.createTexture(rachis::TextureDesc{
            .width = 1,
            .height = 1,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
            .label = "Pixel-read mip target",
        });
        REQUIRE(upload.has_value());
        REQUIRE(readback.has_value());
        REQUIRE(sampledTexture.has_value());
        REQUIRE(copiedTexture.has_value());
        REQUIRE(target.has_value());
        writeSeed(device, *upload);

        auto sampledView = device.createTextureView(rachis::TextureViewDesc{
            .texture = *sampledTexture,
            .usage = rachis::TextureViewUsage::sampled,
            .label = "Pixel-read generated chain view",
        });
        auto targetView = device.createTextureView(rachis::TextureViewDesc{
            .texture = *target,
            .usage = rachis::TextureViewUsage::renderTarget,
        });
        REQUIRE(sampledView.has_value());
        REQUIRE(targetView.has_value());

        const auto vertexBytecode = loadShader(RACHIS_TEST_MIPMAP_SHADER_READ_VERTEX_SHADER_PATH);
        const auto pixelBytecode = loadShader(RACHIS_TEST_MIPMAP_SHADER_READ_PIXEL_SHADER_PATH);
        REQUIRE_FALSE(vertexBytecode.empty());
        REQUIRE_FALSE(pixelBytecode.empty());
        constexpr std::array pixelBindings{
            rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::sampledTexture, 1},
        };
        auto vertexShader = device.createShaderModule(rachis::ShaderModuleDesc{
            .stage = rachis::ShaderStage::vertex,
            .bytecode = vertexBytecode,
            .entryPoint = "fullscreenVertex",
            .shaderInterface = {},
        });
        auto pixelShader = device.createShaderModule(rachis::ShaderModuleDesc{
            .stage = rachis::ShaderStage::pixel,
            .bytecode = pixelBytecode,
            .entryPoint = "pixelMain",
            .shaderInterface = rachis::test::shaderInterface(pixelBindings),
        });
        REQUIRE(vertexShader.has_value());
        REQUIRE(pixelShader.has_value());
        const std::array bindingEntries{
            rachis::BindingLayoutEntry{0, rachis::BindingType::sampledTexture, rachis::ShaderStage::pixel},
        };
        auto bindingLayout = device.createBindingLayout(rachis::BindingLayoutDesc{
            .entries = bindingEntries,
            .label = "Pixel-read mip binding layout",
        });
        REQUIRE(bindingLayout.has_value());
        const std::array writes{
            rachis::BindingWrite{.binding = 0, .resource = *sampledView},
        };
        auto bindingSet = rachis::test::createBindingSet(device,
                                                         {
                                                             .layout = *bindingLayout,
                                                             .writes = writes,
                                                             .label = "Pixel-read mip binding set",
                                                         });
        REQUIRE(bindingSet.has_value());
        const std::array bindingLayouts{*bindingLayout};
        auto pipelineLayout = device.createPipelineLayout(rachis::PipelineLayoutDesc{
            .bindingLayouts = bindingLayouts,
            .label = "Pixel-read mip pipeline layout",
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
        });
        REQUIRE(pipeline.has_value());

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        seedShaderResourceBase(device, *command, *sampledTexture, *upload, graphicsShaderReads);
        seedShaderResourceBase(device, *command, *copiedTexture, *upload, graphicsShaderReads);
        REQUIRE(device
                    .generateMipmaps(*command,
                                     rachis::TextureMipmapGeneration{
                                         .texture = *sampledTexture,
                                         .subresources = mipRange(0, mipLevels),
                                         .baseLayout = rachis::TextureLayout::shaderResource,
                                     })
                    .has_value());
        REQUIRE(device
                    .generateMipmaps(*command,
                                     rachis::TextureMipmapGeneration{
                                         .texture = *copiedTexture,
                                         .subresources = mipRange(0, mipLevels),
                                         .baseLayout = rachis::TextureLayout::shaderResource,
                                         .finalLayout = rachis::TextureLayout::copySource,
                                     })
                    .has_value());

        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
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
                .clearValue = {},
            },
        };
        REQUIRE(device.beginRendering(*command, rachis::RenderingDesc{.colorAttachments = attachments}).has_value());
        REQUIRE(device.bindGraphicsPipeline(*command, *pipeline).has_value());
        REQUIRE(device.bindGraphicsBindingSet(*command, 0, *bindingSet).has_value());
        REQUIRE(device.setViewport(*command, rachis::Viewport{.width = 1.0F, .height = 1.0F}).has_value());
        REQUIRE(device.setScissor(*command, rachis::ScissorRect{.width = 1, .height = 1}).has_value());
        REQUIRE(device.draw(*command, 3).has_value());
        REQUIRE(device.endRendering(*command).has_value());

        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = *sampledTexture,
                                        .stagesBefore = graphicsShaderReads,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::shaderResource,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::shaderResource,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                        .subresources = mipRange(0, mipLevels),
                                    })
                    .has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = *target,
                                        .stagesBefore = rachis::PipelineStage::renderTarget,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::renderTarget,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                    })
                    .has_value());
        copyMipToReadback(device, *command, *sampledTexture, *readback, sampledMip1Offset, 1);
        copyMipToReadback(device, *command, *copiedTexture, *readback, copiedMip1Offset, 1);
        copyMipToReadback(device, *command, *copiedTexture, *readback, copiedMip2Offset, 2);
        REQUIRE(device
                    .copyTextureToBuffer(*command,
                                         rachis::BufferTextureCopy{
                                             .buffer = *readback,
                                             .bufferOffset = targetOffset,
                                             .bytesPerRow = rowPitch,
                                             .rowsPerImage = 1,
                                             .texture = *target,
                                             .extent = {1, 1, 1},
                                         })
                    .has_value());

        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());

        auto mapped = device.mapBuffer(*readback);
        REQUIRE(mapped.has_value());
        checkMip1(mapped->data + sampledMip1Offset);
        checkMip1(mapped->data + copiedMip1Offset);
        checkMip2(mapped->data + copiedMip2Offset);
        checkMip2(mapped->data + targetOffset);
        REQUIRE(device.unmapBuffer(*readback).has_value());

        REQUIRE(device.destroyPipeline(*pipeline).has_value());
        REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
        REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
        REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
        REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
        REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
        REQUIRE(device.destroyTextureView(*targetView).has_value());
        REQUIRE(device.destroyTextureView(*sampledView).has_value());
        REQUIRE(device.destroyTexture(*target).has_value());
        REQUIRE(device.destroyTexture(*copiedTexture).has_value());
        REQUIRE(device.destroyTexture(*sampledTexture).has_value());
        REQUIRE(device.destroyBuffer(*readback).has_value());
        REQUIRE(device.destroyBuffer(*upload).has_value());
        checkCleanDiagnostics(device);
    }

    void verifyComputeQueueReadsGeneratedChain(bool legacyBarriers)
    {
        auto device = createWarpDevice(legacyBarriers);

        constexpr std::uint64_t mip1Offset = 0;
        constexpr std::uint64_t readValueOffset = placementAlignment;
        constexpr std::uint64_t readValueSize = 4 * sizeof(float);
        auto upload = device.createBuffer(rachis::BufferDesc{
            .size = static_cast<std::uint64_t>(rowPitch) * baseHeight,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
            .label = "Compute-read mip seed upload",
        });
        auto output = device.createBuffer(rachis::BufferDesc{
            .size = readValueSize,
            .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
            .label = "Compute-read mip output",
        });
        auto readback = device.createBuffer(rachis::BufferDesc{
            .size = readValueOffset + readValueSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Compute-read mip readback",
        });
        auto texture = createChainTexture(device, "Compute-read generated mip texture");
        REQUIRE(upload.has_value());
        REQUIRE(output.has_value());
        REQUIRE(readback.has_value());
        REQUIRE(texture.has_value());
        writeSeed(device, *upload);

        auto sampledView = device.createTextureView(rachis::TextureViewDesc{
            .texture = *texture,
            .usage = rachis::TextureViewUsage::sampled,
            .label = "Compute-read generated chain view",
        });
        REQUIRE(sampledView.has_value());
        const auto bytecode = loadShader(RACHIS_TEST_MIPMAP_SHADER_READ_COMPUTE_SHADER_PATH);
        REQUIRE_FALSE(bytecode.empty());
        auto shader = device.createShaderModule(rachis::ShaderModuleDesc{
            .stage = rachis::ShaderStage::compute,
            .bytecode = bytecode,
            .entryPoint = "computeMain",
            .label = "Compute-read mip shader",
            .shaderInterface = rachis::test::computeShaderInterface(
                {1, 1, 1},
                std::array{
                    rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::sampledTexture, 1},
                    rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
                }),
        });
        REQUIRE(shader.has_value());
        const std::array bindingEntries{
            rachis::BindingLayoutEntry{0, rachis::BindingType::sampledTexture, rachis::ShaderStage::compute},
            rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
        };
        auto bindingLayout = device.createBindingLayout(rachis::BindingLayoutDesc{
            .entries = bindingEntries,
            .label = "Compute-read mip binding layout",
        });
        REQUIRE(bindingLayout.has_value());
        const std::array writes{
            rachis::BindingWrite{.binding = 0, .resource = *sampledView},
            rachis::BindingWrite{
                .binding = 1,
                .resource =
                    rachis::BufferBinding{
                        .buffer = *output,
                        .size = readValueSize,
                        .structureByteStride = readValueSize,
                    },
            },
        };
        auto bindingSet = rachis::test::createBindingSet(device,
                                                         {
                                                             .layout = *bindingLayout,
                                                             .writes = writes,
                                                             .label = "Compute-read mip binding set",
                                                         });
        REQUIRE(bindingSet.has_value());
        const std::array bindingLayouts{*bindingLayout};
        auto pipelineLayout = device.createPipelineLayout(rachis::PipelineLayoutDesc{
            .bindingLayouts = bindingLayouts,
            .label = "Compute-read mip pipeline layout",
        });
        REQUIRE(pipelineLayout.has_value());
        auto pipeline = device.createComputePipeline(rachis::ComputePipelineDesc{
            .layout = *pipelineLayout,
            .computeShader = *shader,
            .label = "Compute-read mip pipeline",
        });
        REQUIRE(pipeline.has_value());

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command.has_value());
        seedShaderResourceBase(device, *command, *texture, *upload, rachis::PipelineStage::computeShader);
        REQUIRE(device
                    .generateMipmaps(*command,
                                     rachis::TextureMipmapGeneration{
                                         .texture = *texture,
                                         .subresources = mipRange(0, mipLevels),
                                         .baseLayout = rachis::TextureLayout::shaderResource,
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
        REQUIRE(device
                    .copyBuffer(*command,
                                {
                                    .source = *output,
                                    .destination = *readback,
                                    .destinationOffset = readValueOffset,
                                    .size = readValueSize,
                                })
                    .has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = *texture,
                                        .stagesBefore = rachis::PipelineStage::computeShader,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::shaderResource,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::shaderResource,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                        .subresources = mipRange(0, mipLevels),
                                    })
                    .has_value());
        copyMipToReadback(device, *command, *texture, *readback, mip1Offset, 1);

        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());

        auto mapped = device.mapBuffer(*readback);
        REQUIRE(mapped.has_value());
        checkMip1(mapped->data + mip1Offset);
        std::array<float, 4> readValue{};
        std::memcpy(readValue.data(), mapped->data + readValueOffset, sizeof(readValue));
        for (std::size_t channel = 0; channel < readValue.size(); ++channel)
        {
            CAPTURE(channel, mip2Expected[channel], readValue[channel]);
            CHECK(nearUnorm(readValue[channel], mip2Expected[channel]));
        }
        REQUIRE(device.unmapBuffer(*readback).has_value());

        REQUIRE(device.destroyPipeline(*pipeline).has_value());
        REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
        REQUIRE(device.destroyBindingSet(*bindingSet).has_value());
        REQUIRE(device.destroyBindingLayout(*bindingLayout).has_value());
        REQUIRE(device.destroyShaderModule(*shader).has_value());
        REQUIRE(device.destroyTextureView(*sampledView).has_value());
        REQUIRE(device.destroyTexture(*texture).has_value());
        REQUIRE(device.destroyBuffer(*readback).has_value());
        REQUIRE(device.destroyBuffer(*output).has_value());
        REQUIRE(device.destroyBuffer(*upload).has_value());
        checkCleanDiagnostics(device);
    }

} // namespace

TEST_CASE("mipmap generation hands a shader-resource chain to graphics-queue pixel shaders",
          "[size:medium][rhi][texture][mipmap][barrier][warp]")
{
    SECTION("legacy barriers")
    {
        verifyPixelShaderReadsGeneratedChain(true);
    }
    SECTION("enhanced barriers")
    {
        verifyPixelShaderReadsGeneratedChain(false);
    }
}

TEST_CASE("mipmap generation hands a shader-resource chain to compute-queue compute shaders",
          "[size:medium][rhi][texture][mipmap][barrier][compute][warp]")
{
    SECTION("legacy barriers")
    {
        verifyComputeQueueReadsGeneratedChain(true);
    }
    SECTION("enhanced barriers")
    {
        verifyComputeQueueReadsGeneratedChain(false);
    }
}
