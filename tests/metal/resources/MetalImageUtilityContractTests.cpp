#include "Rachis/Device.h"
#include "RhiTestSupport.h"

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
#include <string_view>
#include <utility>
#include <vector>

namespace
{

    constexpr std::uint32_t copyWidth = 4;
    constexpr std::uint32_t copyHeight = 4;
    constexpr std::uint32_t bytesPerPixel = 4;
    constexpr std::uint32_t mipLevels = 3;
    constexpr std::uint32_t renderedSize = 2048;
    constexpr std::uint32_t renderedMipLevels = 12;
    constexpr std::uint32_t renderedLastMip = renderedMipLevels - 1;
    constexpr std::uint32_t renderedIterations = 4;

    [[nodiscard]] rachis::DeviceDesc metalDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
    }

    [[nodiscard]] std::byte texelByte(std::uint32_t x, std::uint32_t y, std::uint32_t channel) noexcept
    {
        return static_cast<std::byte>((x * 31U + y * 47U + channel * 59U + 11U) & 0xFFU);
    }

    [[nodiscard]] std::array<std::uint8_t, 4> baseTexel(std::uint32_t x, std::uint32_t y) noexcept
    {
        return {
            static_cast<std::uint8_t>(x * 40U),
            static_cast<std::uint8_t>(y * 40U),
            static_cast<std::uint8_t>((x + y) * 20U),
            255,
        };
    }

    [[nodiscard]] bool nearByte(std::byte actual, std::uint8_t expected, std::uint8_t slack = 1) noexcept
    {
        const auto value = std::to_integer<std::uint8_t>(actual);
        const auto lower = expected < slack ? 0U : static_cast<std::uint32_t>(expected) - slack;
        const auto upper = (std::min)(static_cast<std::uint32_t>(expected) + slack, 255U);
        return value >= lower && value <= upper;
    }

    struct SrgbMipmapCase final
    {
        rachis::Format format = rachis::Format::unknown;
        rachis::Format linearView = rachis::Format::unknown;
        bool placed = false;
    };

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

    [[nodiscard]] std::filesystem::path pnmsPath(std::string_view metallibName)
    {
        return std::filesystem::path(RACHIS_METALLIB_DIR) /
               std::filesystem::path(metallibName).replace_extension(".pnms");
    }

    [[nodiscard]] rachis::PipelineHandle createRenderedMipPipeline(rachis::Device& device,
                                                                   std::string_view pixelMetallib,
                                                                   std::string_view pixelEntry,
                                                                   const rachis::BlendState& blend)
    {
        const auto vertexBytes = loadBytes(pnmsPath("MipmapAfterRendering.vs.metallib"));
        const auto pixelBytes = loadBytes(pnmsPath(pixelMetallib));
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
            .label = "Metal rendered mip pipeline",
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
            .label = "Metal rendered mip chain",
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

} // namespace

#if RACHIS_ENABLE_METAL
TEST_CASE("Metal texture copies move color texels and reject overlapping destinations",
          "[size:large][rhi][metal][copy][texture]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    auto source = device.createTexture({
        .width = copyWidth,
        .height = copyHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Metal image-copy source",
    });
    auto destination = device.createTexture({
        .width = copyWidth,
        .height = copyHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Metal image-copy destination",
    });
    REQUIRE(source);
    REQUIRE(destination);

    const rachis::BufferTextureCopyRegion region{.extent = {.width = copyWidth, .height = copyHeight}};
    auto layout = device.queryTextureTransferLayout(*source, region);
    REQUIRE(layout);
    auto upload = device.createBuffer({
        .size = layout->requiredBufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto readback = device.createBuffer({
        .size = layout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(upload);
    REQUIRE(readback);

    std::vector<std::byte> pixels(static_cast<std::size_t>(layout->requiredBufferSize));
    for (std::uint32_t y = 0; y < copyHeight; ++y)
    {
        for (std::uint32_t x = 0; x < copyWidth; ++x)
        {
            for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
            {
                pixels[static_cast<std::size_t>(y) * layout->bytesPerRow + static_cast<std::size_t>(x) * bytesPerPixel +
                       channel] = texelByte(x, y, channel);
            }
        }
    }
    REQUIRE(device.writeBuffer(*upload, 0, pixels).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *source,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    REQUIRE(device.copyBufferToTexture(*command, *upload, *source, std::array{region}).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *source,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::copyDestination,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *destination,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());

    const std::span<const rachis::TextureCopyRegion> empty;
    CHECK_FALSE(device.copyTexture(*command, *source, *destination, empty).has_value());
    CHECK_FALSE(device
                    .copyTexture(*command,
                                 {
                                     .source = *source,
                                     .destination = *source,
                                     .extent = {.width = copyWidth, .height = copyHeight},
                                 })
                    .has_value());
    const std::array overlapping{
        rachis::TextureCopyRegion{.extent = {.width = 2, .height = 2}},
        rachis::TextureCopyRegion{.destinationOrigin = {.x = 1, .y = 1}, .extent = {.width = 2, .height = 2}},
    };
    CHECK_FALSE(device.copyTexture(*command, *source, *destination, overlapping).has_value());
    REQUIRE(device
                .copyTexture(*command,
                             {
                                 .source = *source,
                                 .destination = *destination,
                                 .extent = {.width = copyWidth, .height = copyHeight},
                             })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *destination,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::copyDestination,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device.copyTextureToBuffer(*command, *destination, *readback, std::array{region}).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped);
    CHECK(std::memcmp(mapped->data, pixels.data(), pixels.size()) == 0);
    REQUIRE(device.unmapBuffer(*readback).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal color resolve averages samples, preserves the MSAA source, and rejects min/max",
          "[size:large][rhi][metal][resolve][color]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto format = device.queryFormatProperties(rachis::Format::rgba8Unorm);
    REQUIRE(rachis::supportsResolveMode(format.colorResolveModes, rachis::ResolveMode::average));
    REQUIRE_FALSE(rachis::supportsResolveMode(format.colorResolveModes, rachis::ResolveMode::minimum));

    auto source = device.createTexture({
        .width = 2,
        .height = 2,
        .sampleCount = 4,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource,
        .label = "Metal MSAA color source",
    });
    auto destination = device.createTexture({
        .width = 2,
        .height = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource |
                 rachis::TextureUsage::copyDestination,
        .label = "Metal resolved color",
    });
    auto second = device.createTexture({
        .width = 2,
        .height = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource |
                 rachis::TextureUsage::copyDestination,
        .label = "Metal resolved color replay",
    });
    REQUIRE(source);
    REQUIRE(destination);
    REQUIRE(second);
    auto view = device.createTextureView({
        .texture = *source,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    REQUIRE(view);

    const rachis::BufferTextureCopyRegion region{.extent = {.width = 2, .height = 2}};
    auto layout = device.queryTextureTransferLayout(*destination, region);
    REQUIRE(layout);
    auto readback = device.createBuffer({
        .size = layout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    auto replayReadback = device.createBuffer({
        .size = layout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(readback);
    REQUIRE(replayReadback);

    auto copyCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(copyCommand);
    CHECK_FALSE(device.resolveTexture(*copyCommand, {.source = *source, .destination = *destination}).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *source,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array attachments{
        rachis::ColorAttachment{
            .view = *view,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.25F, 0.5F, 0.75F, 1.0F},
        },
    };
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
    CHECK_FALSE(device.resolveTexture(*command, {.source = *source, .destination = *destination}).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *source,
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
                                {
                                    .texture = *destination,
                                    .stagesAfter = rachis::PipelineStage::resolve,
                                    .accessAfter = rachis::Access::resolveDestination,
                                    .layoutAfter = rachis::TextureLayout::resolveDestination,
                                })
                .has_value());
    CHECK_FALSE(device
                    .resolveTexture(*command,
                                    {
                                        .source = *source,
                                        .destination = *destination,
                                        .mode = rachis::ResolveMode::minimum,
                                    })
                    .has_value());
    REQUIRE(device.resolveTexture(*command, {.source = *source, .destination = *destination}).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *second,
                                    .stagesAfter = rachis::PipelineStage::resolve,
                                    .accessAfter = rachis::Access::resolveDestination,
                                    .layoutAfter = rachis::TextureLayout::resolveDestination,
                                })
                .has_value());
    REQUIRE(device.resolveTexture(*command, {.source = *source, .destination = *second}).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *destination,
                                    .stagesBefore = rachis::PipelineStage::resolve,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::resolveDestination,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::resolveDestination,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *second,
                                    .stagesBefore = rachis::PipelineStage::resolve,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::resolveDestination,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::resolveDestination,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device.copyTextureToBuffer(*command, *destination, *readback, std::array{region}).has_value());
    REQUIRE(device.copyTextureToBuffer(*command, *second, *replayReadback, std::array{region}).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped);
    CHECK(nearByte(mapped->data[0], 64));
    CHECK(nearByte(mapped->data[1], 128));
    CHECK(nearByte(mapped->data[2], 191));
    CHECK(mapped->data[3] == std::byte{255});
    auto replay = device.mapBuffer(*replayReadback);
    REQUIRE(replay);
    for (std::uint32_t y = 0; y < 2; ++y)
    {
        const auto* left = mapped->data + static_cast<std::size_t>(y) * layout->bytesPerRow;
        const auto* right = replay->data + static_cast<std::size_t>(y) * layout->bytesPerRow;
        CHECK(std::memcmp(left, right, std::size_t{2} * bytesPerPixel) == 0);
    }
    REQUIRE(device.unmapBuffer(*replayReadback).has_value());
    REQUIRE(device.unmapBuffer(*readback).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal depth resolve uses min/max on d32Float", "[size:large][rhi][metal][resolve][depth]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto depth = device.queryFormatProperties(rachis::Format::d32Float);
    if (!rachis::supportsResolveMode(depth.depthResolveModes, rachis::ResolveMode::minimum))
    {
        return;
    }

    auto source = device.createTexture({
        .width = 2,
        .height = 2,
        .sampleCount = 4,
        .format = rachis::Format::d32Float,
        .usage = rachis::TextureUsage::depthStencil | rachis::TextureUsage::copySource,
        .label = "Metal MSAA depth source",
    });
    auto destination = device.createTexture({
        .width = 2,
        .height = 2,
        .format = rachis::Format::d32Float,
        .usage = rachis::TextureUsage::depthStencil | rachis::TextureUsage::copySource |
                 rachis::TextureUsage::copyDestination,
        .label = "Metal resolved depth",
    });
    REQUIRE(source);
    REQUIRE(destination);
    auto view = device.createTextureView({
        .texture = *source,
        .usage = rachis::TextureViewUsage::depthStencil,
        .aspect = rachis::TextureAspect::depth,
    });
    REQUIRE(view);

    const rachis::BufferTextureCopyRegion region{
        .textureSubresource = {.aspect = rachis::TextureAspect::depth},
        .extent = {.width = 2, .height = 2},
    };
    auto layout = device.queryTextureTransferLayout(*destination, region);
    REQUIRE(layout);
    auto readback = device.createBuffer({
        .size = layout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(readback);
    const auto beforeResolve = device.captureResourceMemorySnapshot();
    REQUIRE(beforeResolve);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *source,
                                    .stagesAfter = rachis::PipelineStage::depthStencil,
                                    .accessAfter = rachis::Access::depthStencilWrite,
                                    .layoutAfter = rachis::TextureLayout::depthStencilWrite,
                                    .subresources = {.aspect = rachis::TextureAspect::depth},
                                })
                .has_value());
    const rachis::DepthStencilAttachment depthAttachment{
        .view = *view,
        .depthLoadOperation = rachis::LoadOperation::clear,
        .depthClearValue = 0.25F,
        .stencilLoadOperation = rachis::LoadOperation::discard,
    };
    REQUIRE(device.beginRendering(*command, {.depthStencilAttachment = depthAttachment}).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *source,
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
                                {
                                    .texture = *destination,
                                    .stagesAfter = rachis::PipelineStage::resolve,
                                    .accessAfter = rachis::Access::resolveDestination,
                                    .layoutAfter = rachis::TextureLayout::resolveDestination,
                                    .subresources = {.aspect = rachis::TextureAspect::depth},
                                })
                .has_value());
    CHECK_FALSE(device
                    .resolveTexture(*command,
                                    {
                                        .source = *source,
                                        .sourceSubresource = {.aspect = rachis::TextureAspect::depth},
                                        .destination = *destination,
                                        .destinationSubresource = {.aspect = rachis::TextureAspect::depth},
                                        .mode = rachis::ResolveMode::average,
                                    })
                    .has_value());
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
                .textureBarrier(*command,
                                {
                                    .texture = *destination,
                                    .stagesBefore = rachis::PipelineStage::resolve,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::resolveDestination,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::resolveDestination,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                    .subresources = {.aspect = rachis::TextureAspect::depth},
                                })
                .has_value());
    REQUIRE(device.copyTextureToBuffer(*command, *destination, *readback, std::array{region}).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    const auto afterResolve = device.captureResourceMemorySnapshot();
    REQUIRE(afterResolve);
    CHECK(afterResolve->entries.size() == beforeResolve->entries.size());
    CHECK(afterResolve->liveAllocationCount == beforeResolve->liveAllocationCount);
    CHECK(afterResolve->liveAllocationBytes == beforeResolve->liveAllocationBytes);
    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped);
    float depthValue = 0.0F;
    std::memcpy(&depthValue, mapped->data, sizeof(depthValue));
    CHECK(depthValue > 0.20F);
    CHECK(depthValue < 0.30F);
    REQUIRE(device.unmapBuffer(*readback).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal mipmap generation rejects integer textures without mutating the command buffer",
          "[size:large][rhi][metal][mipmap][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command);
    for (const auto format : {rachis::Format::r8Uint,
                              rachis::Format::rgba8Sint,
                              rachis::Format::r16Uint,
                              rachis::Format::rgba16Sint,
                              rachis::Format::r32Uint,
                              rachis::Format::rgba32Sint})
    {
        auto texture = device.createTexture({
            .width = 4,
            .height = 4,
            .mipLevels = 3,
            .format = format,
            .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
        });
        REQUIRE(texture);
        auto generated = device.generateMipmaps(*command,
                                                {
                                                    .texture = *texture,
                                                    .baseLayout = rachis::TextureLayout::shaderResource,
                                                    .finalLayout = rachis::TextureLayout::shaderResource,
                                                });
        REQUIRE_FALSE(generated);
        CHECK(generated.error().code == rachis::ErrorCode::unsupported);
        REQUIRE(device.destroyTexture(*texture));
    }
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted));
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal mipmap generation reports 1D textures and 1D arrays as unsupported",
          "[size:large][rhi][metal][mipmap][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command);
    for (const std::uint32_t layers : {1U, 4U})
    {
        CAPTURE(layers);
        auto texture = device.createTexture({
            .dimension = rachis::TextureDimension::texture1D,
            .width = 64,
            .height = 1,
            .depthOrArrayLayers = layers,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
            .label = "Metal single-mip 1D",
        });
        REQUIRE(texture);
        auto generated = device.generateMipmaps(*command,
                                                {
                                                    .texture = *texture,
                                                    .baseLayout = rachis::TextureLayout::shaderResource,
                                                    .finalLayout = rachis::TextureLayout::shaderResource,
                                                });
        REQUIRE_FALSE(generated);
        CHECK(generated.error().code == rachis::ErrorCode::unsupported);
        REQUIRE(device.destroyTexture(*texture));
    }
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted));
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal mipmap generation filters complete chains and odd extents", "[size:large][rhi][metal][mipmap]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    const auto flags = GENERATE(rachis::CommandPoolFlag::none, rachis::CommandPoolFlag::unretainedResources);
    auto sampledOnly = device.createTexture({
        .width = copyWidth,
        .height = copyHeight,
        .mipLevels = mipLevels,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::copyDestination,
        .label = "Metal sampled-only mips",
    });
    auto texture = device.createTexture({
        .width = copyWidth,
        .height = copyHeight,
        .mipLevels = mipLevels,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
        .label = "Metal mipmap source",
    });
    REQUIRE(sampledOnly);
    REQUIRE(texture);

    const rachis::BufferTextureCopyRegion seed{.extent = {.width = copyWidth, .height = copyHeight}};
    auto seedLayout = device.queryTextureTransferLayout(*texture, seed);
    REQUIRE(seedLayout);
    auto upload = device.createBuffer({
        .size = seedLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    const rachis::BufferTextureCopyRegion mip1{
        .textureSubresource = {.mipLevel = 1},
        .extent = {.width = 2, .height = 2},
    };
    auto mip1Layout = device.queryTextureTransferLayout(*texture, mip1);
    REQUIRE(mip1Layout);
    auto mip1Readback = device.createBuffer({
        .size = mip1Layout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    const rachis::BufferTextureCopyRegion mip2{
        .textureSubresource = {.mipLevel = 2},
        .extent = {.width = 1, .height = 1},
    };
    auto mipLayout = device.queryTextureTransferLayout(*texture, mip2);
    REQUIRE(mipLayout);
    auto readback = device.createBuffer({
        .size = mipLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(upload);
    REQUIRE(mip1Readback);
    REQUIRE(readback);

    std::vector<std::byte> pixels(static_cast<std::size_t>(seedLayout->requiredBufferSize));
    for (std::uint32_t y = 0; y < copyHeight; ++y)
    {
        for (std::uint32_t x = 0; x < copyWidth; ++x)
        {
            const auto texel = baseTexel(x, y);
            for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
            {
                pixels[static_cast<std::size_t>(y) * seedLayout->bytesPerRow +
                       static_cast<std::size_t>(x) * bytesPerPixel + channel] = static_cast<std::byte>(texel[channel]);
            }
        }
    }
    REQUIRE(device.writeBuffer(*upload, 0, pixels).has_value());

    auto copyCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(copyCommand);
    CHECK_FALSE(device
                    .generateMipmaps(*copyCommand,
                                     {
                                         .texture = *texture,
                                         .baseLayout = rachis::TextureLayout::copyDestination,
                                     })
                    .has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics, flags);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *texture,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    REQUIRE(device.copyBufferToTexture(*command, *upload, *texture, std::array{seed}).has_value());
    CHECK_FALSE(device
                    .generateMipmaps(*command,
                                     {
                                         .texture = *sampledOnly,
                                         .baseLayout = rachis::TextureLayout::shaderResource,
                                     })
                    .has_value());
    auto generated = device.generateMipmaps(*command,
                                            {
                                                .texture = *texture,
                                                .baseLayout = rachis::TextureLayout::copyDestination,
                                                .finalLayout = rachis::TextureLayout::copySource,
                                            });
    if (!generated)
    {
        UNSCOPED_INFO(generated.error().message);
    }
    REQUIRE(generated.has_value());
    REQUIRE(device.copyTextureToBuffer(*command, *texture, *mip1Readback, std::array{mip1}).has_value());
    REQUIRE(device.copyTextureToBuffer(*command, *texture, *readback, std::array{mip2}).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyTexture(*sampledOnly).has_value());
    auto mip1Mapped = device.mapBuffer(*mip1Readback);
    REQUIRE(mip1Mapped);
    const auto* mip1Corner = mip1Mapped->data + static_cast<std::size_t>(mip1Layout->bytesPerRow) + 4;
    CHECK(nearByte(mip1Mapped->data[0], 20, 2));
    CHECK(nearByte(mip1Mapped->data[1], 20, 2));
    CHECK(nearByte(mip1Corner[0], 100, 2));
    CHECK(nearByte(mip1Corner[1], 100, 2));
    REQUIRE(device.unmapBuffer(*mip1Readback).has_value());
    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped);
    CHECK(nearByte(mapped->data[0], 60, 2));
    CHECK(nearByte(mapped->data[1], 60, 2));
    CHECK(nearByte(mapped->data[2], 60, 2));
    CHECK(mapped->data[3] == std::byte{255});
    REQUIRE(device.unmapBuffer(*readback).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal mipmap generation filters an odd 3x3 extent to a 1x1 mip", "[size:large][rhi][metal][mipmap][odd]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    constexpr std::uint32_t oddSize = 3;
    auto texture = device.createTexture({
        .width = oddSize,
        .height = oddSize,
        .mipLevels = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
        .label = "Metal odd-extent mips",
    });
    REQUIRE(texture);

    const rachis::BufferTextureCopyRegion seed{.extent = {.width = oddSize, .height = oddSize}};
    auto seedLayout = device.queryTextureTransferLayout(*texture, seed);
    REQUIRE(seedLayout);
    auto upload = device.createBuffer({
        .size = seedLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    const rachis::BufferTextureCopyRegion mip1{
        .textureSubresource = {.mipLevel = 1},
        .extent = {.width = 1, .height = 1},
    };
    auto mipLayout = device.queryTextureTransferLayout(*texture, mip1);
    REQUIRE(mipLayout);
    auto readback = device.createBuffer({
        .size = mipLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(upload);
    REQUIRE(readback);

    std::vector<std::byte> pixels(static_cast<std::size_t>(seedLayout->requiredBufferSize));
    for (std::uint32_t y = 0; y < oddSize; ++y)
    {
        for (std::uint32_t x = 0; x < oddSize; ++x)
        {
            const auto texel = baseTexel(x, y);
            for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
            {
                pixels[static_cast<std::size_t>(y) * seedLayout->bytesPerRow +
                       static_cast<std::size_t>(x) * bytesPerPixel + channel] = static_cast<std::byte>(texel[channel]);
            }
        }
    }
    REQUIRE(device.writeBuffer(*upload, 0, pixels).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *texture,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    REQUIRE(device.copyBufferToTexture(*command, *upload, *texture, std::array{seed}).has_value());
    REQUIRE(device
                .generateMipmaps(*command,
                                 {
                                     .texture = *texture,
                                     .baseLayout = rachis::TextureLayout::copyDestination,
                                     .finalLayout = rachis::TextureLayout::copySource,
                                 })
                .has_value());
    REQUIRE(device.copyTextureToBuffer(*command, *texture, *readback, std::array{mip1}).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped);
    CHECK(nearByte(mapped->data[0], 40, 8));
    CHECK(nearByte(mapped->data[1], 40, 8));
    CHECK(nearByte(mapped->data[2], 40, 8));
    CHECK(mapped->data[3] == std::byte{255});
    REQUIRE(device.unmapBuffer(*readback).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal mipmap generation filters every axis of a 3D texture", "[size:large][rhi][metal][mipmap]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    constexpr std::uint32_t volumeSize = 2;
    auto texture = device.createTexture({
        .dimension = rachis::TextureDimension::texture3D,
        .width = volumeSize,
        .height = volumeSize,
        .depthOrArrayLayers = volumeSize,
        .mipLevels = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
        .label = "Metal 3D mips",
    });
    REQUIRE(texture);

    const rachis::BufferTextureCopyRegion seed{
        .extent = {.width = volumeSize, .height = volumeSize, .depth = volumeSize},
    };
    auto seedLayout = device.queryTextureTransferLayout(*texture, seed);
    REQUIRE(seedLayout);
    auto upload = device.createBuffer({
        .size = seedLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    const rachis::BufferTextureCopyRegion mip1{
        .textureSubresource = {.mipLevel = 1},
        .extent = {.width = 1, .height = 1, .depth = 1},
    };
    auto mipLayout = device.queryTextureTransferLayout(*texture, mip1);
    REQUIRE(mipLayout);
    auto readback = device.createBuffer({
        .size = mipLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(upload);
    REQUIRE(readback);

    std::vector<std::byte> voxels(static_cast<std::size_t>(seedLayout->requiredBufferSize));
    for (std::uint32_t z = 0; z < volumeSize; ++z)
    {
        for (std::uint32_t y = 0; y < volumeSize; ++y)
        {
            for (std::uint32_t x = 0; x < volumeSize; ++x)
            {
                const auto offset = static_cast<std::size_t>(z) * seedLayout->bytesPerRow * seedLayout->rowsPerImage +
                                    static_cast<std::size_t>(y) * seedLayout->bytesPerRow +
                                    static_cast<std::size_t>(x) * bytesPerPixel;
                voxels[offset] = static_cast<std::byte>(x * 200U);
                voxels[offset + 1] = static_cast<std::byte>(y * 200U);
                voxels[offset + 2] = static_cast<std::byte>(z * 200U);
                voxels[offset + 3] = std::byte{255};
            }
        }
    }
    REQUIRE(device.writeBuffer(*upload, 0, voxels).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *texture,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    REQUIRE(device.copyBufferToTexture(*command, *upload, *texture, std::array{seed}).has_value());
    REQUIRE(device
                .generateMipmaps(*command,
                                 {
                                     .texture = *texture,
                                     .baseLayout = rachis::TextureLayout::copyDestination,
                                     .finalLayout = rachis::TextureLayout::copySource,
                                 })
                .has_value());
    REQUIRE(device.copyTextureToBuffer(*command, *texture, *readback, std::array{mip1}).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped);
    CHECK(nearByte(mapped->data[0], 100, 4));
    CHECK(nearByte(mapped->data[1], 100, 4));
    CHECK(nearByte(mapped->data[2], 100, 4));
    CHECK(mapped->data[3] == std::byte{255});
    REQUIRE(device.unmapBuffer(*readback).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal mipmap generation filters sRGB textures in linear light",
          "[size:large][rhi][metal][mipmap][srgb][regression]")
{
    const auto testCase =
        GENERATE(SrgbMipmapCase{.format = rachis::Format::rgba8UnormSrgb, .linearView = rachis::Format::rgba8Unorm},
                 SrgbMipmapCase{
                     .format = rachis::Format::rgba8UnormSrgb,
                     .linearView = rachis::Format::rgba8Unorm,
                     .placed = true,
                 });
    CAPTURE(static_cast<int>(testCase.format), testCase.placed);

    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);

    const std::array viewFormats{testCase.linearView};
    const rachis::TextureDesc desc{
        .width = copyWidth,
        .height = copyHeight,
        .mipLevels = 2,
        .format = testCase.format,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
        .viewFormats = viewFormats,
        .label = "Metal sRGB mipmap texture",
    };
    rachis::ResourceHeapHandle heap{};
    if (testCase.placed)
    {
        const auto requirements = device.queryTextureMemoryRequirements(desc);
        REQUIRE(requirements);
        auto createdHeap = device.createResourceHeap({
            .size = requirements->size * 2,
            .alignment = requirements->alignment,
            .memoryType = rachis::MemoryType::deviceLocal,
            .resourceClass = rachis::ResourceHeapClass::nonAttachmentTexture,
            .label = "Metal sRGB mipmap heap",
        });
        REQUIRE(createdHeap);
        heap = *createdHeap;
    }
    auto texture =
        testCase.placed ? device.createPlacedTexture({.heap = heap, .offset = 0}, desc) : device.createTexture(desc);
    REQUIRE(texture);

    const rachis::BufferTextureCopyRegion seed{.extent = {.width = copyWidth, .height = copyHeight}};
    auto seedLayout = device.queryTextureTransferLayout(*texture, seed);
    REQUIRE(seedLayout);
    const rachis::BufferTextureCopyRegion mip1{
        .textureSubresource = {.mipLevel = 1},
        .extent = {.width = 2, .height = 2},
    };
    auto mipLayout = device.queryTextureTransferLayout(*texture, mip1);
    REQUIRE(mipLayout);
    auto upload = device.createBuffer({
        .size = seedLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto readback = device.createBuffer({
        .size = mipLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(upload);
    REQUIRE(readback);

    std::vector<std::byte> pixels(static_cast<std::size_t>(seedLayout->requiredBufferSize));
    for (std::uint32_t y = 0; y < copyHeight; ++y)
    {
        for (std::uint32_t x = 0; x < copyWidth; ++x)
        {
            const auto value = ((x + y) & 1U) == 0 ? std::byte{0} : std::byte{255};
            const auto offset =
                static_cast<std::size_t>(y) * seedLayout->bytesPerRow + static_cast<std::size_t>(x) * bytesPerPixel;
            pixels[offset] = value;
            pixels[offset + 1] = value;
            pixels[offset + 2] = value;
            pixels[offset + 3] = std::byte{255};
        }
    }
    REQUIRE(device.writeBuffer(*upload, 0, pixels).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *texture,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    REQUIRE(device.copyBufferToTexture(*command, *upload, *texture, std::array{seed}).has_value());
    auto generated = device.generateMipmaps(*command,
                                            {
                                                .texture = *texture,
                                                .baseLayout = rachis::TextureLayout::copyDestination,
                                                .finalLayout = rachis::TextureLayout::copySource,
                                            });
    if (!generated)
    {
        UNSCOPED_INFO(generated.error().message);
    }
    REQUIRE(generated.has_value());
    REQUIRE(device.copyTextureToBuffer(*command, *texture, *readback, std::array{mip1}).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped);
    constexpr std::uint8_t encodedHalfLinear = 188;
    for (std::uint32_t y = 0; y < 2; ++y)
    {
        for (std::uint32_t x = 0; x < 2; ++x)
        {
            const auto* texel =
                mapped->data + static_cast<std::size_t>(y) * mipLayout->bytesPerRow + static_cast<std::size_t>(x) * 4;
            for (std::uint32_t channel = 0; channel < 3; ++channel)
            {
                CAPTURE(x, y, channel, std::to_integer<std::uint32_t>(texel[channel]));
                CHECK(nearByte(texel[channel], encodedHalfLinear, 3));
            }
            CHECK(texel[3] == std::byte{255});
        }
    }
    REQUIRE(device.unmapBuffer(*readback).has_value());
    REQUIRE(device.destroyTexture(*texture).has_value());
    if (testCase.placed)
    {
        REQUIRE(device.destroyResourceHeap(heap).has_value());
    }
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal mipmap generation keeps its helper resources and pipeline out of application state",
          "[size:large][rhi][metal][mipmap][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    auto texture = device.createTexture({
        .width = copyWidth,
        .height = copyHeight,
        .mipLevels = mipLevels,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::storage,
        .label = "Metal mipmap helper isolation",
    });
    REQUIRE(texture);
    const auto before = device.captureResourceMemorySnapshot();
    REQUIRE(before);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *texture,
                                    .stagesAfter = rachis::PipelineStage::computeShader,
                                    .accessAfter = rachis::Access::shaderResource,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::shaderResource,
                                })
                .has_value());
    REQUIRE(device
                .generateMipmaps(*command,
                                 {
                                     .texture = *texture,
                                     .baseLayout = rachis::TextureLayout::shaderResource,
                                     .finalLayout = rachis::TextureLayout::shaderResource,
                                 })
                .has_value());
    const std::array<std::byte, 4> pushConstants{};
    auto pushed = device.setComputePushConstants(*command, 2, pushConstants);
    REQUIRE_FALSE(pushed);
    CHECK(pushed.error().code == rachis::ErrorCode::invalidState);
    auto dispatched = device.dispatch(*command, 1, 1, 1);
    REQUIRE_FALSE(dispatched);
    CHECK(dispatched.error().code == rachis::ErrorCode::invalidState);
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());

    const auto after = device.captureResourceMemorySnapshot();
    REQUIRE(after);
    CHECK(after->entries.size() == before->entries.size());
    CHECK(after->liveAllocationCount == before->liveAllocationCount);
    CHECK(after->liveAllocationBytes == before->liveAllocationBytes);
    REQUIRE(device.destroyTexture(*texture).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal mipmap generation waits for the render pass that wrote its base level",
          "[size:large][rhi][metal][mipmap][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto seed = createRenderedMipPipeline(device, "MipmapAfterRendering.slow.ps.metallib", "slowPixel", {});

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
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal mipmap generation finishes before a later render pass blends into its last mip",
          "[size:large][rhi][metal][mipmap][regression]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto seed = createRenderedMipPipeline(device, "MipmapAfterRendering.slow.ps.metallib", "slowPixel", {});
    constexpr rachis::BlendComponent additive{
        .sourceFactor = rachis::BlendFactor::one,
        .destinationFactor = rachis::BlendFactor::one,
    };
    const auto blend = createRenderedMipPipeline(device,
                                                 "MipmapAfterRendering.blend.ps.metallib",
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
    CHECK(device.diagnosticCounts().errors == 0);
}
#endif
