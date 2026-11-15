#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

    constexpr std::uint32_t targetWidth = 64;
    constexpr std::uint32_t targetHeight = 64;
    constexpr std::uint32_t valueCount = 64;
    constexpr std::uint64_t outputSize = valueCount * sizeof(std::uint32_t);
    constexpr std::array tint{0.125F, 0.5F, 0.75F, 1.0F};
    constexpr std::array<std::uint8_t, 4> tintBytes{32, 128, 191, 255};
    constexpr std::array<std::uint8_t, 4> tealBytes{0, 200, 200, 255};
    constexpr std::array<std::uint8_t, 4> redBytes{255, 0, 0, 255};
    constexpr std::array<std::uint8_t, 4> blueBytes{0, 0, 255, 255};

    [[nodiscard]] rachis::DeviceDesc metalDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
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

    [[nodiscard]] std::filesystem::path pnmsPath(std::string_view metallibName)
    {
        return std::filesystem::path(RACHIS_METALLIB_DIR) /
               std::filesystem::path(metallibName).replace_extension(".pnms");
    }

    [[nodiscard]] bool withinTolerance(std::byte actual, std::uint8_t expected, std::uint8_t tolerance) noexcept
    {
        const auto value = std::to_integer<std::uint8_t>(actual);
        const auto lower = expected <= tolerance ? 0 : static_cast<int>(expected) - tolerance;
        const auto upper = expected >= 255 - tolerance ? 255 : static_cast<int>(expected) + tolerance;
        return std::cmp_greater_equal(value, lower) && std::cmp_less_equal(value, upper);
    }

    [[nodiscard]] const std::byte* pixelAt(const std::byte* data,
                                           std::uint32_t pitch,
                                           std::uint32_t x,
                                           std::uint32_t y) noexcept
    {
        return data + static_cast<std::uint64_t>(y) * pitch + static_cast<std::uint64_t>(x) * 4;
    }

    [[nodiscard]] bool pixelMatches(const std::byte* pixel,
                                    std::span<const std::uint8_t, 4> expected,
                                    std::uint8_t tolerance) noexcept
    {
        for (std::uint32_t channel = 0; channel < 4; ++channel)
        {
            if (!withinTolerance(pixel[channel], expected[channel], tolerance))
            {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool isClear(const std::byte* pixel) noexcept
    {
        return std::to_integer<std::uint8_t>(pixel[0]) < 4 && std::to_integer<std::uint8_t>(pixel[1]) < 4 &&
               std::to_integer<std::uint8_t>(pixel[2]) < 4;
    }

    void writePpm(const std::filesystem::path& path, const std::byte* data, std::uint32_t pitch)
    {
        std::error_code error;
        std::filesystem::create_directories(path.parent_path(), error);
        std::ofstream stream(path, std::ios::binary);
        stream << "P6\n" << targetWidth << ' ' << targetHeight << "\n255\n";
        for (std::uint32_t y = 0; y < targetHeight; ++y)
        {
            for (std::uint32_t x = 0; x < targetWidth; ++x)
            {
                const auto* pixel = pixelAt(data, pitch, x, y);
                stream.put(static_cast<char>(pixel[0]));
                stream.put(static_cast<char>(pixel[1]));
                stream.put(static_cast<char>(pixel[2]));
            }
        }
    }

    void writeReport(const std::filesystem::path& path, std::string_view body)
    {
        std::error_code error;
        std::filesystem::create_directories(path.parent_path(), error);
        std::ofstream stream(path, std::ios::binary);
        stream << body;
    }

    [[nodiscard]] std::filesystem::path evidenceDir()
    {
#ifdef RACHIS_PARITY_EVIDENCE_DIR
        return {RACHIS_PARITY_EVIDENCE_DIR};
#else
        return std::filesystem::temp_directory_path() / "rachis-metal-parity";
#endif
    }

    [[nodiscard]] std::string adapterLabel(const rachis::AdapterInfo& adapter)
    {
        std::string label = adapter.type == rachis::AdapterType::software ? "software" : "hardware";
        label += " backend=metal vendor=0x";
        char hex[16]{};
        std::snprintf(hex, sizeof(hex), "%04x/%04x", adapter.vendorId, adapter.deviceId);
        label += hex;
        label += " name=";
        label += adapter.name;
        return label;
    }

    struct TriangleShaders final
    {
        std::vector<std::byte> vertex;
        std::vector<std::byte> farVertex;
        std::vector<std::byte> pixel;
    };

    [[nodiscard]] TriangleShaders metalTriangleShaders()
    {
        return {
            .vertex = loadBytes(pnmsPath("MetalHeadlessParity.direct.vs.metallib")),
            .farVertex = loadBytes(pnmsPath("MetalHeadlessParity.depth.vs.metallib")),
            .pixel = loadBytes(pnmsPath("MetalHeadlessParity.ps.metallib")),
        };
    }

    struct ColorReadback final
    {
        std::vector<std::byte> pixels;
        std::uint32_t pitch = 0;
    };

    struct TriangleShaderModules final
    {
        rachis::ShaderModuleHandle vertex{};
        rachis::ShaderModuleHandle pixel{};
    };

    [[nodiscard]] TriangleShaderModules createTriangleShaderModules(rachis::Device& device,
                                                                    const TriangleShaders& shaders,
                                                                    bool useFarVertex)
    {
        REQUIRE_FALSE(shaders.vertex.empty());
        REQUIRE_FALSE(shaders.pixel.empty());
        if (useFarVertex)
        {
            REQUIRE_FALSE(shaders.farVertex.empty());
        }

        auto vertexShader = device.createShaderModule({
            .stage = rachis::ShaderStage::vertex,
            .bytecode = useFarVertex ? std::span<const std::byte>(shaders.farVertex)
                                     : std::span<const std::byte>(shaders.vertex),
            .entryPoint = useFarVertex ? "depthBiasVertex" : "directVertex",
        });
        if (!vertexShader)
        {
            UNSCOPED_INFO(vertexShader.error().message);
        }
        auto pixelShader = device.createShaderModule({
            .stage = rachis::ShaderStage::pixel,
            .bytecode = shaders.pixel,
            .entryPoint = "pixelMain",
            .shaderInterface = rachis::test::graphicsPixelShaderInterface(),
        });
        if (!pixelShader)
        {
            UNSCOPED_INFO(pixelShader.error().message);
        }
        REQUIRE(vertexShader);
        REQUIRE(pixelShader);
        return {.vertex = *vertexShader, .pixel = *pixelShader};
    }

    [[nodiscard]] rachis::PipelineHandle createTintedTrianglePipeline(rachis::Device& device,
                                                                      const TriangleShaderModules& modules,
                                                                      rachis::CullMode cullMode,
                                                                      rachis::BlendState blend,
                                                                      std::uint32_t sampleCount)
    {
        const std::array pushConstantRanges{
            rachis::PushConstantRange{rachis::ShaderStage::pixel, 0, 4 * sizeof(float)},
        };
        auto layout = device.createPipelineLayout({.pushConstantRanges = pushConstantRanges});
        REQUIRE(layout);
        const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm, .blend = blend}};
        auto pipeline = device.createGraphicsPipeline({
            .layout = *layout,
            .vertexShader = modules.vertex,
            .pixelShader = modules.pixel,
            .cullMode = cullMode,
            .colorTargets = colorTargets,
            .depthStencil =
                {
                    .format = rachis::Format::d32Float,
                    .depthTestEnabled = true,
                    .depthWriteEnabled = true,
                    .depthCompare = rachis::CompareOperation::lessEqual,
                },
            .sampleCount = sampleCount,
        });
        if (!pipeline)
        {
            UNSCOPED_INFO(pipeline.error().message);
        }
        REQUIRE(pipeline);
        return *pipeline;
    }

    struct TriangleTargets final
    {
        rachis::TextureHandle color{};
        rachis::TextureHandle depth{};
        rachis::TextureViewHandle colorView{};
        rachis::TextureViewHandle depthView{};
        rachis::TextureHandle resolved{};
        rachis::TextureHandle readSource{};
    };

    [[nodiscard]] TriangleTargets createTriangleTargets(rachis::Device& device, std::uint32_t sampleCount)
    {
        auto color = device.createTexture({
            .width = targetWidth,
            .height = targetHeight,
            .sampleCount = sampleCount,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
            .label = "Metal parity color",
        });
        auto depth = device.createTexture({
            .width = targetWidth,
            .height = targetHeight,
            .sampleCount = sampleCount,
            .format = rachis::Format::d32Float,
            .usage = rachis::TextureUsage::depthStencil,
            .label = "Metal parity depth",
        });
        REQUIRE(color);
        REQUIRE(depth);
        auto colorView = device.createTextureView({.texture = *color, .usage = rachis::TextureViewUsage::renderTarget});
        auto depthView = device.createTextureView({
            .texture = *depth,
            .usage = rachis::TextureViewUsage::depthStencil,
            .aspect = rachis::TextureAspect::depth,
        });
        REQUIRE(colorView);
        REQUIRE(depthView);

        rachis::TextureHandle readSource = *color;
        rachis::TextureHandle resolved{};
        if (sampleCount > 1)
        {
            auto resolveTarget = device.createTexture({
                .width = targetWidth,
                .height = targetHeight,
                .format = rachis::Format::rgba8Unorm,
                .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget |
                         rachis::TextureUsage::copyDestination,
                .label = "Metal parity resolved color",
            });
            REQUIRE(resolveTarget);
            resolved = *resolveTarget;
            readSource = resolved;
        }
        return {
            .color = *color,
            .depth = *depth,
            .colorView = *colorView,
            .depthView = *depthView,
            .resolved = resolved,
            .readSource = readSource,
        };
    }

    void transitionTargetsForRendering(rachis::Device& device,
                                       rachis::CommandBuffer& command,
                                       const TriangleTargets& targets)
    {
        REQUIRE(device
                    .textureBarrier(command,
                                    {
                                        .texture = targets.color,
                                        .stagesAfter = rachis::PipelineStage::renderTarget,
                                        .accessAfter = rachis::Access::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::renderTarget,
                                    })
                    .has_value());
        REQUIRE(device
                    .textureBarrier(command,
                                    {
                                        .texture = targets.depth,
                                        .stagesAfter = rachis::PipelineStage::depthStencil,
                                        .accessAfter = rachis::Access::depthStencilWrite,
                                        .layoutAfter = rachis::TextureLayout::depthStencilWrite,
                                        .subresources = {.aspect = rachis::TextureAspect::depth},
                                    })
                    .has_value());
    }

    void recordTintedTriangleDraw(rachis::Device& device,
                                  rachis::CommandBuffer& command,
                                  const TriangleTargets& targets,
                                  rachis::PipelineHandle pipeline,
                                  const rachis::ScissorRect& scissor,
                                  const std::array<float, 4>& tintColor)
    {
        const std::array attachments{
            rachis::ColorAttachment{.view = targets.colorView, .loadOperation = rachis::LoadOperation::clear},
        };
        REQUIRE(device
                    .beginRendering(command,
                                    {
                                        .colorAttachments = attachments,
                                        .depthStencilAttachment =
                                            rachis::DepthStencilAttachment{
                                                .view = targets.depthView,
                                                .depthLoadOperation = rachis::LoadOperation::clear,
                                                .depthClearValue = 1.0F,
                                            },
                                    })
                    .has_value());
        REQUIRE(device.bindGraphicsPipeline(command, pipeline).has_value());
        REQUIRE(device
                    .setViewport(command,
                                 {
                                     .width = static_cast<float>(targetWidth),
                                     .height = static_cast<float>(targetHeight),
                                 })
                    .has_value());
        REQUIRE(device.setScissor(command, scissor).has_value());
        REQUIRE(
            device.setGraphicsPushConstants(command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(tintColor)))
                .has_value());
        REQUIRE(device.draw(command, 3).has_value());
        REQUIRE(device.endRendering(command).has_value());
    }

    void resolveColorForCopy(rachis::Device& device, rachis::CommandBuffer& command, const TriangleTargets& targets)
    {
        REQUIRE(device
                    .textureBarrier(command,
                                    {
                                        .texture = targets.color,
                                        .stagesBefore = rachis::PipelineStage::renderTarget,
                                        .stagesAfter = rachis::PipelineStage::resolve,
                                        .accessBefore = rachis::Access::renderTarget,
                                        .accessAfter = rachis::Access::resolveSource,
                                        .layoutBefore = rachis::TextureLayout::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::resolveSource,
                                    })
                    .has_value());
        REQUIRE(device
                    .textureBarrier(command,
                                    {
                                        .texture = targets.resolved,
                                        .stagesAfter = rachis::PipelineStage::resolve,
                                        .accessAfter = rachis::Access::resolveDestination,
                                        .layoutAfter = rachis::TextureLayout::resolveDestination,
                                    })
                    .has_value());
        REQUIRE(device.resolveTexture(command, {.source = targets.color, .destination = targets.resolved}).has_value());
        REQUIRE(device
                    .textureBarrier(command,
                                    {
                                        .texture = targets.resolved,
                                        .stagesBefore = rachis::PipelineStage::resolve,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::resolveDestination,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::resolveDestination,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                    })
                    .has_value());
    }

    void transitionColorForCopy(rachis::Device& device, rachis::CommandBuffer& command, const TriangleTargets& targets)
    {
        REQUIRE(device
                    .textureBarrier(command,
                                    {
                                        .texture = targets.color,
                                        .stagesBefore = rachis::PipelineStage::renderTarget,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::renderTarget,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                    })
                    .has_value());
    }

    [[nodiscard]] ColorReadback renderTintedTriangle(rachis::Device& device,
                                                     const TriangleShaders& shaders,
                                                     const rachis::ScissorRect& scissor,
                                                     const std::array<float, 4>& tintColor,
                                                     bool useFarVertex,
                                                     rachis::CullMode cullMode,
                                                     rachis::BlendState blend = {},
                                                     std::uint32_t sampleCount = 1)
    {
        const auto modules = createTriangleShaderModules(device, shaders, useFarVertex);
        const auto pipeline = createTintedTrianglePipeline(device, modules, cullMode, blend, sampleCount);
        const auto targets = createTriangleTargets(device, sampleCount);

        const rachis::BufferTextureCopyRegion region{
            .extent = {.width = targetWidth, .height = targetHeight, .depth = 1},
        };
        auto transfer = device.queryTextureTransferLayout(targets.readSource, region);
        REQUIRE(transfer);
        auto readback = device.createBuffer({
            .size = transfer->requiredBufferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
            .label = "Metal parity readback",
        });
        REQUIRE(readback);

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command);
        transitionTargetsForRendering(device, *command, targets);
        recordTintedTriangleDraw(device, *command, targets, pipeline, scissor, tintColor);
        if (sampleCount > 1)
        {
            resolveColorForCopy(device, *command, targets);
        }
        else
        {
            transitionColorForCopy(device, *command, targets);
        }
        const rachis::BufferTextureCopyRegion copyRegion{
            .bytesPerRow = transfer->bytesPerRow,
            .rowsPerImage = transfer->rowsPerImage,
            .extent = {.width = targetWidth, .height = targetHeight, .depth = 1},
        };
        REQUIRE(
            device.copyTextureToBuffer(*command, targets.readSource, *readback, std::array{copyRegion}).has_value());
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted);
        REQUIRE(device.wait(*submitted).has_value());
        auto mapped = device.mapBuffer(*readback);
        REQUIRE(mapped);
        ColorReadback result{
            .pixels = std::vector<std::byte>(mapped->data,
                                             mapped->data + static_cast<std::size_t>(transfer->requiredBufferSize)),
            .pitch = transfer->bytesPerRow,
        };
        REQUIRE(device.unmapBuffer(*readback).has_value());
        return result;
    }

    void requireOffscreenTriangle(const ColorReadback& image, std::uint8_t tolerance)
    {
        const auto* center = pixelAt(image.pixels.data(), image.pitch, targetWidth / 2, targetHeight / 2);
        const auto* upright = pixelAt(image.pixels.data(), image.pitch, 14, 52);
        const auto* flipped = pixelAt(image.pixels.data(), image.pitch, 14, 12);
        CAPTURE(std::to_integer<unsigned int>(center[0]),
                std::to_integer<unsigned int>(center[1]),
                std::to_integer<unsigned int>(center[2]),
                std::to_integer<unsigned int>(center[3]));
        REQUIRE(pixelMatches(center, tintBytes, tolerance));
        REQUIRE(pixelMatches(upright, tintBytes, tolerance));
        REQUIRE(isClear(flipped));
    }

} // namespace

TEST_CASE("Metal multisample resolves order rendered samples before readback",
          "[size:medium][rhi][metal][parity][hardware]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    const auto shaders = metalTriangleShaders();

    for (std::uint32_t iteration = 0; iteration < 16; ++iteration)
    {
        CAPTURE(iteration);
        const auto image = renderTintedTriangle(device,
                                                shaders,
                                                {.width = targetWidth, .height = targetHeight},
                                                tint,
                                                false,
                                                rachis::CullMode::none,
                                                {},
                                                4);
        requireOffscreenTriangle(image, 2);
    }
}

TEST_CASE(
    "Metal hardware offscreenTriangle, depth, blend, clip, sRGB, texture, MSAA, and compute match deterministic values",
    "[size:large][rhi][metal][parity][hardware]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created);
    auto device = std::move(*created);
    REQUIRE(device.adapterInfo().has_value());
    const auto adapter = *device.adapterInfo();
    CAPTURE(adapterLabel(adapter));
    REQUIRE(adapter.backend == rachis::Backend::metal);
    REQUIRE(adapter.type != rachis::AdapterType::software);
    REQUIRE(adapter.vendorId == 0x106b);

    const auto shaders = metalTriangleShaders();
    const auto pixels = renderTintedTriangle(
        device, shaders, {.width = targetWidth, .height = targetHeight}, tint, false, rachis::CullMode::none);
    requireOffscreenTriangle(pixels, 2);
    writePpm(evidenceDir() / "metal-hardware-offscreenTriangle.ppm", pixels.pixels.data(), pixels.pitch);

    const auto frontKept = renderTintedTriangle(
        device, shaders, {.width = targetWidth, .height = targetHeight}, tint, false, rachis::CullMode::front);
    const auto backCulled = renderTintedTriangle(
        device, shaders, {.width = targetWidth, .height = targetHeight}, tint, false, rachis::CullMode::back);
    CHECK(pixelMatches(
        pixelAt(frontKept.pixels.data(), frontKept.pitch, targetWidth / 2, targetHeight / 2), tintBytes, 2));
    CHECK(isClear(pixelAt(backCulled.pixels.data(), backCulled.pitch, targetWidth / 2, targetHeight / 2)));
    writePpm(evidenceDir() / "metal-hardware-frontFace.ppm", frontKept.pixels.data(), frontKept.pitch);

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    auto color = device.createTexture({
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
    });
    auto depth = device.createTexture({
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::d32Float,
        .usage = rachis::TextureUsage::depthStencil,
    });
    REQUIRE(color);
    REQUIRE(depth);
    auto colorView = device.createTextureView({.texture = *color, .usage = rachis::TextureViewUsage::renderTarget});
    auto depthView = device.createTextureView({
        .texture = *depth,
        .usage = rachis::TextureViewUsage::depthStencil,
        .aspect = rachis::TextureAspect::depth,
    });
    REQUIRE(colorView);
    REQUIRE(depthView);
    auto nearShader = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = shaders.vertex,
        .entryPoint = "directVertex",
    });
    auto farShader = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = shaders.farVertex,
        .entryPoint = "depthBiasVertex",
    });
    auto pixelShader = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = shaders.pixel,
        .entryPoint = "pixelMain",
        .shaderInterface = rachis::test::graphicsPixelShaderInterface(),
    });
    REQUIRE(nearShader);
    REQUIRE(farShader);
    REQUIRE(pixelShader);
    const std::array pushConstantRanges{
        rachis::PushConstantRange{rachis::ShaderStage::pixel, 0, 4 * sizeof(float)},
    };
    auto layout = device.createPipelineLayout({.pushConstantRanges = pushConstantRanges});
    REQUIRE(layout);
    const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
    const rachis::DepthStencilState depthState{
        .format = rachis::Format::d32Float,
        .depthTestEnabled = true,
        .depthWriteEnabled = true,
        .depthCompare = rachis::CompareOperation::lessEqual,
    };
    auto nearPipeline = device.createGraphicsPipeline({
        .layout = *layout,
        .vertexShader = *nearShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .depthStencil = depthState,
    });
    auto farPipeline = device.createGraphicsPipeline({
        .layout = *layout,
        .vertexShader = *farShader,
        .pixelShader = *pixelShader,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
        .depthStencil = depthState,
    });
    REQUIRE(nearPipeline);
    REQUIRE(farPipeline);
    const rachis::BufferTextureCopyRegion region{
        .extent = {.width = targetWidth, .height = targetHeight, .depth = 1},
    };
    auto transfer = device.queryTextureTransferLayout(*color, region);
    REQUIRE(transfer);
    auto readback = device.createBuffer({
        .size = transfer->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(readback);
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *color,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *depth,
                                    .stagesAfter = rachis::PipelineStage::depthStencil,
                                    .accessAfter = rachis::Access::depthStencilWrite,
                                    .layoutAfter = rachis::TextureLayout::depthStencilWrite,
                                    .subresources = {.aspect = rachis::TextureAspect::depth},
                                })
                .has_value());
    const std::array attachments{
        rachis::ColorAttachment{.view = *colorView, .loadOperation = rachis::LoadOperation::clear},
    };
    REQUIRE(device
                .beginRendering(*command,
                                {
                                    .colorAttachments = attachments,
                                    .depthStencilAttachment =
                                        rachis::DepthStencilAttachment{
                                            .view = *depthView,
                                            .depthLoadOperation = rachis::LoadOperation::clear,
                                            .depthClearValue = 1.0F,
                                        },
                                })
                .has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *farPipeline).has_value());
    REQUIRE(device
                .setViewport(*command,
                             {
                                 .width = static_cast<float>(targetWidth),
                                 .height = static_cast<float>(targetHeight),
                             })
                .has_value());
    REQUIRE(device.setScissor(*command, {.width = targetWidth, .height = targetHeight}).has_value());
    const std::array farTint{1.0F, 0.0F, 0.0F, 1.0F};
    const std::array nearTint{0.0F, 0.784F, 0.784F, 1.0F};
    REQUIRE(device.setGraphicsPushConstants(*command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(farTint)))
                .has_value());
    REQUIRE(device.draw(*command, 3).has_value());
    REQUIRE(device.bindGraphicsPipeline(*command, *nearPipeline).has_value());
    REQUIRE(device.setGraphicsPushConstants(*command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(nearTint)))
                .has_value());
    REQUIRE(device.draw(*command, 3).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *color,
                                    .stagesBefore = rachis::PipelineStage::renderTarget,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::renderTarget,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    const rachis::BufferTextureCopyRegion copyRegion{
        .bytesPerRow = transfer->bytesPerRow,
        .rowsPerImage = transfer->rowsPerImage,
        .extent = {.width = targetWidth, .height = targetHeight, .depth = 1},
    };
    REQUIRE(device.copyTextureToBuffer(*command, *color, *readback, std::array{copyRegion}).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped);
    const auto* depthCenter = pixelAt(mapped->data, transfer->bytesPerRow, targetWidth / 2, targetHeight / 2);
    CHECK(pixelMatches(depthCenter, tealBytes, 2));
    CHECK_FALSE(pixelMatches(depthCenter, redBytes, 2));
    writePpm(evidenceDir() / "metal-hardware-depth.ppm", mapped->data, transfer->bytesPerRow);
    REQUIRE(device.unmapBuffer(*readback).has_value());

    const auto clipped = renderTintedTriangle(
        device, shaders, {.x = 32, .width = 32, .height = targetHeight}, tint, false, rachis::CullMode::none);
    CHECK(isClear(pixelAt(clipped.pixels.data(), clipped.pitch, 14, 52)));
    CHECK(pixelMatches(pixelAt(clipped.pixels.data(), clipped.pitch, targetWidth / 2, targetHeight / 2), tintBytes, 2));
    writePpm(evidenceDir() / "metal-hardware-scissor.ppm", clipped.pixels.data(), clipped.pitch);

    const rachis::BlendState srcAlpha{
        .enabled = true,
        .color =
            {
                .sourceFactor = rachis::BlendFactor::sourceAlpha,
                .destinationFactor = rachis::BlendFactor::oneMinusSourceAlpha,
            },
        .alpha =
            {
                .sourceFactor = rachis::BlendFactor::one,
                .destinationFactor = rachis::BlendFactor::oneMinusSourceAlpha,
            },
    };
    const std::array halfRed{1.0F, 0.0F, 0.0F, 0.5F};
    const auto blended = renderTintedTriangle(device,
                                              shaders,
                                              {.width = targetWidth, .height = targetHeight},
                                              halfRed,
                                              false,
                                              rachis::CullMode::none,
                                              srcAlpha);
    const auto* blendCenter = pixelAt(blended.pixels.data(), blended.pitch, targetWidth / 2, targetHeight / 2);
    CHECK(withinTolerance(blendCenter[0], 128, 2));
    CHECK(withinTolerance(blendCenter[1], 0, 2));
    CHECK_FALSE(withinTolerance(blendCenter[0], 255, 2));
    writePpm(evidenceDir() / "metal-hardware-blend.ppm", blended.pixels.data(), blended.pitch);

    auto linear = device.createTexture({
        .width = 2,
        .height = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
    });
    auto srgb = device.createTexture({
        .width = 2,
        .height = 2,
        .format = rachis::Format::rgba8UnormSrgb,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
    });
    REQUIRE(linear);
    REQUIRE(srgb);
    auto linearView = device.createTextureView({.texture = *linear, .usage = rachis::TextureViewUsage::renderTarget});
    auto srgbView = device.createTextureView({.texture = *srgb, .usage = rachis::TextureViewUsage::renderTarget});
    REQUIRE(linearView);
    REQUIRE(srgbView);
    const rachis::BufferTextureCopyRegion small{.extent = {.width = 2, .height = 2}};
    auto linearLayout = device.queryTextureTransferLayout(*linear, small);
    auto srgbLayout = device.queryTextureTransferLayout(*srgb, small);
    REQUIRE(linearLayout);
    REQUIRE(srgbLayout);
    auto linearRead = device.createBuffer({
        .size = linearLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    auto srgbRead = device.createBuffer({
        .size = srgbLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(linearRead);
    REQUIRE(srgbRead);
    auto clearCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(clearCommand);
    REQUIRE(device
                .textureBarrier(*clearCommand,
                                {
                                    .texture = *linear,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    REQUIRE(device
                .textureBarrier(*clearCommand,
                                {
                                    .texture = *srgb,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array linearAttachments{
        rachis::ColorAttachment{
            .view = *linearView,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.5F, 0.5F, 0.5F, 1.0F},
        },
    };
    REQUIRE(device.beginRendering(*clearCommand, {.colorAttachments = linearAttachments}).has_value());
    REQUIRE(device.endRendering(*clearCommand).has_value());
    const std::array srgbAttachments{
        rachis::ColorAttachment{
            .view = *srgbView,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.5F, 0.5F, 0.5F, 1.0F},
        },
    };
    REQUIRE(device.beginRendering(*clearCommand, {.colorAttachments = srgbAttachments}).has_value());
    REQUIRE(device.endRendering(*clearCommand).has_value());
    REQUIRE(device
                .textureBarrier(*clearCommand,
                                {
                                    .texture = *linear,
                                    .stagesBefore = rachis::PipelineStage::renderTarget,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::renderTarget,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device
                .textureBarrier(*clearCommand,
                                {
                                    .texture = *srgb,
                                    .stagesBefore = rachis::PipelineStage::renderTarget,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::renderTarget,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device.copyTextureToBuffer(*clearCommand, *linear, *linearRead, std::array{small}).has_value());
    REQUIRE(device.copyTextureToBuffer(*clearCommand, *srgb, *srgbRead, std::array{small}).has_value());
    auto clearSubmitted = device.submit(std::move(*clearCommand));
    REQUIRE(clearSubmitted);
    REQUIRE(device.wait(*clearSubmitted).has_value());
    auto linearMapped = device.mapBuffer(*linearRead);
    auto srgbMapped = device.mapBuffer(*srgbRead);
    REQUIRE(linearMapped);
    REQUIRE(srgbMapped);
    const auto linearByte = std::to_integer<std::uint8_t>(linearMapped->data[0]);
    const auto srgbByte = std::to_integer<std::uint8_t>(srgbMapped->data[0]);
    CAPTURE(linearByte, srgbByte);
    CHECK(withinTolerance(linearMapped->data[0], 128, 2));
    CHECK(withinTolerance(srgbMapped->data[0], 188, 2));
    CHECK_FALSE(withinTolerance(srgbMapped->data[0], 128, 2));
    REQUIRE(device.unmapBuffer(*linearRead).has_value());
    REQUIRE(device.unmapBuffer(*srgbRead).has_value());

    const auto texturedVs = loadBytes(pnmsPath("MetalHeadlessParity.textured.vs.metallib"));
    const auto texturedPs = loadBytes(pnmsPath("MetalHeadlessParity.textured.ps.metallib"));
    REQUIRE_FALSE(texturedVs.empty());
    REQUIRE_FALSE(texturedPs.empty());
    static constexpr std::array texturedBindings{
        rachis::ShaderBindingRequirement{1, 0, rachis::ShaderResourceType::sampledTexture, 1},
        rachis::ShaderBindingRequirement{3, 1, rachis::ShaderResourceType::sampler, 1},
    };
    auto texturedVertex = device.createShaderModule({
        .stage = rachis::ShaderStage::vertex,
        .bytecode = texturedVs,
        .entryPoint = "texturedVertex",
    });
    auto texturedPixel = device.createShaderModule({
        .stage = rachis::ShaderStage::pixel,
        .bytecode = texturedPs,
        .entryPoint = "texturedPixel",
        .shaderInterface = rachis::test::shaderInterface(texturedBindings),
    });
    REQUIRE(texturedVertex);
    REQUIRE(texturedPixel);
    const std::array textureLayoutEntries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::sampledTexture, rachis::ShaderStage::pixel},
    };
    auto textureBindingLayout = device.createBindingLayout({.entries = textureLayoutEntries});
    REQUIRE(textureBindingLayout);
    auto emptyBindingLayout = device.createBindingLayout({});
    auto samplerBindingLayout = device.createBindingLayout({
        .entries = std::array{rachis::BindingLayoutEntry{1, rachis::BindingType::sampler, rachis::ShaderStage::pixel}},
    });
    REQUIRE(emptyBindingLayout);
    REQUIRE(samplerBindingLayout);
    auto texturePipelineLayout = device.createPipelineLayout({
        .bindingLayouts =
            std::array{*emptyBindingLayout, *textureBindingLayout, *emptyBindingLayout, *samplerBindingLayout},
    });
    REQUIRE(texturePipelineLayout);
    auto texturedPipeline = device.createGraphicsPipeline({
        .layout = *texturePipelineLayout,
        .vertexShader = *texturedVertex,
        .pixelShader = *texturedPixel,
        .cullMode = rachis::CullMode::none,
        .colorTargets = colorTargets,
    });
    if (!texturedPipeline)
    {
        UNSCOPED_INFO(texturedPipeline.error().message);
    }
    REQUIRE(texturedPipeline);
    auto albedo = device.createTexture({
        .width = 2,
        .height = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copyDestination | rachis::TextureUsage::sampled,
    });
    REQUIRE(albedo);
    const rachis::BufferTextureCopyRegion albedoRegion{.extent = {.width = 2, .height = 2}};
    auto albedoLayout = device.queryTextureTransferLayout(*albedo, albedoRegion);
    REQUIRE(albedoLayout);
    auto upload = device.createBuffer({
        .size = albedoLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    REQUIRE(upload);
    std::vector<std::byte> checker(static_cast<std::size_t>(albedoLayout->requiredBufferSize));
    const std::array<std::array<std::uint8_t, 4>, 4> texels{
        std::array<std::uint8_t, 4>{255, 0, 0, 255},
        std::array<std::uint8_t, 4>{0, 255, 0, 255},
        std::array<std::uint8_t, 4>{0, 0, 255, 255},
        std::array<std::uint8_t, 4>{255, 255, 255, 255},
    };
    for (std::uint32_t y = 0; y < 2; ++y)
    {
        for (std::uint32_t x = 0; x < 2; ++x)
        {
            const auto& texel = texels[y * 2 + x];
            for (std::uint32_t channel = 0; channel < 4; ++channel)
            {
                checker[static_cast<std::size_t>(y) * albedoLayout->bytesPerRow + static_cast<std::size_t>(x) * 4 +
                        channel] = static_cast<std::byte>(texel[channel]);
            }
        }
    }
    REQUIRE(device.writeBuffer(*upload, 0, checker).has_value());
    auto albedoView = device.createTextureView({.texture = *albedo, .usage = rachis::TextureViewUsage::sampled});
    auto sampler = device.createSampler({
        .minFilter = rachis::FilterMode::nearest,
        .magFilter = rachis::FilterMode::nearest,
        .mipFilter = rachis::MipmapFilterMode::nearest,
    });
    REQUIRE(albedoView);
    REQUIRE(sampler);
    const std::array textureWrites{
        rachis::BindingWrite{.binding = 0, .resource = *albedoView},
    };
    auto textureSet = rachis::test::createBindingSet(device,
                                                     {
                                                         .layout = *textureBindingLayout,
                                                         .writes = textureWrites,
                                                     });
    REQUIRE(textureSet);
    auto samplerSet = rachis::test::createBindingSet(
        device,
        {
            .layout = *samplerBindingLayout,
            .writes = std::array{rachis::BindingWrite{.binding = 1, .resource = *sampler}},
        });
    REQUIRE(samplerSet);
    auto textureTarget = device.createTexture({
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
    });
    REQUIRE(textureTarget);
    auto textureTargetView =
        device.createTextureView({.texture = *textureTarget, .usage = rachis::TextureViewUsage::renderTarget});
    REQUIRE(textureTargetView);
    auto textureTransfer = device.queryTextureTransferLayout(*textureTarget, region);
    REQUIRE(textureTransfer);
    auto textureReadback = device.createBuffer({
        .size = textureTransfer->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(textureReadback);
    auto textureCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(textureCommand);
    REQUIRE(device
                .textureBarrier(*textureCommand,
                                {
                                    .texture = *albedo,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    REQUIRE(device.copyBufferToTexture(*textureCommand, *upload, *albedo, std::array{albedoRegion}).has_value());
    REQUIRE(device
                .textureBarrier(*textureCommand,
                                {
                                    .texture = *albedo,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::pixelShader,
                                    .accessBefore = rachis::Access::copyDestination,
                                    .accessAfter = rachis::Access::shaderResource,
                                    .layoutBefore = rachis::TextureLayout::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::shaderResource,
                                })
                .has_value());
    REQUIRE(device
                .textureBarrier(*textureCommand,
                                {
                                    .texture = *textureTarget,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array textureAttachments{
        rachis::ColorAttachment{.view = *textureTargetView, .loadOperation = rachis::LoadOperation::clear},
    };
    REQUIRE(device.beginRendering(*textureCommand, {.colorAttachments = textureAttachments}).has_value());
    REQUIRE(device.bindGraphicsPipeline(*textureCommand, *texturedPipeline).has_value());
    REQUIRE(device.bindGraphicsBindingSet(*textureCommand, 1, *textureSet).has_value());
    REQUIRE(device.bindGraphicsBindingSet(*textureCommand, 3, *samplerSet).has_value());
    REQUIRE(device
                .setViewport(*textureCommand,
                             {
                                 .width = static_cast<float>(targetWidth),
                                 .height = static_cast<float>(targetHeight),
                             })
                .has_value());
    REQUIRE(device.setScissor(*textureCommand, {.width = targetWidth, .height = targetHeight}).has_value());
    REQUIRE(device.draw(*textureCommand, 3).has_value());
    REQUIRE(device.endRendering(*textureCommand).has_value());
    REQUIRE(device
                .textureBarrier(*textureCommand,
                                {
                                    .texture = *textureTarget,
                                    .stagesBefore = rachis::PipelineStage::renderTarget,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::renderTarget,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    const rachis::BufferTextureCopyRegion textureCopy{
        .bytesPerRow = textureTransfer->bytesPerRow,
        .rowsPerImage = textureTransfer->rowsPerImage,
        .extent = {.width = targetWidth, .height = targetHeight, .depth = 1},
    };
    REQUIRE(device.copyTextureToBuffer(*textureCommand, *textureTarget, *textureReadback, std::array{textureCopy})
                .has_value());
    auto textureSubmitted = device.submit(std::move(*textureCommand));
    REQUIRE(textureSubmitted);
    REQUIRE(device.wait(*textureSubmitted).has_value());
    auto textureMapped = device.mapBuffer(*textureReadback);
    REQUIRE(textureMapped);
    CHECK(pixelMatches(
        pixelAt(textureMapped->data, textureTransfer->bytesPerRow, targetWidth / 2, targetHeight / 2), redBytes, 2));
    CHECK_FALSE(pixelMatches(
        pixelAt(textureMapped->data, textureTransfer->bytesPerRow, targetWidth / 2, targetHeight / 2), blueBytes, 2));
    writePpm(evidenceDir() / "metal-hardware-textured.ppm", textureMapped->data, textureTransfer->bytesPerRow);
    REQUIRE(device.unmapBuffer(*textureReadback).has_value());

    const auto msaa = renderTintedTriangle(
        device, shaders, {.width = targetWidth, .height = targetHeight}, tint, false, rachis::CullMode::none, {}, 4);
    requireOffscreenTriangle(msaa, 2);
    writePpm(evidenceDir() / "metal-hardware-msaa.ppm", msaa.pixels.data(), msaa.pitch);

    const auto computeBytecode = loadBytes(pnmsPath("MetalHeadlessParity.cs.metallib"));
    REQUIRE_FALSE(computeBytecode.empty());
    const std::array computeBindings{
        rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::storageBuffer, 1},
    };
    const std::array computePushReq{rachis::ShaderPushConstantRequirement{0, 16}};
    auto computeShader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = computeBytecode,
        .entryPoint = "computeMain",
        .shaderInterface = rachis::test::computeShaderInterface({64, 1, 1}, computeBindings, computePushReq),
    });
    if (!computeShader)
    {
        UNSCOPED_INFO(computeShader.error().message);
    }
    REQUIRE(computeShader);
    const std::array computeEntries{
        rachis::BindingLayoutEntry{1, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
    };
    auto computeBindingLayout = device.createBindingLayout({.entries = computeEntries});
    REQUIRE(computeBindingLayout);
    const std::array computePush{rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 16}};
    auto computeLayout = device.createPipelineLayout({
        .bindingLayouts = std::array{*computeBindingLayout},
        .pushConstantRanges = computePush,
    });
    REQUIRE(computeLayout);
    auto computePipeline = device.createComputePipeline({.layout = *computeLayout, .computeShader = *computeShader});
    REQUIRE(computePipeline);
    auto storage = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::deviceLocal,
    });
    auto computeReadback = device.createBuffer({
        .size = outputSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(storage);
    REQUIRE(computeReadback);
    const std::array storageWrites{
        rachis::BindingWrite{
            .binding = 1,
            .resource =
                rachis::BufferBinding{
                    .buffer = *storage,
                    .size = outputSize,
                    .structureByteStride = sizeof(std::uint32_t),
                },
        },
    };
    auto storageSet =
        rachis::test::createBindingSet(device, {.layout = *computeBindingLayout, .writes = storageWrites});
    REQUIRE(storageSet);
    auto computeCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(computeCommand);
    REQUIRE(device
                .bufferBarrier(*computeCommand,
                               {
                                   .buffer = *storage,
                                   .stagesAfter = rachis::PipelineStage::computeShader,
                                   .accessAfter = rachis::Access::unorderedAccess,
                               })
                .has_value());
    REQUIRE(device.bindComputePipeline(*computeCommand, *computePipeline).has_value());
    REQUIRE(device.bindComputeBindingSet(*computeCommand, 0, *storageSet).has_value());
    const std::array constants{3U, 7U, 0U, 0U};
    REQUIRE(device.setComputePushConstants(*computeCommand, 0, std::as_bytes(std::span(constants))).has_value());
    REQUIRE(device.dispatch(*computeCommand, 1).has_value());
    REQUIRE(device
                .bufferBarrier(*computeCommand,
                               {
                                   .buffer = *storage,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(
        device.copyBuffer(*computeCommand, {.source = *storage, .destination = *computeReadback, .size = outputSize})
            .has_value());
    auto computeSubmitted = device.submit(std::move(*computeCommand));
    REQUIRE(computeSubmitted);
    REQUIRE(device.wait(*computeSubmitted).has_value());
    auto computeMapped = device.mapBuffer(*computeReadback);
    REQUIRE(computeMapped);
    const auto* values = reinterpret_cast<const std::uint32_t*>(computeMapped->data);
    CHECK(values[0] == 7);
    CHECK(values[1] == 10);
    CHECK(values[63] == 196);
    REQUIRE(device.unmapBuffer(*computeReadback).has_value());

    writeReport(evidenceDir() / "metal-hardware-report.txt",
                adapterLabel(adapter) + "\nfixture=offscreenTriangle tolerance=2\nlinearClear0.5=" +
                    std::to_string(linearByte) + "\nsrgbClear0.5=" + std::to_string(srgbByte) + "\n");
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
