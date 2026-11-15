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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

    constexpr std::uint32_t targetWidth = 64;
    constexpr std::uint32_t targetHeight = 64;
    constexpr std::uint32_t rowPitch = 256;
    constexpr std::uint64_t readbackSize = static_cast<std::uint64_t>(rowPitch) * targetHeight;
    constexpr std::uint32_t valueCount = 64;
    constexpr std::uint64_t outputSize = valueCount * sizeof(std::uint32_t);
    constexpr std::array tint{0.125F, 0.5F, 0.75F, 1.0F};
    constexpr std::array<std::uint8_t, 4> tintBytes{32, 128, 191, 255};
    constexpr std::array<std::uint8_t, 4> tealBytes{0, 200, 200, 255};
    constexpr std::array<std::uint8_t, 4> redBytes{255, 0, 0, 255};

    [[nodiscard]] rachis::DeviceDesc deviceDesc(rachis::Backend backend, rachis::AdapterPreference preference) noexcept
    {
        return rachis::DeviceDesc{
            .backend = backend,
            .adapterPreference = preference,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = backend == rachis::Backend::d3d12,
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

    [[nodiscard]] bool withinTolerance(std::byte actual, std::uint8_t expected, std::uint8_t tolerance) noexcept
    {
        const auto value = std::to_integer<std::uint8_t>(actual);
        const auto lower = expected <= tolerance ? 0 : static_cast<int>(expected) - tolerance;
        const auto upper = expected >= 255 - tolerance ? 255 : static_cast<int>(expected) + tolerance;
        return std::cmp_greater_equal(value, lower) && std::cmp_less_equal(value, upper);
    }

    [[nodiscard]] const std::byte* pixelAt(const std::byte* data, std::uint32_t x, std::uint32_t y) noexcept
    {
        return data + static_cast<std::uint64_t>(y) * rowPitch + static_cast<std::uint64_t>(x) * 4;
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

    void writePpm(const std::filesystem::path& path, const std::byte* data)
    {
        std::error_code error;
        std::filesystem::create_directories(path.parent_path(), error);
        std::ofstream stream(path, std::ios::binary);
        stream << "P6\n" << targetWidth << ' ' << targetHeight << "\n255\n";
        for (std::uint32_t y = 0; y < targetHeight; ++y)
        {
            for (std::uint32_t x = 0; x < targetWidth; ++x)
            {
                const auto* pixel = pixelAt(data, x, y);
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
        return std::filesystem::temp_directory_path() / "rachis-vulkan-parity";
#endif
    }

    [[nodiscard]] std::string adapterLabel(const rachis::AdapterInfo& adapter, bool warp)
    {
        std::string label = adapter.type == rachis::AdapterType::software ? "software" : "hardware";
        if (warp)
        {
            label = "warp";
        }
        label += " backend=";
        label += adapter.backend == rachis::Backend::vulkan ? "vulkan" : "d3d12";
        label += " vendor=0x";
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
        const char* vertexEntry = "directVertex";
        const char* farEntry = "depthBiasVertex";
        const char* pixelEntry = "pixelMain";
    };

    [[nodiscard]] TriangleShaders vulkanTriangleShaders()
    {
        const auto dir = std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR);
        return {
            .vertex = loadBytes(dir / "VulkanGraphicsTriangle.direct.vs.spv"),
            .farVertex = loadBytes(dir / "VulkanGraphicsTriangle.depth.vs.spv"),
            .pixel = loadBytes(dir / "VulkanGraphicsTriangle.ps.spv"),
        };
    }

#if RACHIS_ENABLE_D3D12
    [[nodiscard]] std::optional<rachis::Device> tryCreate(const rachis::DeviceDesc& desc)
    {
        auto created = rachis::createDevice(desc);
        if (!created)
        {
            return std::nullopt;
        }
        return std::move(*created);
    }

    [[nodiscard]] TriangleShaders d3d12TriangleShaders()
    {
        return {
            .vertex = loadBytes(RACHIS_TEST_GRAPHICS_DIRECT_VERTEX_SHADER_PATH),
            .farVertex = loadBytes(RACHIS_TEST_GRAPHICS_DEPTH_BIAS_VERTEX_SHADER_PATH),
            .pixel = loadBytes(RACHIS_TEST_GRAPHICS_PIXEL_SHADER_PATH),
        };
    }
#endif

    [[nodiscard]] rachis::Result<rachis::ShaderModuleHandle> createVertexShader(rachis::Device& device,
                                                                                std::span<const std::byte> bytecode,
                                                                                std::string_view entryPoint)
    {
        return device.createShaderModule({
            .stage = rachis::ShaderStage::vertex,
            .bytecode = bytecode,
            .entryPoint = entryPoint,
        });
    }

    [[nodiscard]] rachis::Result<rachis::ShaderModuleHandle> createTintPixelShader(rachis::Device& device,
                                                                                   std::span<const std::byte> bytecode,
                                                                                   std::string_view entryPoint)
    {
        return device.createShaderModule({
            .stage = rachis::ShaderStage::pixel,
            .bytecode = bytecode,
            .entryPoint = entryPoint,
            .shaderInterface = rachis::test::graphicsPixelShaderInterface(),
        });
    }

    [[nodiscard]] rachis::PipelineLayoutHandle createTintPipelineLayout(rachis::Device& device)
    {
        const std::array pushConstantRanges{
            rachis::PushConstantRange{rachis::ShaderStage::pixel, 0, 4 * sizeof(float)},
        };
        auto layout = device.createPipelineLayout({.pushConstantRanges = pushConstantRanges});
        REQUIRE(layout);
        return *layout;
    }

    [[nodiscard]] rachis::Result<rachis::PipelineHandle> createDepthTestedPipeline(
        rachis::Device& device,
        rachis::PipelineLayoutHandle layout,
        rachis::ShaderModuleHandle vertexShader,
        rachis::ShaderModuleHandle pixelShader)
    {
        const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
        return device.createGraphicsPipeline({
            .layout = layout,
            .vertexShader = vertexShader,
            .pixelShader = pixelShader,
            .cullMode = rachis::CullMode::none,
            .colorTargets = colorTargets,
            .depthStencil =
                {
                    .format = rachis::Format::d32Float,
                    .depthTestEnabled = true,
                    .depthWriteEnabled = true,
                    .depthCompare = rachis::CompareOperation::lessEqual,
                },
        });
    }

    [[nodiscard]] rachis::PipelineHandle createTintedTrianglePipeline(rachis::Device& device,
                                                                      const TriangleShaders& shaders,
                                                                      bool useFarVertex)
    {
        auto vertexShader = useFarVertex ? createVertexShader(device, shaders.farVertex, shaders.farEntry)
                                         : createVertexShader(device, shaders.vertex, shaders.vertexEntry);
        auto pixelShader = createTintPixelShader(device, shaders.pixel, shaders.pixelEntry);
        REQUIRE(vertexShader);
        REQUIRE(pixelShader);
        const auto layout = createTintPipelineLayout(device);
        auto pipeline = createDepthTestedPipeline(device, layout, *vertexShader, *pixelShader);
        REQUIRE(pipeline);
        return *pipeline;
    }

    struct DepthTestedTarget final
    {
        rachis::TextureHandle color{};
        rachis::TextureHandle depth{};
        rachis::TextureViewHandle colorView{};
        rachis::TextureViewHandle depthView{};
    };

    [[nodiscard]] DepthTestedTarget createDepthTestedTarget(rachis::Device& device)
    {
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
        return {.color = *color, .depth = *depth, .colorView = *colorView, .depthView = *depthView};
    }

    [[nodiscard]] rachis::Result<rachis::BufferHandle> createReadbackBuffer(rachis::Device& device, std::uint64_t size)
    {
        return device.createBuffer({
            .size = size,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
        });
    }

    void transitionToRenderTarget(rachis::Device& device, rachis::CommandBuffer& command, rachis::TextureHandle texture)
    {
        REQUIRE(device
                    .textureBarrier(command,
                                    {
                                        .texture = texture,
                                        .stagesAfter = rachis::PipelineStage::renderTarget,
                                        .accessAfter = rachis::Access::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::renderTarget,
                                    })
                    .has_value());
    }

    void transitionToDepthWrite(rachis::Device& device, rachis::CommandBuffer& command, rachis::TextureHandle depth)
    {
        REQUIRE(device
                    .textureBarrier(command,
                                    {
                                        .texture = depth,
                                        .stagesAfter = rachis::PipelineStage::depthStencil,
                                        .accessAfter = rachis::Access::depthStencilWrite,
                                        .layoutAfter = rachis::TextureLayout::depthStencilWrite,
                                        .subresources = {.aspect = rachis::TextureAspect::depth},
                                    })
                    .has_value());
    }

    void beginClearedDepthTestedRendering(rachis::Device& device,
                                          rachis::CommandBuffer& command,
                                          const DepthTestedTarget& target)
    {
        const std::array attachments{
            rachis::ColorAttachment{.view = target.colorView, .loadOperation = rachis::LoadOperation::clear},
        };
        REQUIRE(device
                    .beginRendering(command,
                                    {
                                        .colorAttachments = attachments,
                                        .depthStencilAttachment =
                                            rachis::DepthStencilAttachment{
                                                .view = target.depthView,
                                                .depthLoadOperation = rachis::LoadOperation::clear,
                                                .depthClearValue = 1.0F,
                                            },
                                    })
                    .has_value());
    }

    void setFullTargetViewport(rachis::Device& device, rachis::CommandBuffer& command)
    {
        REQUIRE(device
                    .setViewport(command,
                                 {
                                     .width = static_cast<float>(targetWidth),
                                     .height = static_cast<float>(targetHeight),
                                 })
                    .has_value());
    }

    void drawTintedTriangle(rachis::Device& device,
                            rachis::CommandBuffer& command,
                            const std::array<float, 4>& tintColor)
    {
        REQUIRE(
            device.setGraphicsPushConstants(command, rachis::ShaderStage::pixel, 0, std::as_bytes(std::span(tintColor)))
                .has_value());
        REQUIRE(device.draw(command, 3).has_value());
    }

    void transitionRenderTargetToCopySource(rachis::Device& device,
                                            rachis::CommandBuffer& command,
                                            rachis::TextureHandle texture)
    {
        REQUIRE(device
                    .textureBarrier(command,
                                    {
                                        .texture = texture,
                                        .stagesBefore = rachis::PipelineStage::renderTarget,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::renderTarget,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                    })
                    .has_value());
    }

    void copyRenderTargetToReadback(rachis::Device& device,
                                    rachis::CommandBuffer& command,
                                    rachis::TextureHandle renderTarget,
                                    rachis::BufferHandle readback)
    {
        transitionRenderTargetToCopySource(device, command, renderTarget);
        REQUIRE(device
                    .copyTextureToBuffer(command,
                                         {
                                             .buffer = readback,
                                             .bytesPerRow = rowPitch,
                                             .rowsPerImage = targetHeight,
                                             .texture = renderTarget,
                                             .extent = {targetWidth, targetHeight, 1},
                                         })
                    .has_value());
    }

    void destroyDepthTestedTarget(rachis::Device& device, const DepthTestedTarget& target)
    {
        REQUIRE(device.destroyTextureView(target.colorView).has_value());
        REQUIRE(device.destroyTextureView(target.depthView).has_value());
        REQUIRE(device.destroyTexture(target.color).has_value());
        REQUIRE(device.destroyTexture(target.depth).has_value());
    }

    void submitAndWait(rachis::Device& device, rachis::CommandBuffer&& command)
    {
        auto submitted = device.submit(std::move(command));
        REQUIRE(submitted);
        REQUIRE(device.wait(*submitted).has_value());
    }

    [[nodiscard]] std::vector<std::byte> renderTintedTriangle(rachis::Device& device,
                                                              const TriangleShaders& shaders,
                                                              const rachis::ScissorRect& scissor,
                                                              const std::array<float, 4>& tintColor,
                                                              bool useFarVertex)
    {
        REQUIRE_FALSE(shaders.vertex.empty());
        REQUIRE_FALSE(shaders.pixel.empty());
        if (useFarVertex)
        {
            REQUIRE_FALSE(shaders.farVertex.empty());
        }

        const auto pipeline = createTintedTrianglePipeline(device, shaders, useFarVertex);
        const auto target = createDepthTestedTarget(device);
        auto readback = createReadbackBuffer(device, readbackSize);
        REQUIRE(readback);

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command);
        transitionToRenderTarget(device, *command, target.color);
        transitionToDepthWrite(device, *command, target.depth);
        beginClearedDepthTestedRendering(device, *command, target);
        REQUIRE(device.bindGraphicsPipeline(*command, pipeline).has_value());
        setFullTargetViewport(device, *command);
        REQUIRE(device.setScissor(*command, scissor).has_value());
        drawTintedTriangle(device, *command, tintColor);
        REQUIRE(device.endRendering(*command).has_value());
        copyRenderTargetToReadback(device, *command, target.color, *readback);
        destroyDepthTestedTarget(device, target);
        submitAndWait(device, std::move(*command));

        auto mapped = device.mapBuffer(*readback);
        REQUIRE(mapped);
        std::vector<std::byte> pixels(mapped->data, mapped->data + readbackSize);
        REQUIRE(device.unmapBuffer(*readback).has_value());
        return pixels;
    }

    void requireOffscreenTriangle(const std::byte* pixels, std::uint8_t tolerance)
    {
        const auto* center = pixelAt(pixels, targetWidth / 2, targetHeight / 2);
        const auto* upright = pixelAt(pixels, 14, 52);
        const auto* flipped = pixelAt(pixels, 14, 12);
        REQUIRE(pixelMatches(center, tintBytes, tolerance));
        REQUIRE(pixelMatches(upright, tintBytes, tolerance));
        REQUIRE(isClear(flipped));
    }

    struct DepthOrderedPipelines final
    {
        rachis::PipelineHandle nearTriangle{};
        rachis::PipelineHandle farTriangle{};
    };

    [[nodiscard]] DepthOrderedPipelines createDepthOrderedPipelines(rachis::Device& device,
                                                                    const TriangleShaders& shaders)
    {
        auto nearShader = createVertexShader(device, shaders.vertex, "directVertex");
        auto farShader = createVertexShader(device, shaders.farVertex, "depthBiasVertex");
        auto pixelShader = createTintPixelShader(device, shaders.pixel, "pixelMain");
        REQUIRE(nearShader);
        REQUIRE(farShader);
        REQUIRE(pixelShader);
        const auto layout = createTintPipelineLayout(device);
        auto nearPipeline = createDepthTestedPipeline(device, layout, *nearShader, *pixelShader);
        auto farPipeline = createDepthTestedPipeline(device, layout, *farShader, *pixelShader);
        REQUIRE(nearPipeline);
        REQUIRE(farPipeline);
        return {.nearTriangle = *nearPipeline, .farTriangle = *farPipeline};
    }

    void drawFarThenNearTriangles(rachis::Device& device,
                                  rachis::CommandBuffer& command,
                                  const DepthOrderedPipelines& pipelines)
    {
        REQUIRE(device.bindGraphicsPipeline(command, pipelines.farTriangle).has_value());
        setFullTargetViewport(device, command);
        REQUIRE(device.setScissor(command, {.width = targetWidth, .height = targetHeight}).has_value());
        const std::array farTint{1.0F, 0.0F, 0.0F, 1.0F};
        const std::array nearTint{0.0F, 0.784F, 0.784F, 1.0F};
        drawTintedTriangle(device, command, farTint);
        REQUIRE(device.bindGraphicsPipeline(command, pipelines.nearTriangle).has_value());
        drawTintedTriangle(device, command, nearTint);
    }

    void checkDepthAndScissor(rachis::Device& device, const TriangleShaders& shaders)
    {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command);

        const auto target = createDepthTestedTarget(device);
        const auto pipelines = createDepthOrderedPipelines(device, shaders);
        auto readback = createReadbackBuffer(device, readbackSize);
        REQUIRE(readback);

        transitionToRenderTarget(device, *command, target.color);
        transitionToDepthWrite(device, *command, target.depth);
        beginClearedDepthTestedRendering(device, *command, target);
        drawFarThenNearTriangles(device, *command, pipelines);
        REQUIRE(device.endRendering(*command).has_value());
        copyRenderTargetToReadback(device, *command, target.color, *readback);
        destroyDepthTestedTarget(device, target);
        submitAndWait(device, std::move(*command));

        auto mapped = device.mapBuffer(*readback);
        REQUIRE(mapped);
        const auto* depthCenter = pixelAt(mapped->data, targetWidth / 2, targetHeight / 2);
        CHECK(pixelMatches(depthCenter, tealBytes, 2));
        CHECK_FALSE(pixelMatches(depthCenter, redBytes, 2));
        writePpm(evidenceDir() / "vulkan-hardware-depth.ppm", mapped->data);
        REQUIRE(device.unmapBuffer(*readback).has_value());

        const auto clipped =
            renderTintedTriangle(device, shaders, {.x = 32, .width = 32, .height = targetHeight}, tint, false);
        CHECK(isClear(pixelAt(clipped.data(), 14, 52)));
        CHECK(pixelMatches(pixelAt(clipped.data(), targetWidth / 2, targetHeight / 2), tintBytes, 2));
        writePpm(evidenceDir() / "vulkan-hardware-scissor.ppm", clipped.data());
    }

    [[nodiscard]] rachis::PipelineHandle createAlphaBlendPipeline(rachis::Device& device,
                                                                  const TriangleShaders& shaders)
    {
        const std::array blendTargets{
            rachis::ColorTargetState{
                .format = rachis::Format::rgba8Unorm,
                .blend =
                    {
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
                    },
            },
        };
        auto vertexShader = createVertexShader(device, shaders.vertex, "directVertex");
        auto pixelShader = createTintPixelShader(device, shaders.pixel, "pixelMain");
        REQUIRE(vertexShader);
        REQUIRE(pixelShader);
        const auto layout = createTintPipelineLayout(device);
        auto pipeline = device.createGraphicsPipeline({
            .layout = layout,
            .vertexShader = *vertexShader,
            .pixelShader = *pixelShader,
            .cullMode = rachis::CullMode::none,
            .colorTargets = blendTargets,
        });
        REQUIRE(pipeline);
        return *pipeline;
    }

    struct ColorTarget final
    {
        rachis::TextureHandle texture{};
        rachis::TextureViewHandle view{};
    };

    [[nodiscard]] ColorTarget createColorTarget(rachis::Device& device)
    {
        auto texture = device.createTexture({
            .width = targetWidth,
            .height = targetHeight,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
        });
        REQUIRE(texture);
        auto view = device.createTextureView({.texture = *texture, .usage = rachis::TextureViewUsage::renderTarget});
        REQUIRE(view);
        return {.texture = *texture, .view = *view};
    }

    void beginClearedColorRendering(rachis::Device& device,
                                    rachis::CommandBuffer& command,
                                    rachis::TextureViewHandle view)
    {
        const std::array attachments{
            rachis::ColorAttachment{.view = view, .loadOperation = rachis::LoadOperation::clear},
        };
        REQUIRE(device.beginRendering(command, {.colorAttachments = attachments}).has_value());
    }

    void checkBlend(rachis::Device& device, const TriangleShaders& shaders)
    {
        const auto pipeline = createAlphaBlendPipeline(device, shaders);
        const auto target = createColorTarget(device);
        auto readback = createReadbackBuffer(device, readbackSize);
        REQUIRE(readback);

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command);
        transitionToRenderTarget(device, *command, target.texture);
        beginClearedColorRendering(device, *command, target.view);
        REQUIRE(device.bindGraphicsPipeline(*command, pipeline).has_value());
        setFullTargetViewport(device, *command);
        REQUIRE(device.setScissor(*command, {.width = targetWidth, .height = targetHeight}).has_value());
        const std::array halfRed{1.0F, 0.0F, 0.0F, 0.5F};
        drawTintedTriangle(device, *command, halfRed);
        REQUIRE(device.endRendering(*command).has_value());
        copyRenderTargetToReadback(device, *command, target.texture, *readback);
        REQUIRE(device.destroyTextureView(target.view).has_value());
        REQUIRE(device.destroyTexture(target.texture).has_value());
        submitAndWait(device, std::move(*command));

        auto mapped = device.mapBuffer(*readback);
        REQUIRE(mapped);
        const auto* blendCenter = pixelAt(mapped->data, targetWidth / 2, targetHeight / 2);
        CHECK(withinTolerance(blendCenter[0], 128, 2));
        CHECK(withinTolerance(blendCenter[1], 0, 2));
        CHECK_FALSE(withinTolerance(blendCenter[0], 255, 2));
        writePpm(evidenceDir() / "vulkan-hardware-blend.ppm", mapped->data);
        REQUIRE(device.unmapBuffer(*readback).has_value());
    }

    struct ClearBytes final
    {
        std::uint8_t linear{};
        std::uint8_t srgb{};
    };

    [[nodiscard]] rachis::Result<rachis::TextureHandle> createTwoByTwoRenderTarget(rachis::Device& device,
                                                                                   rachis::Format format)
    {
        return device.createTexture({
            .width = 2,
            .height = 2,
            .format = format,
            .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
        });
    }

    void clearToHalfIntensity(rachis::Device& device, rachis::CommandBuffer& command, rachis::TextureViewHandle view)
    {
        const std::array attachments{
            rachis::ColorAttachment{
                .view = view,
                .loadOperation = rachis::LoadOperation::clear,
                .clearValue = {0.5F, 0.5F, 0.5F, 1.0F},
            },
        };
        REQUIRE(device.beginRendering(command, {.colorAttachments = attachments}).has_value());
        REQUIRE(device.endRendering(command).has_value());
    }

    [[nodiscard]] ClearBytes checkSrgbClear(rachis::Device& device)
    {
        auto linear = createTwoByTwoRenderTarget(device, rachis::Format::rgba8Unorm);
        auto srgb = createTwoByTwoRenderTarget(device, rachis::Format::rgba8UnormSrgb);
        REQUIRE(linear);
        REQUIRE(srgb);
        auto linearView =
            device.createTextureView({.texture = *linear, .usage = rachis::TextureViewUsage::renderTarget});
        auto srgbView = device.createTextureView({.texture = *srgb, .usage = rachis::TextureViewUsage::renderTarget});
        REQUIRE(linearView);
        REQUIRE(srgbView);
        const rachis::BufferTextureCopyRegion small{.extent = {.width = 2, .height = 2}};
        auto linearLayout = device.queryTextureTransferLayout(*linear, small);
        auto srgbLayout = device.queryTextureTransferLayout(*srgb, small);
        REQUIRE(linearLayout);
        REQUIRE(srgbLayout);
        auto linearRead = createReadbackBuffer(device, linearLayout->requiredBufferSize);
        auto srgbRead = createReadbackBuffer(device, srgbLayout->requiredBufferSize);
        REQUIRE(linearRead);
        REQUIRE(srgbRead);

        auto clearCommand = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(clearCommand);
        transitionToRenderTarget(device, *clearCommand, *linear);
        transitionToRenderTarget(device, *clearCommand, *srgb);
        clearToHalfIntensity(device, *clearCommand, *linearView);
        clearToHalfIntensity(device, *clearCommand, *srgbView);
        transitionRenderTargetToCopySource(device, *clearCommand, *linear);
        transitionRenderTargetToCopySource(device, *clearCommand, *srgb);
        REQUIRE(device.copyTextureToBuffer(*clearCommand, *linear, *linearRead, std::array{small}).has_value());
        REQUIRE(device.copyTextureToBuffer(*clearCommand, *srgb, *srgbRead, std::array{small}).has_value());
        REQUIRE(device.destroyTextureView(*linearView).has_value());
        REQUIRE(device.destroyTextureView(*srgbView).has_value());
        REQUIRE(device.destroyTexture(*linear).has_value());
        REQUIRE(device.destroyTexture(*srgb).has_value());
        submitAndWait(device, std::move(*clearCommand));

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
        return {.linear = linearByte, .srgb = srgbByte};
    }

    struct TexturedPipeline final
    {
        rachis::BindingLayoutHandle bindingLayout{};
        rachis::PipelineHandle pipeline{};
    };

    [[nodiscard]] TexturedPipeline createTexturedPipeline(rachis::Device& device)
    {
        const auto vertexBytecode =
            loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanHeadlessParity.textured.vs.spv");
        const auto pixelBytecode =
            loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanHeadlessParity.textured.ps.spv");
        REQUIRE_FALSE(vertexBytecode.empty());
        REQUIRE_FALSE(pixelBytecode.empty());
        static constexpr std::array bindingRequirements{
            rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::sampledTexture, 1},
            rachis::ShaderBindingRequirement{0, 1, rachis::ShaderResourceType::sampler, 1},
        };
        auto vertexShader = createVertexShader(device, vertexBytecode, "texturedVertex");
        auto pixelShader = device.createShaderModule({
            .stage = rachis::ShaderStage::pixel,
            .bytecode = pixelBytecode,
            .entryPoint = "texturedPixel",
            .shaderInterface = rachis::test::shaderInterface(bindingRequirements),
        });
        REQUIRE(vertexShader);
        REQUIRE(pixelShader);
        const std::array layoutEntries{
            rachis::BindingLayoutEntry{0, rachis::BindingType::sampledTexture, rachis::ShaderStage::pixel},
            rachis::BindingLayoutEntry{1, rachis::BindingType::sampler, rachis::ShaderStage::pixel},
        };
        auto bindingLayout = device.createBindingLayout({.entries = layoutEntries});
        REQUIRE(bindingLayout);
        auto pipelineLayout = device.createPipelineLayout({.bindingLayouts = std::array{*bindingLayout}});
        REQUIRE(pipelineLayout);
        const std::array colorTargets{rachis::ColorTargetState{.format = rachis::Format::rgba8Unorm}};
        auto pipeline = device.createGraphicsPipeline({
            .layout = *pipelineLayout,
            .vertexShader = *vertexShader,
            .pixelShader = *pixelShader,
            .cullMode = rachis::CullMode::none,
            .colorTargets = colorTargets,
        });
        REQUIRE(pipeline);
        return {.bindingLayout = *bindingLayout, .pipeline = *pipeline};
    }

    [[nodiscard]] std::vector<std::byte> checkerTexels(const rachis::TextureTransferLayout& layout)
    {
        std::vector<std::byte> checker(static_cast<std::size_t>(layout.requiredBufferSize));
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
                    checker[static_cast<std::size_t>(y) * layout.bytesPerRow + static_cast<std::size_t>(x) * 4 +
                            channel] = static_cast<std::byte>(texel[channel]);
                }
            }
        }
        return checker;
    }

    struct CheckerAlbedo final
    {
        rachis::TextureHandle texture{};
        rachis::BufferHandle upload{};
    };

    [[nodiscard]] CheckerAlbedo createCheckerAlbedo(rachis::Device& device,
                                                    const rachis::BufferTextureCopyRegion& region)
    {
        auto albedo = device.createTexture({
            .width = 2,
            .height = 2,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::copyDestination | rachis::TextureUsage::sampled,
        });
        REQUIRE(albedo);
        auto albedoLayout = device.queryTextureTransferLayout(*albedo, region);
        REQUIRE(albedoLayout);
        auto upload = device.createBuffer({
            .size = albedoLayout->requiredBufferSize,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
        });
        REQUIRE(upload);
        const auto checker = checkerTexels(*albedoLayout);
        REQUIRE(device.writeBuffer(*upload, 0, checker).has_value());
        return {.texture = *albedo, .upload = *upload};
    }

    struct SampledAlbedo final
    {
        rachis::TextureViewHandle view{};
        rachis::BindingSetHandle bindingSet{};
    };

    [[nodiscard]] SampledAlbedo createSampledAlbedo(rachis::Device& device,
                                                    rachis::BindingLayoutHandle bindingLayout,
                                                    rachis::TextureHandle albedo)
    {
        auto albedoView = device.createTextureView({.texture = albedo, .usage = rachis::TextureViewUsage::sampled});
        auto sampler = device.createSampler({
            .minFilter = rachis::FilterMode::nearest,
            .magFilter = rachis::FilterMode::nearest,
            .mipFilter = rachis::MipmapFilterMode::nearest,
        });
        REQUIRE(albedoView);
        REQUIRE(sampler);
        const std::array writes{
            rachis::BindingWrite{.binding = 0, .resource = *albedoView},
            rachis::BindingWrite{.binding = 1, .resource = *sampler},
        };
        auto bindingSet = rachis::test::createBindingSet(device,
                                                         {
                                                             .layout = bindingLayout,
                                                             .writes = writes,
                                                         });
        REQUIRE(bindingSet);
        return {.view = *albedoView, .bindingSet = *bindingSet};
    }

    void uploadCheckerAlbedo(rachis::Device& device,
                             rachis::CommandBuffer& command,
                             const CheckerAlbedo& albedo,
                             const rachis::BufferTextureCopyRegion& region)
    {
        REQUIRE(device
                    .textureBarrier(command,
                                    {
                                        .texture = albedo.texture,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessAfter = rachis::Access::copyDestination,
                                        .layoutAfter = rachis::TextureLayout::copyDestination,
                                    })
                    .has_value());
        REQUIRE(device.copyBufferToTexture(command, albedo.upload, albedo.texture, std::array{region}).has_value());
        REQUIRE(device
                    .textureBarrier(command,
                                    {
                                        .texture = albedo.texture,
                                        .stagesBefore = rachis::PipelineStage::copy,
                                        .stagesAfter = rachis::PipelineStage::pixelShader,
                                        .accessBefore = rachis::Access::copyDestination,
                                        .accessAfter = rachis::Access::shaderResource,
                                        .layoutBefore = rachis::TextureLayout::copyDestination,
                                        .layoutAfter = rachis::TextureLayout::shaderResource,
                                    })
                    .has_value());
    }

    void checkTexturedDraw(rachis::Device& device)
    {
        const auto textured = createTexturedPipeline(device);
        const rachis::BufferTextureCopyRegion albedoRegion{.extent = {.width = 2, .height = 2}};
        const auto albedo = createCheckerAlbedo(device, albedoRegion);
        const auto sampledAlbedo = createSampledAlbedo(device, textured.bindingLayout, albedo.texture);
        const auto target = createColorTarget(device);
        auto readback = createReadbackBuffer(device, readbackSize);
        REQUIRE(readback);

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command);
        uploadCheckerAlbedo(device, *command, albedo, albedoRegion);
        transitionToRenderTarget(device, *command, target.texture);
        beginClearedColorRendering(device, *command, target.view);
        REQUIRE(device.bindGraphicsPipeline(*command, textured.pipeline).has_value());
        REQUIRE(device.bindGraphicsBindingSet(*command, 0, sampledAlbedo.bindingSet).has_value());
        setFullTargetViewport(device, *command);
        REQUIRE(device.setScissor(*command, {.width = targetWidth, .height = targetHeight}).has_value());
        REQUIRE(device.draw(*command, 3).has_value());
        REQUIRE(device.endRendering(*command).has_value());
        copyRenderTargetToReadback(device, *command, target.texture, *readback);
        REQUIRE(device.destroyTextureView(target.view).has_value());
        REQUIRE(device.destroyTextureView(sampledAlbedo.view).has_value());
        REQUIRE(device.destroyTexture(target.texture).has_value());
        REQUIRE(device.destroyTexture(albedo.texture).has_value());
        submitAndWait(device, std::move(*command));

        auto mapped = device.mapBuffer(*readback);
        REQUIRE(mapped);
        CHECK(pixelMatches(pixelAt(mapped->data, targetWidth / 2, targetHeight / 2), redBytes, 2));
        writePpm(evidenceDir() / "vulkan-hardware-textured.ppm", mapped->data);
        REQUIRE(device.unmapBuffer(*readback).has_value());
    }

    struct ComputeWritePipeline final
    {
        rachis::BindingLayoutHandle bindingLayout{};
        rachis::PipelineHandle pipeline{};
    };

    [[nodiscard]] ComputeWritePipeline createComputeWritePipeline(rachis::Device& device)
    {
        const auto bytecode = loadBytes(std::filesystem::path(RACHIS_RENDERER_SPIRV_DIR) / "VulkanComputeWrite.spv");
        REQUIRE_FALSE(bytecode.empty());
        auto shader = device.createShaderModule({
            .stage = rachis::ShaderStage::compute,
            .bytecode = bytecode,
            .entryPoint = "main",
            .shaderInterface = rachis::test::computeWriteShaderInterface(),
        });
        REQUIRE(shader);
        const std::array layoutEntries{
            rachis::BindingLayoutEntry{0, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute},
        };
        auto bindingLayout = device.createBindingLayout({.entries = layoutEntries});
        REQUIRE(bindingLayout);
        const std::array pushConstantRanges{
            rachis::PushConstantRange{rachis::ShaderStage::compute, 0, 2 * sizeof(std::uint32_t)},
        };
        auto pipelineLayout = device.createPipelineLayout({
            .bindingLayouts = std::array{*bindingLayout},
            .pushConstantRanges = pushConstantRanges,
        });
        REQUIRE(pipelineLayout);
        auto pipeline = device.createComputePipeline({.layout = *pipelineLayout, .computeShader = *shader});
        REQUIRE(pipeline);
        return {.bindingLayout = *bindingLayout, .pipeline = *pipeline};
    }

    [[nodiscard]] rachis::BindingSetHandle createStorageBindingSet(rachis::Device& device,
                                                                   rachis::BindingLayoutHandle bindingLayout,
                                                                   rachis::BufferHandle storage)
    {
        const std::array writes{
            rachis::BindingWrite{
                .binding = 0,
                .resource =
                    rachis::BufferBinding{
                        .buffer = storage,
                        .size = outputSize,
                        .structureByteStride = sizeof(std::uint32_t),
                    },
            },
        };
        auto bindingSet = rachis::test::createBindingSet(device, {.layout = bindingLayout, .writes = writes});
        REQUIRE(bindingSet);
        return *bindingSet;
    }

    void transitionToUnorderedAccess(rachis::Device& device,
                                     rachis::CommandBuffer& command,
                                     rachis::BufferHandle storage)
    {
        REQUIRE(device
                    .bufferBarrier(command,
                                   {
                                       .buffer = storage,
                                       .stagesAfter = rachis::PipelineStage::computeShader,
                                       .accessAfter = rachis::Access::unorderedAccess,
                                   })
                    .has_value());
    }

    void dispatchComputeWrite(rachis::Device& device,
                              rachis::CommandBuffer& command,
                              rachis::PipelineHandle pipeline,
                              rachis::BindingSetHandle storageSet)
    {
        REQUIRE(device.bindComputePipeline(command, pipeline).has_value());
        REQUIRE(device.bindComputeBindingSet(command, 0, storageSet).has_value());
        const std::array constants{3U, 7U};
        REQUIRE(device.setComputePushConstants(command, 0, std::as_bytes(std::span(constants))).has_value());
        REQUIRE(device.dispatch(command, 1).has_value());
    }

    void copyStorageToReadback(rachis::Device& device,
                               rachis::CommandBuffer& command,
                               rachis::BufferHandle storage,
                               rachis::BufferHandle readback)
    {
        REQUIRE(device
                    .bufferBarrier(command,
                                   {
                                       .buffer = storage,
                                       .stagesBefore = rachis::PipelineStage::computeShader,
                                       .stagesAfter = rachis::PipelineStage::copy,
                                       .accessBefore = rachis::Access::unorderedAccess,
                                       .accessAfter = rachis::Access::copySource,
                                   })
                    .has_value());
        REQUIRE(device
                    .copyBuffer(command,
                                {
                                    .source = storage,
                                    .destination = readback,
                                    .size = outputSize,
                                })
                    .has_value());
    }

    void checkComputeWrite(rachis::Device& device)
    {
        const auto computeWrite = createComputeWritePipeline(device);
        auto storage = device.createBuffer({
            .size = outputSize,
            .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::deviceLocal,
        });
        auto readback = createReadbackBuffer(device, outputSize);
        REQUIRE(storage);
        REQUIRE(readback);
        const auto storageSet = createStorageBindingSet(device, computeWrite.bindingLayout, *storage);

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
        REQUIRE(command);
        transitionToUnorderedAccess(device, *command, *storage);
        dispatchComputeWrite(device, *command, computeWrite.pipeline, storageSet);
        copyStorageToReadback(device, *command, *storage, *readback);
        REQUIRE(device.destroyBuffer(*storage).has_value());
        submitAndWait(device, std::move(*command));

        auto mapped = device.mapBuffer(*readback);
        REQUIRE(mapped);
        const auto* values = reinterpret_cast<const std::uint32_t*>(mapped->data);
        CHECK(values[0] == 7);
        CHECK(values[1] == 10);
        CHECK(values[63] == 196);
        REQUIRE(device.unmapBuffer(*readback).has_value());
    }

#if RACHIS_ENABLE_D3D12
    [[nodiscard]] bool pixelsDiffer(const std::byte* left, const std::byte* right, int tolerance) noexcept
    {
        for (std::uint32_t channel = 0; channel < 4; ++channel)
        {
            const auto leftValue = std::to_integer<std::uint8_t>(left[channel]);
            const auto rightValue = std::to_integer<std::uint8_t>(right[channel]);
            if (std::abs(static_cast<int>(leftValue) - static_cast<int>(rightValue)) > tolerance)
            {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] std::uint32_t countMismatchedPixels(const std::byte* leftPixels,
                                                      const std::byte* rightPixels,
                                                      int tolerance) noexcept
    {
        std::uint32_t mismatches = 0;
        for (std::uint32_t y = 0; y < targetHeight; ++y)
        {
            for (std::uint32_t x = 0; x < targetWidth; ++x)
            {
                if (pixelsDiffer(pixelAt(leftPixels, x, y), pixelAt(rightPixels, x, y), tolerance))
                {
                    ++mismatches;
                }
            }
        }
        return mismatches;
    }
#endif
} // namespace

TEST_CASE(
    "Vulkan hardware offscreenTriangle, depth, blend, clip, sRGB, texture, and compute match deterministic values",
    "[size:medium][rhi][vulkan][parity][hardware]")
{
    auto created =
        rachis::createDevice(deviceDesc(rachis::Backend::vulkan, rachis::AdapterPreference::highPerformance));
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    REQUIRE(device.adapterInfo().has_value());
    const auto adapter = *device.adapterInfo();
    CAPTURE(adapterLabel(adapter, false));
    REQUIRE(adapter.backend == rachis::Backend::vulkan);
    REQUIRE(adapter.type != rachis::AdapterType::software);
    REQUIRE_FALSE(adapter.name.contains("Dozen"));

    const auto shaders = vulkanTriangleShaders();
    const auto pixels =
        renderTintedTriangle(device, shaders, {.width = targetWidth, .height = targetHeight}, tint, false);
    requireOffscreenTriangle(pixels.data(), 2);
    writePpm(evidenceDir() / "vulkan-hardware-offscreenTriangle.ppm", pixels.data());

    checkDepthAndScissor(device, shaders);
    checkBlend(device, shaders);
    const auto clear = checkSrgbClear(device);
    checkTexturedDraw(device);
    checkComputeWrite(device);

    writeReport(evidenceDir() / "vulkan-hardware-report.txt",
                adapterLabel(adapter, false) + "\nfixture=offscreenTriangle tolerance=2\nlinearClear0.5=" +
                    std::to_string(clear.linear) + "\nsrgbClear0.5=" + std::to_string(clear.srgb) + "\n");
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

#if RACHIS_ENABLE_D3D12
TEST_CASE("D3D12 hardware baseline matches Vulkan offscreenTriangle and is labeled separately from WARP",
          "[size:medium][requires:HardwareD3D12][requires:HardwareVulkan][rhi][vulkan][parity][d3d12]")
{
    auto vulkan = tryCreate(deviceDesc(rachis::Backend::vulkan, rachis::AdapterPreference::highPerformance));
    auto d3d12 = tryCreate(deviceDesc(rachis::Backend::d3d12, rachis::AdapterPreference::highPerformance));
    REQUIRE(vulkan);
    REQUIRE(d3d12);
    REQUIRE(rachis::test::requireAdapterInfo(*vulkan).type != rachis::AdapterType::software);
    REQUIRE(rachis::test::requireAdapterInfo(*d3d12).backend == rachis::Backend::d3d12);
    CAPTURE(adapterLabel(*vulkan->adapterInfo(), false));
    CAPTURE(adapterLabel(*d3d12->adapterInfo(), false));

    const auto vulkanPixels = renderTintedTriangle(
        *vulkan, vulkanTriangleShaders(), {.width = targetWidth, .height = targetHeight}, tint, false);
    const auto d3d12Pixels = renderTintedTriangle(
        *d3d12, d3d12TriangleShaders(), {.width = targetWidth, .height = targetHeight}, tint, false);
    requireOffscreenTriangle(vulkanPixels.data(), 2);
    requireOffscreenTriangle(d3d12Pixels.data(), 2);
    writePpm(evidenceDir() / "d3d12-hardware-offscreenTriangle.ppm", d3d12Pixels.data());

    const auto mismatches = countMismatchedPixels(vulkanPixels.data(), d3d12Pixels.data(), 2);
    CHECK(mismatches == 0);

    auto warp = tryCreate(deviceDesc(rachis::Backend::d3d12, rachis::AdapterPreference::warp));
    REQUIRE(warp);
    REQUIRE(rachis::test::requireAdapterInfo(*warp).type == rachis::AdapterType::software);
    const auto warpPixels = renderTintedTriangle(
        *warp, d3d12TriangleShaders(), {.width = targetWidth, .height = targetHeight}, tint, false);
    requireOffscreenTriangle(warpPixels.data(), 2);
    writePpm(evidenceDir() / "d3d12-warp-offscreenTriangle.ppm", warpPixels.data());
    writeReport(evidenceDir() / "d3d12-warp-report.txt",
                adapterLabel(*warp->adapterInfo(), true) + "\nlabel=warp\nnot-vulkan-hardware-proof=true\n");

    auto vulkanWarp = rachis::createDevice(deviceDesc(rachis::Backend::vulkan, rachis::AdapterPreference::warp));
    REQUIRE_FALSE(vulkanWarp.has_value());
}
#endif
