#pragma once

#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace rachis::test::fill_and_clear
{

    struct TriangleShaders final
    {
        std::span<const std::byte> inputVertex;
        std::span<const std::byte> pixel;
    };

    inline constexpr std::uint32_t targetSize = 64;
    inline constexpr std::uint32_t readbackRowPitch = 256;

    [[nodiscard]] inline BufferHandle requireBuffer(Device& device, const BufferDesc& desc)
    {
        auto buffer = device.createBuffer(desc);
        INFO((buffer ? std::string{} : buffer.error().message));
        REQUIRE(buffer.has_value());
        return *buffer;
    }

    [[nodiscard]] inline std::vector<std::byte> readBytes(Device& device, BufferHandle buffer, std::size_t size)
    {
        auto mapped = device.mapBuffer(buffer);
        REQUIRE(mapped.has_value());
        std::vector<std::byte> bytes(size);
        std::memcpy(bytes.data(), mapped->data, size);
        REQUIRE(device.unmapBuffer(buffer).has_value());
        return bytes;
    }

    template<typename T>
    [[nodiscard]] T readAt(std::span<const std::byte> bytes, std::size_t offset)
    {
        T value{};
        REQUIRE(offset + sizeof(T) <= bytes.size());
        std::memcpy(&value, bytes.data() + offset, sizeof(T));
        return value;
    }

    inline void submitAndWait(Device& device, CommandBuffer&& command)
    {
        auto submitted = device.submit(std::move(command));
        INFO((submitted ? std::string{} : submitted.error().message));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());
    }

    inline void checkNoNativeErrors(const Device& device)
    {
        const auto diagnostics = device.diagnosticCounts();
        CHECK(diagnostics.errors == 0);
        CHECK(diagnostics.corruptions == 0);
    }

    inline void requireBufferBarrier(Device& device,
                                     CommandBuffer& command,
                                     BufferHandle buffer,
                                     PipelineStage stagesBefore,
                                     Access accessBefore,
                                     PipelineStage stagesAfter,
                                     Access accessAfter)
    {
        REQUIRE(device
                    .bufferBarrier(command,
                                   BufferBarrier{
                                       .buffer = buffer,
                                       .stagesBefore = stagesBefore,
                                       .stagesAfter = stagesAfter,
                                       .accessBefore = accessBefore,
                                       .accessAfter = accessAfter,
                                   })
                    .has_value());
    }

    inline void requireTextureBarrier(Device& device,
                                      CommandBuffer& command,
                                      TextureHandle texture,
                                      PipelineStage stagesBefore,
                                      Access accessBefore,
                                      TextureLayout layoutBefore,
                                      PipelineStage stagesAfter,
                                      Access accessAfter,
                                      TextureLayout layoutAfter)
    {
        REQUIRE(device
                    .textureBarrier(command,
                                    TextureBarrier{
                                        .texture = texture,
                                        .stagesBefore = stagesBefore,
                                        .stagesAfter = stagesAfter,
                                        .accessBefore = accessBefore,
                                        .accessAfter = accessAfter,
                                        .layoutBefore = layoutBefore,
                                        .layoutAfter = layoutAfter,
                                    })
                    .has_value());
    }

    inline void requireUnorderedAccessFillBarrier(Device& device,
                                                  CommandBuffer& command,
                                                  TextureHandle texture,
                                                  const TextureSubresourceRange& subresources)
    {
        REQUIRE(device
                    .textureBarrier(command,
                                    TextureBarrier{
                                        .texture = texture,
                                        .stagesBefore = PipelineStage::computeShader,
                                        .stagesAfter = PipelineStage::computeShader,
                                        .accessBefore = Access::unorderedAccess,
                                        .accessAfter = Access::unorderedAccess,
                                        .layoutBefore = TextureLayout::unorderedAccess,
                                        .layoutAfter = TextureLayout::unorderedAccess,
                                        .subresources = subresources,
                                    })
                    .has_value());
    }

    [[nodiscard]] inline bool isRed(std::span<const std::byte> pixels, std::uint32_t x, std::uint32_t y) noexcept
    {
        const auto offset = static_cast<std::size_t>(y) * readbackRowPitch + static_cast<std::size_t>(x) * 4U;
        return std::to_integer<std::uint8_t>(pixels[offset]) > 180 &&
               std::to_integer<std::uint8_t>(pixels[offset + 1]) < 100 &&
               std::to_integer<std::uint8_t>(pixels[offset + 2]) < 100;
    }

    [[nodiscard]] inline bool isBlank(std::span<const std::byte> pixels, std::uint32_t x, std::uint32_t y) noexcept
    {
        const auto offset = static_cast<std::size_t>(y) * readbackRowPitch + static_cast<std::size_t>(x) * 4U;
        return std::to_integer<std::uint8_t>(pixels[offset]) < 4 &&
               std::to_integer<std::uint8_t>(pixels[offset + 1]) < 4 &&
               std::to_integer<std::uint8_t>(pixels[offset + 2]) < 4;
    }

    inline void verifyCountedDrawAfterFill(Device& device, const TriangleShaders& shaders)
    {
        const auto capabilities = requireCapabilities(device);
        if (!capabilities.drawIndirectCount)
        {
            SKIP("The device does not support counted indirect draws.");
        }

        auto vertexShader = device.createShaderModule(ShaderModuleDesc{
            .stage = ShaderStage::vertex,
            .bytecode = shaders.inputVertex,
            .entryPoint = "inputVertex",
            .shaderInterface = graphicsInputVertexShaderInterface(),
        });
        auto pixelShader = device.createShaderModule(ShaderModuleDesc{
            .stage = ShaderStage::pixel,
            .bytecode = shaders.pixel,
            .entryPoint = "pixelMain",
            .shaderInterface = graphicsPixelShaderInterface(),
        });
        REQUIRE(vertexShader.has_value());
        REQUIRE(pixelShader.has_value());
        const std::array pushConstantRanges{
            PushConstantRange{ShaderStage::pixel, 0, 4 * sizeof(float)},
        };
        auto layout = device.createPipelineLayout(PipelineLayoutDesc{.pushConstantRanges = pushConstantRanges});
        REQUIRE(layout.has_value());
        const std::array colorTargets{ColorTargetState{.format = Format::rgba8Unorm}};
        const std::array vertexAttributes{
            VertexAttribute{.format = VertexFormat::float32x2, .shaderLocation = 0},
        };
        const std::array vertexLayouts{
            VertexBufferLayout{.stride = 2 * sizeof(float), .attributes = vertexAttributes},
        };
        auto pipeline = device.createGraphicsPipeline(GraphicsPipelineDesc{
            .layout = *layout,
            .vertexShader = *vertexShader,
            .pixelShader = *pixelShader,
            .vertexBuffers = vertexLayouts,
            .cullMode = CullMode::none,
            .colorTargets = colorTargets,
        });
        REQUIRE(pipeline.has_value());

        auto target = device.createTexture(TextureDesc{
            .width = targetSize,
            .height = targetSize,
            .format = Format::rgba8Unorm,
            .usage = TextureUsage::copySource | TextureUsage::renderTarget,
        });
        REQUIRE(target.has_value());
        auto targetView = device.createTextureView(TextureViewDesc{
            .texture = *target,
            .usage = TextureViewUsage::renderTarget,
        });
        REQUIRE(targetView.has_value());

        const std::array vertices{std::array{0.0F, 0.8F}, std::array{0.8F, -0.8F}, std::array{-0.8F, -0.8F}};
        const auto vertexBuffer = requireBuffer(device,
                                                BufferDesc{
                                                    .size = sizeof(vertices),
                                                    .usage = BufferUsage::vertex,
                                                    .memoryType = MemoryType::upload,
                                                });
        const DrawIndirectArguments draw{.vertexCount = 3, .instanceCount = 1};
        const auto arguments = requireBuffer(device,
                                             BufferDesc{
                                                 .size = sizeof(draw),
                                                 .usage = BufferUsage::indirect,
                                                 .memoryType = MemoryType::upload,
                                             });
        {
            auto mappedVertices = device.mapBuffer(vertexBuffer);
            auto mappedArguments = device.mapBuffer(arguments);
            REQUIRE(mappedVertices.has_value());
            REQUIRE(mappedArguments.has_value());
            std::memcpy(mappedVertices->data, vertices.data(), sizeof(vertices));
            std::memcpy(mappedArguments->data, &draw, sizeof(draw));
            REQUIRE(device.unmapBuffer(vertexBuffer).has_value());
            REQUIRE(device.unmapBuffer(arguments).has_value());
        }

        constexpr std::uint64_t countBytes = 4 * sizeof(std::uint32_t);
        const auto count = requireBuffer(device,
                                         BufferDesc{
                                             .size = countBytes,
                                             .usage = BufferUsage::storage | BufferUsage::copyDestination |
                                                      BufferUsage::copySource | BufferUsage::indirect,
                                             .label = "Fill contract count buffer",
                                         });
        constexpr std::uint64_t imageBytes = std::uint64_t{readbackRowPitch} * targetSize;
        const std::array drawnReadbacks{
            requireBuffer(device,
                          BufferDesc{
                              .size = imageBytes,
                              .usage = BufferUsage::copyDestination,
                              .memoryType = MemoryType::readback,
                          }),
            requireBuffer(device,
                          BufferDesc{
                              .size = imageBytes,
                              .usage = BufferUsage::copyDestination,
                              .memoryType = MemoryType::readback,
                          }),
        };
        const auto countReadback = requireBuffer(device,
                                                 BufferDesc{
                                                     .size = countBytes,
                                                     .usage = BufferUsage::copyDestination,
                                                     .memoryType = MemoryType::readback,
                                                 });

        auto command = acquireCommandBuffer(device, QueueType::graphics);
        REQUIRE(command.has_value());
        const auto renderCountedDraw = [&](std::size_t pass) {
            requireTextureBarrier(device,
                                  *command,
                                  *target,
                                  pass == 0 ? PipelineStage::all : PipelineStage::copy,
                                  pass == 0 ? Access::none : Access::copySource,
                                  pass == 0 ? TextureLayout::undefined : TextureLayout::copySource,
                                  PipelineStage::renderTarget,
                                  Access::renderTarget,
                                  TextureLayout::renderTarget);
            const std::array attachments{
                ColorAttachment{
                    .view = *targetView,
                    .loadOperation = LoadOperation::clear,
                    .clearValue = {},
                },
            };
            REQUIRE(device.beginRendering(*command, RenderingDesc{.colorAttachments = attachments}).has_value());
            const auto insideRendering = device.fillBuffer(*command, count, 0, countBytes, 0);
            REQUIRE_FALSE(insideRendering.has_value());
            CHECK(insideRendering.error().code == ErrorCode::invalidState);
            REQUIRE(device.bindGraphicsPipeline(*command, *pipeline).has_value());
            REQUIRE(device.bindVertexBuffer(*command, 0, vertexBuffer).has_value());
            REQUIRE(device
                        .setViewport(*command,
                                     Viewport{
                                         .width = static_cast<float>(targetSize),
                                         .height = static_cast<float>(targetSize),
                                     })
                        .has_value());
            REQUIRE(device.setScissor(*command, ScissorRect{.width = targetSize, .height = targetSize}).has_value());
            const std::array tint{0.9F, 0.2F, 0.1F, 1.0F};
            REQUIRE(device.setGraphicsPushConstants(*command, ShaderStage::pixel, 0, std::as_bytes(std::span(tint)))
                        .has_value());
            REQUIRE(device
                        .drawIndirect(*command,
                                      IndirectDrawDesc{
                                          .argumentBuffer = arguments,
                                          .maxDrawCount = 1,
                                          .countBuffer = count,
                                          .countOffset = sizeof(std::uint32_t),
                                      })
                        .has_value());
            REQUIRE(device.endRendering(*command).has_value());
            requireTextureBarrier(device,
                                  *command,
                                  *target,
                                  PipelineStage::renderTarget,
                                  Access::renderTarget,
                                  TextureLayout::renderTarget,
                                  PipelineStage::copy,
                                  Access::copySource,
                                  TextureLayout::copySource);
            REQUIRE(device
                        .copyTextureToBuffer(*command,
                                             BufferTextureCopy{
                                                 .buffer = drawnReadbacks[pass],
                                                 .bytesPerRow = readbackRowPitch,
                                                 .rowsPerImage = targetSize,
                                                 .texture = *target,
                                                 .extent = {targetSize, targetSize, 1},
                                             })
                        .has_value());
        };

        requireBufferBarrier(device,
                             *command,
                             count,
                             PipelineStage::all,
                             Access::common,
                             PipelineStage::computeShader,
                             Access::unorderedAccess);
        REQUIRE(device.fillBuffer(*command, count, 0, wholeResource, 1).has_value());
        requireBufferBarrier(device,
                             *command,
                             count,
                             PipelineStage::computeShader,
                             Access::unorderedAccess,
                             PipelineStage::indirect,
                             Access::indirectArgument);
        renderCountedDraw(0);

        requireBufferBarrier(device,
                             *command,
                             count,
                             PipelineStage::indirect,
                             Access::indirectArgument,
                             PipelineStage::computeShader,
                             Access::unorderedAccess);
        REQUIRE(device.fillBuffer(*command, count, 0, countBytes, 0).has_value());
        requireBufferBarrier(device,
                             *command,
                             count,
                             PipelineStage::computeShader,
                             Access::unorderedAccess,
                             PipelineStage::indirect,
                             Access::indirectArgument);
        renderCountedDraw(1);

        requireBufferBarrier(device,
                             *command,
                             count,
                             PipelineStage::indirect,
                             Access::indirectArgument,
                             PipelineStage::computeShader,
                             Access::unorderedAccess);
        REQUIRE(device.fillBuffer(*command, count, 4, 8, 0xA5A5A5A5U).has_value());
        requireBufferBarrier(device,
                             *command,
                             count,
                             PipelineStage::computeShader,
                             Access::unorderedAccess,
                             PipelineStage::copy,
                             Access::copySource);
        REQUIRE(
            device.copyBuffer(*command, BufferCopy{.source = count, .destination = countReadback, .size = countBytes})
                .has_value());
        submitAndWait(device, std::move(*command));

        const auto onePass = readBytes(device, drawnReadbacks[0], imageBytes);
        const auto zeroPass = readBytes(device, drawnReadbacks[1], imageBytes);
        CHECK(isRed(onePass, targetSize / 2, targetSize / 2));
        CHECK(isBlank(onePass, 1, 1));
        CHECK(isBlank(zeroPass, targetSize / 2, targetSize / 2));
        CHECK(isBlank(zeroPass, 1, 1));
        const auto counts = readBytes(device, countReadback, countBytes);
        CHECK(readAt<std::uint32_t>(counts, 0) == 0U);
        CHECK(readAt<std::uint32_t>(counts, 4) == 0xA5A5A5A5U);
        CHECK(readAt<std::uint32_t>(counts, 8) == 0xA5A5A5A5U);
        CHECK(readAt<std::uint32_t>(counts, 12) == 0U);
        checkNoNativeErrors(device);

        REQUIRE(device.destroyBuffer(countReadback).has_value());
        for (const auto readback : drawnReadbacks)
        {
            REQUIRE(device.destroyBuffer(readback).has_value());
        }
        REQUIRE(device.destroyBuffer(count).has_value());
        REQUIRE(device.destroyBuffer(arguments).has_value());
        REQUIRE(device.destroyBuffer(vertexBuffer).has_value());
        REQUIRE(device.destroyTextureView(*targetView).has_value());
        REQUIRE(device.destroyTexture(*target).has_value());
        REQUIRE(device.destroyPipeline(*pipeline).has_value());
        REQUIRE(device.destroyPipelineLayout(*layout).has_value());
        REQUIRE(device.destroyShaderModule(*pixelShader).has_value());
        REQUIRE(device.destroyShaderModule(*vertexShader).has_value());
    }

    inline void verifyLargeBufferFill(Device& device, QueueType queueType)
    {
        constexpr std::uint64_t size = 3ULL * 1024ULL * 1024ULL + 64ULL;
        constexpr std::uint64_t offset = 4;
        const auto buffer =
            requireBuffer(device,
                          BufferDesc{
                              .size = size,
                              .usage = BufferUsage::storage | BufferUsage::copyDestination | BufferUsage::copySource,
                          });
        const auto readback = requireBuffer(device,
                                            BufferDesc{
                                                .size = size,
                                                .usage = BufferUsage::copyDestination,
                                                .memoryType = MemoryType::readback,
                                            });
        auto command = acquireCommandBuffer(device, queueType);
        REQUIRE(command.has_value());
        requireBufferBarrier(device,
                             *command,
                             buffer,
                             PipelineStage::all,
                             Access::common,
                             PipelineStage::computeShader,
                             Access::unorderedAccess);
        REQUIRE(device.fillBuffer(*command, buffer, 0, wholeResource, 0x01020304U).has_value());
        requireBufferBarrier(device,
                             *command,
                             buffer,
                             PipelineStage::computeShader,
                             Access::unorderedAccess,
                             PipelineStage::computeShader,
                             Access::unorderedAccess);
        REQUIRE(device.fillBuffer(*command, buffer, offset, size - 2 * offset, 0xCAFEF00DU).has_value());
        requireBufferBarrier(device,
                             *command,
                             buffer,
                             PipelineStage::computeShader,
                             Access::unorderedAccess,
                             PipelineStage::copy,
                             Access::copySource);
        REQUIRE(device.copyBuffer(*command, BufferCopy{.source = buffer, .destination = readback, .size = size})
                    .has_value());
        submitAndWait(device, std::move(*command));

        const auto bytes = readBytes(device, readback, size);
        CHECK(readAt<std::uint32_t>(bytes, 0) == 0x01020304U);
        CHECK(readAt<std::uint32_t>(bytes, size - 4) == 0x01020304U);
        std::size_t mismatches = 0;
        for (std::uint64_t word = offset; word < size - offset; word += 4)
        {
            mismatches += readAt<std::uint32_t>(bytes, word) == 0xCAFEF00DU ? 0U : 1U;
        }
        CHECK(mismatches == 0);
        checkNoNativeErrors(device);
        REQUIRE(device.destroyBuffer(readback).has_value());
        REQUIRE(device.destroyBuffer(buffer).has_value());
    }

    struct ClearedTexture final
    {
        TextureHandle texture{};
        TextureDesc desc{};
    };

    [[nodiscard]] inline ClearedTexture createStorageTexture(Device& device, TextureDesc desc)
    {
        desc.usage = desc.usage | TextureUsage::storage | TextureUsage::copyDestination | TextureUsage::copySource;
        auto texture = device.createTexture(desc);
        INFO((texture ? std::string{} : texture.error().message));
        REQUIRE(texture.has_value());
        return ClearedTexture{.texture = *texture, .desc = desc};
    }

    [[nodiscard]] inline TextureViewHandle requireStorageView(Device& device,
                                                              TextureHandle texture,
                                                              TextureViewDimension dimension,
                                                              std::uint32_t mipLevel,
                                                              std::uint32_t firstLayer = 0,
                                                              std::uint32_t layerCount = 1)
    {
        auto view = device.createTextureView(TextureViewDesc{
            .texture = texture,
            .usage = TextureViewUsage::storage,
            .dimension = dimension,
            .firstMipLevel = mipLevel,
            .mipLevelCount = 1,
            .firstArrayLayer = firstLayer,
            .arrayLayerCount = layerCount,
        });
        INFO((view ? std::string{} : view.error().message));
        REQUIRE(view.has_value());
        return *view;
    }

    struct SubresourceReadback final
    {
        BufferHandle buffer{};
        std::uint32_t width = 1;
        std::uint32_t height = 1;
        std::uint32_t depth = 1;
        std::uint64_t size = 0;
    };

    [[nodiscard]] inline SubresourceReadback recordSubresourceReadback(Device& device,
                                                                       CommandBuffer& command,
                                                                       const ClearedTexture& texture,
                                                                       std::uint32_t mipLevel,
                                                                       std::uint32_t arrayLayer)
    {
        const auto width = std::max(1U, texture.desc.width >> mipLevel);
        const auto height = std::max(1U, texture.desc.height >> mipLevel);
        const auto depth = texture.desc.dimension == TextureDimension::texture3D
                               ? std::max(1U, texture.desc.depthOrArrayLayers >> mipLevel)
                               : 1U;
        const std::uint64_t size = std::uint64_t{readbackRowPitch} * height * depth;
        const auto buffer = requireBuffer(device,
                                          BufferDesc{
                                              .size = size,
                                              .usage = BufferUsage::copyDestination,
                                              .memoryType = MemoryType::readback,
                                          });
        REQUIRE(device
                    .copyTextureToBuffer(command,
                                         BufferTextureCopy{
                                             .buffer = buffer,
                                             .bytesPerRow = readbackRowPitch,
                                             .rowsPerImage = height,
                                             .texture = texture.texture,
                                             .textureSubresource = {.mipLevel = mipLevel, .arrayLayer = arrayLayer},
                                             .extent = {width, height, depth},
                                         })
                    .has_value());
        return SubresourceReadback{.buffer = buffer, .width = width, .height = height, .depth = depth, .size = size};
    }

    template<typename Texel>
    [[nodiscard]] inline std::size_t countMismatchedTexels(Device& device,
                                                           const SubresourceReadback& readback,
                                                           const Texel& expected)
    {
        const auto bytes = readBytes(device, readback.buffer, readback.size);
        std::size_t mismatches = 0;
        for (std::uint32_t z = 0; z < readback.depth; ++z)
        {
            for (std::uint32_t y = 0; y < readback.height; ++y)
            {
                for (std::uint32_t x = 0; x < readback.width; ++x)
                {
                    const auto offset = (static_cast<std::size_t>(z) * readback.height + y) * readbackRowPitch +
                                        static_cast<std::size_t>(x) * sizeof(Texel);
                    if (std::memcmp(bytes.data() + offset, &expected, sizeof(Texel)) != 0)
                    {
                        ++mismatches;
                    }
                }
            }
        }
        REQUIRE(device.destroyBuffer(readback.buffer).has_value());
        return mismatches;
    }

    [[nodiscard]] inline StorageTextureClearValue floatClear(float red,
                                                             float green = 0,
                                                             float blue = 0,
                                                             float alpha = 0)
    {
        return StorageTextureClearValue{
            .type = StorageTextureClearType::float32,
            .float32 = {red, green, blue, alpha},
        };
    }

    [[nodiscard]] inline StorageTextureClearValue uintClear(std::uint32_t red,
                                                            std::uint32_t green = 0,
                                                            std::uint32_t blue = 0,
                                                            std::uint32_t alpha = 0)
    {
        return StorageTextureClearValue{
            .type = StorageTextureClearType::uint32,
            .uint32 = {red, green, blue, alpha},
        };
    }

    [[nodiscard]] inline StorageTextureClearValue sintClear(std::int32_t red,
                                                            std::int32_t green = 0,
                                                            std::int32_t blue = 0,
                                                            std::int32_t alpha = 0)
    {
        return StorageTextureClearValue{
            .type = StorageTextureClearType::sint32,
            .sint32 = {red, green, blue, alpha},
        };
    }

    inline void verifyStorageTextureClears(Device& device, QueueType queueType)
    {
        const auto layered = createStorageTexture(device,
                                                  TextureDesc{
                                                      .width = 8,
                                                      .height = 8,
                                                      .depthOrArrayLayers = 3,
                                                      .mipLevels = 2,
                                                      .format = Format::r32Uint,
                                                  });
        const auto normalized = createStorageTexture(device,
                                                     TextureDesc{
                                                         .width = 4,
                                                         .height = 4,
                                                         .format = Format::rgba8Unorm,
                                                     });
        const auto floating = createStorageTexture(device,
                                                   TextureDesc{
                                                       .width = 4,
                                                       .height = 4,
                                                       .format = Format::r32Float,
                                                   });
        const auto half = createStorageTexture(device,
                                               TextureDesc{
                                                   .width = 4,
                                                   .height = 4,
                                                   .format = Format::rgba16Float,
                                               });
        const auto signedWide = createStorageTexture(device,
                                                     TextureDesc{
                                                         .width = 4,
                                                         .height = 4,
                                                         .format = Format::r32Sint,
                                                     });
        const auto signedNarrow = createStorageTexture(device,
                                                       TextureDesc{
                                                           .width = 4,
                                                           .height = 4,
                                                           .format = Format::rgba8Sint,
                                                       });
        const auto volume = createStorageTexture(device,
                                                 TextureDesc{
                                                     .dimension = TextureDimension::texture3D,
                                                     .width = 4,
                                                     .height = 4,
                                                     .depthOrArrayLayers = 2,
                                                     .format = Format::r32Uint,
                                                 });
        const std::array textures{layered, normalized, floating, half, signedWide, signedNarrow, volume};

        const auto wholeArray =
            requireStorageView(device, layered.texture, TextureViewDimension::texture2DArray, 0, 0, 3);
        const auto middleLayer =
            requireStorageView(device, layered.texture, TextureViewDimension::texture2DArray, 0, 1, 1);
        const auto secondMip =
            requireStorageView(device, layered.texture, TextureViewDimension::texture2DArray, 1, 0, 3);
        const auto normalizedView = requireStorageView(device, normalized.texture, TextureViewDimension::texture2D, 0);
        const auto floatingView = requireStorageView(device, floating.texture, TextureViewDimension::texture2D, 0);
        const auto halfView = requireStorageView(device, half.texture, TextureViewDimension::texture2D, 0);
        const auto signedWideView = requireStorageView(device, signedWide.texture, TextureViewDimension::texture2D, 0);
        const auto signedNarrowView =
            requireStorageView(device, signedNarrow.texture, TextureViewDimension::texture2D, 0);
        const auto volumeView = requireStorageView(device, volume.texture, TextureViewDimension::texture3D, 0);
        const std::array views{
            wholeArray,
            middleLayer,
            secondMip,
            normalizedView,
            floatingView,
            halfView,
            signedWideView,
            signedNarrowView,
            volumeView,
        };

        auto command = acquireCommandBuffer(device, queueType);
        REQUIRE(command.has_value());
        for (const auto& texture : textures)
        {
            requireTextureBarrier(device,
                                  *command,
                                  texture.texture,
                                  PipelineStage::all,
                                  Access::none,
                                  TextureLayout::undefined,
                                  PipelineStage::computeShader,
                                  Access::unorderedAccess,
                                  TextureLayout::unorderedAccess);
        }
        REQUIRE(device.clearStorageTexture(*command, wholeArray, uintClear(7)).has_value());
        requireUnorderedAccessFillBarrier(device, *command, layered.texture, {.firstMipLevel = 0, .mipLevelCount = 1});
        REQUIRE(device.clearStorageTexture(*command, middleLayer, uintClear(0xDEADBEEFU)).has_value());
        REQUIRE(device.clearStorageTexture(*command, secondMip, uintClear(42)).has_value());
        REQUIRE(device.clearStorageTexture(*command, normalizedView, floatClear(1.0F, 0.2F, 0.0F, 0.6F)).has_value());
        REQUIRE(device.clearStorageTexture(*command, floatingView, floatClear(-2.5F)).has_value());
        REQUIRE(device.clearStorageTexture(*command, halfView, floatClear(0.5F, -1.0F, 2.0F, 0.25F)).has_value());
        REQUIRE(device.clearStorageTexture(*command, signedWideView, sintClear(-5)).has_value());
        REQUIRE(device.clearStorageTexture(*command, signedNarrowView, sintClear(-128, 127, -1, 3)).has_value());
        REQUIRE(device.clearStorageTexture(*command, volumeView, uintClear(9)).has_value());
        for (const auto& texture : textures)
        {
            requireTextureBarrier(device,
                                  *command,
                                  texture.texture,
                                  PipelineStage::computeShader,
                                  Access::unorderedAccess,
                                  TextureLayout::unorderedAccess,
                                  PipelineStage::copy,
                                  Access::copySource,
                                  TextureLayout::copySource);
        }
        const std::array layeredReadbacks{
            recordSubresourceReadback(device, *command, layered, 0, 0),
            recordSubresourceReadback(device, *command, layered, 0, 1),
            recordSubresourceReadback(device, *command, layered, 0, 2),
            recordSubresourceReadback(device, *command, layered, 1, 0),
            recordSubresourceReadback(device, *command, layered, 1, 1),
            recordSubresourceReadback(device, *command, layered, 1, 2),
        };
        const auto normalizedReadback = recordSubresourceReadback(device, *command, normalized, 0, 0);
        const auto floatingReadback = recordSubresourceReadback(device, *command, floating, 0, 0);
        const auto halfReadback = recordSubresourceReadback(device, *command, half, 0, 0);
        const auto signedWideReadback = recordSubresourceReadback(device, *command, signedWide, 0, 0);
        const auto signedNarrowReadback = recordSubresourceReadback(device, *command, signedNarrow, 0, 0);
        const auto volumeReadback = recordSubresourceReadback(device, *command, volume, 0, 0);
        submitAndWait(device, std::move(*command));

        CHECK(countMismatchedTexels(device, layeredReadbacks[0], std::uint32_t{7}) == 0);
        CHECK(countMismatchedTexels(device, layeredReadbacks[1], std::uint32_t{0xDEADBEEFU}) == 0);
        CHECK(countMismatchedTexels(device, layeredReadbacks[2], std::uint32_t{7}) == 0);
        for (std::size_t layer = 3; layer < layeredReadbacks.size(); ++layer)
        {
            CHECK(countMismatchedTexels(device, layeredReadbacks[layer], std::uint32_t{42}) == 0);
        }
        CHECK(countMismatchedTexels(device, normalizedReadback, std::array<std::uint8_t, 4>{255, 51, 0, 153}) == 0);
        CHECK(countMismatchedTexels(device, floatingReadback, -2.5F) == 0);
        CHECK(countMismatchedTexels(
                  device, halfReadback, std::array<std::uint16_t, 4>{0x3800, 0xBC00, 0x4000, 0x3400}) == 0);
        CHECK(countMismatchedTexels(device, signedWideReadback, std::int32_t{-5}) == 0);
        CHECK(countMismatchedTexels(device, signedNarrowReadback, std::array<std::int8_t, 4>{-128, 127, -1, 3}) == 0);
        CHECK(countMismatchedTexels(device, volumeReadback, std::uint32_t{9}) == 0);
        checkNoNativeErrors(device);

        for (const auto view : views)
        {
            REQUIRE(device.destroyTextureView(view).has_value());
        }
        for (const auto& texture : textures)
        {
            REQUIRE(device.destroyTexture(texture.texture).has_value());
        }
    }

    inline void verifyFillAndClearRejections(Device& device)
    {
        const auto storageOnly = requireBuffer(device, BufferDesc{.size = 16, .usage = BufferUsage::storage});
        const auto copyOnly = requireBuffer(
            device, BufferDesc{.size = 16, .usage = BufferUsage::copyDestination | BufferUsage::indirect});
        const auto fillable =
            requireBuffer(device, BufferDesc{.size = 16, .usage = BufferUsage::storage | BufferUsage::copyDestination});
        const auto ragged =
            requireBuffer(device, BufferDesc{.size = 18, .usage = BufferUsage::storage | BufferUsage::copyDestination});
        const auto stale =
            requireBuffer(device, BufferDesc{.size = 16, .usage = BufferUsage::storage | BufferUsage::copyDestination});
        REQUIRE(device.destroyBuffer(stale).has_value());

        const auto storageUint = createStorageTexture(device,
                                                      TextureDesc{
                                                          .width = 4,
                                                          .height = 4,
                                                          .format = Format::rgba8Uint,
                                                          .usage = TextureUsage::sampled,
                                                      });
        const auto signedNarrow = createStorageTexture(device,
                                                       TextureDesc{
                                                           .width = 4,
                                                           .height = 4,
                                                           .format = Format::rgba8Sint,
                                                       });
        auto withoutCopyDestination = device.createTexture(TextureDesc{
            .width = 4,
            .height = 4,
            .format = Format::r32Uint,
            .usage = TextureUsage::storage | TextureUsage::copySource,
        });
        REQUIRE(withoutCopyDestination.has_value());
        const auto storageView = requireStorageView(device, storageUint.texture, TextureViewDimension::texture2D, 0);
        const auto signedView = requireStorageView(device, signedNarrow.texture, TextureViewDimension::texture2D, 0);
        const auto noCopyView = requireStorageView(device, *withoutCopyDestination, TextureViewDimension::texture2D, 0);
        auto sampledView = device.createTextureView(TextureViewDesc{
            .texture = storageUint.texture,
            .usage = TextureViewUsage::sampled,
        });
        REQUIRE(sampledView.has_value());
        const auto staleView = requireStorageView(device, signedNarrow.texture, TextureViewDimension::texture2D, 0);
        REQUIRE(device.destroyTextureView(staleView).has_value());

        const auto expectCode = [](const Status& status, ErrorCode code) {
            REQUIRE_FALSE(status.has_value());
            INFO(status.error().message);
            CHECK(status.error().code == code);
        };

        auto copyCommand = acquireCommandBuffer(device, QueueType::copy);
        REQUIRE(copyCommand.has_value());
        expectCode(device.fillBuffer(*copyCommand, fillable, 0, 16, 0), ErrorCode::invalidState);
        expectCode(device.clearStorageTexture(*copyCommand, storageView, uintClear(1)), ErrorCode::invalidState);

        auto command = acquireCommandBuffer(device, QueueType::compute);
        REQUIRE(command.has_value());
        expectCode(device.fillBuffer(*command, storageOnly, 0, 16, 0), ErrorCode::invalidArgument);
        expectCode(device.fillBuffer(*command, copyOnly, 0, 16, 0), ErrorCode::invalidArgument);
        expectCode(device.fillBuffer(*command, fillable, 2, 8, 0), ErrorCode::invalidArgument);
        expectCode(device.fillBuffer(*command, fillable, 0, 6, 0), ErrorCode::invalidArgument);
        expectCode(device.fillBuffer(*command, fillable, 0, 0, 0), ErrorCode::invalidArgument);
        expectCode(device.fillBuffer(*command, fillable, 16, wholeResource, 0), ErrorCode::invalidArgument);
        expectCode(device.fillBuffer(*command, fillable, 8, 12, 0), ErrorCode::invalidArgument);
        expectCode(device.fillBuffer(*command, ragged, 0, wholeResource, 0), ErrorCode::invalidArgument);
        expectCode(device.fillBuffer(*command, stale, 0, 16, 0), ErrorCode::invalidArgument);
        requireBufferBarrier(device,
                             *command,
                             ragged,
                             PipelineStage::all,
                             Access::common,
                             PipelineStage::computeShader,
                             Access::unorderedAccess);
        REQUIRE(device.fillBuffer(*command, ragged, 0, 16, 0).has_value());

        requireTextureBarrier(device,
                              *command,
                              storageUint.texture,
                              PipelineStage::all,
                              Access::none,
                              TextureLayout::undefined,
                              PipelineStage::computeShader,
                              Access::unorderedAccess,
                              TextureLayout::unorderedAccess);
        requireTextureBarrier(device,
                              *command,
                              signedNarrow.texture,
                              PipelineStage::all,
                              Access::none,
                              TextureLayout::undefined,
                              PipelineStage::computeShader,
                              Access::unorderedAccess,
                              TextureLayout::unorderedAccess);
        requireTextureBarrier(device,
                              *command,
                              *withoutCopyDestination,
                              PipelineStage::all,
                              Access::none,
                              TextureLayout::undefined,
                              PipelineStage::computeShader,
                              Access::unorderedAccess,
                              TextureLayout::unorderedAccess);
        expectCode(device.clearStorageTexture(*command, *sampledView, uintClear(1)), ErrorCode::invalidArgument);
        expectCode(device.clearStorageTexture(*command, noCopyView, uintClear(1)), ErrorCode::invalidArgument);
        expectCode(device.clearStorageTexture(*command, storageView, floatClear(1.0F)), ErrorCode::invalidArgument);
        expectCode(device.clearStorageTexture(*command, storageView, sintClear(1)), ErrorCode::invalidArgument);
        expectCode(device.clearStorageTexture(*command, storageView, uintClear(255, 256)), ErrorCode::invalidArgument);
        expectCode(device.clearStorageTexture(*command, signedView, sintClear(-129)), ErrorCode::invalidArgument);
        expectCode(device.clearStorageTexture(*command, signedView, sintClear(0, 128)), ErrorCode::invalidArgument);
        expectCode(
            device.clearStorageTexture(*command, signedView, floatClear(std::numeric_limits<float>::quiet_NaN())),
            ErrorCode::invalidArgument);
        auto unknownType = uintClear(1);
        unknownType.type = static_cast<StorageTextureClearType>(9);
        expectCode(device.clearStorageTexture(*command, storageView, unknownType), ErrorCode::invalidArgument);
        expectCode(device.clearStorageTexture(*command, staleView, sintClear(1)), ErrorCode::invalidArgument);
        REQUIRE(device.clearStorageTexture(*command, storageView, uintClear(255, 0, 1, 2)).has_value());
        REQUIRE(device.clearStorageTexture(*command, signedView, sintClear(-128, 127)).has_value());

        requireTextureBarrier(device,
                              *command,
                              storageUint.texture,
                              PipelineStage::computeShader,
                              Access::unorderedAccess,
                              TextureLayout::unorderedAccess,
                              PipelineStage::computeShader,
                              Access::shaderResource,
                              TextureLayout::shaderResource);
        expectCode(device.clearStorageTexture(*command, storageView, uintClear(1)), ErrorCode::invalidState);
        submitAndWait(device, std::move(*command));

        auto untransitioned = acquireCommandBuffer(device, QueueType::graphics);
        REQUIRE(untransitioned.has_value());
        const auto recorded = device.clearStorageTexture(*untransitioned, storageView, uintClear(1));
        if (recorded)
        {
            auto submitted = device.submit(std::move(*untransitioned));
            REQUIRE_FALSE(submitted.has_value());
            CHECK(submitted.error().code == ErrorCode::invalidState);
        }
        else
        {
            CHECK(recorded.error().code == ErrorCode::invalidState);
        }
        checkNoNativeErrors(device);

        REQUIRE(device.destroyTextureView(*sampledView).has_value());
        REQUIRE(device.destroyTextureView(noCopyView).has_value());
        REQUIRE(device.destroyTextureView(signedView).has_value());
        REQUIRE(device.destroyTextureView(storageView).has_value());
        REQUIRE(device.destroyTexture(*withoutCopyDestination).has_value());
        REQUIRE(device.destroyTexture(signedNarrow.texture).has_value());
        REQUIRE(device.destroyTexture(storageUint.texture).has_value());
        REQUIRE(device.destroyBuffer(ragged).has_value());
        REQUIRE(device.destroyBuffer(fillable).has_value());
        REQUIRE(device.destroyBuffer(copyOnly).has_value());
        REQUIRE(device.destroyBuffer(storageOnly).has_value());
    }

} // namespace rachis::test::fill_and_clear
