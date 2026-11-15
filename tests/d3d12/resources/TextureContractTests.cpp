#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace
{

    constexpr std::uint32_t textureWidth = 7;
    constexpr std::uint32_t textureHeight = 5;
    constexpr std::uint32_t bytesPerPixel = 4;
    constexpr std::uint32_t rowPitch = 256;
    constexpr std::uint64_t transferSize = static_cast<std::uint64_t>(rowPitch) * textureHeight;

    struct AllocationTotals final
    {
        std::uint64_t bytes = 0;
        std::uint64_t count = 0;
    };

    [[nodiscard]] std::byte texelByte(std::uint32_t x, std::uint32_t y, std::uint32_t channel) noexcept
    {
        return static_cast<std::byte>((x * 29U + y * 47U + channel * 61U + 13U) & 0xFFU);
    }

    [[nodiscard]] bool nearByte(std::byte actual, std::uint8_t expected) noexcept
    {
        const auto value = std::to_integer<int>(actual);
        return value >= static_cast<int>(expected) - 1 && value <= static_cast<int>(expected) + 1;
    }

    void printDiagnostic(const rachis::DiagnosticMessage& message, void*) noexcept
    {
        static_cast<void>(std::fprintf(stderr, "D3D12 [%u]: ", static_cast<unsigned int>(message.nativeId)));
        if (!message.description.empty())
        {
            static_cast<void>(std::fwrite(message.description.data(), 1, message.description.size(), stderr));
        }
        static_cast<void>(std::fputc('\n', stderr));
    }

    [[nodiscard]] AllocationTotals allocationTotals(rachis::Device& device)
    {
        const auto budgets = device.queryMemoryBudgets();
        REQUIRE(budgets.has_value());

        AllocationTotals totals;
        for (const auto& budget : *budgets)
        {
            REQUIRE(rachis::hasAny(budget.features, rachis::MemoryBudgetFeature::allocationStatistics));
            totals.bytes += budget.allocationBytes;
            totals.count += budget.allocationCount;
        }
        return totals;
    }

} // namespace

TEST_CASE("texture upload copy and readback preserve texels", "[size:medium][rhi][texture][transfer][warp]")
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
    const auto baselineAllocations = allocationTotals(device);

    CHECK_FALSE(device
                    .createTexture(rachis::TextureDesc{
                        .width = textureWidth,
                        .height = textureHeight,
                        .format = rachis::Format::unknown,
                        .usage = rachis::TextureUsage::copyDestination,
                    })
                    .has_value());

    auto upload = device.createBuffer(rachis::BufferDesc{
        .size = transferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Texture contract upload",
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = transferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Texture contract readback",
    });
    const rachis::TextureDesc textureDesc{
        .dimension = rachis::TextureDimension::texture2D,
        .width = textureWidth,
        .height = textureHeight,
        .depthOrArrayLayers = 2,
        .mipLevels = 2,
        .sampleCount = 1,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Texture contract resource",
    };
    auto source = device.createTexture(textureDesc);
    auto destination = device.createTexture(textureDesc);
    REQUIRE(upload.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(source.has_value());
    REQUIRE(destination.has_value());
    const auto liveAllocations = allocationTotals(device);
    CHECK(liveAllocations.count == baselineAllocations.count + 4);
    CHECK(liveAllocations.bytes > baselineAllocations.bytes);

    auto mappedUpload = device.mapBuffer(*upload);
    REQUIRE(mappedUpload.has_value());
    std::fill_n(mappedUpload->data, mappedUpload->size, std::byte{0xCD});
    for (std::uint32_t y = 0; y < textureHeight; ++y)
    {
        for (std::uint32_t x = 0; x < textureWidth; ++x)
        {
            for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
            {
                mappedUpload->data[static_cast<std::uint64_t>(y) * rowPitch +
                                   static_cast<std::uint64_t>(x) * bytesPerPixel + channel] = texelByte(x, y, channel);
            }
        }
    }
    REQUIRE(device.unmapBuffer(*upload).has_value());

    const rachis::TextureSubresource selectedSubresource{
        .mipLevel = 0,
        .arrayLayer = 1,
        .aspect = rachis::TextureAspect::color,
    };
    const rachis::TextureSubresourceRange selectedRange{
        .firstMipLevel = 0,
        .mipLevelCount = 1,
        .firstArrayLayer = 1,
        .arrayLayerCount = 1,
        .aspect = rachis::TextureAspect::color,
    };
    const rachis::Extent3D copyExtent{
        .width = textureWidth,
        .height = textureHeight,
        .depth = 1,
    };

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    CHECK_FALSE(device
                    .copyBufferToTexture(*command,
                                         rachis::BufferTextureCopy{
                                             .buffer = *upload,
                                             .bytesPerRow = textureWidth * bytesPerPixel,
                                             .rowsPerImage = textureHeight,
                                             .texture = *source,
                                             .textureSubresource = selectedSubresource,
                                             .extent = copyExtent,
                                         })
                    .has_value());
    CHECK_FALSE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = *source,
                                        .stagesBefore = rachis::PipelineStage::all,
                                        .stagesAfter = rachis::PipelineStage::pixelShader,
                                        .accessBefore = rachis::Access::none,
                                        .accessAfter = rachis::Access::shaderResource,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::shaderResource,
                                        .subresources = selectedRange,
                                    })
                    .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *source,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                    .subresources = selectedRange,
                                })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *destination,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                    .subresources = selectedRange,
                                })
                .has_value());
    REQUIRE(device
                .copyBufferToTexture(*command,
                                     rachis::BufferTextureCopy{
                                         .buffer = *upload,
                                         .bufferOffset = 0,
                                         .bytesPerRow = rowPitch,
                                         .rowsPerImage = textureHeight,
                                         .texture = *source,
                                         .textureSubresource = selectedSubresource,
                                         .textureOrigin = {},
                                         .extent = copyExtent,
                                     })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *source,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::copyDestination,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                    .subresources = selectedRange,
                                })
                .has_value());
    REQUIRE(device
                .copyTexture(*command,
                             rachis::TextureCopy{
                                 .source = *source,
                                 .sourceSubresource = selectedSubresource,
                                 .sourceOrigin = {},
                                 .destination = *destination,
                                 .destinationSubresource = selectedSubresource,
                                 .destinationOrigin = {},
                                 .extent = copyExtent,
                             })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *destination,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::copyDestination,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                    .subresources = selectedRange,
                                })
                .has_value());
    REQUIRE(device
                .copyTextureToBuffer(*command,
                                     rachis::BufferTextureCopy{
                                         .buffer = *readback,
                                         .bufferOffset = 0,
                                         .bytesPerRow = rowPitch,
                                         .rowsPerImage = textureHeight,
                                         .texture = *destination,
                                         .textureSubresource = selectedSubresource,
                                         .textureOrigin = {},
                                         .extent = copyExtent,
                                     })
                .has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());

    const auto staleSource = *source;
    const auto staleDestination = *destination;
    REQUIRE(device.destroyTexture(*source).has_value());
    REQUIRE(device.destroyTexture(*destination).has_value());
    CHECK_FALSE(device.destroyTexture(staleSource).has_value());
    CHECK_FALSE(device.destroyTexture(staleDestination).has_value());
    const auto retainedAllocations = allocationTotals(device);
    CHECK(retainedAllocations.count == liveAllocations.count);
    CHECK(retainedAllocations.bytes == liveAllocations.bytes);

    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();
    const auto collectedAllocations = allocationTotals(device);
    CHECK(collectedAllocations.count == baselineAllocations.count + 2);
    CHECK(collectedAllocations.bytes < liveAllocations.bytes);

    auto mappedReadback = device.mapBuffer(*readback);
    REQUIRE(mappedReadback.has_value());
    std::uint64_t firstMismatch = transferSize;
    for (std::uint32_t y = 0; y < textureHeight && firstMismatch == transferSize; ++y)
    {
        for (std::uint32_t x = 0; x < textureWidth && firstMismatch == transferSize; ++x)
        {
            for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
            {
                const auto offset =
                    static_cast<std::uint64_t>(y) * rowPitch + static_cast<std::uint64_t>(x) * bytesPerPixel + channel;
                if (mappedReadback->data[offset] != texelByte(x, y, channel))
                {
                    firstMismatch = offset;
                    break;
                }
            }
        }
    }
    CAPTURE(firstMismatch);
    CHECK(firstMismatch == transferSize);

    REQUIRE(device.unmapBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    const auto releasedAllocations = allocationTotals(device);
    CHECK(releasedAllocations.count == baselineAllocations.count);
    CHECK(releasedAllocations.bytes == baselineAllocations.bytes);

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("declared texture view formats preserve typed color semantics",
          "[size:medium][rhi][texture][view][format][warp]")
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

    const auto rgbaProperties = device.queryFormatProperties(rachis::Format::rgba8Unorm);
    REQUIRE(rachis::hasAny(rgbaProperties.features, rachis::FormatFeature::viewFormatReinterpretation));

    const std::array repeatedBase{rachis::Format::rgba8Unorm};
    CHECK_FALSE(device
                    .createTexture(rachis::TextureDesc{
                        .format = rachis::Format::rgba8Unorm,
                        .usage = rachis::TextureUsage::sampled,
                        .viewFormats = repeatedBase,
                    })
                    .has_value());
    const std::array incompatible{rachis::Format::bgra8Unorm};
    CHECK_FALSE(device
                    .createTexture(rachis::TextureDesc{
                        .format = rachis::Format::rgba8Unorm,
                        .usage = rachis::TextureUsage::sampled,
                        .viewFormats = incompatible,
                    })
                    .has_value());
    const std::array duplicateAlternates{rachis::Format::rgba8UnormSrgb, rachis::Format::rgba8UnormSrgb};
    CHECK_FALSE(device
                    .createTexture(rachis::TextureDesc{
                        .format = rachis::Format::rgba8Unorm,
                        .usage = rachis::TextureUsage::sampled,
                        .viewFormats = duplicateAlternates,
                    })
                    .has_value());

    auto ordinary = device.createTexture(rachis::TextureDesc{
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled,
        .label = "Typed texture without alternate views",
    });
    REQUIRE(ordinary.has_value());
    CHECK_FALSE(device
                    .createTextureView(rachis::TextureViewDesc{
                        .texture = *ordinary,
                        .usage = rachis::TextureViewUsage::sampled,
                        .format = rachis::Format::rgba8UnormSrgb,
                    })
                    .has_value());

    const std::array alternateFormats{rachis::Format::rgba8UnormSrgb};
    auto mutableColor = device.createTexture(rachis::TextureDesc{
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::storage | rachis::TextureUsage::renderTarget,
        .viewFormats = alternateFormats,
        .label = "Declared linear and sRGB texture",
    });
    REQUIRE(mutableColor.has_value());
    auto linearRenderTarget = device.createTextureView(rachis::TextureViewDesc{
        .texture = *mutableColor,
        .usage = rachis::TextureViewUsage::renderTarget,
        .format = rachis::Format::rgba8Unorm,
        .label = "Linear RTV",
    });
    auto srgbRenderTarget = device.createTextureView(rachis::TextureViewDesc{
        .texture = *mutableColor,
        .usage = rachis::TextureViewUsage::renderTarget,
        .format = rachis::Format::rgba8UnormSrgb,
        .label = "sRGB RTV",
    });
    REQUIRE(linearRenderTarget.has_value());
    REQUIRE(srgbRenderTarget.has_value());
    CHECK_FALSE(device
                    .createTextureView(rachis::TextureViewDesc{
                        .texture = *mutableColor,
                        .usage = rachis::TextureViewUsage::sampled,
                        .format = rachis::Format::bgra8Unorm,
                    })
                    .has_value());
    CHECK_FALSE(device
                    .createTextureView(rachis::TextureViewDesc{
                        .texture = *mutableColor,
                        .usage = rachis::TextureViewUsage::storage,
                        .format = rachis::Format::rgba8UnormSrgb,
                    })
                    .has_value());

    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = rowPitch,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Mutable texture readback",
    });
    REQUIRE(readback.has_value());
    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *mutableColor,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array attachments{
        rachis::ColorAttachment{
            .view = *srgbRenderTarget,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.5F, 0.25F, 0.0F, 0.75F},
        },
    };
    REQUIRE(device.beginRendering(*command, rachis::RenderingDesc{.colorAttachments = attachments}).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                rachis::TextureBarrier{
                                    .texture = *mutableColor,
                                    .stagesBefore = rachis::PipelineStage::renderTarget,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::renderTarget,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::renderTarget,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device
                .copyTextureToBuffer(*command,
                                     rachis::BufferTextureCopy{
                                         .buffer = *readback,
                                         .bytesPerRow = rowPitch,
                                         .rowsPerImage = 1,
                                         .texture = *mutableColor,
                                     })
                .has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto mapped = device.mapBuffer(*readback);
    REQUIRE(mapped.has_value());
    CAPTURE(std::to_integer<std::uint8_t>(mapped->data[0]),
            std::to_integer<std::uint8_t>(mapped->data[1]),
            std::to_integer<std::uint8_t>(mapped->data[2]),
            std::to_integer<std::uint8_t>(mapped->data[3]));
    CHECK(nearByte(mapped->data[0], 188));
    CHECK(nearByte(mapped->data[1], 137));
    CHECK(nearByte(mapped->data[2], 0));
    CHECK(nearByte(mapped->data[3], 191));
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyTexture(*ordinary).has_value());
    REQUIRE(device.destroyTexture(*mutableColor).has_value());
    REQUIRE(device.destroyTextureView(*linearRenderTarget).has_value());
    REQUIRE(device.destroyTextureView(*srgbRenderTarget).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    device.collectGarbage();

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("texture views validate subresources and retain their textures", "[size:medium][rhi][texture][view][warp]")
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

    auto color = device.createTexture(rachis::TextureDesc{
        .dimension = rachis::TextureDimension::texture2D,
        .width = 32,
        .height = 16,
        .depthOrArrayLayers = 4,
        .mipLevels = 4,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::storage | rachis::TextureUsage::renderTarget,
        .label = "Texture view color resource",
    });
    REQUIRE(color.has_value());

    auto sampled = device.createTextureView(rachis::TextureViewDesc{
        .texture = *color,
        .usage = rachis::TextureViewUsage::sampled,
        .dimension = rachis::TextureViewDimension::texture2DArray,
        .firstMipLevel = 1,
        .mipLevelCount = 2,
        .firstArrayLayer = 1,
        .arrayLayerCount = 3,
        .label = "Array SRV",
    });
    auto storage = device.createTextureView(rachis::TextureViewDesc{
        .texture = *color,
        .usage = rachis::TextureViewUsage::storage,
        .dimension = rachis::TextureViewDimension::texture2DArray,
        .firstMipLevel = 2,
        .firstArrayLayer = 3,
        .arrayLayerCount = 1,
        .label = "Single-layer UAV",
    });
    auto renderTarget = device.createTextureView(rachis::TextureViewDesc{
        .texture = *color,
        .usage = rachis::TextureViewUsage::renderTarget,
        .dimension = rachis::TextureViewDimension::texture2DArray,
        .firstMipLevel = 1,
        .firstArrayLayer = 1,
        .arrayLayerCount = 2,
        .label = "Array RTV",
    });
    REQUIRE(sampled.has_value());
    REQUIRE(storage.has_value());
    REQUIRE(renderTarget.has_value());

    CHECK_FALSE(device
                    .createTextureView(rachis::TextureViewDesc{
                        .texture = *color,
                        .usage = rachis::TextureViewUsage::storage,
                        .dimension = rachis::TextureViewDimension::texture2DArray,
                        .mipLevelCount = 2,
                        .arrayLayerCount = 1,
                    })
                    .has_value());
    CHECK_FALSE(device
                    .createTextureView(rachis::TextureViewDesc{
                        .texture = *color,
                        .usage = rachis::TextureViewUsage::sampled,
                        .dimension = rachis::TextureViewDimension::texture3D,
                    })
                    .has_value());
    CHECK_FALSE(device
                    .createTextureView(rachis::TextureViewDesc{
                        .texture = *color,
                        .usage = rachis::TextureViewUsage::sampled,
                        .dimension = rachis::TextureViewDimension::texture2D,
                    })
                    .has_value());
    CHECK_FALSE(device
                    .createTextureView(rachis::TextureViewDesc{
                        .texture = *color,
                        .usage = rachis::TextureViewUsage::sampled,
                        .dimension = rachis::TextureViewDimension::texture2D,
                        .format = rachis::Format::bgra8Unorm,
                    })
                    .has_value());

    auto depth = device.createTexture(rachis::TextureDesc{
        .width = 32,
        .height = 16,
        .mipLevels = 3,
        .format = rachis::Format::d32Float,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::depthStencil,
        .label = "Texture view depth resource",
    });
    REQUIRE(depth.has_value());
    auto sampledDepth = device.createTextureView(rachis::TextureViewDesc{
        .texture = *depth,
        .usage = rachis::TextureViewUsage::sampled,
        .dimension = rachis::TextureViewDimension::texture2D,
        .aspect = rachis::TextureAspect::depth,
        .label = "Depth SRV",
    });
    auto depthStencil = device.createTextureView(rachis::TextureViewDesc{
        .texture = *depth,
        .usage = rachis::TextureViewUsage::depthStencil,
        .dimension = rachis::TextureViewDimension::texture2D,
        .aspect = rachis::TextureAspect::depth,
        .firstMipLevel = 1,
        .label = "DSV",
    });
    REQUIRE(sampledDepth.has_value());
    REQUIRE(depthStencil.has_value());

    auto depthStencilTexture = device.createTexture(rachis::TextureDesc{
        .width = 8,
        .height = 8,
        .format = rachis::Format::d24UnormS8Uint,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::depthStencil,
        .label = "Texture view stencil resource",
    });
    REQUIRE(depthStencilTexture.has_value());
    auto sampledStencil = device.createTextureView(rachis::TextureViewDesc{
        .texture = *depthStencilTexture,
        .usage = rachis::TextureViewUsage::sampled,
        .dimension = rachis::TextureViewDimension::texture2D,
        .aspect = rachis::TextureAspect::stencil,
        .label = "Stencil SRV",
    });
    REQUIRE(sampledStencil.has_value());

    auto cubeTexture = device.createTexture(rachis::TextureDesc{
        .width = 16,
        .height = 16,
        .depthOrArrayLayers = 12,
        .mipLevels = 3,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled,
        .label = "Texture view cube resource",
    });
    REQUIRE(cubeTexture.has_value());
    auto cube = device.createTextureView(rachis::TextureViewDesc{
        .texture = *cubeTexture,
        .usage = rachis::TextureViewUsage::sampled,
        .dimension = rachis::TextureViewDimension::cube,
        .label = "Cube SRV",
    });
    auto cubeArray = device.createTextureView(rachis::TextureViewDesc{
        .texture = *cubeTexture,
        .usage = rachis::TextureViewUsage::sampled,
        .dimension = rachis::TextureViewDimension::cubeArray,
        .firstArrayLayer = 6,
        .arrayLayerCount = 6,
        .label = "Cube-array SRV",
    });
    REQUIRE(cube.has_value());
    REQUIRE(cubeArray.has_value());
    CHECK_FALSE(device
                    .createTextureView(rachis::TextureViewDesc{
                        .texture = *cubeTexture,
                        .usage = rachis::TextureViewUsage::sampled,
                        .dimension = rachis::TextureViewDimension::cube,
                        .firstArrayLayer = 6,
                        .arrayLayerCount = 6,
                    })
                    .has_value());
    CHECK_FALSE(device
                    .createTextureView(rachis::TextureViewDesc{
                        .texture = *cubeTexture,
                        .usage = rachis::TextureViewUsage::sampled,
                        .dimension = rachis::TextureViewDimension::cubeArray,
                        .firstArrayLayer = 1,
                        .arrayLayerCount = 6,
                    })
                    .has_value());

    std::vector<rachis::TextureViewHandle> pagedViews;
    pagedViews.reserve(300);
    for (std::uint32_t index = 0; index < 300; ++index)
    {
        auto view = device.createTextureView(rachis::TextureViewDesc{
            .texture = *color,
            .usage = rachis::TextureViewUsage::sampled,
            .dimension = rachis::TextureViewDimension::texture2DArray,
            .arrayLayerCount = 1,
        });
        REQUIRE(view.has_value());
        pagedViews.push_back(*view);
    }

    const auto staleColor = *color;
    REQUIRE(device.destroyTexture(*color).has_value());
    CHECK_FALSE(device
                    .createTextureView(rachis::TextureViewDesc{
                        .texture = staleColor,
                    })
                    .has_value());

    for (const auto view : pagedViews)
    {
        REQUIRE(device.destroyTextureView(view).has_value());
    }
    const auto staleSampled = *sampled;
    REQUIRE(device.destroyTextureView(*sampled).has_value());
    CHECK_FALSE(device.destroyTextureView(staleSampled).has_value());
    REQUIRE(device.destroyTextureView(*storage).has_value());
    REQUIRE(device.destroyTextureView(*renderTarget).has_value());

    REQUIRE(device.destroyTexture(*depth).has_value());
    REQUIRE(device.destroyTextureView(*sampledDepth).has_value());
    REQUIRE(device.destroyTextureView(*depthStencil).has_value());
    REQUIRE(device.destroyTexture(*depthStencilTexture).has_value());
    REQUIRE(device.destroyTextureView(*sampledStencil).has_value());
    REQUIRE(device.destroyTexture(*cubeTexture).has_value());
    REQUIRE(device.destroyTextureView(*cube).has_value());
    REQUIRE(device.destroyTextureView(*cubeArray).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
