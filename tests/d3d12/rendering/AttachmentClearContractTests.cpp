#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <span>

namespace
{

    constexpr std::uint32_t targetWidth = 4;
    constexpr std::uint32_t targetHeight = 4;
    constexpr std::uint32_t rowPitch = 256;
    constexpr std::uint64_t planeBytes = static_cast<std::uint64_t>(rowPitch) * targetHeight;
    constexpr std::uint64_t readbackBytes = planeBytes * 3;

    void printDiagnostic(const rachis::DiagnosticMessage& message, void*) noexcept
    {
        static_cast<void>(std::fprintf(stderr, "D3D12 [%u]: ", static_cast<unsigned int>(message.nativeId)));
        if (!message.description.empty())
        {
            static_cast<void>(std::fwrite(message.description.data(), 1, message.description.size(), stderr));
        }
        static_cast<void>(std::fputc('\n', stderr));
    }

    [[nodiscard]] const std::byte* pixel(const std::array<std::byte, readbackBytes>& data,
                                         std::uint64_t planeOffset,
                                         std::uint32_t x,
                                         std::uint32_t y) noexcept
    {
        return data.data() + planeOffset + static_cast<std::uint64_t>(y) * rowPitch + static_cast<std::uint64_t>(x) * 4;
    }

} // namespace

TEST_CASE("texture optimized clear metadata validates attachment ownership", "[size:medium][rhi][texture][clear][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::disabled,
        .requireValidation = false,
    });
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    const auto rejected = [&](rachis::TextureDesc desc) {
        const auto result = device.createTexture(desc);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().code == rachis::ErrorCode::invalidArgument);
    };

    rejected({
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copyDestination,
        .optimizedClearValue = {.kind = rachis::OptimizedClearValueKind::color},
    });
    rejected({
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .optimizedClearValue =
            {
                .kind = rachis::OptimizedClearValueKind::depthStencil,
            },
    });
    rejected({
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .optimizedClearValue =
            {
                .kind = rachis::OptimizedClearValueKind::color,
                .color = {std::numeric_limits<float>::infinity(), 0.0F, 0.0F, 0.0F},
            },
    });
    rejected({
        .format = rachis::Format::d32Float,
        .usage = rachis::TextureUsage::depthStencil,
        .optimizedClearValue =
            {
                .kind = rachis::OptimizedClearValueKind::depthStencil,
                .depth = -0.25F,
            },
    });
    rejected({
        .format = rachis::Format::d32Float,
        .usage = rachis::TextureUsage::depthStencil,
        .optimizedClearValue =
            {
                .kind = rachis::OptimizedClearValueKind::depthStencil,
                .stencil = 1,
            },
    });
    rejected({
        .format = rachis::Format::d24UnormS8Uint,
        .usage = rachis::TextureUsage::depthStencil,
        .optimizedClearValue =
            {
                .kind = rachis::OptimizedClearValueKind::depthStencil,
                .stencil = 256,
            },
    });
    rejected({
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .optimizedClearValue =
            {
                .kind = static_cast<rachis::OptimizedClearValueKind>(0xFF),
            },
    });

    constexpr std::array alternateFormats{rachis::Format::rgba8UnormSrgb};
    rachis::TextureDesc mutableColor{
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .viewFormats = alternateFormats,
    };
    auto automaticMutable = device.createTexture(mutableColor);
    REQUIRE(automaticMutable.has_value());
    REQUIRE(device.destroyTexture(*automaticMutable).has_value());
    mutableColor.optimizedClearValue = {
        .kind = rachis::OptimizedClearValueKind::color,
        .color = {0.25F, 0.5F, 0.75F, 1.0F},
    };
    rejected(mutableColor);

    const auto snapshot = device.captureResourceMemorySnapshot();
    REQUIRE(snapshot.has_value());
    CHECK(snapshot->entries.empty());
}

TEST_CASE("active rendering attachment clears validate rectangles and preserve exact regions",
          "[size:medium][rhi][graphics][clear][warp]")
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

    auto colorZero = device.createTexture({
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
        .optimizedClearValue =
            {
                .kind = rachis::OptimizedClearValueKind::color,
                .color = {0.0F, 0.0F, 1.0F, 1.0F},
            },
        .label = "Partial attachment clear color",
    });
    auto colorOne = device.createTexture({
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::renderTarget,
        .optimizedClearValue =
            {
                .kind = rachis::OptimizedClearValueKind::color,
                .color = {0.0F, 0.0F, 0.0F, 1.0F},
            },
        .label = "Full attachment clear color",
    });
    auto depth = device.createTexture({
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::d32Float,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::depthStencil,
        .optimizedClearValue =
            {
                .kind = rachis::OptimizedClearValueKind::depthStencil,
                .depth = 0.25F,
            },
        .label = "Partial attachment clear depth",
    });
    auto depthStencil = device.createTexture({
        .width = targetWidth,
        .height = targetHeight,
        .format = rachis::Format::d24UnormS8Uint,
        .usage = rachis::TextureUsage::depthStencil,
        .optimizedClearValue =
            {
                .kind = rachis::OptimizedClearValueKind::depthStencil,
                .depth = 0.0F,
            },
        .label = "Combined attachment clear depth stencil",
    });
    REQUIRE(colorZero.has_value());
    REQUIRE(colorOne.has_value());
    REQUIRE(depth.has_value());
    REQUIRE(depthStencil.has_value());

    auto colorZeroView = device.createTextureView({
        .texture = *colorZero,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    auto colorOneView = device.createTextureView({
        .texture = *colorOne,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    auto depthView = device.createTextureView({
        .texture = *depth,
        .usage = rachis::TextureViewUsage::depthStencil,
        .aspect = rachis::TextureAspect::depth,
    });
    auto depthStencilView = device.createTextureView({
        .texture = *depthStencil,
        .usage = rachis::TextureViewUsage::depthStencil,
        .aspect = rachis::TextureAspect::depth,
    });
    REQUIRE(colorZeroView.has_value());
    REQUIRE(colorOneView.has_value());
    REQUIRE(depthView.has_value());
    REQUIRE(depthStencilView.has_value());

    auto readback = device.createBuffer({
        .size = readbackBytes,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Attachment clear readback",
    });
    REQUIRE(readback.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    auto outsideColor = device.clearColorAttachment(*command, 0, {});
    auto outsideDepth = device.clearDepthStencilAttachment(*command, rachis::DepthStencilClearAspect::depth, 1.0F, 0);
    REQUIRE_FALSE(outsideColor.has_value());
    REQUIRE_FALSE(outsideDepth.has_value());
    CHECK(outsideColor.error().code == rachis::ErrorCode::invalidState);
    CHECK(outsideDepth.error().code == rachis::ErrorCode::invalidState);

    for (const auto color : {*colorZero, *colorOne})
    {
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = color,
                                        .stagesBefore = rachis::PipelineStage::all,
                                        .stagesAfter = rachis::PipelineStage::renderTarget,
                                        .accessAfter = rachis::Access::renderTarget,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::renderTarget,
                                    })
                    .has_value());
    }
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *depth,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::depthStencil,
                                    .accessAfter = rachis::Access::depthStencilWrite,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::depthStencilWrite,
                                    .subresources = {.aspect = rachis::TextureAspect::depth},
                                })
                .has_value());

    const std::array colorAttachments{
        rachis::ColorAttachment{
            .view = *colorZeroView,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.0F, 0.0F, 1.0F, 1.0F},
        },
        rachis::ColorAttachment{
            .view = *colorOneView,
            .loadOperation = rachis::LoadOperation::clear,
            .clearValue = {0.0F, 0.0F, 0.0F, 1.0F},
        },
    };
    const rachis::DepthStencilAttachment depthAttachment{
        .view = *depthView,
        .depthLoadOperation = rachis::LoadOperation::clear,
        .depthClearValue = 0.25F,
    };
    REQUIRE(
        device
            .beginRendering(*command, {.colorAttachments = colorAttachments, .depthStencilAttachment = depthAttachment})
            .has_value());

    const std::array clearRectangle{
        rachis::ScissorRect{.x = 1, .y = 1, .width = 2, .height = 2},
    };
    const std::array negativeRectangle{
        rachis::ScissorRect{.x = -1, .width = 1, .height = 1},
    };
    const std::array emptyRectangle{
        rachis::ScissorRect{.x = 1, .y = 1, .height = 1},
    };
    const std::array outOfBoundsRectangle{
        rachis::ScissorRect{.x = 3, .y = 3, .width = 2, .height = 2},
    };
    CHECK_FALSE(device.clearColorAttachment(*command, 2, {}).has_value());
    CHECK_FALSE(
        device.clearColorAttachment(*command, 0, {.red = std::numeric_limits<float>::infinity()}, clearRectangle)
            .has_value());
    CHECK_FALSE(device.clearColorAttachment(*command, 0, {}, negativeRectangle).has_value());
    CHECK_FALSE(device.clearColorAttachment(*command, 0, {}, emptyRectangle).has_value());
    CHECK_FALSE(device.clearColorAttachment(*command, 0, {}, outOfBoundsRectangle).has_value());
    CHECK_FALSE(
        device.clearDepthStencilAttachment(*command, rachis::DepthStencilClearAspect::none, 1.0F, 0).has_value());
    CHECK_FALSE(
        device.clearDepthStencilAttachment(*command, static_cast<rachis::DepthStencilClearAspect>(1U << 7U), 1.0F, 0)
            .has_value());
    CHECK_FALSE(
        device.clearDepthStencilAttachment(*command, rachis::DepthStencilClearAspect::depth, -0.5F, 0).has_value());
    auto unsupportedStencil =
        device.clearDepthStencilAttachment(*command, rachis::DepthStencilClearAspect::stencil, 0.0F, 1);
    REQUIRE_FALSE(unsupportedStencil.has_value());
    CHECK(unsupportedStencil.error().code == rachis::ErrorCode::unsupported);
    CHECK_FALSE(device
                    .clearDepthStencilAttachment(
                        *command, rachis::DepthStencilClearAspect::depth, 0.5F, 0, outOfBoundsRectangle)
                    .has_value());

    REQUIRE(device.clearColorAttachment(*command, 0, {1.0F, 0.0F, 0.0F, 1.0F}, clearRectangle).has_value());
    REQUIRE(device.clearColorAttachment(*command, 1, {0.0F, 1.0F, 0.0F, 1.0F}).has_value());
    REQUIRE(
        device.clearDepthStencilAttachment(*command, rachis::DepthStencilClearAspect::depth, 0.75F, 0, clearRectangle)
            .has_value());
    REQUIRE(device.endRendering(*command).has_value());

    for (const auto color : {*colorZero, *colorOne})
    {
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = color,
                                        .stagesBefore = rachis::PipelineStage::renderTarget,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::renderTarget,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::renderTarget,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                    })
                    .has_value());
    }
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *depth,
                                    .stagesBefore = rachis::PipelineStage::depthStencil,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::depthStencilWrite,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::depthStencilWrite,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                    .subresources = {.aspect = rachis::TextureAspect::depth},
                                })
                .has_value());

    const std::array copiedTextures{*colorZero, *colorOne, *depth};
    const std::array copiedAspects{
        rachis::TextureAspect::color,
        rachis::TextureAspect::color,
        rachis::TextureAspect::depth,
    };
    for (std::size_t index = 0; index < copiedTextures.size(); ++index)
    {
        REQUIRE(device
                    .copyTextureToBuffer(*command,
                                         {
                                             .buffer = *readback,
                                             .bufferOffset = static_cast<std::uint64_t>(index) * planeBytes,
                                             .bytesPerRow = rowPitch,
                                             .rowsPerImage = targetHeight,
                                             .texture = copiedTextures[index],
                                             .textureSubresource = {.aspect = copiedAspects[index]},
                                             .extent = {targetWidth, targetHeight, 1},
                                         })
                    .has_value());
    }

    for (const auto aspect : {rachis::TextureAspect::depth, rachis::TextureAspect::stencil})
    {
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *depthStencil,
                                        .stagesBefore = rachis::PipelineStage::all,
                                        .stagesAfter = rachis::PipelineStage::depthStencil,
                                        .accessAfter = rachis::Access::depthStencilWrite,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::depthStencilWrite,
                                        .subresources = {.aspect = aspect},
                                    })
                    .has_value());
    }
    const rachis::DepthStencilAttachment combinedAttachment{
        .view = *depthStencilView,
        .depthLoadOperation = rachis::LoadOperation::clear,
        .depthClearValue = 0.0F,
        .stencilLoadOperation = rachis::LoadOperation::clear,
        .stencilClearValue = 0,
    };
    REQUIRE(device.beginRendering(*command, {.depthStencilAttachment = combinedAttachment}).has_value());
    CHECK_FALSE(
        device.clearDepthStencilAttachment(*command, rachis::DepthStencilClearAspect::stencil, 1.0F, 256).has_value());
    REQUIRE(
        device
            .clearDepthStencilAttachment(
                *command, rachis::DepthStencilClearAspect::depth | rachis::DepthStencilClearAspect::stencil, 1.0F, 0x7F)
            .has_value());
    REQUIRE(device.endRendering(*command).has_value());

    REQUIRE(device.destroyTexture(*colorZero).has_value());
    REQUIRE(device.destroyTexture(*colorOne).has_value());
    REQUIRE(device.destroyTexture(*depth).has_value());
    REQUIRE(device.destroyTexture(*depthStencil).has_value());
    REQUIRE(device.destroyTextureView(*colorZeroView).has_value());
    REQUIRE(device.destroyTextureView(*colorOneView).has_value());
    REQUIRE(device.destroyTextureView(*depthView).has_value());
    REQUIRE(device.destroyTextureView(*depthStencilView).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    std::array<std::byte, readbackBytes> data{};
    REQUIRE(device.readBuffer(*readback, 0, data).has_value());
    for (std::uint32_t y = 0; y < targetHeight; ++y)
    {
        for (std::uint32_t x = 0; x < targetWidth; ++x)
        {
            const bool inside = x >= 1 && x < 3 && y >= 1 && y < 3;
            const auto* partialColor = pixel(data, 0, x, y);
            CHECK(partialColor[0] == (inside ? std::byte{0xFF} : std::byte{0x00}));
            CHECK(partialColor[1] == std::byte{0x00});
            CHECK(partialColor[2] == (inside ? std::byte{0x00} : std::byte{0xFF}));
            CHECK(partialColor[3] == std::byte{0xFF});

            const auto* fullColor = pixel(data, planeBytes, x, y);
            CHECK(fullColor[0] == std::byte{0x00});
            CHECK(fullColor[1] == std::byte{0xFF});
            CHECK(fullColor[2] == std::byte{0x00});
            CHECK(fullColor[3] == std::byte{0xFF});

            float depthValue = 0.0F;
            std::memcpy(&depthValue, pixel(data, planeBytes * 2, x, y), sizeof(depthValue));
            CHECK(depthValue == (inside ? 0.75F : 0.25F));
        }
    }
    REQUIRE(device.destroyBuffer(*readback).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
