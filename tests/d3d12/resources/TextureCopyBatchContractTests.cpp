#include "HeapAllocationCounter.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string_view>
#include <utility>

namespace
{

    constexpr std::uint32_t textureWidth = 8;
    constexpr std::uint32_t textureHeight = 4;
    constexpr std::uint32_t bytesPerPixel = 4;
    constexpr std::uint32_t rowPitch = 256;
    constexpr std::uint64_t bufferSize = 1536;

    void printDiagnostic(const rachis::DiagnosticMessage& message, void*) noexcept
    {
        static_cast<void>(std::fprintf(stderr, "D3D12 [%u]: ", static_cast<unsigned int>(message.nativeId)));
        if (!message.description.empty())
        {
            static_cast<void>(std::fwrite(message.description.data(), 1, message.description.size(), stderr));
        }
        static_cast<void>(std::fputc('\n', stderr));
    }

    [[nodiscard]] constexpr std::byte texelByte(std::uint32_t x, std::uint32_t y, std::uint32_t channel) noexcept
    {
        return static_cast<std::byte>((x * 31U + y * 47U + channel * 59U + 11U) & 0xFFU);
    }

    void writeRegion(std::byte* destination, std::uint64_t bufferOffset, std::uint32_t textureX, std::uint32_t textureY)
    {
        for (std::uint32_t y = 0; y < 2; ++y)
        {
            for (std::uint32_t x = 0; x < 4; ++x)
            {
                for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
                {
                    destination[bufferOffset + static_cast<std::uint64_t>(y) * rowPitch +
                                static_cast<std::uint64_t>(x) * bytesPerPixel + channel] =
                        texelByte(textureX + x, textureY + y, channel);
                }
            }
        }
    }

    [[nodiscard]] bool regionMatches(const std::byte* actual,
                                     std::uint64_t bufferOffset,
                                     std::uint32_t textureX,
                                     std::uint32_t textureY) noexcept
    {
        for (std::uint32_t y = 0; y < 2; ++y)
        {
            for (std::uint32_t x = 0; x < 4; ++x)
            {
                for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
                {
                    const auto offset = bufferOffset + static_cast<std::uint64_t>(y) * rowPitch +
                                        static_cast<std::uint64_t>(x) * bytesPerPixel + channel;
                    if (actual[offset] != texelByte(textureX + x, textureY + y, channel))
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    [[nodiscard]] bool regionIsZero(const std::byte* actual, std::uint64_t bufferOffset) noexcept
    {
        for (std::uint32_t y = 0; y < 2; ++y)
        {
            for (std::uint32_t byteIndex = 0; byteIndex < 4 * bytesPerPixel; ++byteIndex)
            {
                if (actual[bufferOffset + static_cast<std::uint64_t>(y) * rowPitch + byteIndex] != std::byte{0})
                {
                    return false;
                }
            }
        }
        return true;
    }

} // namespace

TEST_CASE("texture-copy region batches validate and record atomically",
          "[size:medium][rhi][texture][transfer][batch][warp]")
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

    auto upload = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Texture-copy batch upload",
    });
    auto zeroUpload = device.createBuffer({
        .size = 1024,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Texture-copy batch zero upload",
    });
    auto readback = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Texture-copy batch readback",
    });
    const rachis::TextureDesc textureDesc{
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
    };
    auto source = device.createTexture(textureDesc);
    auto destination = device.createTexture(textureDesc);
    auto multisampled = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .sampleCount = 4,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::renderTarget,
        .label = "Texture-copy batch multisample validation",
    });
    auto multisampledDestination = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .sampleCount = 4,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copyDestination | rachis::TextureUsage::renderTarget,
        .label = "Texture-copy batch multisample destination",
    });
    auto renderTarget = device.createTexture({
        .width = 1,
        .height = 1,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .label = "Texture-copy batch rendering-scope validation",
    });
    REQUIRE(upload.has_value());
    REQUIRE(zeroUpload.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(source.has_value());
    REQUIRE(destination.has_value());
    REQUIRE(multisampled.has_value());
    REQUIRE(multisampledDestination.has_value());
    REQUIRE(renderTarget.has_value());
    auto renderTargetView = device.createTextureView({
        .texture = *renderTarget,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    REQUIRE(renderTargetView.has_value());

    auto mappedUpload = device.mapBuffer(*upload);
    auto mappedZero = device.mapBuffer(*zeroUpload);
    REQUIRE(mappedUpload.has_value());
    REQUIRE(mappedZero.has_value());
    std::fill_n(mappedUpload->data, mappedUpload->size, std::byte{0});
    std::fill_n(mappedZero->data, mappedZero->size, std::byte{0});
    writeRegion(mappedUpload->data, 0, 0, 0);
    writeRegion(mappedUpload->data, 512, 4, 2);
    REQUIRE(device.unmapBuffer(*upload).has_value());
    REQUIRE(device.unmapBuffer(*zeroUpload).has_value());

    const rachis::TextureSubresource subresource{.aspect = rachis::TextureAspect::color};
    const rachis::TextureSubresourceRange range{
        .mipLevelCount = 1,
        .arrayLayerCount = 1,
        .aspect = rachis::TextureAspect::color,
    };
    const std::array uploadRegions{
        rachis::BufferTextureCopyRegion{
            .bufferOffset = 0,
            .bytesPerRow = rowPitch,
            .rowsPerImage = 2,
            .textureSubresource = subresource,
            .textureOrigin = {.x = 0, .y = 0},
            .extent = {.width = 4, .height = 2},
        },
        rachis::BufferTextureCopyRegion{
            .bufferOffset = 512,
            .bytesPerRow = rowPitch,
            .rowsPerImage = 2,
            .textureSubresource = subresource,
            .textureOrigin = {.x = 4, .y = 2},
            .extent = {.width = 4, .height = 2},
        },
    };
    const std::array textureRegions{
        rachis::TextureCopyRegion{
            .sourceSubresource = subresource,
            .sourceOrigin = {.x = 4, .y = 0},
            .destinationSubresource = subresource,
            .destinationOrigin = {.x = 0, .y = 0},
            .extent = {.width = 4, .height = 2},
        },
        rachis::TextureCopyRegion{
            .sourceSubresource = subresource,
            .sourceOrigin = {.x = 0, .y = 0},
            .destinationSubresource = subresource,
            .destinationOrigin = {.x = 4, .y = 0},
            .extent = {.width = 4, .height = 2},
        },
        rachis::TextureCopyRegion{
            .sourceSubresource = subresource,
            .sourceOrigin = {.x = 4, .y = 2},
            .destinationSubresource = subresource,
            .destinationOrigin = {.x = 0, .y = 2},
            .extent = {.width = 4, .height = 2},
        },
    };
    const std::array readbackRegions{
        rachis::BufferTextureCopyRegion{
            .bufferOffset = 0,
            .bytesPerRow = rowPitch,
            .rowsPerImage = 2,
            .textureSubresource = subresource,
            .textureOrigin = {.x = 0, .y = 0},
            .extent = {.width = 4, .height = 2},
        },
        rachis::BufferTextureCopyRegion{
            .bufferOffset = 512,
            .bytesPerRow = rowPitch,
            .rowsPerImage = 2,
            .textureSubresource = subresource,
            .textureOrigin = {.x = 4, .y = 0},
            .extent = {.width = 4, .height = 2},
        },
        rachis::BufferTextureCopyRegion{
            .bufferOffset = 1024,
            .bytesPerRow = rowPitch,
            .rowsPerImage = 2,
            .textureSubresource = subresource,
            .textureOrigin = {.x = 0, .y = 2},
            .extent = {.width = 4, .height = 2},
        },
    };

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    const std::span<const rachis::BufferTextureCopyRegion> emptyBufferTextureRegions;
    const std::span<const rachis::TextureCopyRegion> emptyTextureRegions;
    CHECK_FALSE(device.copyBufferToTexture(*command, *upload, *source, emptyBufferTextureRegions).has_value());
    CHECK_FALSE(device.copyTextureToBuffer(*command, *source, *readback, emptyBufferTextureRegions).has_value());
    CHECK_FALSE(device.copyTexture(*command, *source, *destination, emptyTextureRegions).has_value());

    auto overlappingUploads = uploadRegions;
    overlappingUploads[1].textureOrigin = uploadRegions[0].textureOrigin;
    CHECK_FALSE(device.copyBufferToTexture(*command, *upload, *source, overlappingUploads).has_value());
    auto overlappingReadbacks = readbackRegions;
    overlappingReadbacks[1].bufferOffset = 0;
    CHECK_FALSE(device.copyTextureToBuffer(*command, *destination, *readback, overlappingReadbacks).has_value());
    auto overlappingTextures = textureRegions;
    overlappingTextures[1].destinationOrigin = textureRegions[0].destinationOrigin;
    CHECK_FALSE(device.copyTexture(*command, *source, *destination, overlappingTextures).has_value());
    const std::array sameSubresource{textureRegions[0]};
    CHECK_FALSE(device.copyTexture(*command, *source, *source, sameSubresource).has_value());
    const std::array multisampleLinear{
        rachis::BufferTextureCopyRegion{
            .bytesPerRow = rowPitch,
            .rowsPerImage = textureHeight,
            .textureSubresource = subresource,
            .extent = {.width = textureWidth, .height = textureHeight},
        },
    };
    CHECK_FALSE(device.copyBufferToTexture(*command, *upload, *multisampled, multisampleLinear).has_value());
    const std::array wholeTexture{
        rachis::TextureCopyRegion{
            .sourceSubresource = subresource,
            .destinationSubresource = subresource,
            .extent = {.width = textureWidth, .height = textureHeight},
        },
    };
    CHECK_FALSE(device.copyTexture(*command, *multisampled, *destination, wholeTexture).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *multisampled,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                    .subresources = range,
                                })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *multisampledDestination,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                    .subresources = range,
                                })
                .has_value());
    REQUIRE(device.copyTexture(*command, *multisampled, *multisampledDestination, wholeTexture).has_value());

    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *renderTarget,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array attachments{rachis::ColorAttachment{
        .view = *renderTargetView,
        .loadOperation = rachis::LoadOperation::clear,
    }};
    REQUIRE(device.beginRendering(*command, {.colorAttachments = attachments}).has_value());
    const auto renderingUpload = device.copyBufferToTexture(*command, *upload, *source, uploadRegions);
    const auto renderingReadback = device.copyTextureToBuffer(*command, *destination, *readback, readbackRegions);
    const auto renderingTextureCopy = device.copyTexture(*command, *source, *destination, textureRegions);
    REQUIRE_FALSE(renderingUpload.has_value());
    REQUIRE_FALSE(renderingReadback.has_value());
    REQUIRE_FALSE(renderingTextureCopy.has_value());
    CHECK(renderingUpload.error().code == rachis::ErrorCode::invalidState);
    CHECK(renderingReadback.error().code == rachis::ErrorCode::invalidState);
    CHECK(renderingTextureCopy.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.endRendering(*command).has_value());

    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *source,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                    .subresources = range,
                                })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *destination,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                    .subresources = range,
                                })
                .has_value());
    const std::array zeroSource{
        rachis::BufferTextureCopyRegion{
            .bytesPerRow = rowPitch,
            .rowsPerImage = textureHeight,
            .textureSubresource = subresource,
            .extent = {.width = textureWidth, .height = textureHeight},
        },
    };
    REQUIRE(device.copyBufferToTexture(*command, *zeroUpload, *source, zeroSource).has_value());
    auto partiallyInvalid = uploadRegions;
    partiallyInvalid[0].textureOrigin = {.x = 4, .y = 0};
    partiallyInvalid[1].bufferOffset = bufferSize;
    CHECK_FALSE(device.copyBufferToTexture(*command, *upload, *source, partiallyInvalid).has_value());
    REQUIRE(device.copyBufferToTexture(*command, *upload, *source, uploadRegions).has_value());
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
                                    .subresources = range,
                                })
                .has_value());
    REQUIRE(device.copyTexture(*command, *source, *destination, textureRegions).has_value());
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
                                    .subresources = range,
                                })
                .has_value());
    REQUIRE(device.copyTextureToBuffer(*command, *destination, *readback, readbackRegions).has_value());

    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*zeroUpload).has_value());
    REQUIRE(device.destroyTexture(*source).has_value());
    REQUIRE(device.destroyTexture(*destination).has_value());
    REQUIRE(device.destroyTexture(*multisampled).has_value());
    REQUIRE(device.destroyTexture(*multisampledDestination).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    auto mappedReadback = device.mapBuffer(*readback);
    REQUIRE(mappedReadback.has_value());
    CHECK(regionIsZero(mappedReadback->data, 0));
    CHECK(regionMatches(mappedReadback->data, 512, 0, 0));
    CHECK(regionMatches(mappedReadback->data, 1024, 4, 2));
    REQUIRE(device.unmapBuffer(*readback).has_value());

    REQUIRE(device.destroyTextureView(*renderTargetView).has_value());
    REQUIRE(device.destroyTexture(*renderTarget).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("single-region texture copies record without per-copy scratch allocations",
          "[size:medium][rhi][texture][transfer][batch][warp]")
{
    if (!rachis::test::ThreadHeapAllocationCounter::available())
    {
        SKIP("Heap allocations are counted through the MSVC debug heap.");
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

    auto upload = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Allocation-counted texture upload",
    });
    auto readback = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Allocation-counted texture readback",
    });
    const rachis::TextureDesc textureDesc{
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
    };
    auto source = device.createTexture(textureDesc);
    auto destination = device.createTexture(textureDesc);
    REQUIRE(upload.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(source.has_value());
    REQUIRE(destination.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    for (const auto texture : {*source, *destination})
    {
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = texture,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessAfter = rachis::Access::copyDestination,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::copyDestination,
                                    })
                    .has_value());
    }

    const rachis::BufferTextureCopy uploadCopy{
        .buffer = *upload,
        .bytesPerRow = rowPitch,
        .rowsPerImage = textureHeight,
        .texture = *source,
        .extent = {.width = textureWidth, .height = textureHeight},
    };
    const rachis::TextureCopy textureCopy{
        .source = *source,
        .destination = *destination,
        .extent = {.width = textureWidth, .height = textureHeight},
    };
    const rachis::BufferTextureCopy readbackCopy{
        .buffer = *readback,
        .bytesPerRow = rowPitch,
        .rowsPerImage = textureHeight,
        .texture = *source,
        .extent = {.width = textureWidth, .height = textureHeight},
    };

    constexpr std::size_t iterations = 64;
    std::size_t recorded = 0;
    const auto recordUploads = [&] {
        for (std::size_t iteration = 0; iteration < iterations; ++iteration)
        {
            recorded += device.copyBufferToTexture(*command, uploadCopy).has_value() ? 1U : 0U;
        }
    };
    const auto recordTextureCopies = [&] {
        for (std::size_t iteration = 0; iteration < iterations; ++iteration)
        {
            recorded += device.copyTexture(*command, textureCopy).has_value() ? 1U : 0U;
        }
    };
    const auto recordReadbacks = [&] {
        for (std::size_t iteration = 0; iteration < iterations; ++iteration)
        {
            recorded += device.copyTextureToBuffer(*command, readbackCopy).has_value() ? 1U : 0U;
        }
    };

    rachis::test::ThreadHeapAllocationCounter counter;
    recordUploads();
    const auto uploadAllocations = counter.count(recordUploads);
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
    recordTextureCopies();
    const auto textureCopyAllocations = counter.count(recordTextureCopies);
    recordReadbacks();
    const auto readbackAllocations = counter.count(recordReadbacks);
    REQUIRE(recorded == 6 * iterations);

    constexpr std::size_t retentionGrowthAllowance = 8;
    CAPTURE(uploadAllocations, textureCopyAllocations, readbackAllocations);
    CHECK(uploadAllocations <= iterations + retentionGrowthAllowance);
    CHECK(textureCopyAllocations <= iterations + retentionGrowthAllowance);
    CHECK(readbackAllocations <= iterations + retentionGrowthAllowance);

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();
    REQUIRE(device.destroyTexture(*destination).has_value());
    REQUIRE(device.destroyTexture(*source).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("large texture-copy region batches report the first overlapping destination or shared texel",
          "[size:medium][rhi][texture][transfer][batch][warp]")
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
    REQUIRE(rachis::test::requireCapabilities(device).enhancedBarriers);

    constexpr std::uint32_t regionCount = 16;
    constexpr std::uint64_t regionStride = 512;
    constexpr std::uint64_t stagingSize = regionStride * regionCount;
    auto upload = device.createBuffer({
        .size = stagingSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Large texture-copy batch upload",
    });
    auto readback = device.createBuffer({
        .size = stagingSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Large texture-copy batch readback",
    });
    const rachis::TextureDesc textureDesc{
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
    };
    auto source = device.createTexture(textureDesc);
    auto destination = device.createTexture(textureDesc);
    auto mipmapped = device.createTexture({
        .width = 2 * textureWidth,
        .height = 2 * textureHeight,
        .mipLevels = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Large same-texture copy batch",
    });
    REQUIRE(upload.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(source.has_value());
    REQUIRE(destination.has_value());
    REQUIRE(mipmapped.has_value());

    std::array<rachis::BufferTextureCopyRegion, regionCount> transferRegions{};
    std::array<rachis::TextureCopyRegion, regionCount> textureRegions{};
    for (std::uint32_t index = 0; index < regionCount; ++index)
    {
        const rachis::Origin3D texel{.x = index % textureWidth, .y = index / textureWidth};
        transferRegions[index] = rachis::BufferTextureCopyRegion{
            .bufferOffset = regionStride * index,
            .bytesPerRow = rowPitch,
            .rowsPerImage = 1,
            .textureOrigin = texel,
            .extent = {.width = 1, .height = 1},
        };
        textureRegions[index] = rachis::TextureCopyRegion{
            .sourceOrigin = texel,
            .destinationOrigin = {.x = texel.x, .y = texel.y + 2},
            .extent = {.width = 1, .height = 1},
        };
    }
    std::array<rachis::TextureCopyRegion, 12> mipRegions{};
    for (std::uint32_t index = 0; index < 8; ++index)
    {
        mipRegions[index] = rachis::TextureCopyRegion{
            .sourceSubresource = {.mipLevel = 0},
            .sourceOrigin = {.x = index, .y = 0},
            .destinationSubresource = {.mipLevel = 1},
            .destinationOrigin = {.x = index, .y = 0},
            .extent = {.width = 1, .height = 1},
        };
    }
    for (std::uint32_t index = 0; index < 4; ++index)
    {
        mipRegions[8 + index] = rachis::TextureCopyRegion{
            .sourceSubresource = {.mipLevel = 1},
            .sourceOrigin = {.x = index, .y = 1},
            .destinationSubresource = {.mipLevel = 0},
            .destinationOrigin = {.x = index, .y = 2 * textureHeight - 1},
            .extent = {.width = 1, .height = 1},
        };
    }
    constexpr std::string_view overlappingTexelsMessage = "overlapping destination texels";
    constexpr std::string_view overlappingBytesMessage = "overlapping destination bytes";
    constexpr std::string_view sharedTexelMessage = "one texel as both source and destination";

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    const auto checkRejected = [](const rachis::Status& rejected, std::string_view reason, bool expected) {
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
        CAPTURE(rejected.error().message);
        CHECK(rejected.error().message.contains(reason) == expected);
    };

    auto overlappingUploads = transferRegions;
    overlappingUploads[13].textureOrigin = transferRegions[3].textureOrigin;
    checkRejected(
        device.copyBufferToTexture(*command, *upload, *source, overlappingUploads), overlappingTexelsMessage, true);
    auto overlapBeforeInvalidUpload = transferRegions;
    overlapBeforeInvalidUpload[5].textureOrigin = transferRegions[1].textureOrigin;
    overlapBeforeInvalidUpload[10].textureOrigin = {.x = textureWidth, .y = 0};
    checkRejected(device.copyBufferToTexture(*command, *upload, *source, overlapBeforeInvalidUpload),
                  overlappingTexelsMessage,
                  true);
    auto invalidBeforeOverlapUpload = transferRegions;
    invalidBeforeOverlapUpload[4].textureOrigin = {.x = textureWidth, .y = 0};
    invalidBeforeOverlapUpload[10].textureOrigin = transferRegions[1].textureOrigin;
    checkRejected(device.copyBufferToTexture(*command, *upload, *source, invalidBeforeOverlapUpload),
                  overlappingTexelsMessage,
                  false);
    auto overlappingReadbacks = transferRegions;
    overlappingReadbacks[12].bufferOffset = transferRegions[2].bufferOffset;
    checkRejected(
        device.copyTextureToBuffer(*command, *source, *readback, overlappingReadbacks), overlappingBytesMessage, true);
    auto overlappingTextureCopies = textureRegions;
    overlappingTextureCopies[14].destinationOrigin = textureRegions[6].destinationOrigin;
    checkRejected(
        device.copyTexture(*command, *source, *destination, overlappingTextureCopies), overlappingTexelsMessage, true);
    auto sharedTexel = mipRegions;
    sharedTexel[8].sourceOrigin = mipRegions[0].destinationOrigin;
    checkRejected(device.copyTexture(*command, *mipmapped, *mipmapped, sharedTexel), sharedTexelMessage, true);

    for (const auto texture : {*source, *destination})
    {
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = texture,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessAfter = rachis::Access::copyDestination,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::copyDestination,
                                    })
                    .has_value());
    }
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *mipmapped,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copySource | rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::common,
                                })
                .has_value());
    REQUIRE(device.copyBufferToTexture(*command, *upload, *source, transferRegions).has_value());
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
    REQUIRE(device.copyTexture(*command, *source, *destination, textureRegions).has_value());
    REQUIRE(device.copyTextureToBuffer(*command, *source, *readback, transferRegions).has_value());
    REQUIRE(device.copyTexture(*command, *mipmapped, *mipmapped, mipRegions).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();
    REQUIRE(device.destroyTexture(*mipmapped).has_value());
    REQUIRE(device.destroyTexture(*destination).has_value());
    REQUIRE(device.destroyTexture(*source).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("large texture-copy batches of full-width rows and shared-pitch tiles validate and round-trip",
          "[size:medium][rhi][texture][transfer][batch][warp]")
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
    REQUIRE(rachis::test::requireCapabilities(device).enhancedBarriers);

    constexpr std::uint32_t width = 512;
    constexpr std::uint32_t height = 256;
    constexpr std::uint32_t pitch = width * bytesPerPixel;
    constexpr std::uint64_t imageSize = static_cast<std::uint64_t>(pitch) * height;
    constexpr std::uint32_t tileWidth = 128;
    constexpr std::uint32_t tileHeight = 16;
    constexpr std::uint32_t tilesPerRow = width / tileWidth;
    constexpr std::uint32_t tileCount = tilesPerRow * (height / tileHeight);
    constexpr std::uint32_t mipRowCount = height / 4;

    auto upload = device.createBuffer({
        .size = imageSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Row-strip upload",
    });
    auto readback = device.createBuffer({
        .size = imageSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Shared-pitch tile readback",
    });
    auto texture = device.createTexture({
        .width = width,
        .height = height,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Row-strip and tile texture",
    });
    auto mipmapped = device.createTexture({
        .width = width,
        .height = height,
        .mipLevels = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Row-strip same-texture copies",
    });
    REQUIRE(upload.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(texture.has_value());
    REQUIRE(mipmapped.has_value());

    auto mappedUpload = device.mapBuffer(*upload);
    REQUIRE(mappedUpload.has_value());
    REQUIRE(mappedUpload->size >= imageSize);
    for (std::uint32_t y = 0; y < height; ++y)
    {
        for (std::uint32_t x = 0; x < width; ++x)
        {
            for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
            {
                mappedUpload->data[static_cast<std::uint64_t>(y) * pitch + x * bytesPerPixel + channel] =
                    texelByte(x, y, channel);
            }
        }
    }
    REQUIRE(device.unmapBuffer(*upload).has_value());

    std::array<rachis::BufferTextureCopyRegion, height> rows{};
    for (std::uint32_t y = 0; y < height; ++y)
    {
        rows[y] = rachis::BufferTextureCopyRegion{
            .bufferOffset = static_cast<std::uint64_t>(y) * pitch,
            .bytesPerRow = pitch,
            .rowsPerImage = 1,
            .textureOrigin = {.y = y},
            .extent = {.width = width, .height = 1},
        };
    }
    std::array<rachis::BufferTextureCopyRegion, tileCount> tiles{};
    for (std::uint32_t tile = 0; tile < tileCount; ++tile)
    {
        const rachis::Origin3D origin{.x = tile % tilesPerRow * tileWidth, .y = tile / tilesPerRow * tileHeight};
        tiles[tile] = rachis::BufferTextureCopyRegion{
            .bufferOffset = static_cast<std::uint64_t>(origin.y) * pitch + origin.x * bytesPerPixel,
            .bytesPerRow = pitch,
            .rowsPerImage = tileHeight,
            .textureOrigin = origin,
            .extent = {.width = tileWidth, .height = tileHeight},
        };
    }
    std::array<rachis::TextureCopyRegion, mipRowCount * 2U> mipRows{};
    for (std::uint32_t y = 0; y < mipRowCount; ++y)
    {
        mipRows[y] = rachis::TextureCopyRegion{
            .sourceSubresource = {.mipLevel = 0},
            .sourceOrigin = {.y = y},
            .destinationSubresource = {.mipLevel = 1},
            .destinationOrigin = {.y = y},
            .extent = {.width = width / 2, .height = 1},
        };
        mipRows[mipRowCount + y] = rachis::TextureCopyRegion{
            .sourceSubresource = {.mipLevel = 1},
            .sourceOrigin = {.y = mipRowCount + y},
            .destinationSubresource = {.mipLevel = 0},
            .destinationOrigin = {.y = height / 2 + y},
            .extent = {.width = width / 2, .height = 1},
        };
    }
    constexpr std::string_view overlappingTexelsMessage = "overlapping destination texels";
    constexpr std::string_view overlappingBytesMessage = "overlapping destination bytes";
    constexpr std::string_view sharedTexelMessage = "one texel as both source and destination";

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    const auto checkRejected = [](const rachis::Status& rejected, std::string_view reason) {
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
        CAPTURE(rejected.error().message);
        CHECK(rejected.error().message.contains(reason));
    };

    auto overlappingRows = rows;
    overlappingRows[200].textureOrigin = rows[37].textureOrigin;
    checkRejected(device.copyBufferToTexture(*command, *upload, *texture, overlappingRows), overlappingTexelsMessage);
    auto overlappingTiles = tiles;
    overlappingTiles[41].bufferOffset = tiles[37].bufferOffset + pitch;
    checkRejected(device.copyTextureToBuffer(*command, *texture, *readback, overlappingTiles), overlappingBytesMessage);
    auto sharedTexel = mipRows;
    sharedTexel[mipRowCount + 3U].destinationOrigin = {.y = 5};
    checkRejected(device.copyTexture(*command, *mipmapped, *mipmapped, sharedTexel), sharedTexelMessage);

    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *texture,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    REQUIRE(device.copyBufferToTexture(*command, *upload, *texture, rows).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *texture,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::copyDestination,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device.copyTextureToBuffer(*command, *texture, *readback, tiles).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *mipmapped,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copySource | rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::common,
                                })
                .has_value());
    REQUIRE(device.copyTexture(*command, *mipmapped, *mipmapped, mipRows).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    auto mappedReadback = device.mapBuffer(*readback);
    REQUIRE(mappedReadback.has_value());
    REQUIRE(mappedReadback->size >= imageSize);
    std::size_t mismatches = 0;
    for (std::uint32_t y = 0; y < height; ++y)
    {
        for (std::uint32_t x = 0; x < width; ++x)
        {
            for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
            {
                const auto actual =
                    mappedReadback->data[static_cast<std::uint64_t>(y) * pitch + x * bytesPerPixel + channel];
                mismatches += actual == texelByte(x, y, channel) ? 0U : 1U;
            }
        }
    }
    CHECK(mismatches == 0U);
    REQUIRE(device.unmapBuffer(*readback).has_value());

    device.collectGarbage();
    REQUIRE(device.destroyTexture(*mipmapped).has_value());
    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
