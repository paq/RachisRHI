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
#include <cstring>
#include <initializer_list>
#include <span>
#include <utility>
#include <vector>

namespace
{

    constexpr std::uint64_t bufferSize = 64;
    constexpr std::uint32_t textureWidth = 5;
    constexpr std::uint32_t textureHeight = 3;
    constexpr std::uint32_t bytesPerPixel = 4;

    [[nodiscard]] rachis::DeviceDesc vulkanDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::vulkan,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
    }

    [[nodiscard]] std::byte sourceValue(std::uint64_t index) noexcept
    {
        return static_cast<std::byte>((index * 29U + 7U) & 0xFFU);
    }

    [[nodiscard]] std::byte texelByte(std::uint32_t x, std::uint32_t y, std::uint32_t channel) noexcept
    {
        return static_cast<std::byte>((x * 31U + y * 47U + channel * 59U + 11U) & 0xFFU);
    }

    [[nodiscard]] rachis::Result<rachis::Device> createVulkanDevice()
    {
        return rachis::createDevice(vulkanDesc());
    }

    void writeTexels(std::byte* destination,
                     const rachis::TextureTransferLayout& layout,
                     std::uint32_t originX,
                     std::uint32_t originY,
                     std::uint32_t width,
                     std::uint32_t height)
    {
        for (std::uint32_t y = 0; y < height; ++y)
        {
            for (std::uint32_t x = 0; x < width; ++x)
            {
                for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
                {
                    destination[static_cast<std::size_t>(y) * layout.bytesPerRow +
                                static_cast<std::size_t>(x) * bytesPerPixel + channel] =
                        texelByte(originX + x, originY + y, channel);
                }
            }
        }
    }

    [[nodiscard]] bool texelsMatch(const std::byte* actual,
                                   const rachis::TextureTransferLayout& layout,
                                   std::uint32_t originX,
                                   std::uint32_t originY,
                                   std::uint32_t width,
                                   std::uint32_t height) noexcept
    {
        for (std::uint32_t y = 0; y < height; ++y)
        {
            for (std::uint32_t x = 0; x < width; ++x)
            {
                for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
                {
                    const auto offset = static_cast<std::size_t>(y) * layout.bytesPerRow +
                                        static_cast<std::size_t>(x) * bytesPerPixel + channel;
                    if (actual[offset] != texelByte(originX + x, originY + y, channel))
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    }

} // namespace

TEST_CASE("Vulkan buffer-copy batches validate atomically and complete without idle",
          "[size:medium][rhi][vulkan][transfer][buffer]")
{
    auto created = createVulkanDevice();
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    auto source = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan buffer-copy source",
    });
    auto destination = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan buffer-copy destination",
    });
    auto sameBuffer = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .label = "Vulkan buffer-copy overlap",
    });
    REQUIRE(source);
    REQUIRE(destination);
    REQUIRE(sameBuffer);

    std::array<std::byte, bufferSize> sourceBytes{};
    for (std::uint64_t index = 0; index < bufferSize; ++index)
    {
        sourceBytes[index] = sourceValue(index);
    }
    REQUIRE(device.writeBuffer(*source, 0, sourceBytes).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(command);

    const std::span<const rachis::BufferCopyRegion> emptyRegions;
    CHECK_FALSE(device.copyBuffer(*command, *source, *destination, emptyRegions).has_value());

    const std::array overlappingDestinations{
        rachis::BufferCopyRegion{.sourceOffset = 0, .destinationOffset = 48, .size = 8},
        rachis::BufferCopyRegion{.sourceOffset = 8, .destinationOffset = 52, .size = 8},
    };
    CHECK_FALSE(device.copyBuffer(*command, *source, *destination, overlappingDestinations).has_value());

    const std::array sameBufferCrossOverlap{
        rachis::BufferCopyRegion{.sourceOffset = 0, .destinationOffset = 16, .size = 8},
        rachis::BufferCopyRegion{.sourceOffset = 16, .destinationOffset = 32, .size = 8},
    };
    CHECK_FALSE(device.copyBuffer(*command, *sameBuffer, *sameBuffer, sameBufferCrossOverlap).has_value());

    const std::array partiallyInvalid{
        rachis::BufferCopyRegion{.sourceOffset = 48, .destinationOffset = 56, .size = 8},
        rachis::BufferCopyRegion{.sourceOffset = 60, .destinationOffset = 0, .size = 8},
    };
    CHECK_FALSE(device.copyBuffer(*command, *source, *destination, partiallyInvalid).has_value());

    const std::array validRegions{
        rachis::BufferCopyRegion{.sourceOffset = 0, .destinationOffset = 8, .size = 8},
        rachis::BufferCopyRegion{.sourceOffset = 16, .destinationOffset = 24, .size = 8},
        rachis::BufferCopyRegion{.sourceOffset = 32, .destinationOffset = 40, .size = 8},
    };
    REQUIRE(device.copyBuffer(*command, *source, *destination, validRegions).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    std::array<std::byte, bufferSize> actual{};
    REQUIRE(device.readBuffer(*destination, 0, actual).has_value());
    std::array<std::byte, bufferSize> expected{};
    for (const auto& region : validRegions)
    {
        for (std::uint64_t byteIndex = 0; byteIndex < region.size; ++byteIndex)
        {
            expected[region.destinationOffset + byteIndex] = sourceValue(region.sourceOffset + byteIndex);
        }
    }
    CHECK(actual == expected);

    REQUIRE(device.destroyBuffer(*source).has_value());
    REQUIRE(device.destroyBuffer(*sameBuffer).has_value());
    REQUIRE(device.destroyBuffer(*destination).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan layouts stay local until submission and preserve untouched subresources",
          "[size:medium][requires:HardwareVulkan][rhi][vulkan][transfer][layout]")
{
    using namespace rachis;
    auto created = createVulkanDevice();
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    auto texture = device.createTexture({
        .width = 4,
        .height = 4,
        .depthOrArrayLayers = 2,
        .mipLevels = 2,
        .format = Format::rgba8Unorm,
        .usage = TextureUsage::copySource | TextureUsage::copyDestination,
    });
    REQUIRE(texture);
    const BufferTextureCopyRegion region{
        .textureSubresource = {.mipLevel = 1, .arrayLayer = 1},
        .extent = {.width = 2, .height = 2},
    };
    auto layout = device.queryTextureTransferLayout(*texture, region);
    REQUIRE(layout);
    auto upload = device.createBuffer(
        {.size = layout->requiredBufferSize, .usage = BufferUsage::copySource, .memoryType = MemoryType::upload});
    auto readback = device.createBuffer({.size = layout->requiredBufferSize,
                                         .usage = BufferUsage::copyDestination,
                                         .memoryType = MemoryType::readback});
    REQUIRE(upload);
    REQUIRE(readback);
    std::vector<std::byte> bytes(static_cast<std::size_t>(layout->requiredBufferSize));
    writeTexels(bytes.data(), *layout, 0, 0, 2, 2);
    REQUIRE(device.writeBuffer(*upload, 0, bytes));
    const TextureBarrier toDestination{
        .texture = *texture,
        .stagesAfter = PipelineStage::copy,
        .accessAfter = Access::copyDestination,
        .layoutAfter = TextureLayout::copyDestination,
        .subresources = {.firstMipLevel = 1, .mipLevelCount = 1, .firstArrayLayer = 1, .arrayLayerCount = 1},
    };
    auto first = test::acquireCommandBuffer(device, QueueType::graphics, CommandPoolFlag::allowIndividualReset);
    auto second = test::acquireCommandBuffer(device, QueueType::graphics);
    REQUIRE(first);
    REQUIRE(second);
    REQUIRE(device.textureBarrier(*first, toDestination));
    REQUIRE(device.copyBufferToTexture(*first, *upload, *texture, std::array{region}));

    CHECK_FALSE(device.copyBufferToTexture(*second, *upload, *texture, std::array{region}));
    REQUIRE(device.resetCommandBuffer(*first));
    CHECK_FALSE(device.copyBufferToTexture(*first, *upload, *texture, std::array{region}));
    REQUIRE(device.textureBarrier(*first, toDestination));
    REQUIRE(device.copyBufferToTexture(*first, *upload, *texture, std::array{region}));
    auto uploaded = device.submit(std::move(*first));
    REQUIRE(uploaded);
    REQUIRE(device.wait(*uploaded));

    REQUIRE(device.copyBufferToTexture(*second, *upload, *texture, std::array{region}));
    auto untouched = region;
    untouched.textureSubresource.arrayLayer = 0;
    CHECK_FALSE(device.copyBufferToTexture(*second, *upload, *texture, std::array{untouched}));
    untouched.textureSubresource = {};
    CHECK_FALSE(device.copyBufferToTexture(*second, *upload, *texture, std::array{untouched}));
    const TextureBarrier toSource{
        .texture = *texture,
        .stagesBefore = PipelineStage::copy,
        .stagesAfter = PipelineStage::copy,
        .accessBefore = Access::copyDestination,
        .accessAfter = Access::copySource,
        .layoutBefore = TextureLayout::copyDestination,
        .layoutAfter = TextureLayout::copySource,
        .subresources = toDestination.subresources,
    };
    REQUIRE(device.textureBarrier(*second, toSource));

    *second = CommandBuffer{};
    auto final = test::acquireCommandBuffer(device, QueueType::graphics);
    REQUIRE(final);
    CHECK_FALSE(device.copyTextureToBuffer(*final, *texture, *readback, std::array{region}));
    REQUIRE(device.textureBarrier(*final, toSource));
    REQUIRE(device.copyTextureToBuffer(*final, *texture, *readback, std::array{region}));
    auto copied = device.submit(std::move(*final));
    REQUIRE(copied);
    REQUIRE(device.wait(*copied));
    std::vector<std::byte> actual(bytes.size());
    REQUIRE(device.readBuffer(*readback, 0, actual));
    CHECK(texelsMatch(actual.data(), *layout, 0, 0, 2, 2));
    REQUIRE(device.destroyTexture(*texture));
    REQUIRE(device.destroyBuffer(*upload));
    REQUIRE(device.destroyBuffer(*readback));
    device.collectGarbage();
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Vulkan transfer layouts publish Vulkan alignments and precise footprints",
          "[size:medium][rhi][vulkan][transfer][layout]")
{
    auto created = createVulkanDevice();
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    auto texture = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Vulkan transfer layout texture",
    });
    REQUIRE(texture);

    const rachis::BufferTextureCopyRegion tight{
        .extent = {.width = textureWidth, .height = textureHeight},
    };
    auto layout = device.queryTextureTransferLayout(*texture, tight);
    REQUIRE(layout);
    CHECK(layout->bufferOffsetAlignment == 4);
    CHECK(layout->rowPitchAlignment == 4);
    CHECK(layout->bytesPerRow == textureWidth * bytesPerPixel);
    CHECK(layout->rowsPerImage == textureHeight);
    CHECK(layout->rowCount == textureHeight);
    CHECK(layout->rowDataSize == std::uint64_t{textureWidth} * bytesPerPixel);
    CHECK(layout->requiredBufferSize == static_cast<std::uint64_t>(layout->bytesPerRow) * textureHeight);

    const rachis::BufferTextureCopyRegion padded{
        .bufferOffset = 8,
        .bytesPerRow = 32,
        .rowsPerImage = 5,
        .textureOrigin = {.x = 1, .y = 1},
        .extent = {.width = 3, .height = 2},
    };
    auto paddedLayout = device.queryTextureTransferLayout(*texture, padded);
    REQUIRE(paddedLayout);
    CHECK(paddedLayout->bytesPerRow == 32);
    CHECK(paddedLayout->rowsPerImage == 5);
    CHECK(paddedLayout->rowCount == 2);
    CHECK(paddedLayout->rowDataSize == 12);
    CHECK(paddedLayout->bytesPerImage == 160);
    CHECK(paddedLayout->requiredBufferSize == 8 + 32 + 12);

    CHECK_FALSE(device
                    .queryTextureTransferLayout(*texture,
                                                {
                                                    .bufferOffset = 2,
                                                    .extent = {.width = 1, .height = 1},
                                                })
                    .has_value());
    CHECK_FALSE(device
                    .queryTextureTransferLayout(*texture,
                                                {
                                                    .bytesPerRow = 10,
                                                    .extent = {.width = 1, .height = 1},
                                                })
                    .has_value());
    CHECK_FALSE(device
                    .queryTextureTransferLayout(*texture,
                                                {
                                                    .textureOrigin = {.x = 4, .y = 0},
                                                    .extent = {.width = 2, .height = 1},
                                                })
                    .has_value());

    auto layerTexture = device.createTexture({
        .width = 2,
        .height = 2,
        .depthOrArrayLayers = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copyDestination,
        .label = "Vulkan transfer layout layers",
    });
    auto volume = device.createTexture({
        .dimension = rachis::TextureDimension::texture3D,
        .width = 2,
        .height = 2,
        .depthOrArrayLayers = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copyDestination,
        .label = "Vulkan transfer layout 3D",
    });
    REQUIRE(layerTexture);
    REQUIRE(volume);
    auto layerLayout = device.queryTextureTransferLayout(*layerTexture,
                                                         {
                                                             .textureSubresource = {.arrayLayer = 1},
                                                             .extent = {.width = 2, .height = 2},
                                                         });
    REQUIRE(layerLayout);
    CHECK(layerLayout->requiredBufferSize == 16);
    auto volumeLayout = device.queryTextureTransferLayout(*volume,
                                                          {
                                                              .extent = {.width = 2, .height = 2, .depth = 2},
                                                          });
    REQUIRE(volumeLayout);
    CHECK(volumeLayout->bytesPerImage == 16);
    CHECK(volumeLayout->requiredBufferSize == 32);

    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyTexture(*layerTexture).has_value());
    REQUIRE(device.destroyTexture(*volume).has_value());
}

TEST_CASE("Vulkan buffer-image copies use queried footprints and leave failed destinations unchanged",
          "[size:medium][rhi][vulkan][transfer][texture]")
{
    auto created = createVulkanDevice();
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    auto texture = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Vulkan buffer-image texture",
    });
    auto layerTexture = device.createTexture({
        .width = 2,
        .height = 2,
        .depthOrArrayLayers = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Vulkan buffer-image layered",
    });
    auto volume = device.createTexture({
        .dimension = rachis::TextureDimension::texture3D,
        .width = 2,
        .height = 2,
        .depthOrArrayLayers = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Vulkan buffer-image 3D",
    });
    auto multisampled = device.createTexture({
        .width = 2,
        .height = 2,
        .sampleCount = 4,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copyDestination | rachis::TextureUsage::renderTarget,
        .label = "Vulkan buffer-image MSAA",
    });
    REQUIRE(texture);
    REQUIRE(layerTexture);
    REQUIRE(volume);
    REQUIRE(multisampled);

    const rachis::BufferTextureCopyRegion colorRegion{
        .extent = {.width = textureWidth, .height = textureHeight},
    };
    auto colorLayout = device.queryTextureTransferLayout(*texture, colorRegion);
    REQUIRE(colorLayout);

    const rachis::BufferTextureCopyRegion layerRegion{
        .textureSubresource = {.arrayLayer = 1},
        .extent = {.width = 2, .height = 2},
    };
    auto layerLayout = device.queryTextureTransferLayout(*layerTexture, layerRegion);
    REQUIRE(layerLayout);

    const rachis::BufferTextureCopyRegion volumeRegion{
        .extent = {.width = 2, .height = 2, .depth = 2},
    };
    auto volumeLayout = device.queryTextureTransferLayout(*volume, volumeRegion);
    REQUIRE(volumeLayout);

    const auto colorOffset = 0U;
    const auto layerOffset = colorLayout->requiredBufferSize;
    const auto volumeOffset = layerOffset + layerLayout->requiredBufferSize;
    const auto stagingSize = volumeOffset + volumeLayout->requiredBufferSize;
    auto upload = device.createBuffer({
        .size = stagingSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan buffer-image upload",
    });
    auto readback = device.createBuffer({
        .size = stagingSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan buffer-image readback",
    });
    REQUIRE(upload);
    REQUIRE(readback);

    const rachis::BufferTextureCopyRegion colorUpload = colorRegion;
    const rachis::BufferTextureCopyRegion layerUpload{
        .bufferOffset = layerOffset,
        .textureSubresource = layerRegion.textureSubresource,
        .extent = layerRegion.extent,
    };
    const rachis::BufferTextureCopyRegion volumeUpload{
        .bufferOffset = volumeOffset,
        .extent = volumeRegion.extent,
    };
    auto colorUploadLayout = device.queryTextureTransferLayout(*texture, colorUpload);
    auto layerUploadLayout = device.queryTextureTransferLayout(*layerTexture, layerUpload);
    auto volumeUploadLayout = device.queryTextureTransferLayout(*volume, volumeUpload);
    REQUIRE(colorUploadLayout);
    REQUIRE(layerUploadLayout);
    REQUIRE(volumeUploadLayout);

    std::vector<std::byte> uploadBytes(static_cast<std::size_t>(stagingSize));
    writeTexels(
        uploadBytes.data() + static_cast<std::size_t>(colorOffset), *colorLayout, 0, 0, textureWidth, textureHeight);
    writeTexels(uploadBytes.data() + static_cast<std::size_t>(layerOffset), *layerLayout, 2, 1, 2, 2);
    for (std::uint32_t slice = 0; slice < 2; ++slice)
    {
        writeTexels(uploadBytes.data() + static_cast<std::size_t>(volumeOffset + slice * volumeLayout->bytesPerImage),
                    *volumeLayout,
                    slice,
                    0,
                    2,
                    2);
    }
    REQUIRE(device.writeBuffer(*upload, 0, uploadBytes).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);

    const std::span<const rachis::BufferTextureCopyRegion> emptyRegions;
    CHECK_FALSE(device.copyBufferToTexture(*command, *upload, *texture, emptyRegions).has_value());
    CHECK_FALSE(device.copyTextureToBuffer(*command, *texture, *readback, emptyRegions).has_value());
    CHECK_FALSE(device.copyBufferToTexture(*command, *upload, *multisampled, std::array{colorRegion}).has_value());

    const std::array overlapping{
        rachis::BufferTextureCopyRegion{
            .extent = {.width = 2, .height = 2},
        },
        rachis::BufferTextureCopyRegion{
            .bufferOffset = 32,
            .textureOrigin = {.x = 1, .y = 1},
            .extent = {.width = 2, .height = 2},
        },
    };
    CHECK_FALSE(device.copyBufferToTexture(*command, *upload, *texture, overlapping).has_value());

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

    auto stale = device.createTexture({
        .width = 1,
        .height = 1,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copyDestination,
    });
    REQUIRE(stale);
    const auto staleHandle = *stale;
    REQUIRE(device.destroyTexture(*stale).has_value());
    const std::array staleRegion{colorRegion};
    CHECK_FALSE(device.copyBufferToTexture(*command, *upload, staleHandle, staleRegion).has_value());

    const std::array validColor{colorUpload};
    auto partial = validColor;
    partial[0].textureOrigin = {.x = 4, .y = 0};
    CHECK_FALSE(device.copyBufferToTexture(*command, *upload, *texture, partial).has_value());
    REQUIRE(device.copyBufferToTexture(*command, *upload, *texture, validColor).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *layerTexture,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *volume,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    REQUIRE(device.copyBufferToTexture(*command, *upload, *layerTexture, std::array{layerUpload}).has_value());
    REQUIRE(device.copyBufferToTexture(*command, *upload, *volume, std::array{volumeUpload}).has_value());

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
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *layerTexture,
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
                                    .texture = *volume,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::copyDestination,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device.copyTextureToBuffer(*command, *texture, *readback, validColor).has_value());
    REQUIRE(device.copyTextureToBuffer(*command, *layerTexture, *readback, std::array{layerUpload}).has_value());
    REQUIRE(device.copyTextureToBuffer(*command, *volume, *readback, std::array{volumeUpload}).has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    std::vector<std::byte> actual(static_cast<std::size_t>(stagingSize));
    REQUIRE(device.readBuffer(*readback, 0, actual).has_value());
    CHECK(texelsMatch(
        actual.data() + static_cast<std::size_t>(colorOffset), *colorLayout, 0, 0, textureWidth, textureHeight));
    CHECK(texelsMatch(actual.data() + static_cast<std::size_t>(layerOffset), *layerLayout, 2, 1, 2, 2));
    CHECK(texelsMatch(actual.data() + static_cast<std::size_t>(volumeOffset), *volumeLayout, 0, 0, 2, 2));
    CHECK(texelsMatch(actual.data() + static_cast<std::size_t>(volumeOffset + volumeLayout->bytesPerImage),
                      *volumeLayout,
                      1,
                      0,
                      2,
                      2));

    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyTexture(*layerTexture).has_value());
    REQUIRE(device.destroyTexture(*volume).has_value());
    REQUIRE(device.destroyTexture(*multisampled).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan general-layout transitions preserve earlier copies to other textures",
          "[size:medium][rhi][vulkan][transfer][texture]")
{
    auto created = createVulkanDevice();
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    auto color = device.createTexture({
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Vulkan earlier copy destination",
    });
    auto layer = device.createTexture({
        .width = 2,
        .height = 2,
        .depthOrArrayLayers = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Vulkan general-layout transition",
    });
    REQUIRE(color);
    REQUIRE(layer);
    const rachis::BufferTextureCopyRegion colorRegion{
        .extent = {.width = textureWidth, .height = textureHeight},
    };
    auto colorLayout = device.queryTextureTransferLayout(*color, colorRegion);
    REQUIRE(colorLayout);
    const rachis::BufferTextureCopyRegion layerRegion{
        .bufferOffset = colorLayout->requiredBufferSize,
        .textureSubresource = {.arrayLayer = 1},
        .extent = {.width = 2, .height = 2},
    };
    auto layerLayout = device.queryTextureTransferLayout(*layer, layerRegion);
    REQUIRE(layerLayout);
    auto upload = device.createBuffer({
        .size = layerLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    auto readback = device.createBuffer({
        .size = layerLayout->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(upload);
    REQUIRE(readback);
    std::vector<std::byte> bytes(static_cast<std::size_t>(layerLayout->requiredBufferSize));
    writeTexels(bytes.data(), *colorLayout, 0, 0, textureWidth, textureHeight);
    writeTexels(bytes.data() + static_cast<std::size_t>(layerRegion.bufferOffset), *layerLayout, 2, 1, 2, 2);
    REQUIRE(device.writeBuffer(*upload, 0, bytes).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command);
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *color,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *layer,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .layoutAfter = rachis::TextureLayout::common,
                                })
                .has_value());
    REQUIRE(device.copyBufferToTexture(*command, *upload, *color, std::array{colorRegion}).has_value());

    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *layer,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::common,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    REQUIRE(device.copyBufferToTexture(*command, *upload, *layer, std::array{layerRegion}).has_value());
    for (const auto texture : std::array{*color, *layer})
    {
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = texture,
                                        .stagesBefore = rachis::PipelineStage::copy,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::copyDestination,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::copyDestination,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                    })
                    .has_value());
    }
    REQUIRE(device.copyTextureToBuffer(*command, *color, *readback, std::array{colorRegion}).has_value());
    REQUIRE(device.copyTextureToBuffer(*command, *layer, *readback, std::array{layerRegion}).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted);
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    std::vector<std::byte> actual(bytes.size());
    REQUIRE(device.readBuffer(*readback, 0, actual).has_value());
    CHECK(texelsMatch(actual.data(), *colorLayout, 0, 0, textureWidth, textureHeight));
    CHECK(texelsMatch(actual.data() + static_cast<std::size_t>(layerRegion.bufferOffset), *layerLayout, 2, 1, 2, 2));
    REQUIRE(device.destroyTexture(*color).has_value());
    REQUIRE(device.destroyTexture(*layer).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan staging slices stay valid across in-flight frames without queue idle",
          "[size:medium][rhi][vulkan][transfer][staging]")
{
    auto created = createVulkanDevice();
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    auto textureA = device.createTexture({
        .width = 2,
        .height = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Vulkan in-flight texture A",
    });
    auto textureB = device.createTexture({
        .width = 2,
        .height = 2,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Vulkan in-flight texture B",
    });
    REQUIRE(textureA);
    REQUIRE(textureB);

    const rachis::BufferTextureCopyRegion sliceA{
        .extent = {.width = 2, .height = 2},
    };
    auto layoutA = device.queryTextureTransferLayout(*textureA, sliceA);
    REQUIRE(layoutA);
    const rachis::BufferTextureCopyRegion sliceB{
        .bufferOffset = layoutA->requiredBufferSize,
        .extent = {.width = 2, .height = 2},
    };
    auto layoutB = device.queryTextureTransferLayout(*textureB, sliceB);
    REQUIRE(layoutB);

    auto staging = device.createBuffer({
        .size = layoutB->requiredBufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan in-flight staging",
    });
    auto readback = device.createBuffer({
        .size = layoutA->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan in-flight readback",
    });
    REQUIRE(staging);
    REQUIRE(readback);

    std::vector<std::byte> stagingBytes(static_cast<std::size_t>(layoutB->requiredBufferSize));
    writeTexels(stagingBytes.data(), *layoutA, 0, 0, 2, 2);
    writeTexels(stagingBytes.data() + static_cast<std::size_t>(sliceB.bufferOffset), *layoutA, 3, 1, 2, 2);
    REQUIRE(device.writeBuffer(*staging, 0, stagingBytes).has_value());

    auto first = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    auto second = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(first);
    REQUIRE(second);

    REQUIRE(device
                .textureBarrier(*first,
                                {
                                    .texture = *textureA,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    REQUIRE(device.copyBufferToTexture(*first, *staging, *textureA, std::array{sliceA}).has_value());
    auto firstPoint = device.submit(std::move(*first));
    REQUIRE(firstPoint);

    REQUIRE(device
                .textureBarrier(*second,
                                {
                                    .texture = *textureB,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    REQUIRE(device.copyBufferToTexture(*second, *staging, *textureB, std::array{sliceB}).has_value());
    auto secondPoint = device.submit(std::move(*second));
    REQUIRE(secondPoint);

    REQUIRE(device.wait(*firstPoint).has_value());
    REQUIRE(device.wait(*secondPoint).has_value());
    device.collectGarbage();

    auto readA = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(readA);
    REQUIRE(device
                .textureBarrier(*readA,
                                {
                                    .texture = *textureA,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::copyDestination,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device.copyTextureToBuffer(*readA, *textureA, *readback, std::array{sliceA}).has_value());
    auto readAPoint = device.submit(std::move(*readA));
    REQUIRE(readAPoint);
    REQUIRE(device.wait(*readAPoint).has_value());
    std::vector<std::byte> actual(static_cast<std::size_t>(layoutA->requiredBufferSize));
    REQUIRE(device.readBuffer(*readback, 0, actual).has_value());
    CHECK(texelsMatch(actual.data(), *layoutA, 0, 0, 2, 2));

    auto readB = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(readB);
    REQUIRE(device
                .textureBarrier(*readB,
                                {
                                    .texture = *textureB,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::copyDestination,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device.copyTextureToBuffer(*readB, *textureB, *readback, std::array{sliceA}).has_value());
    auto readBPoint = device.submit(std::move(*readB));
    REQUIRE(readBPoint);
    REQUIRE(device.wait(*readBPoint).has_value());
    REQUIRE(device.readBuffer(*readback, 0, actual).has_value());
    CHECK(texelsMatch(actual.data(), *layoutA, 3, 1, 2, 2));

    REQUIRE(device.destroyTexture(*textureA).has_value());
    REQUIRE(device.destroyTexture(*textureB).has_value());
    REQUIRE(device.destroyBuffer(*staging).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan compressed and depth transfers publish block and plane footprints",
          "[size:medium][rhi][vulkan][transfer][format]")
{
    auto created = createVulkanDevice();
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    const auto depthFeatures = device.queryFormatProperties(rachis::Format::d32Float);
    if (rachis::hasAll(depthFeatures.features,
                       rachis::FormatFeature::copySource | rachis::FormatFeature::copyDestination))
    {
        auto depth = device.createTexture({
            .width = 2,
            .height = 2,
            .format = rachis::Format::d32Float,
            .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                     rachis::TextureUsage::depthStencil,
            .label = "Vulkan depth transfer",
        });
        REQUIRE(depth);
        const rachis::BufferTextureCopyRegion depthRegion{
            .textureSubresource = {.aspect = rachis::TextureAspect::depth},
            .extent = {.width = 2, .height = 2},
        };
        auto depthLayout = device.queryTextureTransferLayout(*depth, depthRegion);
        REQUIRE(depthLayout);
        CHECK(depthLayout->rowDataSize == 8);
        CHECK(depthLayout->bytesPerRow == 8);
        CHECK_FALSE(device
                        .queryTextureTransferLayout(*depth,
                                                    {
                                                        .extent = {.width = 2, .height = 2},
                                                    })
                        .has_value());

        auto upload = device.createBuffer({
            .size = depthLayout->requiredBufferSize,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
        });
        auto readback = device.createBuffer({
            .size = depthLayout->requiredBufferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
        });
        REQUIRE(upload);
        REQUIRE(readback);
        const std::array<float, 4> depths{0.125F, 0.25F, 0.5F, 0.75F};
        REQUIRE(device.writeBuffer(*upload, 0, std::as_bytes(std::span{depths})).has_value());

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command);
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *depth,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessAfter = rachis::Access::copyDestination,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::copyDestination,
                                        .subresources = {.aspect = rachis::TextureAspect::depth},
                                    })
                    .has_value());
        REQUIRE(device.copyBufferToTexture(*command, *upload, *depth, std::array{depthRegion}).has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *depth,
                                        .stagesBefore = rachis::PipelineStage::copy,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::copyDestination,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::copyDestination,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                        .subresources = {.aspect = rachis::TextureAspect::depth},
                                    })
                    .has_value());
        REQUIRE(device.copyTextureToBuffer(*command, *depth, *readback, std::array{depthRegion}).has_value());
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted);
        REQUIRE(device.wait(*submitted).has_value());
        std::array<float, 4> actual{};
        REQUIRE(device.readBuffer(*readback, 0, std::as_writable_bytes(std::span{actual})).has_value());
        CHECK(actual == depths);
        REQUIRE(device.destroyTexture(*depth).has_value());
        REQUIRE(device.destroyBuffer(*upload).has_value());
        REQUIRE(device.destroyBuffer(*readback).has_value());
    }

    const auto bcFeatures = device.queryFormatProperties(rachis::Format::bc1RgbaUnorm);
    if (rachis::hasAll(bcFeatures.features, rachis::FormatFeature::copySource | rachis::FormatFeature::copyDestination))
    {
        auto compressed = device.createTexture({
            .width = 8,
            .height = 4,
            .format = rachis::Format::bc1RgbaUnorm,
            .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                     rachis::TextureUsage::sampled,
            .label = "Vulkan BC1 transfer",
        });
        REQUIRE(compressed);
        const rachis::BufferTextureCopyRegion compressedRegion{
            .extent = {.width = 8, .height = 4},
        };
        auto compressedLayout = device.queryTextureTransferLayout(*compressed, compressedRegion);
        REQUIRE(compressedLayout);
        CHECK(compressedLayout->bufferOffsetAlignment == 8);
        CHECK(compressedLayout->rowPitchAlignment == 8);
        CHECK(compressedLayout->bytesPerRow == 16);
        CHECK(compressedLayout->rowCount == 1);
        CHECK(compressedLayout->rowDataSize == 16);
        CHECK_FALSE(device
                        .queryTextureTransferLayout(*compressed,
                                                    {
                                                        .textureOrigin = {.x = 1},
                                                        .extent = {.width = 4, .height = 4},
                                                    })
                        .has_value());

        auto upload = device.createBuffer({
            .size = compressedLayout->requiredBufferSize,
            .usage = rachis::BufferUsage::copySource,
            .memoryType = rachis::MemoryType::upload,
        });
        auto readback = device.createBuffer({
            .size = compressedLayout->requiredBufferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::readback,
        });
        REQUIRE(upload);
        REQUIRE(readback);
        std::vector<std::byte> blocks(static_cast<std::size_t>(compressedLayout->requiredBufferSize));
        for (std::size_t index = 0; index < blocks.size(); ++index)
        {
            blocks[index] = static_cast<std::byte>(index + 3U);
        }
        REQUIRE(device.writeBuffer(*upload, 0, blocks).has_value());

        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
        REQUIRE(command);
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *compressed,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessAfter = rachis::Access::copyDestination,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::copyDestination,
                                    })
                    .has_value());
        REQUIRE(device.copyBufferToTexture(*command, *upload, *compressed, std::array{compressedRegion}).has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *compressed,
                                        .stagesBefore = rachis::PipelineStage::copy,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::copyDestination,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::copyDestination,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                    })
                    .has_value());
        REQUIRE(device.copyTextureToBuffer(*command, *compressed, *readback, std::array{compressedRegion}).has_value());
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted);
        REQUIRE(device.wait(*submitted).has_value());
        std::vector<std::byte> actual(blocks.size());
        REQUIRE(device.readBuffer(*readback, 0, actual).has_value());
        CHECK(actual == blocks);
        REQUIRE(device.destroyTexture(*compressed).has_value());
        REQUIRE(device.destroyBuffer(*upload).has_value());
        REQUIRE(device.destroyBuffer(*readback).has_value());
    }

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan copy queues reject depth and stencil buffer-texture copies their queue family cannot record",
          "[size:medium][rhi][vulkan][transfer][format]")
{
    auto created = createVulkanDevice();
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    constexpr std::uint32_t extent = 4;
    const auto submitAndWait = [&device](rachis::CommandBuffer& command) {
        auto submitted = device.submit(std::move(command));
        REQUIRE(submitted);
        REQUIRE(device.wait(*submitted).has_value());
    };
    const auto acceptedOrUnsupported = [](const rachis::Status& status) {
        return status.has_value() || status.error().code == rachis::ErrorCode::unsupported;
    };

    for (const auto& [format, aspect] : {std::pair{rachis::Format::d32Float, rachis::TextureAspect::depth},
                                         std::pair{rachis::Format::d24UnormS8Uint, rachis::TextureAspect::stencil}})
    {
        CAPTURE(format, aspect);
        const auto features = device.queryFormatProperties(format);
        if (!rachis::hasAll(features.features,
                            rachis::FormatFeature::copySource | rachis::FormatFeature::copyDestination |
                                rachis::FormatFeature::depthStencilAttachment))
        {
            continue;
        }
        auto texture = device.createTexture({
            .width = extent,
            .height = extent,
            .format = format,
            .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                     rachis::TextureUsage::depthStencil,
            .label = "Vulkan copy-queue depth-stencil transfer",
        });
        REQUIRE(texture);
        const rachis::BufferTextureCopyRegion region{
            .textureSubresource = {.aspect = aspect},
            .extent = {.width = extent, .height = extent},
        };
        auto layout = device.queryTextureTransferLayout(*texture, region);
        REQUIRE(layout);
        const auto texelBytes = static_cast<std::size_t>(layout->rowDataSize / extent);
        const auto texelOffset = [&layout, texelBytes](std::uint32_t x, std::uint32_t y) {
            return (static_cast<std::size_t>(y) * layout->bytesPerRow) + (static_cast<std::size_t>(x) * texelBytes);
        };
        std::vector<std::byte> expected(static_cast<std::size_t>(layout->requiredBufferSize));
        for (std::uint32_t y = 0; y < extent; ++y)
        {
            for (std::uint32_t x = 0; x < extent; ++x)
            {
                if (aspect == rachis::TextureAspect::depth)
                {
                    const auto value = static_cast<float>((y * extent) + x + 1U) / 32.0F;
                    std::memcpy(expected.data() + texelOffset(x, y), &value, sizeof(value));
                }
                else
                {
                    expected[texelOffset(x, y)] = static_cast<std::byte>((y * 53U) + (x * 7U) + 3U);
                }
            }
        }

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
        REQUIRE(device.writeBuffer(*upload, 0, expected).has_value());

        auto copyUpload = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
        REQUIRE(copyUpload);
        REQUIRE(device
                    .textureBarrier(*copyUpload,
                                    {
                                        .texture = *texture,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessAfter = rachis::Access::copyDestination,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::copyDestination,
                                        .subresources = {.aspect = aspect},
                                    })
                    .has_value());
        const auto copyQueueUpload = device.copyBufferToTexture(*copyUpload, *upload, *texture, std::array{region});
        CHECK(acceptedOrUnsupported(copyQueueUpload));
        submitAndWait(*copyUpload);
        if (!copyQueueUpload)
        {
            auto graphicsUpload = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
            REQUIRE(graphicsUpload);
            REQUIRE(device.copyBufferToTexture(*graphicsUpload, *upload, *texture, std::array{region}).has_value());
            submitAndWait(*graphicsUpload);
        }

        auto copyReadback = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
        REQUIRE(copyReadback);
        REQUIRE(device
                    .textureBarrier(*copyReadback,
                                    {
                                        .texture = *texture,
                                        .stagesBefore = rachis::PipelineStage::copy,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::copyDestination,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::copyDestination,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                        .subresources = {.aspect = aspect},
                                    })
                    .has_value());
        const auto copyQueueReadback =
            device.copyTextureToBuffer(*copyReadback, *texture, *readback, std::array{region});
        CHECK(acceptedOrUnsupported(copyQueueReadback));
        CHECK(copyQueueReadback.has_value() == copyQueueUpload.has_value());
        submitAndWait(*copyReadback);
        if (!copyQueueReadback)
        {
            auto graphicsReadback = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
            REQUIRE(graphicsReadback);
            REQUIRE(device.copyTextureToBuffer(*graphicsReadback, *texture, *readback, std::array{region}).has_value());
            submitAndWait(*graphicsReadback);
        }

        std::vector<std::byte> actual(expected.size());
        REQUIRE(device.readBuffer(*readback, 0, actual).has_value());
        for (std::uint32_t y = 0; y < extent; ++y)
        {
            for (std::uint32_t x = 0; x < extent; ++x)
            {
                CAPTURE(x, y);
                CHECK(std::memcmp(actual.data() + texelOffset(x, y), expected.data() + texelOffset(x, y), texelBytes) ==
                      0);
            }
        }
        REQUIRE(device.destroyTexture(*texture).has_value());
        REQUIRE(device.destroyBuffer(*upload).has_value());
        REQUIRE(device.destroyBuffer(*readback).has_value());
    }

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
