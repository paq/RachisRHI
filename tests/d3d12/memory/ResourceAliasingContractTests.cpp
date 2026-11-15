#include "D3D12/D3D12TestHooks.h"
#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace
{

    class ForceLegacyBarriers final
    {
    public:
        explicit ForceLegacyBarriers(bool enabled = true)
            : previous_(rachis::d3d12::test::forceLegacyBarriers().exchange(enabled, std::memory_order_relaxed))
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

} // namespace

TEST_CASE("aliasable placed resources require explicit ordered activation",
          "[size:medium][rhi][memory][aliasing][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).enhancedBarriers);

    constexpr std::uint64_t bufferSize = 257;
    const rachis::BufferDesc placedBufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Aliased placed buffer",
    };
    const auto bufferRequirements = device.queryBufferMemoryRequirements(placedBufferDesc);
    REQUIRE(bufferRequirements.has_value());
    CHECK_FALSE(device
                    .createResourceHeap({
                        .size = bufferRequirements->size,
                        .memoryType = rachis::MemoryType::upload,
                        .flags = rachis::ResourceHeapFlag::allowAliasing,
                    })
                    .has_value());
    const auto bufferHeap = device.createResourceHeap({
        .size = bufferRequirements->size,
        .alignment = bufferRequirements->alignment,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::buffer,
        .flags = rachis::ResourceHeapFlag::allowAliasing,
        .label = "Aliased buffer heap",
    });
    REQUIRE(bufferHeap.has_value());
    const auto firstBuffer = device.createPlacedBuffer({.heap = *bufferHeap}, placedBufferDesc);
    const auto secondBuffer = device.createPlacedBuffer({.heap = *bufferHeap}, placedBufferDesc);
    REQUIRE(firstBuffer.has_value());
    REQUIRE(secondBuffer.has_value());

    const auto otherBufferHeap = device.createResourceHeap({
        .size = bufferRequirements->size,
        .alignment = bufferRequirements->alignment,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::buffer,
        .flags = rachis::ResourceHeapFlag::allowAliasing,
    });
    REQUIRE(otherBufferHeap.has_value());
    const auto otherBuffer = device.createPlacedBuffer({.heap = *otherBufferHeap}, placedBufferDesc);
    REQUIRE(otherBuffer.has_value());

    const auto upload = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    const auto readback = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(upload.has_value());
    REQUIRE(readback.has_value());
    std::array<std::byte, bufferSize> expected{};
    for (std::size_t index = 0; index < expected.size(); ++index)
    {
        expected[index] = static_cast<std::byte>((index * 29U + 7U) & 0xFFU);
    }
    REQUIRE(device.writeBuffer(*upload, 0, expected).has_value());

    const rachis::TextureDesc textureDesc{
        .width = 65,
        .height = 33,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Aliased placed texture",
    };
    const auto textureRequirements = device.queryTextureMemoryRequirements(textureDesc);
    REQUIRE(textureRequirements.has_value());
    const auto textureHeap = device.createResourceHeap({
        .size = textureRequirements->size,
        .alignment = textureRequirements->alignment,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::nonAttachmentTexture,
        .flags = rachis::ResourceHeapFlag::allowAliasing,
        .label = "Aliased texture heap",
    });
    REQUIRE(textureHeap.has_value());
    const auto firstTexture = device.createPlacedTexture({.heap = *textureHeap}, textureDesc);
    const auto secondTexture = device.createPlacedTexture({.heap = *textureHeap}, textureDesc);
    REQUIRE(firstTexture.has_value());
    REQUIRE(secondTexture.has_value());

    const auto textureTransfer =
        device.queryTextureTransferLayout(*secondTexture, {.extent = {textureDesc.width, textureDesc.height, 1}});
    REQUIRE(textureTransfer.has_value());
    const auto textureUpload = device.createBuffer({
        .size = textureTransfer->requiredBufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
    });
    const auto textureReadback = device.createBuffer({
        .size = textureTransfer->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(textureUpload.has_value());
    REQUIRE(textureReadback.has_value());
    {
        auto mapped = device.mapBuffer(*textureUpload);
        REQUIRE(mapped.has_value());
        std::memset(mapped->data, 0, static_cast<std::size_t>(textureTransfer->requiredBufferSize));
        for (std::uint32_t row = 0; row < textureTransfer->rowCount; ++row)
        {
            for (std::uint32_t byteIndex = 0; byteIndex < textureTransfer->rowDataSize; ++byteIndex)
            {
                mapped->data[static_cast<std::size_t>(row) * textureTransfer->bytesPerRow + byteIndex] =
                    static_cast<std::byte>((row * 37U + byteIndex * 11U + 3U) & 0xFFU);
            }
        }
        REQUIRE(device.unmapBuffer(*textureUpload).has_value());
    }

    const rachis::TextureDesc attachmentDesc{
        .width = 16,
        .height = 16,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource,
        .label = "Aliased placed attachment texture",
    };
    const auto attachmentRequirements = device.queryTextureMemoryRequirements(attachmentDesc);
    REQUIRE(attachmentRequirements.has_value());
    const auto attachmentHeap = device.createResourceHeap({
        .size = attachmentRequirements->size,
        .alignment = attachmentRequirements->alignment,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::attachmentTexture,
        .flags = rachis::ResourceHeapFlag::allowAliasing,
        .label = "Aliased attachment texture heap",
    });
    REQUIRE(attachmentHeap.has_value());
    const auto firstAttachment = device.createPlacedTexture({.heap = *attachmentHeap}, attachmentDesc);
    const auto secondAttachment = device.createPlacedTexture({.heap = *attachmentHeap}, attachmentDesc);
    REQUIRE(firstAttachment.has_value());
    REQUIRE(secondAttachment.has_value());
    const auto firstAttachmentView = device.createTextureView({
        .texture = *firstAttachment,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    const auto secondAttachmentView = device.createTextureView({
        .texture = *secondAttachment,
        .usage = rachis::TextureViewUsage::renderTarget,
    });
    REQUIRE(firstAttachmentView.has_value());
    REQUIRE(secondAttachmentView.has_value());
    const auto attachmentTransfer = device.queryTextureTransferLayout(
        *secondAttachment, {.extent = {attachmentDesc.width, attachmentDesc.height, 1}});
    REQUIRE(attachmentTransfer.has_value());
    const auto attachmentReadback = device.createBuffer({
        .size = attachmentTransfer->requiredBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(attachmentReadback.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    const rachis::BufferAliasActivation bufferActivation{
        .before = *firstBuffer,
        .after = *secondBuffer,
        .stagesBefore = rachis::PipelineStage::all,
        .stagesAfter = rachis::PipelineStage::copy,
        .accessBefore = rachis::Access::common,
        .accessAfter = rachis::Access::copyDestination,
    };
    auto invalidBufferActivation = bufferActivation;
    invalidBufferActivation.accessAfter = rachis::Access::copySource;
    CHECK_FALSE(device.activateBufferAlias(*command, invalidBufferActivation).has_value());
    invalidBufferActivation = bufferActivation;
    invalidBufferActivation.after = *otherBuffer;
    CHECK_FALSE(device.activateBufferAlias(*command, invalidBufferActivation).has_value());
    REQUIRE(device.activateBufferAlias(*command, bufferActivation).has_value());
    REQUIRE(
        device.copyBuffer(*command, {.source = *upload, .destination = *secondBuffer, .size = bufferSize}).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *secondBuffer,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device.copyBuffer(*command, {.source = *secondBuffer, .destination = *readback, .size = bufferSize})
                .has_value());

    const rachis::TextureAliasActivation textureActivation{
        .before = *firstTexture,
        .after = *secondTexture,
        .stagesBefore = rachis::PipelineStage::all,
        .stagesAfter = rachis::PipelineStage::copy,
        .accessBefore = rachis::Access::common,
        .accessAfter = rachis::Access::copyDestination,
        .layoutBefore = rachis::TextureLayout::common,
        .layoutAfter = rachis::TextureLayout::copyDestination,
    };
    auto invalidTextureActivation = textureActivation;
    invalidTextureActivation.accessAfter = rachis::Access::copySource;
    invalidTextureActivation.layoutAfter = rachis::TextureLayout::copySource;
    CHECK_FALSE(device.activateTextureAlias(*command, invalidTextureActivation).has_value());
    REQUIRE(device.activateTextureAlias(*command, textureActivation).has_value());
    REQUIRE(device
                .copyBufferToTexture(*command,
                                     {
                                         .buffer = *textureUpload,
                                         .bytesPerRow = textureTransfer->bytesPerRow,
                                         .rowsPerImage = textureTransfer->rowsPerImage,
                                         .texture = *secondTexture,
                                         .extent = {textureDesc.width, textureDesc.height, 1},
                                     })
                .has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *secondTexture,
                                    .stagesBefore = rachis::PipelineStage::copy,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::copyDestination,
                                    .accessAfter = rachis::Access::copySource,
                                    .layoutBefore = rachis::TextureLayout::copyDestination,
                                    .layoutAfter = rachis::TextureLayout::copySource,
                                })
                .has_value());
    REQUIRE(device
                .copyTextureToBuffer(*command,
                                     {
                                         .buffer = *textureReadback,
                                         .bytesPerRow = textureTransfer->bytesPerRow,
                                         .rowsPerImage = textureTransfer->rowsPerImage,
                                         .texture = *secondTexture,
                                         .extent = {textureDesc.width, textureDesc.height, 1},
                                     })
                .has_value());

    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *firstAttachment,
                                    .stagesBefore = rachis::PipelineStage::all,
                                    .stagesAfter = rachis::PipelineStage::renderTarget,
                                    .accessAfter = rachis::Access::renderTarget,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::renderTarget,
                                })
                .has_value());
    const std::array firstPassAttachments{
        rachis::ColorAttachment{
            .view = *firstAttachmentView,
            .loadOperation = rachis::LoadOperation::clear,
            .storeOperation = rachis::StoreOperation::store,
            .clearValue = {1.0F, 0.0F, 0.0F, 1.0F},
        },
    };
    REQUIRE(device.beginRendering(*command, {.colorAttachments = firstPassAttachments}).has_value());
    REQUIRE(device.endRendering(*command).has_value());

    REQUIRE(device
                .activateTextureAlias(*command,
                                      {
                                          .before = *firstAttachment,
                                          .after = *secondAttachment,
                                          .stagesBefore = rachis::PipelineStage::renderTarget,
                                          .stagesAfter = rachis::PipelineStage::renderTarget,
                                          .accessBefore = rachis::Access::renderTarget,
                                          .accessAfter = rachis::Access::renderTarget,
                                          .layoutBefore = rachis::TextureLayout::renderTarget,
                                          .layoutAfter = rachis::TextureLayout::renderTarget,
                                      })
                .has_value());
    const std::array secondPassAttachments{
        rachis::ColorAttachment{
            .view = *secondAttachmentView,
            .loadOperation = rachis::LoadOperation::clear,
            .storeOperation = rachis::StoreOperation::store,
            .clearValue = {0.0F, 1.0F, 0.0F, 1.0F},
        },
    };
    REQUIRE(device.beginRendering(*command, {.colorAttachments = secondPassAttachments}).has_value());
    REQUIRE(device.endRendering(*command).has_value());
    REQUIRE(device
                .textureBarrier(*command,
                                {
                                    .texture = *secondAttachment,
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
                                     {
                                         .buffer = *attachmentReadback,
                                         .bytesPerRow = attachmentTransfer->bytesPerRow,
                                         .rowsPerImage = attachmentTransfer->rowsPerImage,
                                         .texture = *secondAttachment,
                                         .extent = {attachmentDesc.width, attachmentDesc.height, 1},
                                     })
                .has_value());

    REQUIRE(device.destroyBuffer(*firstBuffer).has_value());
    REQUIRE(device.destroyBuffer(*secondBuffer).has_value());
    REQUIRE(device.destroyResourceHeap(*bufferHeap).has_value());
    REQUIRE(device.destroyTexture(*firstTexture).has_value());
    REQUIRE(device.destroyTexture(*secondTexture).has_value());
    REQUIRE(device.destroyResourceHeap(*textureHeap).has_value());
    REQUIRE(device.destroyTextureView(*firstAttachmentView).has_value());
    REQUIRE(device.destroyTextureView(*secondAttachmentView).has_value());
    REQUIRE(device.destroyTexture(*firstAttachment).has_value());
    REQUIRE(device.destroyTexture(*secondAttachment).has_value());
    REQUIRE(device.destroyResourceHeap(*attachmentHeap).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    std::array<std::byte, bufferSize> actual{};
    REQUIRE(device.readBuffer(*readback, 0, actual).has_value());
    CHECK(actual == expected);
    {
        auto mapped = device.mapBuffer(*textureReadback);
        REQUIRE(mapped.has_value());
        bool matched = true;
        for (std::uint32_t row = 0; row < textureTransfer->rowCount && matched; ++row)
        {
            for (std::uint32_t byteIndex = 0; byteIndex < textureTransfer->rowDataSize; ++byteIndex)
            {
                const auto expectedByte = static_cast<std::byte>((row * 37U + byteIndex * 11U + 3U) & 0xFFU);
                if (mapped->data[static_cast<std::size_t>(row) * textureTransfer->bytesPerRow + byteIndex] !=
                    expectedByte)
                {
                    matched = false;
                    break;
                }
            }
        }
        CHECK(matched);
        REQUIRE(device.unmapBuffer(*textureReadback).has_value());
    }
    {
        auto mapped = device.mapBuffer(*attachmentReadback);
        REQUIRE(mapped.has_value());
        CHECK(mapped->data[0] == std::byte{0});
        CHECK(mapped->data[1] == std::byte{0xFF});
        CHECK(mapped->data[2] == std::byte{0});
        CHECK(mapped->data[3] == std::byte{0xFF});
        REQUIRE(device.unmapBuffer(*attachmentReadback).has_value());
    }
    REQUIRE(device.destroyBuffer(*otherBuffer).has_value());
    REQUIRE(device.destroyResourceHeap(*otherBufferHeap).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*textureUpload).has_value());
    REQUIRE(device.destroyBuffer(*textureReadback).has_value());
    REQUIRE(device.destroyBuffer(*attachmentReadback).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("depth placed texture alias activation covers every format aspect before a depth pass",
          "[size:medium][rhi][memory][aliasing][warp]")
{
    const auto format = GENERATE(rachis::Format::d32Float, rachis::Format::d24UnormS8Uint);
    CAPTURE(static_cast<int>(format));
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).enhancedBarriers);

    const bool stencil = format == rachis::Format::d24UnormS8Uint;
    const std::array formatAspects{rachis::TextureAspect::depth, rachis::TextureAspect::stencil};
    const auto aspects = std::span(formatAspects).first(stencil ? 2U : 1U);
    const rachis::TextureDesc depthDesc{
        .width = 16,
        .height = 16,
        .format = format,
        .usage = rachis::TextureUsage::depthStencil | rachis::TextureUsage::sampled,
        .label = "Aliased placed depth texture",
    };
    const auto requirements = device.queryTextureMemoryRequirements(depthDesc);
    REQUIRE(requirements.has_value());
    const auto heap = device.createResourceHeap({
        .size = requirements->size,
        .alignment = requirements->alignment,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::attachmentTexture,
        .flags = rachis::ResourceHeapFlag::allowAliasing,
        .label = "Aliased depth texture heap",
    });
    REQUIRE(heap.has_value());
    const auto first = device.createPlacedTexture({.heap = *heap}, depthDesc);
    const auto second = device.createPlacedTexture({.heap = *heap}, depthDesc);
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());
    const auto firstView = device.createTextureView({
        .texture = *first,
        .usage = rachis::TextureViewUsage::depthStencil,
        .aspect = rachis::TextureAspect::depth,
    });
    const auto secondView = device.createTextureView({
        .texture = *second,
        .usage = rachis::TextureViewUsage::depthStencil,
        .aspect = rachis::TextureAspect::depth,
    });
    REQUIRE(firstView.has_value());
    REQUIRE(secondView.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    for (const auto aspect : aspects)
    {
        REQUIRE(device
                    .textureBarrier(*command,
                                    {
                                        .texture = *first,
                                        .stagesBefore = rachis::PipelineStage::all,
                                        .stagesAfter = rachis::PipelineStage::depthStencil,
                                        .accessAfter = rachis::Access::depthStencilWrite,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::depthStencilWrite,
                                        .subresources = {.aspect = aspect},
                                    })
                    .has_value());
    }
    const auto depthAttachment = [stencil](rachis::TextureViewHandle view, float depth, std::uint32_t stencilValue) {
        return rachis::DepthStencilAttachment{
            .view = view,
            .depthLoadOperation = rachis::LoadOperation::clear,
            .depthClearValue = depth,
            .stencilLoadOperation = stencil ? rachis::LoadOperation::clear : rachis::LoadOperation::load,
            .stencilClearValue = stencil ? stencilValue : 0U,
        };
    };
    REQUIRE(device.beginRendering(*command, {.depthStencilAttachment = depthAttachment(*firstView, 1.0F, 0x11)})
                .has_value());
    REQUIRE(device.endRendering(*command).has_value());

    const auto activated = device.activateTextureAlias(*command,
                                                       {
                                                           .before = *first,
                                                           .after = *second,
                                                           .stagesBefore = rachis::PipelineStage::depthStencil,
                                                           .stagesAfter = rachis::PipelineStage::depthStencil,
                                                           .accessBefore = rachis::Access::depthStencilWrite,
                                                           .accessAfter = rachis::Access::depthStencilWrite,
                                                           .layoutBefore = rachis::TextureLayout::depthStencilWrite,
                                                           .layoutAfter = rachis::TextureLayout::depthStencilWrite,
                                                       });
    if (!activated)
    {
        UNSCOPED_INFO(activated.error().message);
    }
    REQUIRE(activated.has_value());
    REQUIRE(device.beginRendering(*command, {.depthStencilAttachment = depthAttachment(*secondView, 0.25F, 0x5A)})
                .has_value());
    REQUIRE(device.endRendering(*command).has_value());

    REQUIRE(device.destroyTextureView(*firstView).has_value());
    REQUIRE(device.destroyTextureView(*secondView).has_value());
    REQUIRE(device.destroyTexture(*first).has_value());
    REQUIRE(device.destroyTexture(*second).has_value());
    REQUIRE(device.destroyResourceHeap(*heap).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("placed resource alias activation reports unsupported under legacy barriers",
          "[size:medium][rhi][memory][aliasing][warp]")
{
    const ForceLegacyBarriers forceLegacyBarriers;
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    REQUIRE_FALSE(rachis::test::requireCapabilities(device).enhancedBarriers);

    const rachis::BufferDesc bufferDesc{
        .size = 256,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
    };
    const auto bufferRequirements = device.queryBufferMemoryRequirements(bufferDesc);
    REQUIRE(bufferRequirements.has_value());
    const auto bufferHeap = device.createResourceHeap({
        .size = bufferRequirements->size,
        .alignment = bufferRequirements->alignment,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::buffer,
        .flags = rachis::ResourceHeapFlag::allowAliasing,
    });
    REQUIRE(bufferHeap.has_value());
    const auto firstBuffer = device.createPlacedBuffer({.heap = *bufferHeap}, bufferDesc);
    const auto secondBuffer = device.createPlacedBuffer({.heap = *bufferHeap}, bufferDesc);
    REQUIRE(firstBuffer.has_value());
    REQUIRE(secondBuffer.has_value());

    const rachis::TextureDesc textureDesc{
        .width = 16,
        .height = 16,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
    };
    const auto textureRequirements = device.queryTextureMemoryRequirements(textureDesc);
    REQUIRE(textureRequirements.has_value());
    const auto textureHeap = device.createResourceHeap({
        .size = textureRequirements->size,
        .alignment = textureRequirements->alignment,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::nonAttachmentTexture,
        .flags = rachis::ResourceHeapFlag::allowAliasing,
    });
    REQUIRE(textureHeap.has_value());
    const auto firstTexture = device.createPlacedTexture({.heap = *textureHeap}, textureDesc);
    const auto secondTexture = device.createPlacedTexture({.heap = *textureHeap}, textureDesc);
    REQUIRE(firstTexture.has_value());
    REQUIRE(secondTexture.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    const auto bufferResult = device.activateBufferAlias(*command,
                                                         {
                                                             .before = *firstBuffer,
                                                             .after = *secondBuffer,
                                                             .stagesBefore = rachis::PipelineStage::all,
                                                             .stagesAfter = rachis::PipelineStage::copy,
                                                             .accessBefore = rachis::Access::common,
                                                             .accessAfter = rachis::Access::copyDestination,
                                                         });
    REQUIRE_FALSE(bufferResult.has_value());
    CHECK(bufferResult.error().code == rachis::ErrorCode::unsupported);

    const auto textureResult = device.activateTextureAlias(*command,
                                                           {
                                                               .before = *firstTexture,
                                                               .after = *secondTexture,
                                                               .stagesBefore = rachis::PipelineStage::all,
                                                               .stagesAfter = rachis::PipelineStage::copy,
                                                               .accessBefore = rachis::Access::common,
                                                               .accessAfter = rachis::Access::copyDestination,
                                                               .layoutBefore = rachis::TextureLayout::common,
                                                               .layoutAfter = rachis::TextureLayout::copyDestination,
                                                           });
    REQUIRE_FALSE(textureResult.has_value());
    CHECK(textureResult.error().code == rachis::ErrorCode::unsupported);

    REQUIRE(device.destroyBuffer(*firstBuffer).has_value());
    REQUIRE(device.destroyBuffer(*secondBuffer).has_value());
    REQUIRE(device.destroyResourceHeap(*bufferHeap).has_value());
    REQUIRE(device.destroyTexture(*firstTexture).has_value());
    REQUIRE(device.destroyTexture(*secondTexture).has_value());
    REQUIRE(device.destroyResourceHeap(*textureHeap).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
