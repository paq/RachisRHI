#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#ifdef _WINDOWS_
#error Public Rachis headers must not include Windows.h.
#endif

#ifdef VK_VERSION_1_0
#error Public Rachis headers must not include Vulkan headers.
#endif

#include "Internal/VulkanResources.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace
{

    void captureErrors(const rachis::DiagnosticMessage& message, void* userData) noexcept
    {
        if (message.severity == rachis::DiagnosticSeverity::error ||
            message.severity == rachis::DiagnosticSeverity::corruption)
        {
            static_cast<std::atomic_uint64_t*>(userData)->fetch_add(1, std::memory_order_relaxed);
        }
    }

    [[nodiscard]] rachis::DeviceDesc vulkanDesc(std::atomic_uint64_t* errors = nullptr) noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::vulkan,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
            .diagnosticCallback = errors != nullptr ? &captureErrors : nullptr,
            .diagnosticUserData = errors,
        };
    }

    template<typename Create>
    [[nodiscard]] auto createWithBadAlloc(Create&& create)
    {
        rachis::vulkan::injectVulkanBadAlloc(rachis::vulkan::VulkanBadAllocPoint::nativeKeepAlive, true);
        auto created = create();
        rachis::vulkan::injectVulkanBadAlloc(rachis::vulkan::VulkanBadAllocPoint::nativeKeepAlive, false);
        return created;
    }

    [[nodiscard]] const rachis::ResourceMemoryEntry* findByLabel(const rachis::ResourceMemorySnapshot& snapshot,
                                                                 std::string_view label)
    {
        const auto found = std::ranges::find(snapshot.entries, label, &rachis::ResourceMemoryEntry::label);
        return found == snapshot.entries.end() ? nullptr : &*found;
    }

} // namespace

TEST_CASE("Vulkan placed buffer heaps reject retained overlap and outlive their public handle",
          "[size:medium][rhi][vulkan][memory][heap][placed][lifetime]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    const auto baseline = device.captureResourceMemorySnapshot();
    REQUIRE(baseline.has_value());

    const rachis::BufferDesc bufferDesc{
        .size = 4'097,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Placed heap buffer",
    };
    const auto requirements = device.queryBufferMemoryRequirements(bufferDesc);
    REQUIRE(requirements.has_value());
    REQUIRE(requirements->size != 0);

    CHECK_FALSE(device
                    .createResourceHeap({
                        .size = requirements->size + 1,
                        .memoryType = bufferDesc.memoryType,
                    })
                    .has_value());
    const auto heap = device.createResourceHeap({
        .size = requirements->size * 2,
        .memoryType = bufferDesc.memoryType,
        .label = "Placed buffer heap",
    });
    REQUIRE(heap.has_value());

    const auto heapSnapshot = device.captureResourceMemorySnapshot();
    REQUIRE(heapSnapshot.has_value());
    const auto* heapEntry = findByLabel(*heapSnapshot, "Placed buffer heap");
    REQUIRE(heapEntry != nullptr);
    CHECK(heapEntry->kind == rachis::ResourceMemoryKind::resourceHeap);
    CHECK(heapEntry->allocationBytes == requirements->size * 2);
    CHECK(heapEntry->logicalAlive);
    CHECK(heapEntry->dedicatedAllocation);

    auto first = device.createPlacedBuffer({.heap = *heap, .offset = 0}, bufferDesc);
    REQUIRE(first.has_value());

    auto overlap = device.createPlacedBuffer({.heap = *heap, .offset = 0}, bufferDesc);
    REQUIRE_FALSE(overlap.has_value());
    CHECK(overlap.error().code == rachis::ErrorCode::invalidState);
    auto misaligned = device.createPlacedBuffer({.heap = *heap, .offset = 1}, bufferDesc);
    REQUIRE_FALSE(misaligned.has_value());
    CHECK(misaligned.error().code == rachis::ErrorCode::invalidArgument);

    auto mismatchedDesc = bufferDesc;
    mismatchedDesc.memoryType = rachis::MemoryType::upload;
    CHECK_FALSE(device.createPlacedBuffer({.heap = *heap, .offset = requirements->size}, mismatchedDesc).has_value());

    auto second = device.createPlacedBuffer({.heap = *heap, .offset = requirements->size}, bufferDesc);
    REQUIRE(second.has_value());
    {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .bufferBarrier(*command,
                                   rachis::BufferBarrier{
                                       .buffer = *first,
                                       .stagesBefore = rachis::PipelineStage::all,
                                       .stagesAfter = rachis::PipelineStage::copy,
                                       .accessBefore = rachis::Access::common,
                                       .accessAfter = rachis::Access::copyDestination,
                                   })
                    .has_value());
        REQUIRE(device.destroyBuffer(*first).has_value());
        overlap = device.createPlacedBuffer({.heap = *heap, .offset = 0}, bufferDesc);
        REQUIRE_FALSE(overlap.has_value());
        CHECK(overlap.error().code == rachis::ErrorCode::invalidState);
    }

    device.collectGarbage();
    auto replacement = device.createPlacedBuffer({.heap = *heap, .offset = 0}, bufferDesc);
    REQUIRE(replacement.has_value());

    const auto staleHeap = *heap;
    REQUIRE(device.destroyResourceHeap(*heap).has_value());
    CHECK_FALSE(device.createPlacedBuffer({.heap = staleHeap, .offset = 0}, bufferDesc).has_value());
    CHECK_FALSE(device.destroyResourceHeap(staleHeap).has_value());

    const auto retainedHeapSnapshot = device.captureResourceMemorySnapshot();
    REQUIRE(retainedHeapSnapshot.has_value());
    heapEntry = findByLabel(*retainedHeapSnapshot, "Placed buffer heap");
    REQUIRE(heapEntry != nullptr);
    CHECK_FALSE(heapEntry->logicalAlive);

    REQUIRE(device.destroyBuffer(*replacement).has_value());
    REQUIRE(device.destroyBuffer(*second).has_value());
    device.collectGarbage();

    const auto released = device.captureResourceMemorySnapshot();
    REQUIRE(released.has_value());
    CHECK(findByLabel(*released, "Placed buffer heap") == nullptr);
    CHECK(released->liveAllocationBytes == baseline->liveAllocationBytes);
    CHECK(released->retainedAllocationBytes == baseline->retainedAllocationBytes);
    CHECK(released->liveAllocationCount == baseline->liveAllocationCount);
    CHECK(released->retainedAllocationCount == baseline->retainedAllocationCount);
}

TEST_CASE("Vulkan placed texture heaps preserve class and retained-range ownership",
          "[size:medium][rhi][vulkan][memory][heap][placed][texture][lifetime]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    const auto baseline = device.captureResourceMemorySnapshot();
    REQUIRE(baseline.has_value());

    const rachis::TextureDesc sampledDesc{
        .dimension = rachis::TextureDimension::texture2D,
        .width = 65,
        .height = 33,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::copyDestination,
        .label = "Placed sampled texture",
    };
    const auto sampledRequirements = device.queryTextureMemoryRequirements(sampledDesc);
    REQUIRE(sampledRequirements.has_value());

    const rachis::ResourceHeapDesc sampledHeapDesc{
        .size = sampledRequirements->size * 2,
        .alignment = sampledRequirements->alignment,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::nonAttachmentTexture,
        .label = "Placed sampled texture heap",
    };
    auto invalidHeapDesc = sampledHeapDesc;
    invalidHeapDesc.memoryType = rachis::MemoryType::upload;
    CHECK_FALSE(device.createResourceHeap(invalidHeapDesc).has_value());
    invalidHeapDesc = sampledHeapDesc;
    invalidHeapDesc.alignment = 4'096;
    CHECK_FALSE(device.createResourceHeap(invalidHeapDesc).has_value());

    const auto sampledHeap = device.createResourceHeap(sampledHeapDesc);
    REQUIRE(sampledHeap.has_value());
    const auto heapSnapshot = device.captureResourceMemorySnapshot();
    REQUIRE(heapSnapshot.has_value());
    const auto* heapEntry = findByLabel(*heapSnapshot, "Placed sampled texture heap");
    REQUIRE(heapEntry != nullptr);
    CHECK(heapEntry->kind == rachis::ResourceMemoryKind::resourceHeap);
    CHECK(heapEntry->allocationBytes == sampledHeapDesc.size);
    CHECK(heapEntry->logicalAlive);

    auto first = device.createPlacedTexture({.heap = *sampledHeap, .offset = 0}, sampledDesc);
    REQUIRE(first.has_value());
    auto overlap = device.createPlacedTexture({.heap = *sampledHeap, .offset = 0}, sampledDesc);
    REQUIRE_FALSE(overlap.has_value());
    CHECK(overlap.error().code == rachis::ErrorCode::invalidState);
    auto misaligned = device.createPlacedTexture({.heap = *sampledHeap, .offset = 1}, sampledDesc);
    REQUIRE_FALSE(misaligned.has_value());
    CHECK(misaligned.error().code == rachis::ErrorCode::invalidArgument);

    auto second = device.createPlacedTexture({.heap = *sampledHeap, .offset = sampledRequirements->size}, sampledDesc);
    REQUIRE(second.has_value());
    {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = *first,
                                        .stagesBefore = rachis::PipelineStage::all,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::common,
                                        .accessAfter = rachis::Access::copyDestination,
                                        .layoutBefore = rachis::TextureLayout::common,
                                        .layoutAfter = rachis::TextureLayout::copyDestination,
                                    })
                    .has_value());
        REQUIRE(device.destroyTexture(*first).has_value());
        overlap = device.createPlacedTexture({.heap = *sampledHeap, .offset = 0}, sampledDesc);
        REQUIRE_FALSE(overlap.has_value());
        CHECK(overlap.error().code == rachis::ErrorCode::invalidState);
    }

    device.collectGarbage();
    auto replacement = device.createPlacedTexture({.heap = *sampledHeap, .offset = 0}, sampledDesc);
    REQUIRE(replacement.has_value());

    const rachis::TextureDesc attachmentDesc{
        .dimension = rachis::TextureDimension::texture2D,
        .width = 64,
        .height = 32,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget | rachis::TextureUsage::copySource,
        .optimizedClearValue =
            {
                .kind = rachis::OptimizedClearValueKind::color,
                .color = {0.25F, 0.5F, 0.75F, 1.0F},
            },
        .label = "Placed attachment texture",
    };
    CHECK_FALSE(device.createPlacedTexture({.heap = *sampledHeap, .offset = 0}, attachmentDesc).has_value());

    const auto attachmentRequirements = device.queryTextureMemoryRequirements(attachmentDesc);
    REQUIRE(attachmentRequirements.has_value());
    const auto attachmentHeap = device.createResourceHeap({
        .size = attachmentRequirements->size,
        .alignment = attachmentRequirements->alignment,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::attachmentTexture,
        .label = "Placed attachment texture heap",
    });
    REQUIRE(attachmentHeap.has_value());
    const auto attachment = device.createPlacedTexture({.heap = *attachmentHeap, .offset = 0}, attachmentDesc);
    REQUIRE(attachment.has_value());
    CHECK_FALSE(device.createPlacedTexture({.heap = *attachmentHeap, .offset = 0}, sampledDesc).has_value());
    REQUIRE(device.destroyTexture(*attachment).has_value());
    REQUIRE(device.destroyResourceHeap(*attachmentHeap).has_value());

    const auto staleHeap = *sampledHeap;
    REQUIRE(device.destroyResourceHeap(*sampledHeap).has_value());
    CHECK_FALSE(device.createPlacedTexture({.heap = staleHeap, .offset = 0}, sampledDesc).has_value());
    CHECK_FALSE(device.destroyResourceHeap(staleHeap).has_value());

    const auto retainedHeapSnapshot = device.captureResourceMemorySnapshot();
    REQUIRE(retainedHeapSnapshot.has_value());
    heapEntry = findByLabel(*retainedHeapSnapshot, "Placed sampled texture heap");
    REQUIRE(heapEntry != nullptr);
    CHECK_FALSE(heapEntry->logicalAlive);

    REQUIRE(device.destroyTexture(*replacement).has_value());
    REQUIRE(device.destroyTexture(*second).has_value());
    device.collectGarbage();

    const auto released = device.captureResourceMemorySnapshot();
    REQUIRE(released.has_value());
    CHECK(findByLabel(*released, "Placed sampled texture heap") == nullptr);
    CHECK(findByLabel(*released, "Placed attachment texture heap") == nullptr);
    CHECK(released->liveAllocationBytes == baseline->liveAllocationBytes);
    CHECK(released->retainedAllocationBytes == baseline->retainedAllocationBytes);
    CHECK(released->liveAllocationCount == baseline->liveAllocationCount);
    CHECK(released->retainedAllocationCount == baseline->retainedAllocationCount);
}

TEST_CASE("Vulkan placed resources release native objects and ranges when allocation fails",
          "[size:medium][rhi][vulkan][memory][heap][placed][texture][lifetime]")
{
    std::atomic_uint64_t errors{0};
    auto created = rachis::createDevice(vulkanDesc(&errors));
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    const rachis::BufferDesc bufferDesc{
        .size = 4'097,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Placed buffer with a failing keep-alive allocation",
    };
    const auto bufferRequirements = device.queryBufferMemoryRequirements(bufferDesc);
    REQUIRE(bufferRequirements.has_value());
    const auto bufferHeap = device.createResourceHeap({
        .size = bufferRequirements->size,
        .memoryType = bufferDesc.memoryType,
        .label = "Placed buffer heap with a failing keep-alive allocation",
    });
    REQUIRE(bufferHeap.has_value());
    const auto failedBuffer =
        createWithBadAlloc([&] { return device.createPlacedBuffer({.heap = *bufferHeap, .offset = 0}, bufferDesc); });
    REQUIRE_FALSE(failedBuffer.has_value());
    CHECK(failedBuffer.error().code == rachis::ErrorCode::outOfMemory);
    auto buffer = device.createPlacedBuffer({.heap = *bufferHeap, .offset = 0}, bufferDesc);
    REQUIRE(buffer.has_value());

    const rachis::TextureDesc textureDesc{
        .dimension = rachis::TextureDimension::texture2D,
        .width = 65,
        .height = 33,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::copyDestination,
        .label = "Placed texture with a failing keep-alive allocation",
    };
    const auto textureRequirements = device.queryTextureMemoryRequirements(textureDesc);
    REQUIRE(textureRequirements.has_value());
    const auto textureHeap = device.createResourceHeap({
        .size = textureRequirements->size,
        .alignment = textureRequirements->alignment,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::nonAttachmentTexture,
        .label = "Placed texture heap with a failing keep-alive allocation",
    });
    REQUIRE(textureHeap.has_value());
    const auto failedTexture = createWithBadAlloc(
        [&] { return device.createPlacedTexture({.heap = *textureHeap, .offset = 0}, textureDesc); });
    REQUIRE_FALSE(failedTexture.has_value());
    CHECK(failedTexture.error().code == rachis::ErrorCode::outOfMemory);
    auto texture = device.createPlacedTexture({.heap = *textureHeap, .offset = 0}, textureDesc);
    REQUIRE(texture.has_value());

    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyBuffer(*buffer).has_value());
    REQUIRE(device.destroyResourceHeap(*textureHeap).has_value());
    REQUIRE(device.destroyResourceHeap(*bufferHeap).has_value());
    device.collectGarbage();

    CHECK(device.diagnosticCounts().errors == 0);
    device = {};
    CHECK(errors.load() == 0);
}

TEST_CASE("Vulkan aliasable placed resources require explicit ordered activation",
          "[size:medium][rhi][vulkan][memory][aliasing]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
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

    REQUIRE(device.destroyBuffer(*firstBuffer).has_value());
    REQUIRE(device.destroyBuffer(*secondBuffer).has_value());
    REQUIRE(device.destroyResourceHeap(*bufferHeap).has_value());
    REQUIRE(device.destroyTexture(*firstTexture).has_value());
    REQUIRE(device.destroyTexture(*secondTexture).has_value());
    REQUIRE(device.destroyResourceHeap(*textureHeap).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();

    std::array<std::byte, bufferSize> actual{};
    REQUIRE(device.readBuffer(*readback, 0, actual).has_value());
    CHECK(actual == expected);
    REQUIRE(device.destroyBuffer(*otherBuffer).has_value());
    REQUIRE(device.destroyResourceHeap(*otherBufferHeap).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan depth placed texture alias activation covers every format aspect before a depth pass",
          "[size:medium][rhi][vulkan][memory][aliasing]")
{
    const auto format = GENERATE(rachis::Format::d32Float, rachis::Format::d24UnormS8Uint);
    CAPTURE(static_cast<int>(format));
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).enhancedBarriers);
    if (!rachis::hasAll(device.queryFormatProperties(format).features,
                        rachis::FormatFeature::depthStencilAttachment | rachis::FormatFeature::sampled))
    {
        return;
    }

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
    const auto secondView = device.createTextureView({
        .texture = *second,
        .usage = rachis::TextureViewUsage::depthStencil,
        .aspect = rachis::TextureAspect::depth,
    });
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
                                        .stagesAfter = rachis::PipelineStage::pixelShader,
                                        .accessAfter = rachis::Access::shaderResource,
                                        .layoutBefore = rachis::TextureLayout::undefined,
                                        .layoutAfter = rachis::TextureLayout::shaderResource,
                                        .subresources = {.aspect = aspect},
                                    })
                    .has_value());
    }

    const auto activated = device.activateTextureAlias(*command,
                                                       {
                                                           .before = *first,
                                                           .after = *second,
                                                           .stagesBefore = rachis::PipelineStage::pixelShader,
                                                           .stagesAfter = rachis::PipelineStage::depthStencil,
                                                           .accessBefore = rachis::Access::shaderResource,
                                                           .accessAfter = rachis::Access::depthStencilWrite,
                                                           .layoutBefore = rachis::TextureLayout::shaderResource,
                                                           .layoutAfter = rachis::TextureLayout::depthStencilWrite,
                                                       });
    if (!activated)
    {
        UNSCOPED_INFO(activated.error().message);
    }
    REQUIRE(activated.has_value());
    REQUIRE(device
                .beginRendering(*command,
                                {.depthStencilAttachment =
                                     rachis::DepthStencilAttachment{
                                         .view = *secondView,
                                         .depthLoadOperation = rachis::LoadOperation::clear,
                                         .depthClearValue = 0.25F,
                                         .stencilLoadOperation =
                                             stencil ? rachis::LoadOperation::clear : rachis::LoadOperation::load,
                                         .stencilClearValue = stencil ? 0x5AU : 0U,
                                     }})
                .has_value());
    REQUIRE(device.endRendering(*command).has_value());

    REQUIRE(device.destroyTextureView(*secondView).has_value());
    REQUIRE(device.destroyTexture(*first).has_value());
    REQUIRE(device.destroyTexture(*second).has_value());
    REQUIRE(device.destroyResourceHeap(*heap).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    device.collectGarbage();
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan placed upload and readback buffers support one-shot host writes and reads",
          "[size:medium][rhi][vulkan][memory][heap][placed][host]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    constexpr std::uint64_t bufferSize = 1'024;
    const rachis::BufferDesc uploadDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Placed upload buffer",
    };
    const rachis::BufferDesc readbackDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Placed readback buffer",
    };
    const auto uploadRequirements = device.queryBufferMemoryRequirements(uploadDesc);
    const auto readbackRequirements = device.queryBufferMemoryRequirements(readbackDesc);
    REQUIRE(uploadRequirements.has_value());
    REQUIRE(readbackRequirements.has_value());
    const auto uploadHeap = device.createResourceHeap({
        .size = uploadRequirements->size,
        .alignment = uploadRequirements->alignment,
        .memoryType = rachis::MemoryType::upload,
        .resourceClass = rachis::ResourceHeapClass::buffer,
        .label = "Placed upload heap",
    });
    const auto readbackHeap = device.createResourceHeap({
        .size = readbackRequirements->size,
        .alignment = readbackRequirements->alignment,
        .memoryType = rachis::MemoryType::readback,
        .resourceClass = rachis::ResourceHeapClass::buffer,
        .label = "Placed readback heap",
    });
    REQUIRE(uploadHeap.has_value());
    REQUIRE(readbackHeap.has_value());
    const auto upload = device.createPlacedBuffer({.heap = *uploadHeap}, uploadDesc);
    const auto readback = device.createPlacedBuffer({.heap = *readbackHeap}, readbackDesc);
    REQUIRE(upload.has_value());
    REQUIRE(readback.has_value());

    std::array<std::byte, bufferSize> expected{};
    for (std::size_t index = 0; index < expected.size(); ++index)
    {
        expected[index] = static_cast<std::byte>((index * 37U + 11U) & 0xFFU);
    }
    REQUIRE(device.writeBuffer(*upload, 0, expected).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device.copyBuffer(*command, {.source = *upload, .destination = *readback, .size = bufferSize}).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    std::array<std::byte, bufferSize> actual{};
    REQUIRE(device.readBuffer(*readback, 0, actual).has_value());
    CHECK(actual == expected);
    std::array<std::byte, 16> slice{};
    REQUIRE(device.readBuffer(*readback, 512, slice).has_value());
    CHECK(std::equal(slice.begin(), slice.end(), expected.begin() + 512));

    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyResourceHeap(*uploadHeap).has_value());
    REQUIRE(device.destroyResourceHeap(*readbackHeap).has_value());
    device.collectGarbage();
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan placed indirect buffers bind to device-address-capable heap memory",
          "[size:medium][rhi][vulkan][memory][heap][placed][indirect]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    const rachis::BufferDesc indirectDesc{
        .size = 256,
        .usage = rachis::BufferUsage::indirect | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Placed indirect buffer",
    };
    const auto requirements = device.queryBufferMemoryRequirements(indirectDesc);
    REQUIRE(requirements.has_value());
    const auto heap = device.createResourceHeap({
        .size = requirements->size,
        .alignment = requirements->alignment,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::buffer,
        .label = "Placed indirect heap",
    });
    REQUIRE(heap.has_value());
    const auto indirect = device.createPlacedBuffer({.heap = *heap}, indirectDesc);
    REQUIRE(indirect.has_value());

    REQUIRE(device.destroyBuffer(*indirect).has_value());
    REQUIRE(device.destroyResourceHeap(*heap).has_value());
    device.collectGarbage();
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan buffer alias activation orders a fill of the old alias before a fill of the new alias",
          "[size:medium][rhi][vulkan][memory][aliasing]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).enhancedBarriers);

    constexpr std::uint64_t bufferSize = 4096;
    const rachis::BufferDesc placedBufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Aliased filled buffer",
    };
    const auto requirements = device.queryBufferMemoryRequirements(placedBufferDesc);
    REQUIRE(requirements.has_value());
    const auto heap = device.createResourceHeap({
        .size = requirements->size,
        .alignment = requirements->alignment,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::buffer,
        .flags = rachis::ResourceHeapFlag::allowAliasing,
    });
    REQUIRE(heap.has_value());
    const auto first = device.createPlacedBuffer({.heap = *heap}, placedBufferDesc);
    const auto second = device.createPlacedBuffer({.heap = *heap}, placedBufferDesc);
    const auto readback = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());
    REQUIRE(readback.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    REQUIRE(device.fillBuffer(*command, *first, 0, rachis::wholeResource, 0x11111111U).has_value());
    std::vector<rachis::vulkan::VulkanCapturedBarrier> captured;
    rachis::vulkan::captureVulkanResourceBarriers(&captured);
    const auto activated = device.activateBufferAlias(*command,
                                                      {
                                                          .before = *first,
                                                          .after = *second,
                                                          .stagesBefore = rachis::PipelineStage::computeShader,
                                                          .stagesAfter = rachis::PipelineStage::computeShader,
                                                          .accessBefore = rachis::Access::unorderedAccess,
                                                          .accessAfter = rachis::Access::unorderedAccess,
                                                      });
    rachis::vulkan::captureVulkanResourceBarriers(nullptr);
    REQUIRE(activated.has_value());
    REQUIRE(device.fillBuffer(*command, *second, 0, rachis::wholeResource, 0x22222222U).has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               {
                                   .buffer = *second,
                                   .stagesBefore = rachis::PipelineStage::computeShader,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::unorderedAccess,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device.copyBuffer(*command, {.source = *second, .destination = *readback, .size = bufferSize}).has_value());

    const auto ordersFills = std::ranges::any_of(captured, [](const rachis::vulkan::VulkanCapturedBarrier& barrier) {
        constexpr VkPipelineStageFlags2 clear = VK_PIPELINE_STAGE_2_CLEAR_BIT;
        constexpr VkAccessFlags2 transferWrite = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        const bool allCommandsSource = (barrier.srcStageMask & VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT) != 0;
        return barrier.memory && ((barrier.srcStageMask & clear) != 0 || allCommandsSource) &&
               (barrier.dstStageMask & clear) != 0 && (barrier.srcAccessMask & transferWrite) != 0 &&
               (barrier.dstAccessMask & transferWrite) != 0;
    });
    CHECK(ordersFills);

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    std::array<std::uint32_t, bufferSize / sizeof(std::uint32_t)> values{};
    REQUIRE(device.readBuffer(*readback, 0, std::as_writable_bytes(std::span(values))).has_value());
    CHECK(std::ranges::all_of(values, [](std::uint32_t value) { return value == 0x22222222U; }));

    REQUIRE(device.destroyBuffer(*first).has_value());
    REQUIRE(device.destroyBuffer(*second).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    device.collectGarbage();
    REQUIRE(device.destroyResourceHeap(*heap).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
