#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace
{

    [[nodiscard]] rachis::DeviceDesc metalDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
    }

    [[nodiscard]] const rachis::ResourceMemoryEntry* findByLabel(const rachis::ResourceMemorySnapshot& snapshot,
                                                                 std::string_view label)
    {
        const auto found = std::ranges::find(snapshot.entries, label, &rachis::ResourceMemoryEntry::label);
        return found == snapshot.entries.end() ? nullptr : &*found;
    }

} // namespace

TEST_CASE("Metal placed buffer heaps reject retained overlap and outlive their public handle",
          "[size:large][rhi][metal][heap][placed][lifetime]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
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
    REQUIRE(requirements->size % requirements->alignment == 0);

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
    CHECK(device.queryResidency(*heap) == rachis::ResidencyState::automatic);

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
    CHECK(device.queryResidency(*first) == rachis::ResidencyState::automatic);

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

TEST_CASE("Metal placed texture heaps preserve class and retained-range ownership",
          "[size:large][rhi][metal][heap][placed][texture][lifetime]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
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
    invalidHeapDesc.alignment = 3;
    CHECK_FALSE(device.createResourceHeap(invalidHeapDesc).has_value());

    const auto sampledHeap = device.createResourceHeap(sampledHeapDesc);
    REQUIRE(sampledHeap.has_value());
    CHECK(device.queryResidency(*sampledHeap) == rachis::ResidencyState::automatic);
    const auto heapSnapshot = device.captureResourceMemorySnapshot();
    REQUIRE(heapSnapshot.has_value());
    const auto* heapEntry = findByLabel(*heapSnapshot, "Placed sampled texture heap");
    REQUIRE(heapEntry != nullptr);
    CHECK(heapEntry->kind == rachis::ResourceMemoryKind::resourceHeap);
    CHECK(heapEntry->allocationBytes == sampledHeapDesc.size);
    CHECK(heapEntry->logicalAlive);

    auto first = device.createPlacedTexture({.heap = *sampledHeap, .offset = 0}, sampledDesc);
    REQUIRE(first.has_value());
    CHECK(device.queryResidency(*first) == rachis::ResidencyState::automatic);
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

TEST_CASE("Metal aliasable placed resources require explicit ordered activation",
          "[size:large][rhi][metal][heap][aliasing][gpu]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

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

TEST_CASE("Metal keeps sparse resources and pageable residency unsupported",
          "[size:large][rhi][metal][heap][sparse][residency]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    const auto capabilities = device.capabilities();
    REQUIRE(capabilities.has_value());
    CHECK_FALSE(capabilities->sparseBuffers);
    CHECK_FALSE(capabilities->sparseTextures);
    CHECK_FALSE(capabilities->manualResidency);

    const rachis::BufferDesc bufferDesc{
        .size = 65'536,
        .usage = rachis::BufferUsage::copyDestination | rachis::BufferUsage::storage,
        .label = "Sparse buffer",
    };
    auto sparseBuffer = device.createSparseBuffer(bufferDesc);
    REQUIRE_FALSE(sparseBuffer.has_value());
    CHECK(sparseBuffer.error().code == rachis::ErrorCode::unsupported);

    const rachis::TextureDesc textureDesc{
        .width = 64,
        .height = 64,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled | rachis::TextureUsage::copyDestination,
        .label = "Sparse texture",
    };
    auto sparseTexture = device.createSparseTexture(textureDesc);
    REQUIRE_FALSE(sparseTexture.has_value());
    CHECK(sparseTexture.error().code == rachis::ErrorCode::unsupported);

    auto mappings = device.enqueueSparseBufferMappings(rachis::QueueType::graphics, {});
    REQUIRE_FALSE(mappings.has_value());
    CHECK(mappings.error().code == rachis::ErrorCode::unsupported);

    auto sparseHeap = device.createResourceHeap({
        .size = 65'536,
        .memoryType = rachis::MemoryType::deviceLocal,
        .flags = rachis::ResourceHeapFlag::sparseBacking,
        .label = "Sparse-backing heap",
    });
    REQUIRE_FALSE(sparseHeap.has_value());
    CHECK(sparseHeap.error().code == rachis::ErrorCode::unsupported);

    auto residency = device.enqueueMakeResident(rachis::QueueType::graphics, {});
    REQUIRE_FALSE(residency.has_value());
    CHECK(residency.error().code == rachis::ErrorCode::unsupported);
    auto eviction = device.evict({});
    REQUIRE_FALSE(eviction.has_value());
    CHECK(eviction.error().code == rachis::ErrorCode::unsupported);

    auto queryHeap = device.queryResidency(rachis::ResourceHeapHandle{});
    REQUIRE_FALSE(queryHeap.has_value());
    CHECK(queryHeap.error().code == rachis::ErrorCode::invalidArgument);
    auto queryBuffer = device.queryResidency(rachis::BufferHandle{});
    REQUIRE_FALSE(queryBuffer.has_value());
    CHECK(queryBuffer.error().code == rachis::ErrorCode::invalidArgument);
    auto queryTexture = device.queryResidency(rachis::TextureHandle{});
    REQUIRE_FALSE(queryTexture.has_value());
    CHECK(queryTexture.error().code == rachis::ErrorCode::invalidArgument);
}
