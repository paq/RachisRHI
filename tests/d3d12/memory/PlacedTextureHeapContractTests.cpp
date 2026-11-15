#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <string_view>

namespace
{

    [[nodiscard]] const rachis::ResourceMemoryEntry* findByLabel(const rachis::ResourceMemorySnapshot& snapshot,
                                                                 std::string_view label)
    {
        const auto found = std::ranges::find(snapshot.entries, label, &rachis::ResourceMemoryEntry::label);
        return found == snapshot.entries.end() ? nullptr : &*found;
    }

} // namespace

TEST_CASE("typed placed texture heaps preserve class and retained-range ownership",
          "[size:medium][rhi][memory][heap][placed][texture][lifetime][warp]")
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
