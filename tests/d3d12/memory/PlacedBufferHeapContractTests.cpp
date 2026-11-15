#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
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

TEST_CASE("placed buffer heaps reject retained overlap and outlive their public handle",
          "[size:medium][rhi][memory][heap][placed][lifetime][warp]")
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
