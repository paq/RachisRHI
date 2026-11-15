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

TEST_CASE("resource memory snapshots preserve exact retained allocation identity",
          "[size:medium][rhi][memory][snapshot][lifetime][warp]")
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

    auto buffer = device.createBuffer({
        .size = 4'096,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Snapshot retained buffer",
    });
    auto texture = device.createTexture({
        .dimension = rachis::TextureDimension::texture2D,
        .width = 64,
        .height = 64,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled,
        .label = "Snapshot live texture",
    });
    REQUIRE(buffer.has_value());
    REQUIRE(texture.has_value());

    const auto live = device.captureResourceMemorySnapshot();
    REQUIRE(live.has_value());
    CHECK(live->entries.size() == baseline->entries.size() + 2);
    CHECK(live->liveAllocationCount == baseline->liveAllocationCount + 2);
    CHECK(live->retainedAllocationCount == baseline->retainedAllocationCount);
    const auto* liveBuffer = findByLabel(*live, "Snapshot retained buffer");
    const auto* liveTexture = findByLabel(*live, "Snapshot live texture");
    REQUIRE(liveBuffer != nullptr);
    REQUIRE(liveTexture != nullptr);
    CHECK(liveBuffer->kind == rachis::ResourceMemoryKind::buffer);
    CHECK(liveBuffer->memoryType == rachis::MemoryType::deviceLocal);
    CHECK(liveBuffer->logicalAlive);
    CHECK(liveBuffer->allocationBytes >= 4'096);
    CHECK(liveTexture->kind == rachis::ResourceMemoryKind::texture);
    CHECK(liveTexture->logicalAlive);
    CHECK(liveTexture->allocationBytes > 0);
    const auto bufferAllocationId = liveBuffer->allocationId;
    const auto bufferAllocationBytes = liveBuffer->allocationBytes;

    REQUIRE(device.destroyTexture(*texture).has_value());
    {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .bufferBarrier(*command,
                                   rachis::BufferBarrier{
                                       .buffer = *buffer,
                                       .stagesBefore = rachis::PipelineStage::all,
                                       .stagesAfter = rachis::PipelineStage::copy,
                                       .accessBefore = rachis::Access::common,
                                       .accessAfter = rachis::Access::copyDestination,
                                   })
                    .has_value());
        REQUIRE(device.destroyBuffer(*buffer).has_value());

        const auto retained = device.captureResourceMemorySnapshot();
        REQUIRE(retained.has_value());
        CHECK(findByLabel(*retained, "Snapshot live texture") == nullptr);
        const auto* retainedBuffer = findByLabel(*retained, "Snapshot retained buffer");
        REQUIRE(retainedBuffer != nullptr);
        CHECK(retainedBuffer->allocationId == bufferAllocationId);
        CHECK(retainedBuffer->allocationBytes == bufferAllocationBytes);
        CHECK_FALSE(retainedBuffer->logicalAlive);
        CHECK(retained->liveAllocationCount == baseline->liveAllocationCount);
        CHECK(retained->retainedAllocationCount == baseline->retainedAllocationCount + 1);
        CHECK(retained->retainedAllocationBytes == baseline->retainedAllocationBytes + bufferAllocationBytes);
    }

    device.collectGarbage();
    const auto released = device.captureResourceMemorySnapshot();
    REQUIRE(released.has_value());
    CHECK(findByLabel(*released, "Snapshot retained buffer") == nullptr);
    CHECK(released->liveAllocationCount == baseline->liveAllocationCount);
    CHECK(released->retainedAllocationCount == baseline->retainedAllocationCount);
    CHECK(released->liveAllocationBytes == baseline->liveAllocationBytes);
    CHECK(released->retainedAllocationBytes == baseline->retainedAllocationBytes);

    auto replacement = device.createBuffer({
        .size = 4'096,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Snapshot replacement buffer",
    });
    REQUIRE(replacement.has_value());
    const auto replacementSnapshot = device.captureResourceMemorySnapshot();
    REQUIRE(replacementSnapshot.has_value());
    const auto* replacementEntry = findByLabel(*replacementSnapshot, "Snapshot replacement buffer");
    REQUIRE(replacementEntry != nullptr);
    CHECK(replacementEntry->allocationId != bufferAllocationId);
    REQUIRE(device.destroyBuffer(*replacement).has_value());
    device.collectGarbage();
}
