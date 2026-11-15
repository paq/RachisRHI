#include "Rachis/Device.h"
#include "RhiTestSupport.h"
#if RACHIS_ENABLE_METAL
#include "Metal/MetalTestHooks.h"
#endif

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

namespace
{

#if RACHIS_ENABLE_METAL
    constexpr std::uint64_t bufferSize = 4096;
    constexpr std::uint64_t churnBufferSize = 256;
    constexpr std::uint32_t churnBufferCount = 10'000;
    constexpr std::uint32_t churnTextureCount = 1'000;
    constexpr std::uint32_t churnHeapCount = 1'000;
    constexpr std::size_t maxChurnTelemetryEntries = 256;

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

    [[nodiscard]] std::byte pattern(std::uint64_t index) noexcept
    {
        return static_cast<std::byte>((index * 17U + 5U) & 0xFFU);
    }
#endif

} // namespace

#if RACHIS_ENABLE_METAL

TEST_CASE("Metal memory budgets expose working-set statistics without reservation", "[size:large][rhi][metal][budget]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    const auto checkedCapabilities = device.capabilities();
    REQUIRE(checkedCapabilities.has_value());
    REQUIRE(checkedCapabilities->memoryBudgets);

    auto invalidNode = device.queryMemoryBudgets(checkedCapabilities->nodeCount);
    REQUIRE_FALSE(invalidNode.has_value());
    CHECK(invalidNode.error().code == rachis::ErrorCode::invalidArgument);

    const auto budgets = device.queryMemoryBudgets();
    REQUIRE(budgets.has_value());
    REQUIRE_FALSE(budgets->empty());
    for (const auto& budget : *budgets)
    {
        REQUIRE(rachis::hasAny(budget.features, rachis::MemoryBudgetFeature::allocationStatistics));
        CHECK(rachis::hasAny(budget.properties, rachis::MemoryHeapProperty::deviceLocal));
        CHECK(budget.budgetBytes > 0);
    }
}

TEST_CASE("Metal allocation snapshots stay consistent through create fail destroy and retire",
          "[size:large][rhi][metal][budget][snapshot]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    const auto baseline = device.captureResourceMemorySnapshot();
    REQUIRE(baseline.has_value());
    const auto baselineBudgets = device.queryMemoryBudgets();
    REQUIRE(baselineBudgets.has_value());

    auto buffer = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Metal budget live buffer",
    });
    auto texture = device.createTexture({
        .width = 8,
        .height = 8,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::sampled,
        .label = "Metal budget live texture",
    });
    REQUIRE(buffer.has_value());
    REQUIRE(texture.has_value());

    const auto live = device.captureResourceMemorySnapshot();
    REQUIRE(live.has_value());
    CHECK(live->liveAllocationCount == baseline->liveAllocationCount + 2);
    CHECK(live->retainedAllocationCount == baseline->retainedAllocationCount);
    const auto* liveBuffer = findByLabel(*live, "Metal budget live buffer");
    const auto* liveTexture = findByLabel(*live, "Metal budget live texture");
    REQUIRE(liveBuffer != nullptr);
    REQUIRE(liveTexture != nullptr);
    CHECK(liveBuffer->kind == rachis::ResourceMemoryKind::buffer);
    CHECK(liveBuffer->logicalAlive);
    CHECK(liveBuffer->allocationBytes >= bufferSize);
    CHECK(liveBuffer->allocationId != 0);
    CHECK(liveTexture->kind == rachis::ResourceMemoryKind::texture);
    CHECK(liveTexture->logicalAlive);
    CHECK(liveTexture->allocationBytes > 0);
    const auto bufferAllocationId = liveBuffer->allocationId;
    const auto bufferAllocationBytes = liveBuffer->allocationBytes;

    const auto liveBudgets = device.queryMemoryBudgets();
    REQUIRE(liveBudgets.has_value());
    REQUIRE_FALSE(liveBudgets->empty());
    CHECK((*liveBudgets)[0].allocationCount == (*baselineBudgets)[0].allocationCount + 2);
    CHECK((*liveBudgets)[0].allocationBytes >= (*baselineBudgets)[0].allocationBytes + bufferAllocationBytes);

    REQUIRE(device.destroyTexture(*texture).has_value());
    REQUIRE(device.destroyBuffer(*buffer).has_value());
    device.collectGarbage();

    const auto released = device.captureResourceMemorySnapshot();
    REQUIRE(released.has_value());
    CHECK(findByLabel(*released, "Metal budget live buffer") == nullptr);
    CHECK(findByLabel(*released, "Metal budget live texture") == nullptr);
    CHECK(released->liveAllocationCount == baseline->liveAllocationCount);
    CHECK(released->retainedAllocationCount == baseline->retainedAllocationCount);
    CHECK(released->liveAllocationBytes == baseline->liveAllocationBytes);
    CHECK(released->retainedAllocationBytes == baseline->retainedAllocationBytes);

    auto replacement = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Metal budget replacement buffer",
    });
    REQUIRE(replacement.has_value());
    const auto replacementSnapshot = device.captureResourceMemorySnapshot();
    REQUIRE(replacementSnapshot.has_value());
    const auto* replacementEntry = findByLabel(*replacementSnapshot, "Metal budget replacement buffer");
    REQUIRE(replacementEntry != nullptr);
    CHECK(replacementEntry->allocationId != bufferAllocationId);
    REQUIRE(device.destroyBuffer(*replacement).has_value());
    device.collectGarbage();
}

TEST_CASE("Metal destroy after record keeps GPU-retained allocation identity",
          "[size:large][rhi][metal][budget][lifetime]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    const auto baseline = device.captureResourceMemorySnapshot();
    REQUIRE(baseline.has_value());

    auto source = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal budget retained source",
    });
    auto destination = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal budget retained destination",
    });
    REQUIRE(source.has_value());
    REQUIRE(destination.has_value());

    std::vector<std::byte> payload(bufferSize);
    for (std::uint64_t index = 0; index < payload.size(); ++index)
    {
        payload[index] = pattern(index);
    }
    REQUIRE(device.writeBuffer(*source, 0, payload).has_value());

    const auto live = device.captureResourceMemorySnapshot();
    REQUIRE(live.has_value());
    const auto* liveSource = findByLabel(*live, "Metal budget retained source");
    REQUIRE(liveSource != nullptr);
    const auto sourceAllocationId = liveSource->allocationId;
    const auto sourceAllocationBytes = liveSource->allocationBytes;

    std::vector<std::byte> recovered(bufferSize, std::byte{0});
    {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .copyBuffer(*command,
                                {
                                    .source = *source,
                                    .destination = *destination,
                                    .size = bufferSize,
                                })
                    .has_value());
        REQUIRE(device.destroyBuffer(*source).has_value());
        CHECK_FALSE(device.destroyBuffer(*source).has_value());

        const auto retained = device.captureResourceMemorySnapshot();
        REQUIRE(retained.has_value());
        const auto* retainedSource = findByLabel(*retained, "Metal budget retained source");
        REQUIRE(retainedSource != nullptr);
        CHECK(retainedSource->allocationId == sourceAllocationId);
        CHECK(retainedSource->allocationBytes == sourceAllocationBytes);
        CHECK_FALSE(retainedSource->logicalAlive);
        CHECK(retained->liveAllocationCount == baseline->liveAllocationCount + 1);
        CHECK(retained->retainedAllocationCount == baseline->retainedAllocationCount + 1);
        CHECK(retained->retainedAllocationBytes == baseline->retainedAllocationBytes + sourceAllocationBytes);

        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        REQUIRE(device.wait(*submitted).has_value());
        REQUIRE(device.readBuffer(*destination, 0, recovered).has_value());
    }

    CHECK(recovered == payload);
    REQUIRE(device.destroyBuffer(*destination).has_value());
    device.collectGarbage();
    const auto released = device.captureResourceMemorySnapshot();
    REQUIRE(released.has_value());
    CHECK(findByLabel(*released, "Metal budget retained source") == nullptr);
    CHECK(findByLabel(*released, "Metal budget retained destination") == nullptr);
    CHECK(released->liveAllocationCount == baseline->liveAllocationCount);
    CHECK(released->retainedAllocationCount == baseline->retainedAllocationCount);
    CHECK(released->liveAllocationBytes == baseline->liveAllocationBytes);
    CHECK(released->retainedAllocationBytes == baseline->retainedAllocationBytes);
}

TEST_CASE("Metal allocation telemetry stays bounded when resources churn without garbage collection",
          "[size:large][rhi][metal][budget][snapshot]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    const auto baseline = device.captureResourceMemorySnapshot();
    REQUIRE(baseline.has_value());
    const auto heapRequirements = device.queryBufferMemoryRequirements({
        .size = churnBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
    });
    REQUIRE(heapRequirements.has_value());

    auto live = device.createBuffer({
        .size = churnBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Metal churn live buffer",
    });
    auto source = device.createBuffer({
        .size = churnBufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal churn retained source",
    });
    auto destination = device.createBuffer({
        .size = churnBufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal churn retained destination",
    });
    REQUIRE(live.has_value());
    REQUIRE(source.has_value());
    REQUIRE(destination.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(command.has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            {
                                .source = *source,
                                .destination = *destination,
                                .size = churnBufferSize,
                            })
                .has_value());
    REQUIRE(device.destroyBuffer(*source).has_value());

    std::size_t peakCount = 0;
    const auto observe = [&peakCount] {
        const auto count = rachis::metal::boundAllocationTelemetryCount();
        REQUIRE(count.has_value());
        peakCount = std::max(peakCount, *count);
    };
    for (std::uint32_t index = 0; index < churnBufferCount; ++index)
    {
        auto buffer = device.createBuffer({
            .size = churnBufferSize,
            .usage = rachis::BufferUsage::copyDestination,
            .memoryType = rachis::MemoryType::deviceLocal,
        });
        REQUIRE(buffer.has_value());
        REQUIRE(device.destroyBuffer(*buffer).has_value());
        observe();
    }
    for (std::uint32_t index = 0; index < churnTextureCount; ++index)
    {
        auto texture = device.createTexture({
            .width = 4,
            .height = 4,
            .format = rachis::Format::rgba8Unorm,
            .usage = rachis::TextureUsage::sampled,
        });
        REQUIRE(texture.has_value());
        REQUIRE(device.destroyTexture(*texture).has_value());
        observe();
    }
    for (std::uint32_t index = 0; index < churnHeapCount; ++index)
    {
        auto heap = device.createResourceHeap({
            .size = heapRequirements->size,
            .memoryType = rachis::MemoryType::deviceLocal,
        });
        REQUIRE(heap.has_value());
        REQUIRE(device.destroyResourceHeap(*heap).has_value());
        observe();
    }
    CHECK(peakCount < maxChurnTelemetryEntries);

    const auto churned = device.captureResourceMemorySnapshot();
    REQUIRE(churned.has_value());
    CHECK(std::ranges::is_sorted(churned->entries, {}, &rachis::ResourceMemoryEntry::allocationId));
    const auto* liveEntry = findByLabel(*churned, "Metal churn live buffer");
    const auto* retainedEntry = findByLabel(*churned, "Metal churn retained source");
    REQUIRE(liveEntry != nullptr);
    REQUIRE(retainedEntry != nullptr);
    CHECK(liveEntry->logicalAlive);
    CHECK_FALSE(retainedEntry->logicalAlive);
    CHECK(churned->retainedAllocationCount >= 1);
    CHECK(churned->retainedAllocationBytes >= retainedEntry->allocationBytes);

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());
    REQUIRE(device.destroyBuffer(*destination).has_value());
    REQUIRE(device.destroyBuffer(*live).has_value());
    device.collectGarbage();

    const auto settledCount = rachis::metal::boundAllocationTelemetryCount();
    REQUIRE(settledCount.has_value());
    CHECK(*settledCount == baseline->entries.size());
}

TEST_CASE("Metal working-set overflow rejects allocation without publishing counters",
          "[size:large][rhi][metal][budget][pressure]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    const auto checkedCapabilities = device.capabilities();
    REQUIRE(checkedCapabilities.has_value());

    const auto before = device.captureResourceMemorySnapshot();
    REQUIRE(before.has_value());
    const auto budgets = device.queryMemoryBudgets();
    REQUIRE(budgets.has_value());
    REQUIRE_FALSE(budgets->empty());

    std::uint64_t deviceBudget = 0;
    for (const auto& budget : *budgets)
    {
        if (rachis::hasAny(budget.properties, rachis::MemoryHeapProperty::deviceLocal))
        {
            deviceBudget = std::max(deviceBudget, budget.budgetBytes);
        }
    }
    REQUIRE(deviceBudget > 0);
    const auto maxBufferSize = checkedCapabilities->limits.maxBufferSize;
    REQUIRE(maxBufferSize > 0);
    const auto limit = std::min(deviceBudget, maxBufferSize);
    const auto oversized = limit > (std::numeric_limits<std::uint64_t>::max() - 1) ? limit : limit + 1;

    auto pressure = device.createBuffer({
        .size = oversized,
        .usage = rachis::BufferUsage::copyDestination,
        .label = "Bounded Metal memory-pressure buffer",
    });
    REQUIRE_FALSE(pressure.has_value());
    CHECK(pressure.error().code == rachis::ErrorCode::outOfMemory);

    const auto after = device.captureResourceMemorySnapshot();
    REQUIRE(after.has_value());
    CHECK(after->liveAllocationCount == before->liveAllocationCount);
    CHECK(after->retainedAllocationCount == before->retainedAllocationCount);
    CHECK(after->liveAllocationBytes == before->liveAllocationBytes);
    CHECK(after->retainedAllocationBytes == before->retainedAllocationBytes);
    CHECK(findByLabel(*after, "Bounded Metal memory-pressure buffer") == nullptr);

    const auto afterBudgets = device.queryMemoryBudgets();
    REQUIRE(afterBudgets.has_value());
    REQUIRE_FALSE(afterBudgets->empty());
    CHECK((*afterBudgets)[0].allocationCount == (*budgets)[0].allocationCount);
    CHECK((*afterBudgets)[0].allocationBytes == (*budgets)[0].allocationBytes);
}

#endif
