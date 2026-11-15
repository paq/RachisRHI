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
#include <limits>
#include <string_view>

namespace
{

    struct MemoryTotals final
    {
        std::uint64_t blocks = 0;
        std::uint64_t allocations = 0;
        std::uint64_t allocationBytes = 0;
    };

    [[nodiscard]] rachis::DeviceDesc vulkanDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::vulkan,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
    }

    [[nodiscard]] MemoryTotals memoryTotals(rachis::Device& device)
    {
        const auto budgets = device.queryMemoryBudgets();
        REQUIRE(budgets.has_value());

        MemoryTotals result;
        for (const auto& budget : *budgets)
        {
            REQUIRE(rachis::hasAny(budget.features, rachis::MemoryBudgetFeature::allocationStatistics));
            CHECK_FALSE(rachis::hasAny(budget.features, rachis::MemoryBudgetFeature::reservation));
            CHECK(budget.reservationBytes == 0);
            CHECK(budget.availableForReservationBytes == 0);
            CHECK(budget.budgetBytes > 0);
            result.blocks += budget.blockCount;
            result.allocations += budget.allocationCount;
            result.allocationBytes += budget.allocationBytes;
        }
        return result;
    }

    [[nodiscard]] const rachis::ResourceMemoryEntry* findByLabel(const rachis::ResourceMemorySnapshot& snapshot,
                                                                 std::string_view label)
    {
        const auto found = std::ranges::find(snapshot.entries, label, &rachis::ResourceMemoryEntry::label);
        return found == snapshot.entries.end() ? nullptr : &*found;
    }

} // namespace

TEST_CASE("Vulkan memory budgets publish reservation-free heap telemetry",
          "[size:medium][rhi][vulkan][memory][residency]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).memoryBudgets);
    CHECK_FALSE(rachis::test::requireCapabilities(device).manualResidency);

    const auto capabilities = device.capabilities();
    REQUIRE(capabilities.has_value());
    auto invalidNode = device.queryMemoryBudgets(capabilities->nodeCount);
    REQUIRE_FALSE(invalidNode.has_value());
    CHECK(invalidNode.error().code == rachis::ErrorCode::invalidArgument);

    const auto baseline = memoryTotals(device);
    const auto beforeSnapshot = device.captureResourceMemorySnapshot();
    REQUIRE(beforeSnapshot.has_value());

    auto buffer = device.createBuffer({
        .size = 4'096,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .label = "Budget telemetry buffer",
    });
    REQUIRE(buffer.has_value());
    const auto afterCreate = memoryTotals(device);
    CHECK(afterCreate.allocations == baseline.allocations + 1);
    CHECK(afterCreate.allocationBytes >= baseline.allocationBytes);

    const auto liveSnapshot = device.captureResourceMemorySnapshot();
    REQUIRE(liveSnapshot.has_value());
    REQUIRE(findByLabel(*liveSnapshot, "Budget telemetry buffer") != nullptr);
    CHECK(liveSnapshot->liveAllocationCount == beforeSnapshot->liveAllocationCount + 1);

    REQUIRE(device.destroyBuffer(*buffer).has_value());
    const auto afterDestroy = memoryTotals(device);
    CHECK(afterDestroy.allocations == baseline.allocations);
    CHECK(afterDestroy.allocationBytes == baseline.allocationBytes);

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan residency priorities isolate lazy allocation pools", "[size:medium][rhi][vulkan][memory][residency]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    const auto baseline = memoryTotals(device);
    auto invalid = device.createBuffer({
        .size = 4'096,
        .usage = rachis::BufferUsage::copySource,
        .residencyPriority = static_cast<rachis::ResidencyPriority>(255),
        .label = "Invalid residency priority",
    });
    REQUIRE_FALSE(invalid.has_value());
    CHECK(invalid.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(memoryTotals(device).blocks == baseline.blocks);

    auto manual = device.createBuffer({
        .size = 4'096,
        .usage = rachis::BufferUsage::copyDestination,
        .allocationFlags = rachis::ResourceAllocationFlag::manualResidency,
        .label = "Manual residency buffer",
    });
    REQUIRE_FALSE(manual.has_value());
    CHECK(manual.error().code == rachis::ErrorCode::unsupported);

    if (!rachis::test::requireCapabilities(device).residencyPriorities)
    {
        auto explicitPriority = device.createBuffer({
            .size = 4'096,
            .usage = rachis::BufferUsage::copyDestination,
            .residencyPriority = rachis::ResidencyPriority::normal,
            .label = "Unsupported priority buffer",
        });
        REQUIRE_FALSE(explicitPriority.has_value());
        CHECK(explicitPriority.error().code == rachis::ErrorCode::unsupported);
        return;
    }

    auto normalBuffer = device.createBuffer({
        .size = 4'096,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .residencyPriority = rachis::ResidencyPriority::normal,
        .label = "Normal-priority buffer A",
    });
    REQUIRE(normalBuffer.has_value());
    const auto afterNormal = memoryTotals(device);
    CHECK(afterNormal.blocks == baseline.blocks + 1);

    auto secondNormalBuffer = device.createBuffer({
        .size = 4'096,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .residencyPriority = rachis::ResidencyPriority::normal,
        .label = "Normal-priority buffer B",
    });
    REQUIRE(secondNormalBuffer.has_value());
    const auto afterSecondNormal = memoryTotals(device);
    CHECK(afterSecondNormal.blocks == afterNormal.blocks);

    auto lowBuffer = device.createBuffer({
        .size = 4'096,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .residencyPriority = rachis::ResidencyPriority::low,
        .label = "Low-priority buffer",
    });
    REQUIRE(lowBuffer.has_value());
    const auto afterLow = memoryTotals(device);
    CHECK(afterLow.blocks == afterSecondNormal.blocks + 1);

    auto highTexture = device.createTexture({
        .width = 4,
        .height = 4,
        .format = rachis::Format::rgba8Unorm,
        .usage =
            rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination | rachis::TextureUsage::sampled,
        .residencyPriority = rachis::ResidencyPriority::high,
        .label = "High-priority sampled texture",
    });
    REQUIRE(highTexture.has_value());
    const auto afterTexture = memoryTotals(device);
    CHECK(afterTexture.blocks == afterLow.blocks + 1);

    auto highAttachment = device.createTexture({
        .width = 4,
        .height = 4,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::renderTarget,
        .residencyPriority = rachis::ResidencyPriority::high,
        .label = "High-priority attachment",
    });
    REQUIRE(highAttachment.has_value());
    const auto live = memoryTotals(device);
    CHECK(live.blocks == afterTexture.blocks + 1);
    CHECK(live.allocations == baseline.allocations + 5);

    const auto requirements = device.queryBufferMemoryRequirements({
        .size = 4'096,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
    });
    REQUIRE(requirements.has_value());
    auto priorityHeap = device.createResourceHeap({
        .size = requirements->size,
        .alignment = requirements->alignment,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::buffer,
        .residencyPriority = rachis::ResidencyPriority::high,
        .label = "High-priority buffer heap",
    });
    REQUIRE(priorityHeap.has_value());
    REQUIRE(device.destroyResourceHeap(*priorityHeap).has_value());

    REQUIRE(device.destroyTexture(*highAttachment).has_value());
    REQUIRE(device.destroyTexture(*highTexture).has_value());
    REQUIRE(device.destroyBuffer(*lowBuffer).has_value());
    REQUIRE(device.destroyBuffer(*secondNormalBuffer).has_value());
    REQUIRE(device.destroyBuffer(*normalBuffer).has_value());
    CHECK(memoryTotals(device).allocations == baseline.allocations);

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan manual residency stays unsupported and oversized allocations preserve snapshots",
          "[size:medium][rhi][vulkan][memory][residency][failure]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

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

    const auto oversized = std::numeric_limits<std::uint64_t>::max();
    auto pressure = device.createBuffer({
        .size = oversized,
        .usage = rachis::BufferUsage::copyDestination,
        .label = "Rejected oversized buffer",
    });
    REQUIRE_FALSE(pressure.has_value());
    CHECK(pressure.error().code == rachis::ErrorCode::outOfMemory);

    const auto after = device.captureResourceMemorySnapshot();
    REQUIRE(after.has_value());
    CHECK(after->liveAllocationCount == before->liveAllocationCount);
    CHECK(after->retainedAllocationCount == before->retainedAllocationCount);
    CHECK(after->liveAllocationBytes == before->liveAllocationBytes);
    CHECK(after->retainedAllocationBytes == before->retainedAllocationBytes);
    CHECK(findByLabel(*after, "Rejected oversized buffer") == nullptr);

    auto recovered = device.createBuffer({
        .size = 4'096,
        .usage = rachis::BufferUsage::copyDestination,
        .label = "Healthy buffer after pressure",
    });
    REQUIRE(recovered.has_value());
    REQUIRE(device.destroyBuffer(*recovered).has_value());
}
