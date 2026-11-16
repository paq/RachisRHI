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

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace
{

    constexpr std::uint64_t sharedBudgetBufferSize = 64ULL * 1024ULL * 1024ULL;

    [[nodiscard]] std::byte pattern(std::uint64_t index) noexcept
    {
        return static_cast<std::byte>((index * 37U + 11U) & 0xFFU);
    }

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

    struct AllocationTotals final
    {
        std::uint64_t bytes = 0;
        std::uint64_t count = 0;
    };

    [[nodiscard]] AllocationTotals allocationTotals(rachis::Device& device)
    {
        const auto budgets = device.queryMemoryBudgets();
        REQUIRE(budgets.has_value());
        AllocationTotals totals;
        for (const auto& budget : *budgets)
        {
            REQUIRE(rachis::hasAny(budget.features, rachis::MemoryBudgetFeature::allocationStatistics));
            CHECK_FALSE(rachis::hasAny(budget.features, rachis::MemoryBudgetFeature::reservation));
            totals.bytes += budget.allocationBytes;
            totals.count += budget.allocationCount;
        }
        return totals;
    }

    [[nodiscard]] std::vector<rachis::MemoryBudget> requireBudgets(const rachis::Device& device)
    {
        auto budgets = device.queryMemoryBudgets();
        REQUIRE(budgets.has_value());
        return std::move(*budgets);
    }

    [[nodiscard]] std::uint64_t liveSharedAllocationBytes(const rachis::Device& device, std::string_view label)
    {
        const auto snapshot = device.captureResourceMemorySnapshot();
        REQUIRE(snapshot.has_value());
        const auto entry = std::ranges::find(snapshot->entries, label, &rachis::ResourceMemoryEntry::label);
        REQUIRE(entry != snapshot->entries.end());
        CHECK(entry->shared);
        CHECK(entry->logicalAlive);
        CHECK(entry->allocationBytes >= sharedBudgetBufferSize);
        return entry->allocationBytes;
    }

    void requireBudgetsCountAllocation(const std::vector<rachis::MemoryBudget>& baseline,
                                       const std::vector<rachis::MemoryBudget>& live,
                                       std::uint64_t allocationBytes)
    {
        REQUIRE(live.size() == baseline.size());
        std::size_t grownHeaps = 0;
        for (std::size_t index = 0; index < live.size(); ++index)
        {
            CAPTURE(index);
            const auto& before = baseline[index];
            const auto& after = live[index];
            CHECK(after.heapIndex == before.heapIndex);
            CHECK(after.blockBytes >= after.allocationBytes);
            if (after.allocationCount == before.allocationCount)
            {
                CHECK(after.allocationBytes == before.allocationBytes);
                continue;
            }
            ++grownHeaps;
            CHECK(rachis::hasAny(after.properties, rachis::MemoryHeapProperty::deviceLocal));
            CHECK(after.allocationCount == before.allocationCount + 1);
            CHECK(after.blockCount == before.blockCount + 1);
            CHECK(after.allocationBytes >= before.allocationBytes + allocationBytes);
            CHECK(after.blockBytes >= before.blockBytes + allocationBytes);
            CHECK(after.usageBytes >= before.usageBytes + allocationBytes);
        }
        CHECK(grownHeaps == 1);
    }

    void requireBudgetCountersRestored(const std::vector<rachis::MemoryBudget>& baseline,
                                       const std::vector<rachis::MemoryBudget>& released)
    {
        REQUIRE(released.size() == baseline.size());
        for (std::size_t index = 0; index < released.size(); ++index)
        {
            CAPTURE(index);
            CHECK(released[index].blockBytes == baseline[index].blockBytes);
            CHECK(released[index].allocationBytes == baseline[index].allocationBytes);
            CHECK(released[index].blockCount == baseline[index].blockCount);
            CHECK(released[index].allocationCount == baseline[index].allocationCount);
        }
    }

} // namespace

TEST_CASE("Vulkan rejects invalid buffer descriptors before allocation", "[size:medium][rhi][vulkan][buffer]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    auto zero = device.createBuffer({
        .size = 0,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Zero size",
    });
    REQUIRE_FALSE(zero);
    CHECK(zero.error().code == rachis::ErrorCode::invalidArgument);

    auto noUsage = device.createBuffer({
        .size = 64,
        .usage = rachis::BufferUsage::none,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "No usage",
    });
    REQUIRE_FALSE(noUsage);
    CHECK(noUsage.error().code == rachis::ErrorCode::invalidArgument);

    auto residency = device.createBuffer({
        .size = 64,
        .usage = rachis::BufferUsage::copyDestination,
        .allocationFlags = rachis::ResourceAllocationFlag::manualResidency,
        .label = "Manual residency",
    });
    REQUIRE_FALSE(residency);
    CHECK(residency.error().code == rachis::ErrorCode::unsupported);

    auto storageReadback = device.createBuffer({
        .size = 64,
        .usage = rachis::BufferUsage::storage | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Storage readback",
    });
    REQUIRE_FALSE(storageReadback);
    CHECK(storageReadback.error().code == rachis::ErrorCode::unsupported);

    auto storageReadbackRequirements = device.queryBufferMemoryRequirements({
        .size = 64,
        .usage = rachis::BufferUsage::storage,
        .memoryType = rachis::MemoryType::readback,
    });
    REQUIRE_FALSE(storageReadbackRequirements);
    CHECK(storageReadbackRequirements.error().code == rachis::ErrorCode::unsupported);

    auto copyReadback = device.createBuffer({
        .size = 64,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Copy readback",
    });
    REQUIRE(copyReadback);
    REQUIRE(device.destroyBuffer(*copyReadback).has_value());
}

TEST_CASE("Vulkan buffer create map write and destroy restore allocation counts", "[size:medium][rhi][vulkan][buffer]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).memoryBudgets);
    REQUIRE(rachis::test::requireCapabilities(device).persistentMapping);
    REQUIRE(rachis::test::requireCapabilities(device).mappedMemory.nonCoherentAtomSize >= 1);

    const auto baseline = allocationTotals(device);
    const auto baselineSnapshot = device.captureResourceMemorySnapshot();
    REQUIRE(baselineSnapshot);

    auto requirements = device.queryBufferMemoryRequirements({
        .size = 4096,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Requirements",
    });
    REQUIRE(requirements);
    CHECK(requirements->size >= 4096);
    CHECK(requirements->alignment > 0);

    auto upload = device.createBuffer({
        .size = 4096,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Vulkan upload",
    });
    auto gpu = device.createBuffer({
        .size = 4096,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Vulkan device local",
    });
    auto readback = device.createBuffer({
        .size = 4096,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Vulkan readback",
    });
    REQUIRE(upload);
    REQUIRE(gpu);
    REQUIRE(readback);

    const auto live = allocationTotals(device);
    CHECK(live.count == baseline.count + 3);
    CHECK(live.bytes > baseline.bytes);
    const auto liveSnapshot = device.captureResourceMemorySnapshot();
    REQUIRE(liveSnapshot);
    CHECK(liveSnapshot->liveAllocationCount == baselineSnapshot->liveAllocationCount + 3);
    CHECK(liveSnapshot->retainedAllocationCount == 0);

    auto mappedGpu = device.mapBuffer(*gpu);
    REQUIRE_FALSE(mappedGpu);
    CHECK(mappedGpu.error().code == rachis::ErrorCode::unsupported);

    std::vector<std::byte> payload(4096);
    for (std::uint64_t i = 0; i < payload.size(); ++i)
    {
        payload[i] = pattern(i);
    }
    REQUIRE(device.writeBuffer(*upload, 0, payload).has_value());
    REQUIRE(rachis::vulkan::copyVulkanBuffersImmediate(*upload, *gpu, 0, 0, payload.size()).has_value());
    REQUIRE(rachis::vulkan::copyVulkanBuffersImmediate(*gpu, *readback, 0, 0, payload.size()).has_value());

    std::vector<std::byte> recovered(4096, std::byte{0});
    REQUIRE(device.readBuffer(*readback, 0, recovered).has_value());
    CHECK(recovered == payload);

    const auto staleUpload = *upload;
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*gpu).has_value());
    CHECK_FALSE(device.mapBuffer(staleUpload).has_value());
    CHECK_FALSE(device.destroyBuffer(staleUpload).has_value());

    REQUIRE(device.destroyBuffer(*readback).has_value());
    const auto released = allocationTotals(device);
    CHECK(released.count == baseline.count);
    CHECK(released.bytes == baseline.bytes);
    const auto releasedSnapshot = device.captureResourceMemorySnapshot();
    REQUIRE(releasedSnapshot);
    CHECK(releasedSnapshot->liveAllocationCount == baselineSnapshot->liveAllocationCount);
    CHECK(releasedSnapshot->liveAllocationBytes == baselineSnapshot->liveAllocationBytes);

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan buffer creation releases its native buffer when allocation fails",
          "[size:medium][rhi][vulkan][buffer]")
{
    std::atomic_uint64_t errors{0};
    auto created = rachis::createDevice(vulkanDesc(&errors));
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto baseline = allocationTotals(device);

    for (const auto memoryType :
         {rachis::MemoryType::deviceLocal, rachis::MemoryType::upload, rachis::MemoryType::readback})
    {
        CAPTURE(memoryType);
        const rachis::BufferDesc desc{
            .size = 4096,
            .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
            .memoryType = memoryType,
            .label = "Vulkan buffer with a failing keep-alive allocation",
        };
        const auto failed = createWithBadAlloc([&] { return device.createBuffer(desc); });
        REQUIRE_FALSE(failed.has_value());
        CHECK(failed.error().code == rachis::ErrorCode::outOfMemory);
        const auto afterFailure = allocationTotals(device);
        CHECK(afterFailure.count == baseline.count);
        CHECK(afterFailure.bytes == baseline.bytes);

        auto buffer = device.createBuffer(desc);
        REQUIRE(buffer.has_value());
        REQUIRE(device.destroyBuffer(*buffer).has_value());
        device.collectGarbage();
    }

    CHECK(device.diagnosticCounts().errors == 0);
    device = {};
    CHECK(errors.load() == 0);
}

TEST_CASE("Vulkan shared buffers count in memory budgets until their memory is freed",
          "[size:medium][rhi][vulkan][buffer]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);
    const auto capabilities = rachis::test::requireCapabilities(device);
    if (rachis::test::skipWithoutCapability(capabilities.sharedBufferHandles && capabilities.memoryBudgets,
                                            "sharedBufferHandles and memoryBudgets"))
    {
        return;
    }

    const auto baseline = requireBudgets(device);
    constexpr std::string_view label = "Vulkan budgeted shared buffer";
    auto shared = device.createSharedBuffer({
        .size = sharedBudgetBufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = label,
    });
    if (!shared)
    {
        INFO(shared.error().message);
    }
    REQUIRE(shared.has_value());
    const auto allocationBytes = liveSharedAllocationBytes(device, label);
    requireBudgetsCountAllocation(baseline, requireBudgets(device), allocationBytes);

    REQUIRE(device.destroyBuffer(*shared).has_value());
    device.collectGarbage();
    requireBudgetCountersRestored(baseline, requireBudgets(device));

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("Vulkan host buffer IO is directional bounded and mapping-exclusive", "[size:medium][rhi][vulkan][buffer]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    constexpr std::array payload{
        std::byte{0x10},
        std::byte{0x21},
        std::byte{0x32},
        std::byte{0x43},
        std::byte{0x54},
        std::byte{0x65},
        std::byte{0x76},
        std::byte{0x87},
        std::byte{0x98},
        std::byte{0xA9},
        std::byte{0xBA},
        std::byte{0xCB},
        std::byte{0xDC},
        std::byte{0xED},
        std::byte{0xFE},
        std::byte{0x0F},
    };
    auto upload = device.createBuffer({
        .size = payload.size(),
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Host write",
    });
    auto readback = device.createBuffer({
        .size = payload.size(),
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Host read",
    });
    auto deviceLocal = device.createBuffer({
        .size = payload.size(),
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Host inaccessible",
    });
    REQUIRE(upload);
    REQUIRE(readback);
    REQUIRE(deviceLocal);

    REQUIRE(device.writeBuffer(*upload, 0, payload).has_value());
    const std::span<const std::byte> empty;
    auto emptyWrite = device.writeBuffer(*upload, 0, empty);
    REQUIRE_FALSE(emptyWrite);
    CHECK(emptyWrite.error().code == rachis::ErrorCode::invalidArgument);
    auto outOfBoundsWrite = device.writeBuffer(*upload, payload.size() - 1, std::span{payload}.first(2));
    REQUIRE_FALSE(outOfBoundsWrite);
    CHECK(outOfBoundsWrite.error().code == rachis::ErrorCode::invalidArgument);

    std::array<std::byte, payload.size()> destination{};
    auto wrongWriteDirection = device.writeBuffer(*readback, 0, payload);
    auto wrongReadDirection = device.readBuffer(*upload, 0, destination);
    auto deviceLocalWrite = device.writeBuffer(*deviceLocal, 0, payload);
    auto deviceLocalRead = device.readBuffer(*deviceLocal, 0, destination);
    REQUIRE_FALSE(wrongWriteDirection);
    REQUIRE_FALSE(wrongReadDirection);
    REQUIRE_FALSE(deviceLocalWrite);
    REQUIRE_FALSE(deviceLocalRead);
    CHECK(wrongWriteDirection.error().code == rachis::ErrorCode::unsupported);
    CHECK(wrongReadDirection.error().code == rachis::ErrorCode::unsupported);
    CHECK(deviceLocalWrite.error().code == rachis::ErrorCode::unsupported);
    CHECK(deviceLocalRead.error().code == rachis::ErrorCode::unsupported);

    auto mappedUpload = device.mapBuffer(*upload);
    REQUIRE(mappedUpload);
    auto writeWhileMapped = device.writeBuffer(*upload, 0, payload);
    REQUIRE_FALSE(writeWhileMapped);
    CHECK(writeWhileMapped.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.unmapBuffer(*upload).has_value());

    REQUIRE(rachis::vulkan::copyVulkanBuffersImmediate(*upload, *readback, 0, 0, payload.size()).has_value());
    destination.fill(std::byte{0});
    REQUIRE(device.readBuffer(*readback, 0, destination).has_value());
    CHECK(destination == payload);

    constexpr std::uint64_t mappedOffset = 4;
    auto persistent = device.mapBuffer(*upload, mappedOffset, 8);
    REQUIRE(persistent);
    CHECK(persistent->size == 8);
    const std::array ranges{
        rachis::MappedMemoryRange{.offset = mappedOffset, .size = 4},
        rachis::MappedMemoryRange{.offset = mappedOffset + 4, .size = 4},
    };
    REQUIRE(device.flushMappedMemoryRanges(*upload, ranges).has_value());
    const auto emptyFlush = device.flushMappedMemoryRanges(*upload, {});
    REQUIRE_FALSE(emptyFlush);
    CHECK(emptyFlush.error().code == rachis::ErrorCode::invalidArgument);
    const auto outsideFlush = device.flushMappedMemoryRanges(
        *upload, std::array{rachis::MappedMemoryRange{.offset = 0, .size = mappedOffset}});
    REQUIRE_FALSE(outsideFlush);
    CHECK(outsideFlush.error().code == rachis::ErrorCode::invalidArgument);
    const auto wrongFlush = device.flushMappedMemoryRanges(*readback, ranges);
    REQUIRE_FALSE(wrongFlush);
    CHECK(wrongFlush.error().code == rachis::ErrorCode::unsupported);
    REQUIRE(device.unmapBuffer(*upload, mappedOffset, 8).has_value());
    const auto flushAfterUnmap = device.flushMappedMemoryRanges(*upload, ranges);
    REQUIRE_FALSE(flushAfterUnmap);
    CHECK(flushAfterUnmap.error().code == rachis::ErrorCode::invalidState);

    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*deviceLocal).has_value());
}

TEST_CASE("Vulkan unmap ranges must lie inside the active mapping", "[size:medium][rhi][vulkan][buffer]")
{
    auto created = rachis::createDevice(vulkanDesc());
    if (rachis::test::skipWithoutCreatedDevice(created))
    {
        return;
    }
    auto device = std::move(*created);

    constexpr std::uint64_t mappedBufferSize = 64;
    constexpr std::uint64_t mappedOffset = 16;
    constexpr std::uint64_t mappedSize = 16;
    for (const auto memoryType : {rachis::MemoryType::upload, rachis::MemoryType::readback})
    {
        CAPTURE(static_cast<int>(memoryType));
        auto buffer = device.createBuffer({
            .size = mappedBufferSize,
            .usage = memoryType == rachis::MemoryType::upload ? rachis::BufferUsage::copySource
                                                              : rachis::BufferUsage::copyDestination,
            .memoryType = memoryType,
            .label = "Subrange-mapped buffer",
        });
        REQUIRE(buffer.has_value());
        auto mapped = device.mapBuffer(*buffer, mappedOffset, mappedSize);
        REQUIRE(mapped.has_value());
        CHECK(mapped->size == mappedSize);

        const auto beforeMapping = device.unmapBuffer(*buffer, 0, mappedOffset);
        REQUIRE_FALSE(beforeMapping.has_value());
        CHECK(beforeMapping.error().code == rachis::ErrorCode::invalidArgument);
        const auto straddlingEnd = device.unmapBuffer(*buffer, mappedOffset + mappedSize / 2, mappedSize);
        REQUIRE_FALSE(straddlingEnd.has_value());
        CHECK(straddlingEnd.error().code == rachis::ErrorCode::invalidArgument);
        const auto pastEnd = device.unmapBuffer(*buffer, mappedOffset, rachis::wholeResource);
        REQUIRE_FALSE(pastEnd.has_value());
        CHECK(pastEnd.error().code == rachis::ErrorCode::invalidArgument);
        const auto outsideBuffer = device.unmapBuffer(*buffer, mappedBufferSize, 1);
        REQUIRE_FALSE(outsideBuffer.has_value());
        CHECK(outsideBuffer.error().code == rachis::ErrorCode::invalidArgument);
        const auto stillMapped = device.mapBuffer(*buffer);
        REQUIRE_FALSE(stillMapped.has_value());
        CHECK(stillMapped.error().code == rachis::ErrorCode::invalidState);
        REQUIRE(device.unmapBuffer(*buffer, mappedOffset + 4, 8).has_value());

        auto remapped = device.mapBuffer(*buffer, mappedOffset, mappedSize);
        REQUIRE(remapped.has_value());
        REQUIRE(device.unmapBuffer(*buffer).has_value());
        auto tailMapped = device.mapBuffer(*buffer, mappedOffset, rachis::wholeResource);
        REQUIRE(tailMapped.has_value());
        CHECK(tailMapped->size == mappedBufferSize - mappedOffset);
        REQUIRE(device.unmapBuffer(*buffer, mappedOffset, rachis::wholeResource).has_value());
        REQUIRE(device.destroyBuffer(*buffer).has_value());
    }

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
