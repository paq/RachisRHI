#include "Metal/MetalTestHooks.h"
#include "Rachis/Device.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace
{

#if RACHIS_ENABLE_METAL
    [[nodiscard]] std::byte pattern(std::uint64_t index) noexcept
    {
        return static_cast<std::byte>((index * 37U + 11U) & 0xFFU);
    }

    [[nodiscard]] rachis::DeviceDesc metalDesc() noexcept
    {
        return rachis::DeviceDesc{
            .backend = rachis::Backend::metal,
            .validation = rachis::ValidationLevel::standard,
            .requireValidation = false,
            .enableDred = false,
        };
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
#endif

} // namespace

#if RACHIS_ENABLE_METAL

TEST_CASE("Metal rejects invalid buffer descriptors before allocation", "[size:large][rhi][metal][buffer]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
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
    CHECK(residency.error().message.contains("residency"));
}

TEST_CASE("Metal buffer create map write copy and destroy restore allocation counts",
          "[size:large][rhi][metal][buffer]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    const auto checkedCapabilities = device.capabilities();
    REQUIRE(checkedCapabilities.has_value());
    REQUIRE(checkedCapabilities->memoryBudgets);
    REQUIRE(checkedCapabilities->persistentMapping);
    REQUIRE(checkedCapabilities->mappedMemory.hostCoherent);
    REQUIRE(checkedCapabilities->mappedMemory.nonCoherentAtomSize >= 1);

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
        .label = "Metal upload",
    });
    auto gpu = device.createBuffer({
        .size = 4096,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Metal device local",
    });
    auto readback = device.createBuffer({
        .size = 4096,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Metal readback",
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
    REQUIRE(rachis::metal::copyBuffersImmediate(*upload, *gpu, 0, 0, payload.size()).has_value());
    REQUIRE(rachis::metal::copyBuffersImmediate(*gpu, *readback, 0, 0, payload.size()).has_value());

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

TEST_CASE("Metal host buffer IO is directional bounded and mapping-exclusive", "[size:large][rhi][metal][buffer]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
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

    auto emptyWriteWhileMapped = device.writeBuffer(*upload, 0, empty);
    auto outOfBoundsWriteWhileMapped = device.writeBuffer(*upload, payload.size() - 1, std::span{payload}.first(2));
    auto wrappingWriteWhileMapped = device.writeBuffer(*upload, std::numeric_limits<std::uint64_t>::max(), payload);
    auto wrongReadWhileMapped = device.readBuffer(*upload, 0, destination);
    REQUIRE_FALSE(emptyWriteWhileMapped);
    REQUIRE_FALSE(outOfBoundsWriteWhileMapped);
    REQUIRE_FALSE(wrappingWriteWhileMapped);
    REQUIRE_FALSE(wrongReadWhileMapped);
    CHECK(emptyWriteWhileMapped.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(outOfBoundsWriteWhileMapped.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(wrappingWriteWhileMapped.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(wrongReadWhileMapped.error().code == rachis::ErrorCode::unsupported);
    REQUIRE(device.unmapBuffer(*upload).has_value());

    auto mappedReadback = device.mapBuffer(*readback);
    REQUIRE(mappedReadback);
    auto readWhileMapped = device.readBuffer(*readback, 0, destination);
    REQUIRE_FALSE(readWhileMapped);
    CHECK(readWhileMapped.error().code == rachis::ErrorCode::invalidState);
    const std::span<std::byte> emptyMutable;
    auto emptyReadWhileMapped = device.readBuffer(*readback, 0, emptyMutable);
    auto outOfBoundsReadWhileMapped =
        device.readBuffer(*readback, destination.size() - 1, std::span{destination}.first(2));
    auto wrappingReadWhileMapped = device.readBuffer(*readback, std::numeric_limits<std::uint64_t>::max(), destination);
    auto wrongWriteWhileMapped = device.writeBuffer(*readback, 0, payload);
    REQUIRE_FALSE(emptyReadWhileMapped);
    REQUIRE_FALSE(outOfBoundsReadWhileMapped);
    REQUIRE_FALSE(wrappingReadWhileMapped);
    REQUIRE_FALSE(wrongWriteWhileMapped);
    CHECK(emptyReadWhileMapped.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(outOfBoundsReadWhileMapped.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(wrappingReadWhileMapped.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(wrongWriteWhileMapped.error().code == rachis::ErrorCode::unsupported);
    REQUIRE(device.unmapBuffer(*readback).has_value());

    for (const auto target : {*upload, *readback, *deviceLocal})
    {
        auto emptyW = device.writeBuffer(target, 0, empty);
        auto emptyR = device.readBuffer(target, 0, emptyMutable);
        auto pastEndW = device.writeBuffer(target, payload.size(), std::span{payload}.first(1));
        auto pastEndR = device.readBuffer(target, destination.size(), std::span{destination}.first(1));
        auto overflowW = device.writeBuffer(target, payload.size() - 1, std::span{payload}.first(2));
        auto overflowR = device.readBuffer(target, destination.size() - 1, std::span{destination}.first(2));
        auto wrapW = device.writeBuffer(target, std::numeric_limits<std::uint64_t>::max(), payload);
        auto wrapR = device.readBuffer(target, std::numeric_limits<std::uint64_t>::max(), destination);
        REQUIRE_FALSE(emptyW);
        REQUIRE_FALSE(emptyR);
        REQUIRE_FALSE(pastEndW);
        REQUIRE_FALSE(pastEndR);
        REQUIRE_FALSE(overflowW);
        REQUIRE_FALSE(overflowR);
        REQUIRE_FALSE(wrapW);
        REQUIRE_FALSE(wrapR);
        CHECK(emptyW.error().code == rachis::ErrorCode::invalidArgument);
        CHECK(emptyR.error().code == rachis::ErrorCode::invalidArgument);
        CHECK(pastEndW.error().code == rachis::ErrorCode::invalidArgument);
        CHECK(pastEndR.error().code == rachis::ErrorCode::invalidArgument);
        CHECK(overflowW.error().code == rachis::ErrorCode::invalidArgument);
        CHECK(overflowR.error().code == rachis::ErrorCode::invalidArgument);
        CHECK(wrapW.error().code == rachis::ErrorCode::invalidArgument);
        CHECK(wrapR.error().code == rachis::ErrorCode::invalidArgument);
    }

    auto nullWrite = device.writeBuffer({}, 0, empty);
    auto nullRead = device.readBuffer({}, 0, emptyMutable);
    REQUIRE_FALSE(nullWrite);
    REQUIRE_FALSE(nullRead);
    CHECK(nullWrite.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(nullRead.error().code == rachis::ErrorCode::invalidArgument);

    REQUIRE(rachis::metal::copyBuffersImmediate(*upload, *readback, 0, 0, payload.size()).has_value());
    REQUIRE(device.readBuffer(*readback, 0, destination).has_value());
    CHECK(destination == payload);
}

TEST_CASE("Metal mapBuffer rejects out-of-bounds and wrapping offsets without unsigned underflow",
          "[size:large][rhi][metal][buffer]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    constexpr std::uint64_t size = 64;
    auto upload = device.createBuffer({
        .size = size,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Metal map bounds upload",
    });
    REQUIRE(upload);

    auto pastEndWhole = device.mapBuffer(*upload, size + 1, rachis::wholeResource);
    auto pastEndSized = device.mapBuffer(*upload, size + 1, 1);
    auto atEndWhole = device.mapBuffer(*upload, size, rachis::wholeResource);
    auto atEndSized = device.mapBuffer(*upload, size, 1);
    auto zeroSize = device.mapBuffer(*upload, 0, 0);
    auto overflowTail = device.mapBuffer(*upload, size - 4, 8);
    auto maxOffsetWrap = device.mapBuffer(*upload, std::numeric_limits<std::uint64_t>::max(), 16);

    REQUIRE_FALSE(pastEndWhole);
    REQUIRE_FALSE(pastEndSized);
    REQUIRE_FALSE(atEndWhole);
    REQUIRE_FALSE(atEndSized);
    REQUIRE_FALSE(zeroSize);
    REQUIRE_FALSE(overflowTail);
    REQUIRE_FALSE(maxOffsetWrap);

    CHECK(pastEndWhole.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(pastEndSized.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(atEndWhole.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(atEndSized.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(zeroSize.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(overflowTail.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(maxOffsetWrap.error().code == rachis::ErrorCode::invalidArgument);

    auto validTail = device.mapBuffer(*upload, 48, 16);
    REQUIRE(validTail);
    CHECK(validTail->size == 16);
    REQUIRE(device.unmapBuffer(*upload).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal mapped buffers refuse destruction and validate flush and invalidate ranges",
          "[size:large][rhi][metal][buffer]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
    auto device = std::move(*created);

    constexpr std::uint64_t bufferSize = 64;
    constexpr std::uint64_t mappedOffset = 16;
    constexpr std::uint64_t mappedSize = 16;
    auto upload = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Mapped upload",
    });
    auto readback = device.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Mapped readback",
    });
    REQUIRE(upload);
    REQUIRE(readback);

    const std::array mappedRange{rachis::MappedMemoryRange{.offset = mappedOffset, .size = mappedSize}};
    const auto flushUnmapped = device.flushMappedMemoryRanges(*upload, mappedRange);
    REQUIRE_FALSE(flushUnmapped);
    CHECK(flushUnmapped.error().code == rachis::ErrorCode::invalidState);
    const auto invalidateUnmapped = device.invalidateMappedMemoryRanges(*readback, mappedRange);
    REQUIRE_FALSE(invalidateUnmapped);
    CHECK(invalidateUnmapped.error().code == rachis::ErrorCode::invalidState);

    REQUIRE(device.mapBuffer(*upload, mappedOffset, mappedSize));
    REQUIRE(device.mapBuffer(*readback, mappedOffset, mappedSize));
    for (const auto target : {*upload, *readback})
    {
        const auto destroyedWhileMapped = device.destroyBuffer(target);
        REQUIRE_FALSE(destroyedWhileMapped);
        CHECK(destroyedWhileMapped.error().code == rachis::ErrorCode::invalidState);
        const auto stillMapped = device.mapBuffer(target);
        REQUIRE_FALSE(stillMapped);
        CHECK(stillMapped.error().code == rachis::ErrorCode::invalidState);
    }

    REQUIRE(device.flushMappedMemoryRanges(*upload, mappedRange).has_value());
    REQUIRE(device.invalidateMappedMemoryRanges(*readback, mappedRange).has_value());
    const auto emptyFlush = device.flushMappedMemoryRanges(*upload, {});
    REQUIRE_FALSE(emptyFlush);
    CHECK(emptyFlush.error().code == rachis::ErrorCode::invalidArgument);
    const auto emptyInvalidate = device.invalidateMappedMemoryRanges(*readback, {});
    REQUIRE_FALSE(emptyInvalidate);
    CHECK(emptyInvalidate.error().code == rachis::ErrorCode::invalidArgument);
    const auto wrongFlush = device.flushMappedMemoryRanges(*readback, mappedRange);
    REQUIRE_FALSE(wrongFlush);
    CHECK(wrongFlush.error().code == rachis::ErrorCode::unsupported);
    const auto wrongInvalidate = device.invalidateMappedMemoryRanges(*upload, mappedRange);
    REQUIRE_FALSE(wrongInvalidate);
    CHECK(wrongInvalidate.error().code == rachis::ErrorCode::unsupported);

    const std::array rejectedRanges{
        rachis::MappedMemoryRange{.offset = 0, .size = mappedOffset},
        rachis::MappedMemoryRange{.offset = mappedOffset, .size = rachis::wholeResource},
        rachis::MappedMemoryRange{.offset = mappedOffset + mappedSize / 2, .size = mappedSize},
        rachis::MappedMemoryRange{.offset = mappedOffset, .size = 0},
        rachis::MappedMemoryRange{.offset = bufferSize, .size = 1},
    };
    for (const auto& range : rejectedRanges)
    {
        CAPTURE(range.offset, range.size);
        const auto flushed = device.flushMappedMemoryRanges(*upload, std::span{&range, 1});
        REQUIRE_FALSE(flushed);
        CHECK(flushed.error().code == rachis::ErrorCode::invalidArgument);
        const auto invalidated = device.invalidateMappedMemoryRanges(*readback, std::span{&range, 1});
        REQUIRE_FALSE(invalidated);
        CHECK(invalidated.error().code == rachis::ErrorCode::invalidArgument);
    }

    REQUIRE(device.unmapBuffer(*upload).has_value());
    REQUIRE(device.unmapBuffer(*readback).has_value());
    const auto flushAfterUnmap = device.flushMappedMemoryRanges(*upload, mappedRange);
    REQUIRE_FALSE(flushAfterUnmap);
    CHECK(flushAfterUnmap.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    CHECK(device.diagnosticCounts().errors == 0);
}

TEST_CASE("Metal unmap ranges must lie inside the active mapping", "[size:large][rhi][metal][buffer]")
{
    auto created = rachis::createDevice(metalDesc());
    REQUIRE(created.has_value());
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

#endif
