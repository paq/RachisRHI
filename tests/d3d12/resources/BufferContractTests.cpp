#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>

namespace
{

    constexpr std::uint64_t bufferSize = 4096;

    struct AllocationTotals final
    {
        std::uint64_t bytes = 0;
        std::uint64_t count = 0;
    };

    [[nodiscard]] std::byte pattern(std::uint64_t index) noexcept
    {
        return static_cast<std::byte>((index * 37U + 11U) & 0xFFU);
    }

    void printDiagnostic(const rachis::DiagnosticMessage& message, void*) noexcept
    {
        static_cast<void>(std::fprintf(stderr, "D3D12 [%u]: ", static_cast<unsigned int>(message.nativeId)));
        if (!message.description.empty())
        {
            static_cast<void>(std::fwrite(message.description.data(), 1, message.description.size(), stderr));
        }
        static_cast<void>(std::fputc('\n', stderr));
    }

    [[nodiscard]] AllocationTotals allocationTotals(rachis::Device& device)
    {
        const auto budgets = device.queryMemoryBudgets();
        REQUIRE(budgets.has_value());

        AllocationTotals totals;
        for (const auto& budget : *budgets)
        {
            REQUIRE(rachis::hasAny(budget.features, rachis::MemoryBudgetFeature::allocationStatistics));
            totals.bytes += budget.allocationBytes;
            totals.count += budget.allocationCount;
        }
        return totals;
    }

} // namespace

TEST_CASE("buffer upload copy and readback preserve data", "[size:medium][rhi][buffer][transfer][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    const auto baselineAllocations = allocationTotals(device);

    auto upload = device.createBuffer(rachis::BufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Buffer contract upload",
    });
    auto gpu = device.createBuffer(rachis::BufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Buffer contract device local",
    });
    auto readback = device.createBuffer(rachis::BufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Buffer contract readback",
    });
    REQUIRE(upload.has_value());
    REQUIRE(gpu.has_value());
    REQUIRE(readback.has_value());
    const auto liveAllocations = allocationTotals(device);
    CHECK(liveAllocations.count == baselineAllocations.count + 3);
    CHECK(liveAllocations.bytes > baselineAllocations.bytes);

    auto mappedUpload = device.mapBuffer(*upload);
    REQUIRE(mappedUpload.has_value());
    for (std::uint64_t index = 0; index < mappedUpload->size; ++index)
    {
        mappedUpload->data[index] = pattern(index);
    }
    REQUIRE(device.unmapBuffer(*upload).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(command.has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *gpu,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::copyDestination,
                               })
                .has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            rachis::BufferCopy{
                                .source = *upload,
                                .destination = *gpu,
                                .size = bufferSize,
                            })
                .has_value());
    REQUIRE(device
                .bufferBarrier(*command,
                               rachis::BufferBarrier{
                                   .buffer = *gpu,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device
                .copyBuffer(*command,
                            rachis::BufferCopy{
                                .source = *gpu,
                                .destination = *readback,
                                .size = bufferSize,
                            })
                .has_value());

    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());

    const auto staleUpload = *upload;
    const auto staleGpu = *gpu;
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*gpu).has_value());
    CHECK_FALSE(device.mapBuffer(staleUpload).has_value());
    CHECK_FALSE(device.destroyBuffer(staleGpu).has_value());
    const auto retainedAllocations = allocationTotals(device);
    CHECK(retainedAllocations.count == liveAllocations.count);
    CHECK(retainedAllocations.bytes == liveAllocations.bytes);

    REQUIRE(device.wait(*submitted).has_value());
    const auto collectedAllocations = allocationTotals(device);
    CHECK(collectedAllocations.count == baselineAllocations.count + 1);
    CHECK(collectedAllocations.bytes < liveAllocations.bytes);

    auto mappedReadback = device.mapBuffer(*readback);
    REQUIRE(mappedReadback.has_value());
    std::uint64_t firstMismatch = mappedReadback->size;
    for (std::uint64_t index = 0; index < mappedReadback->size; ++index)
    {
        if (mappedReadback->data[index] != pattern(index))
        {
            firstMismatch = index;
            break;
        }
    }
    CAPTURE(firstMismatch);
    CHECK(firstMismatch == mappedReadback->size);
    CHECK_FALSE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.unmapBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    const auto releasedAllocations = allocationTotals(device);
    CHECK(releasedAllocations.count == baselineAllocations.count);
    CHECK(releasedAllocations.bytes == baselineAllocations.bytes);

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("persistent mappings remain coherent across repeated submissions",
          "[size:medium][rhi][buffer][mapping][persistent][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
    REQUIRE(created.has_value());
    auto device = std::move(*created);
    REQUIRE(rachis::test::requireCapabilities(device).persistentMapping);
    REQUIRE(rachis::test::requireCapabilities(device).mappedMemory.hostCoherent);
    REQUIRE(rachis::test::requireCapabilities(device).mappedMemory.nonCoherentAtomSize == 1);

    constexpr std::array firstPayload{
        std::byte{0x10},
        std::byte{0x21},
        std::byte{0x32},
        std::byte{0x43},
        std::byte{0x54},
        std::byte{0x65},
        std::byte{0x76},
        std::byte{0x87},
    };
    constexpr std::array secondPayload{
        std::byte{0xF8},
        std::byte{0xE7},
        std::byte{0xD6},
        std::byte{0xC5},
        std::byte{0xB4},
        std::byte{0xA3},
        std::byte{0x92},
        std::byte{0x81},
    };

    auto upload = device.createBuffer({
        .size = firstPayload.size() + 8,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Persistent upload",
    });
    auto readback = device.createBuffer({
        .size = firstPayload.size() + 8,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Persistent readback",
    });
    REQUIRE(upload.has_value());
    REQUIRE(readback.has_value());

    constexpr std::uint64_t mappedOffset = 4;
    auto mappedUpload = device.mapBuffer(*upload, mappedOffset, firstPayload.size());
    auto mappedReadback = device.mapBuffer(*readback, mappedOffset, firstPayload.size());
    REQUIRE(mappedUpload.has_value());
    REQUIRE(mappedReadback.has_value());
    REQUIRE(mappedUpload->size == firstPayload.size());
    REQUIRE(mappedReadback->size == firstPayload.size());
    auto* const uploadPointer = mappedUpload->data;
    auto* const readbackPointer = mappedReadback->data;
    const std::array synchronizedRanges{
        rachis::MappedMemoryRange{.offset = mappedOffset, .size = firstPayload.size() / 2},
        rachis::MappedMemoryRange{
            .offset = mappedOffset + firstPayload.size() / 2,
            .size = firstPayload.size() / 2,
        },
    };
    const std::array outsideMapping{
        rachis::MappedMemoryRange{.offset = 0, .size = mappedOffset},
    };
    const auto emptyFlush = device.flushMappedMemoryRanges(*upload, {});
    REQUIRE_FALSE(emptyFlush.has_value());
    CHECK(emptyFlush.error().code == rachis::ErrorCode::invalidArgument);
    const auto outsideFlush = device.flushMappedMemoryRanges(*upload, outsideMapping);
    REQUIRE_FALSE(outsideFlush.has_value());
    CHECK(outsideFlush.error().code == rachis::ErrorCode::invalidArgument);
    const auto wrongFlushDirection = device.flushMappedMemoryRanges(*readback, synchronizedRanges);
    const auto wrongInvalidateDirection = device.invalidateMappedMemoryRanges(*upload, synchronizedRanges);
    REQUIRE_FALSE(wrongFlushDirection.has_value());
    REQUIRE_FALSE(wrongInvalidateDirection.has_value());
    CHECK(wrongFlushDirection.error().code == rachis::ErrorCode::unsupported);
    CHECK(wrongInvalidateDirection.error().code == rachis::ErrorCode::unsupported);

    const auto submitCopy = [&]() {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .copyBuffer(*command,
                                {
                                    .source = *upload,
                                    .sourceOffset = mappedOffset,
                                    .destination = *readback,
                                    .destinationOffset = mappedOffset,
                                    .size = firstPayload.size(),
                                })
                    .has_value());
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        return *submitted;
    };

    std::ranges::copy(firstPayload, uploadPointer);
    REQUIRE(device.flushMappedMemoryRanges(*upload, synchronizedRanges).has_value());
    auto firstSubmission = submitCopy();
    auto mappedAgain = device.mapBuffer(*upload);
    REQUIRE_FALSE(mappedAgain.has_value());
    CHECK(mappedAgain.error().code == rachis::ErrorCode::invalidState);
    auto destroyMapped = device.destroyBuffer(*upload);
    REQUIRE_FALSE(destroyMapped.has_value());
    CHECK(destroyMapped.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.wait(firstSubmission).has_value());
    REQUIRE(device.invalidateMappedMemoryRanges(*readback, synchronizedRanges).has_value());
    CHECK(std::ranges::equal(firstPayload, std::span{readbackPointer, firstPayload.size()}));

    std::ranges::copy(secondPayload, uploadPointer);
    REQUIRE(device.flushMappedMemoryRanges(*upload, synchronizedRanges).has_value());
    auto secondSubmission = submitCopy();
    REQUIRE(device.wait(secondSubmission).has_value());
    REQUIRE(device.invalidateMappedMemoryRanges(*readback, synchronizedRanges).has_value());
    CHECK(std::ranges::equal(secondPayload, std::span{readbackPointer, secondPayload.size()}));

    REQUIRE(device.unmapBuffer(*readback, mappedOffset, firstPayload.size()).has_value());
    REQUIRE(device.unmapBuffer(*upload, mappedOffset, firstPayload.size()).has_value());
    const auto flushAfterUnmap = device.flushMappedMemoryRanges(*upload, synchronizedRanges);
    REQUIRE_FALSE(flushAfterUnmap.has_value());
    CHECK(flushAfterUnmap.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*upload).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("buffer unmap ranges stay inside the active mapping", "[size:medium][rhi][buffer][mapping][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
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
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}

TEST_CASE("one-shot host buffer IO is directional bounded and mapping-exclusive",
          "[size:medium][rhi][buffer][host-io][warp]")
{
    auto created = rachis::createDevice(rachis::DeviceDesc{
        .adapterPreference = rachis::AdapterPreference::warp,
        .validation = rachis::ValidationLevel::standard,
        .requireValidation = false,
        .enableDred = true,
        .diagnosticCallback = &printDiagnostic,
    });
    if (!created)
    {
        INFO(created.error().message);
    }
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
        .label = "One-shot host write",
    });
    auto readback = device.createBuffer({
        .size = payload.size(),
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "One-shot host read",
    });
    auto deviceLocal = device.createBuffer({
        .size = payload.size(),
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "One-shot host inaccessible",
    });
    REQUIRE(upload.has_value());
    REQUIRE(readback.has_value());
    REQUIRE(deviceLocal.has_value());

    REQUIRE(device.writeBuffer(*upload, 0, payload).has_value());
    const std::span<const std::byte> empty;
    auto emptyWrite = device.writeBuffer(*upload, 0, empty);
    REQUIRE_FALSE(emptyWrite.has_value());
    CHECK(emptyWrite.error().code == rachis::ErrorCode::invalidArgument);
    auto outOfBoundsWrite = device.writeBuffer(*upload, payload.size() - 1, std::span{payload}.first(2));
    REQUIRE_FALSE(outOfBoundsWrite.has_value());
    CHECK(outOfBoundsWrite.error().code == rachis::ErrorCode::invalidArgument);

    std::array<std::byte, payload.size()> destination{};
    destination.fill(std::byte{0x55});
    const auto unchangedDestination = destination;
    auto outOfBoundsRead = device.readBuffer(*readback, payload.size() - 1, std::span{destination}.first(2));
    REQUIRE_FALSE(outOfBoundsRead.has_value());
    CHECK(outOfBoundsRead.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(destination == unchangedDestination);

    auto wrongWriteDirection = device.writeBuffer(*readback, 0, payload);
    auto wrongReadDirection = device.readBuffer(*upload, 0, destination);
    auto deviceLocalWrite = device.writeBuffer(*deviceLocal, 0, payload);
    auto deviceLocalRead = device.readBuffer(*deviceLocal, 0, destination);
    REQUIRE_FALSE(wrongWriteDirection.has_value());
    REQUIRE_FALSE(wrongReadDirection.has_value());
    REQUIRE_FALSE(deviceLocalWrite.has_value());
    REQUIRE_FALSE(deviceLocalRead.has_value());
    CHECK(wrongWriteDirection.error().code == rachis::ErrorCode::unsupported);
    CHECK(wrongReadDirection.error().code == rachis::ErrorCode::unsupported);
    CHECK(deviceLocalWrite.error().code == rachis::ErrorCode::unsupported);
    CHECK(deviceLocalRead.error().code == rachis::ErrorCode::unsupported);

    auto mappedUpload = device.mapBuffer(*upload);
    REQUIRE(mappedUpload.has_value());
    auto writeWhileMapped = device.writeBuffer(*upload, 0, payload);
    REQUIRE_FALSE(writeWhileMapped.has_value());
    CHECK(writeWhileMapped.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.unmapBuffer(*upload).has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(command.has_value());
    REQUIRE(
        device.copyBuffer(*command, {.source = *upload, .destination = *readback, .size = payload.size()}).has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    destination.fill(std::byte{0});
    REQUIRE(device.readBuffer(*readback, 0, destination).has_value());
    CHECK(destination == payload);

    auto mappedReadback = device.mapBuffer(*readback);
    REQUIRE(mappedReadback.has_value());
    auto readWhileMapped = device.readBuffer(*readback, 0, destination);
    REQUIRE_FALSE(readWhileMapped.has_value());
    CHECK(readWhileMapped.error().code == rachis::ErrorCode::invalidState);
    REQUIRE(device.unmapBuffer(*readback).has_value());

    const auto staleUpload = *upload;
    REQUIRE(device.destroyBuffer(*upload).has_value());
    auto staleWrite = device.writeBuffer(staleUpload, 0, payload);
    REQUIRE_FALSE(staleWrite.has_value());
    CHECK(staleWrite.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(device.destroyBuffer(*readback).has_value());
    REQUIRE(device.destroyBuffer(*deviceLocal).has_value());

    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.warnings == 0);
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
