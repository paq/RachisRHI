#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <Windows.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace
{

    constexpr std::uint64_t bufferSize = 4'096;
    constexpr std::uint64_t sharedBudgetBufferSize = 64ULL * 1024ULL * 1024ULL;
    constexpr std::uint64_t constantBufferSize = 64;
    constexpr std::uint64_t paddedConstantBufferWidth = 256;
    constexpr std::uint64_t dynamicConstantArrayOutputSize = 2 * 4 * sizeof(float);

    void printDiagnostic(const rachis::DiagnosticMessage& message, void*) noexcept
    {
        static_cast<void>(std::fprintf(stderr, "D3D12 [%u]: ", static_cast<unsigned int>(message.nativeId)));
        if (!message.description.empty())
        {
            static_cast<void>(std::fwrite(message.description.data(), 1, message.description.size(), stderr));
        }
        static_cast<void>(std::fputc('\n', stderr));
    }

    [[nodiscard]] rachis::Device createWarpDevice()
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
        return std::move(*created);
    }

    [[nodiscard]] constexpr std::byte payloadByte(std::uint8_t seed, std::uint64_t index) noexcept
    {
        return static_cast<std::byte>((seed + index * 37U + index / 17U * 53U) & 0xFFU);
    }

    void fillUpload(rachis::Device& device, rachis::BufferHandle buffer, std::uint8_t seed)
    {
        auto mapped = device.mapBuffer(buffer);
        REQUIRE(mapped.has_value());
        REQUIRE(mapped->size == bufferSize);
        for (std::uint64_t index = 0; index < mapped->size; ++index)
        {
            mapped->data[index] = payloadByte(seed, index);
        }
        REQUIRE(device.unmapBuffer(buffer).has_value());
    }

    void requireReadback(rachis::Device& device, rachis::BufferHandle buffer, std::uint8_t seed)
    {
        auto mapped = device.mapBuffer(buffer);
        REQUIRE(mapped.has_value());
        REQUIRE(mapped->size == bufferSize);
        for (std::uint64_t index = 0; index < mapped->size; ++index)
        {
            CAPTURE(seed, index);
            REQUIRE(mapped->data[index] == payloadByte(seed, index));
        }
        REQUIRE(device.unmapBuffer(buffer).has_value());
    }

    [[nodiscard]] rachis::TimelinePoint copyBuffer(rachis::Device& device,
                                                   rachis::BufferHandle source,
                                                   rachis::BufferHandle destination)
    {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .copyBuffer(*command,
                                rachis::BufferCopy{
                                    .source = source,
                                    .destination = destination,
                                    .size = bufferSize,
                                })
                    .has_value());
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        return *submitted;
    }

    struct RecordedCopy final
    {
        rachis::CommandBuffer command;
    };

    [[nodiscard]] RecordedCopy recordCopy(rachis::Device& device,
                                          rachis::BufferHandle source,
                                          rachis::BufferHandle destination)
    {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .copyBuffer(*command,
                                rachis::BufferCopy{
                                    .source = source,
                                    .destination = destination,
                                    .size = bufferSize,
                                })
                    .has_value());
        return {.command = std::move(*command)};
    }

    void requireCleanDiagnostics(const rachis::Device& device)
    {
        const auto diagnostics = device.diagnosticCounts();
        CHECK(diagnostics.warnings == 0);
        CHECK(diagnostics.errors == 0);
        CHECK(diagnostics.corruptions == 0);
    }

    [[nodiscard]] const rachis::ResourceMemoryEntry* findByLabel(const rachis::ResourceMemorySnapshot& snapshot,
                                                                 std::string_view label)
    {
        const auto found = std::ranges::find(snapshot.entries, label, &rachis::ResourceMemoryEntry::label);
        return found == snapshot.entries.end() ? nullptr : &*found;
    }

    [[nodiscard]] bool ntHandleOpen(std::uintptr_t value) noexcept
    {
        DWORD flags = 0;
        return GetHandleInformation(std::bit_cast<HANDLE>(value), &flags) != FALSE;
    }

    void requireIndependentExports(rachis::SharedHandle& first, rachis::SharedHandle& second)
    {
        REQUIRE(first.isValid());
        REQUIRE(second.isValid());
        CHECK(first.view().value != second.view().value);
        second.reset();
        CHECK(ntHandleOpen(first.view().value));
        auto* const applicationEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        REQUIRE(applicationEvent != nullptr);
        first.reset();
        const bool eventSurvived = ntHandleOpen(std::bit_cast<std::uintptr_t>(applicationEvent));
        if (eventSurvived)
        {
            static_cast<void>(CloseHandle(applicationEvent));
        }
        CHECK(eventSurvived);
    }

    void handOff(rachis::Device& from, rachis::TimelinePoint point, rachis::Device& to, rachis::QueueType queueType)
    {
        auto shared = from.exportTimelinePoint(point);
        REQUIRE(shared.has_value());
        REQUIRE(to.enqueueSharedTimelineWait(queueType, shared->view()).has_value());
        shared->handle.reset();
    }

    [[nodiscard]] std::vector<rachis::MemoryBudget> requireBudgets(const rachis::Device& device)
    {
        auto budgets = device.queryMemoryBudgets();
        REQUIRE(budgets.has_value());
        return std::move(*budgets);
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

    [[nodiscard]] ComPtr<ID3D12Device> createNativeWarpDevice()
    {
        ComPtr<IDXGIFactory4> factory;
        REQUIRE(SUCCEEDED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))));
        ComPtr<IDXGIAdapter> adapter;
        REQUIRE(SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))));
        ComPtr<ID3D12Device> device;
        REQUIRE(SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device))));
        return device;
    }

    [[nodiscard]] ComPtr<ID3D12Resource> createUnpaddedSharedBuffer(ID3D12Device& device, std::uint64_t width)
    {
        const D3D12_HEAP_PROPERTIES heap{.Type = D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_RESOURCE_DESC desc{
            .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
            .Width = width,
            .Height = 1,
            .DepthOrArraySize = 1,
            .MipLevels = 1,
            .Format = DXGI_FORMAT_UNKNOWN,
            .SampleDesc = {.Count = 1, .Quality = 0},
            .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        };
        ComPtr<ID3D12Resource> resource;
        REQUIRE(SUCCEEDED(device.CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&resource))));
        return resource;
    }

    [[nodiscard]] rachis::Result<rachis::BufferHandle> importNativeBuffer(rachis::Device& device,
                                                                          ID3D12Device& nativeDevice,
                                                                          ID3D12Resource& resource,
                                                                          const rachis::BufferDesc& desc)
    {
        HANDLE handle = nullptr;
        REQUIRE(SUCCEEDED(nativeDevice.CreateSharedHandle(&resource, nullptr, GENERIC_ALL, nullptr, &handle)));
        auto imported = device.importSharedBuffer(
            {.type = rachis::SharedHandleType::windowsNt, .value = reinterpret_cast<std::uintptr_t>(handle)}, desc);
        REQUIRE(CloseHandle(handle) != 0);
        return imported;
    }

    [[nodiscard]] rachis::Result<rachis::BindingSetHandle> bindConstantBuffer(rachis::Device& device,
                                                                              rachis::BindingLayoutHandle layout,
                                                                              rachis::BindingPoolHandle pool,
                                                                              rachis::BufferHandle buffer)
    {
        const std::array writes{
            rachis::BindingWrite{
                .binding = 0,
                .resource = rachis::BufferBinding{.buffer = buffer, .offset = 0, .size = constantBufferSize},
            },
        };
        return device.createBindingSet({
            .layout = layout,
            .writes = writes,
            .label = "Shared constant binding set",
            .pool = pool,
        });
    }

    [[nodiscard]] std::vector<std::byte> loadShader(const char* path)
    {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        const auto end = stream.tellg();
        if (end <= 0)
        {
            return {};
        }
        std::vector<std::byte> result(static_cast<std::size_t>(end));
        stream.seekg(0, std::ios::beg);
        stream.read(reinterpret_cast<char*>(result.data()), end);
        return stream ? result : std::vector<std::byte>{};
    }

    [[nodiscard]] rachis::Result<rachis::BindingSetHandle> bindDynamicConstantArray(rachis::Device& device,
                                                                                    rachis::BindingLayoutHandle layout,
                                                                                    rachis::BindingPoolHandle pool,
                                                                                    rachis::BufferHandle buffer,
                                                                                    rachis::BufferHandle output)
    {
        const rachis::BufferBinding range{.buffer = buffer, .offset = 0, .size = constantBufferSize};
        const std::array writes{
            rachis::BindingWrite{.binding = 0, .arrayElement = 0, .resource = range},
            rachis::BindingWrite{.binding = 0, .arrayElement = 1, .resource = range},
            rachis::BindingWrite{
                .binding = 2,
                .resource =
                    rachis::BufferBinding{
                        .buffer = output,
                        .size = dynamicConstantArrayOutputSize,
                        .structureByteStride = dynamicConstantArrayOutputSize / 2,
                    },
            },
        };
        return device.createBindingSet({
            .layout = layout,
            .writes = writes,
            .label = "Dynamic constant array binding set",
            .pool = pool,
        });
    }

} // namespace

TEST_CASE("shared buffer and timeline imports reject INVALID_HANDLE_VALUE and unknown handle types",
          "[size:medium][rhi][interop][buffer][timeline][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).sharedBufferHandles);
    REQUIRE(rachis::test::requireCapabilities(device).sharedTimelineHandles);

    const rachis::BufferDesc sharedDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Rejected shared interop buffer",
    };
    constexpr auto invalidHandleValue = std::numeric_limits<std::uintptr_t>::max();
    const std::array rejected{
        rachis::SharedHandleView{.type = rachis::SharedHandleType::windowsNt, .value = invalidHandleValue},
        rachis::SharedHandleView{.type = static_cast<rachis::SharedHandleType>(2), .value = invalidHandleValue},
    };
    for (const auto handle : rejected)
    {
        CAPTURE(static_cast<unsigned int>(handle.type), handle.value);
        auto imported = device.importSharedBuffer(handle, sharedDesc);
        REQUIRE_FALSE(imported.has_value());
        CHECK(imported.error().code == rachis::ErrorCode::invalidArgument);
        CHECK(imported.error().nativeCode == 0);

        auto waited = device.enqueueSharedTimelineWait(rachis::QueueType::copy, {.handle = handle, .value = 1});
        REQUIRE_FALSE(waited.has_value());
        CHECK(waited.error().code == rachis::ErrorCode::invalidArgument);
        CHECK(waited.error().nativeCode == 0);
    }

    auto sharedBuffer = device.createSharedBuffer(sharedDesc);
    REQUIRE(sharedBuffer.has_value());
    auto exported = device.exportSharedBuffer(*sharedBuffer);
    REQUIRE(exported.has_value());
    REQUIRE(exported->isValid());
    auto unknownType = device.importSharedBuffer(
        {.type = static_cast<rachis::SharedHandleType>(2), .value = exported->view().value}, sharedDesc);
    REQUIRE_FALSE(unknownType.has_value());
    CHECK(unknownType.error().code == rachis::ErrorCode::invalidArgument);
    CHECK(unknownType.error().nativeCode == 0);
    exported->reset();
    REQUIRE(device.destroyBuffer(*sharedBuffer).has_value());
    device.collectGarbage();
    requireCleanDiagnostics(device);
}

TEST_CASE("shared buffers use explicit cross-device timeline handoffs",
          "[size:medium][rhi][interop][buffer][timeline][warp]")
{
    auto producer = createWarpDevice();
    auto consumer = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(producer).sharedBufferHandles);
    REQUIRE(rachis::test::requireCapabilities(producer).sharedTimelineHandles);
    REQUIRE(rachis::test::requireCapabilities(consumer).sharedBufferHandles);
    REQUIRE(rachis::test::requireCapabilities(consumer).sharedTimelineHandles);

    const rachis::BufferDesc sharedDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Shared interop buffer",
    };
    auto invalidUploadDesc = sharedDesc;
    invalidUploadDesc.memoryType = rachis::MemoryType::upload;
    CHECK_FALSE(producer.createSharedBuffer(invalidUploadDesc).has_value());
    auto invalidPriorityDesc = sharedDesc;
    invalidPriorityDesc.residencyPriority = rachis::ResidencyPriority::high;
    CHECK_FALSE(producer.createSharedBuffer(invalidPriorityDesc).has_value());
    auto invalidManualDesc = sharedDesc;
    invalidManualDesc.allocationFlags = rachis::ResourceAllocationFlag::manualResidency;
    CHECK_FALSE(producer.createSharedBuffer(invalidManualDesc).has_value());
    auto invalidAccelerationStructureDesc = sharedDesc;
    invalidAccelerationStructureDesc.usage = rachis::BufferUsage::accelerationStructureScratch;
    CHECK_FALSE(producer.createSharedBuffer(invalidAccelerationStructureDesc).has_value());
    auto ordinary = producer.createBuffer(sharedDesc);
    REQUIRE(ordinary.has_value());
    CHECK_FALSE(producer.exportSharedBuffer(*ordinary).has_value());
    REQUIRE(producer.destroyBuffer(*ordinary).has_value());
    CHECK_FALSE(consumer.importSharedBuffer({}, sharedDesc).has_value());

    auto producerUpload = producer.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Shared interop producer upload",
    });
    auto producerReadback = producer.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Shared interop producer readback",
    });
    auto consumerUpload = consumer.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Shared interop consumer upload",
    });
    auto consumerReadback = consumer.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Shared interop consumer readback",
    });
    auto sharedBuffer = producer.createSharedBuffer(sharedDesc);
    REQUIRE(producerUpload.has_value());
    REQUIRE(producerReadback.has_value());
    REQUIRE(consumerUpload.has_value());
    REQUIRE(consumerReadback.has_value());
    REQUIRE(sharedBuffer.has_value());

    const auto producerMemory = producer.captureResourceMemorySnapshot();
    REQUIRE(producerMemory.has_value());
    const auto* sharedEntry = findByLabel(*producerMemory, sharedDesc.label);
    REQUIRE(sharedEntry != nullptr);
    CHECK(sharedEntry->dedicatedAllocation);
    CHECK(sharedEntry->shared);

    const auto consumerMemoryBeforeImport = consumer.captureResourceMemorySnapshot();
    REQUIRE(consumerMemoryBeforeImport.has_value());
    auto bufferHandle = producer.exportSharedBuffer(*sharedBuffer);
    REQUIRE(bufferHandle.has_value());
    auto mismatchDesc = sharedDesc;
    mismatchDesc.size += 1;
    CHECK_FALSE(consumer.importSharedBuffer(bufferHandle->view(), mismatchDesc).has_value());
    auto incompatibleUsageDesc = sharedDesc;
    incompatibleUsageDesc.usage = incompatibleUsageDesc.usage | rachis::BufferUsage::storage;
    CHECK_FALSE(consumer.importSharedBuffer(bufferHandle->view(), incompatibleUsageDesc).has_value());
    auto importedBuffer = consumer.importSharedBuffer(bufferHandle->view(), sharedDesc);
    REQUIRE(importedBuffer.has_value());
    bufferHandle->reset();
    const auto consumerMemoryAfterImport = consumer.captureResourceMemorySnapshot();
    REQUIRE(consumerMemoryAfterImport.has_value());
    CHECK(consumerMemoryAfterImport->entries.size() == consumerMemoryBeforeImport->entries.size());

    constexpr std::uint8_t producerSeed = 17;
    fillUpload(producer, *producerUpload, producerSeed);
    const auto producerPoint = copyBuffer(producer, *producerUpload, *sharedBuffer);
    auto sharedProducerPoint = producer.exportTimelinePoint(producerPoint);
    REQUIRE(sharedProducerPoint.has_value());
    REQUIRE(consumer.enqueueSharedTimelineWait(rachis::QueueType::copy, sharedProducerPoint->view()).has_value());
    sharedProducerPoint->handle.reset();

    const auto consumerReadPoint = copyBuffer(consumer, *importedBuffer, *consumerReadback);
    REQUIRE(consumer.wait(consumerReadPoint).has_value());
    requireReadback(consumer, *consumerReadback, producerSeed);

    constexpr std::uint8_t consumerSeed = 193;
    fillUpload(consumer, *consumerUpload, consumerSeed);
    const auto consumerPoint = copyBuffer(consumer, *consumerUpload, *importedBuffer);
    auto sharedConsumerPoint = consumer.exportTimelinePoint(consumerPoint);
    REQUIRE(sharedConsumerPoint.has_value());
    REQUIRE(producer.enqueueSharedTimelineWait(rachis::QueueType::copy, sharedConsumerPoint->view()).has_value());
    sharedConsumerPoint->handle.reset();

    auto producerRead = recordCopy(producer, *sharedBuffer, *producerReadback);
    REQUIRE(producer.destroyBuffer(*sharedBuffer).has_value());
    auto producerReadPoint = producer.submit(std::move(producerRead.command));
    REQUIRE(producerReadPoint.has_value());
    REQUIRE(consumer.destroyBuffer(*importedBuffer).has_value());
    REQUIRE(producer.wait(*producerReadPoint).has_value());
    requireReadback(producer, *producerReadback, consumerSeed);

    REQUIRE(producer.destroyBuffer(*producerUpload).has_value());
    REQUIRE(producer.destroyBuffer(*producerReadback).has_value());
    REQUIRE(consumer.destroyBuffer(*consumerUpload).has_value());
    REQUIRE(consumer.destroyBuffer(*consumerReadback).has_value());
    producer.collectGarbage();
    consumer.collectGarbage();
    requireCleanDiagnostics(producer);
    requireCleanDiagnostics(consumer);
}

TEST_CASE("shared buffer exports are independent NT handles", "[size:medium][rhi][interop][buffer][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).sharedBufferHandles);

    auto sharedBuffer = device.createSharedBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Twice-exported shared buffer",
    });
    REQUIRE(sharedBuffer.has_value());
    auto first = device.exportSharedBuffer(*sharedBuffer);
    auto second = device.exportSharedBuffer(*sharedBuffer);
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());
    requireIndependentExports(*first, *second);

    REQUIRE(device.destroyBuffer(*sharedBuffer).has_value());
    device.collectGarbage();
    requireCleanDiagnostics(device);
}

TEST_CASE("shared buffers count in memory budgets until their memory is freed",
          "[size:medium][rhi][interop][buffer][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).sharedBufferHandles);
    REQUIRE(rachis::test::requireCapabilities(device).memoryBudgets);

    const auto baseline = requireBudgets(device);
    constexpr std::string_view label = "Budgeted shared buffer";
    auto sharedBuffer = device.createSharedBuffer({
        .size = sharedBudgetBufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = label,
    });
    REQUIRE(sharedBuffer.has_value());
    const auto snapshot = device.captureResourceMemorySnapshot();
    REQUIRE(snapshot.has_value());
    const auto* sharedEntry = findByLabel(*snapshot, label);
    REQUIRE(sharedEntry != nullptr);
    CHECK(sharedEntry->shared);
    CHECK(sharedEntry->allocationBytes >= sharedBudgetBufferSize);
    requireBudgetsCountAllocation(baseline, requireBudgets(device), sharedEntry->allocationBytes);

    REQUIRE(device.destroyBuffer(*sharedBuffer).has_value());
    device.collectGarbage();
    requireBudgetCountersRestored(baseline, requireBudgets(device));
    requireCleanDiagnostics(device);
}

TEST_CASE("shared buffer re-exports name their buffer after an earlier export closes",
          "[size:medium][rhi][interop][buffer][timeline][warp]")
{
    auto producer = createWarpDevice();
    auto firstConsumer = createWarpDevice();
    auto secondConsumer = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(producer).sharedBufferHandles);
    REQUIRE(rachis::test::requireCapabilities(producer).sharedTimelineHandles);

    const rachis::BufferDesc sharedDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Re-exported shared buffer",
    };
    auto producerUpload = producer.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Re-export producer upload",
    });
    auto producerReadback = producer.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Re-export producer readback",
    });
    auto consumerUpload = secondConsumer.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Re-export consumer upload",
    });
    auto consumerReadback = secondConsumer.createBuffer({
        .size = bufferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Re-export consumer readback",
    });
    auto exportedBuffer = producer.createSharedBuffer(sharedDesc);
    auto otherBuffer = producer.createSharedBuffer(sharedDesc);
    REQUIRE(producerUpload.has_value());
    REQUIRE(producerReadback.has_value());
    REQUIRE(consumerUpload.has_value());
    REQUIRE(consumerReadback.has_value());
    REQUIRE(exportedBuffer.has_value());
    REQUIRE(otherBuffer.has_value());

    constexpr std::uint8_t exportedSeed = 41;
    constexpr std::uint8_t otherSeed = 97;
    constexpr std::uint8_t consumerSeed = 211;
    fillUpload(producer, *producerUpload, exportedSeed);
    REQUIRE(producer.wait(copyBuffer(producer, *producerUpload, *exportedBuffer)).has_value());
    auto firstExport = producer.exportSharedBuffer(*exportedBuffer);
    REQUIRE(firstExport.has_value());
    auto firstImport = firstConsumer.importSharedBuffer(firstExport->view(), sharedDesc);
    REQUIRE(firstImport.has_value());
    fillUpload(producer, *producerUpload, otherSeed);
    const auto producerPoint = copyBuffer(producer, *producerUpload, *otherBuffer);
    REQUIRE(producer.wait(producerPoint).has_value());
    firstExport->reset();

    auto otherExport = producer.exportSharedBuffer(*otherBuffer);
    REQUIRE(otherExport.has_value());
    auto reexport = producer.exportSharedBuffer(*exportedBuffer);
    REQUIRE(reexport.has_value());
    CHECK(reexport->view().value != otherExport->view().value);
    auto reimported = secondConsumer.importSharedBuffer(reexport->view(), sharedDesc);
    if (!reimported)
    {
        INFO(reimported.error().message);
    }
    REQUIRE(reimported.has_value());
    reexport->reset();
    otherExport->reset();

    handOff(producer, producerPoint, secondConsumer, rachis::QueueType::copy);
    REQUIRE(secondConsumer.wait(copyBuffer(secondConsumer, *reimported, *consumerReadback)).has_value());
    requireReadback(secondConsumer, *consumerReadback, exportedSeed);

    fillUpload(secondConsumer, *consumerUpload, consumerSeed);
    const auto consumerPoint = copyBuffer(secondConsumer, *consumerUpload, *reimported);
    handOff(secondConsumer, consumerPoint, producer, rachis::QueueType::copy);
    REQUIRE(producer.wait(copyBuffer(producer, *otherBuffer, *producerReadback)).has_value());
    requireReadback(producer, *producerReadback, otherSeed);
    REQUIRE(producer.wait(copyBuffer(producer, *exportedBuffer, *producerReadback)).has_value());
    requireReadback(producer, *producerReadback, consumerSeed);

    REQUIRE(secondConsumer.destroyBuffer(*reimported).has_value());
    REQUIRE(firstConsumer.destroyBuffer(*firstImport).has_value());
    REQUIRE(producer.destroyBuffer(*exportedBuffer).has_value());
    REQUIRE(producer.destroyBuffer(*otherBuffer).has_value());
    REQUIRE(producer.destroyBuffer(*producerUpload).has_value());
    REQUIRE(producer.destroyBuffer(*producerReadback).has_value());
    REQUIRE(secondConsumer.destroyBuffer(*consumerUpload).has_value());
    REQUIRE(secondConsumer.destroyBuffer(*consumerReadback).has_value());
    producer.collectGarbage();
    firstConsumer.collectGarbage();
    secondConsumer.collectGarbage();
    requireCleanDiagnostics(producer);
    requireCleanDiagnostics(firstConsumer);
    requireCleanDiagnostics(secondConsumer);
}

TEST_CASE("constant-usage shared buffers pad their native width to 256 bytes for constant-buffer views",
          "[size:medium][rhi][interop][buffer][binding][warp]")
{
    auto producer = createWarpDevice();
    auto consumer = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(producer).sharedBufferHandles);
    REQUIRE(rachis::test::requireCapabilities(consumer).sharedBufferHandles);
    auto nativeDevice = createNativeWarpDevice();

    const rachis::BufferDesc constantDesc{
        .size = constantBufferSize,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Shared 64-byte constant buffer",
    };
    auto sharedBuffer = producer.createSharedBuffer(constantDesc);
    REQUIRE(sharedBuffer.has_value());
    auto exported = producer.exportSharedBuffer(*sharedBuffer);
    REQUIRE(exported.has_value());
    ComPtr<ID3D12Resource> nativeResource;
    REQUIRE(SUCCEEDED(
        nativeDevice->OpenSharedHandle(std::bit_cast<HANDLE>(exported->view().value), IID_PPV_ARGS(&nativeResource))));
    CHECK(nativeResource->GetDesc().Width == paddedConstantBufferWidth);

    auto copyOnlyDesc = constantDesc;
    copyOnlyDesc.usage = rachis::BufferUsage::copySource;
    auto copyOnly = consumer.importSharedBuffer(exported->view(), copyOnlyDesc);
    REQUIRE(copyOnly.has_value());
    REQUIRE(consumer.destroyBuffer(*copyOnly).has_value());
    auto oversizedDesc = constantDesc;
    oversizedDesc.size = paddedConstantBufferWidth + 1;
    CHECK_FALSE(consumer.importSharedBuffer(exported->view(), oversizedDesc).has_value());
    auto imported = consumer.importSharedBuffer(exported->view(), constantDesc);
    REQUIRE(imported.has_value());
    exported->reset();

    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::uniformBuffer, rachis::ShaderStage::compute, 1},
    };
    auto layout = consumer.createBindingLayout({.entries = entries, .label = "Shared constant binding layout"});
    REQUIRE(layout.has_value());
    const std::array poolSizes{rachis::BindingPoolSize{rachis::BindingType::uniformBuffer, 2}};
    auto pool = consumer.createBindingPool({
        .maxBindingSets = 2,
        .sizes = poolSizes,
        .label = "Shared constant binding pool",
    });
    REQUIRE(pool.has_value());

    auto paddedSet = bindConstantBuffer(consumer, *layout, *pool, *imported);
    const std::string paddedFailure = paddedSet ? std::string{} : paddedSet.error().message;
    CAPTURE(paddedFailure);
    REQUIRE(paddedSet.has_value());

    auto unpaddedResource = createUnpaddedSharedBuffer(*nativeDevice.Get(), constantBufferSize);
    auto unpadded = importNativeBuffer(consumer, *nativeDevice.Get(), *unpaddedResource.Get(), constantDesc);
    const std::string importFailure = unpadded ? std::string{} : unpadded.error().message;
    CAPTURE(importFailure);
    REQUIRE(unpadded.has_value());
    auto unpaddedSet = bindConstantBuffer(consumer, *layout, *pool, *unpadded);
    REQUIRE_FALSE(unpaddedSet.has_value());
    CHECK(unpaddedSet.error().code == rachis::ErrorCode::invalidArgument);

    REQUIRE(consumer.destroyBindingSet(*paddedSet).has_value());
    REQUIRE(consumer.destroyBindingPool(*pool).has_value());
    REQUIRE(consumer.destroyBindingLayout(*layout).has_value());
    REQUIRE(consumer.destroyBuffer(*unpadded).has_value());
    REQUIRE(consumer.destroyBuffer(*imported).has_value());
    REQUIRE(producer.destroyBuffer(*sharedBuffer).has_value());
    producer.collectGarbage();
    consumer.collectGarbage();
    requireCleanDiagnostics(producer);
    requireCleanDiagnostics(consumer);
}

TEST_CASE("dynamic constant-buffer arrays reject an unpadded imported buffer once an offset pushes the view past it",
          "[size:medium][rhi][interop][buffer][binding][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).sharedBufferHandles);
    auto nativeDevice = createNativeWarpDevice();

    constexpr std::uint64_t unpaddedWidth = paddedConstantBufferWidth + constantBufferSize;
    const rachis::BufferDesc constantDesc{
        .size = unpaddedWidth,
        .usage = rachis::BufferUsage::constant,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Unpadded 320-byte constant buffer",
    };
    auto unpaddedResource = createUnpaddedSharedBuffer(*nativeDevice.Get(), unpaddedWidth);
    auto unpadded = importNativeBuffer(device, *nativeDevice.Get(), *unpaddedResource.Get(), constantDesc);
    const std::string importFailure = unpadded ? std::string{} : unpadded.error().message;
    CAPTURE(importFailure);
    REQUIRE(unpadded.has_value());
    auto padded = device.createBuffer(constantDesc);
    REQUIRE(padded.has_value());
    auto output = device.createBuffer({
        .size = dynamicConstantArrayOutputSize,
        .usage = rachis::BufferUsage::storage,
        .label = "Dynamic constant array output",
    });
    REQUIRE(output.has_value());

    const std::array entries{
        rachis::BindingLayoutEntry{0, rachis::BindingType::dynamicUniformBuffer, rachis::ShaderStage::compute, 2},
        rachis::BindingLayoutEntry{2, rachis::BindingType::storageBuffer, rachis::ShaderStage::compute, 1},
    };
    auto layout = device.createBindingLayout({.entries = entries, .label = "Dynamic constant array layout"});
    REQUIRE(layout.has_value());
    const std::array poolSizes{
        rachis::BindingPoolSize{rachis::BindingType::dynamicUniformBuffer, 4},
        rachis::BindingPoolSize{rachis::BindingType::storageBuffer, 2},
    };
    auto pool = device.createBindingPool({
        .maxBindingSets = 2,
        .sizes = poolSizes,
        .label = "Dynamic constant array pool",
    });
    REQUIRE(pool.has_value());
    auto unpaddedSet = bindDynamicConstantArray(device, *layout, *pool, *unpadded, *output);
    const std::string unpaddedFailure = unpaddedSet ? std::string{} : unpaddedSet.error().message;
    CAPTURE(unpaddedFailure);
    REQUIRE(unpaddedSet.has_value());
    auto paddedSet = bindDynamicConstantArray(device, *layout, *pool, *padded, *output);
    REQUIRE(paddedSet.has_value());

    const auto bytecode = loadShader(RACHIS_TEST_DYNAMIC_UNIFORM_BUFFER_ARRAY_SHADER_PATH);
    REQUIRE_FALSE(bytecode.empty());
    const std::array requirements{
        rachis::ShaderBindingRequirement{0, 0, rachis::ShaderResourceType::uniformBuffer, 2},
        rachis::ShaderBindingRequirement{0, 2, rachis::ShaderResourceType::storageBuffer, 1},
    };
    auto shader = device.createShaderModule({
        .stage = rachis::ShaderStage::compute,
        .bytecode = bytecode,
        .entryPoint = "main",
        .label = "Dynamic constant array shader",
        .shaderInterface = rachis::test::computeShaderInterface({1, 1, 1}, requirements),
    });
    REQUIRE(shader.has_value());
    const std::array layouts{*layout};
    auto pipelineLayout =
        device.createPipelineLayout({.bindingLayouts = layouts, .label = "Dynamic constant array pipeline layout"});
    REQUIRE(pipelineLayout.has_value());
    auto pipeline = device.createComputePipeline({
        .layout = *pipelineLayout,
        .computeShader = *shader,
        .label = "Dynamic constant array pipeline",
    });
    REQUIRE(pipeline.has_value());

    auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::compute);
    REQUIRE(command.has_value());
    REQUIRE(device.bindComputePipeline(*command, *pipeline).has_value());
    const std::array inside{0U, 0U};
    auto boundInside = device.bindComputeBindingSet(*command, 0, *unpaddedSet, inside);
    const std::string insideFailure = boundInside ? std::string{} : boundInside.error().message;
    CAPTURE(insideFailure);
    REQUIRE(boundInside.has_value());
    for (const auto& offsets : {std::array{256U, 0U}, std::array{0U, 256U}})
    {
        CAPTURE(offsets[0], offsets[1]);
        auto rejected = device.bindComputeBindingSet(*command, 0, *unpaddedSet, offsets);
        REQUIRE_FALSE(rejected.has_value());
        CHECK(rejected.error().code == rachis::ErrorCode::invalidArgument);
        CHECK(rejected.error().message.find("native buffer") != std::string::npos);
    }
    const std::array pushed{256U, 256U};
    auto boundPadded = device.bindComputeBindingSet(*command, 0, *paddedSet, pushed);
    const std::string paddedFailure = boundPadded ? std::string{} : boundPadded.error().message;
    CAPTURE(paddedFailure);
    REQUIRE(boundPadded.has_value());
    auto submitted = device.submit(std::move(*command));
    REQUIRE(submitted.has_value());
    REQUIRE(device.wait(*submitted).has_value());

    REQUIRE(device.destroyPipeline(*pipeline).has_value());
    REQUIRE(device.destroyPipelineLayout(*pipelineLayout).has_value());
    REQUIRE(device.destroyShaderModule(*shader).has_value());
    REQUIRE(device.destroyBindingSet(*paddedSet).has_value());
    REQUIRE(device.destroyBindingSet(*unpaddedSet).has_value());
    REQUIRE(device.destroyBindingPool(*pool).has_value());
    REQUIRE(device.destroyBindingLayout(*layout).has_value());
    REQUIRE(device.destroyBuffer(*output).has_value());
    REQUIRE(device.destroyBuffer(*padded).has_value());
    REQUIRE(device.destroyBuffer(*unpadded).has_value());
    device.collectGarbage();
    requireCleanDiagnostics(device);
}
