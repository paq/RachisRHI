#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>

TEST_CASE("placed resources share explicit heap residency and exact queue lifetime",
          "[size:medium][rhi][memory][residency][heap][warp]")
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
    REQUIRE(rachis::test::requireCapabilities(device).manualResidency);
    REQUIRE(rachis::test::requireCapabilities(device).residencyPriorities);

    constexpr std::uint64_t bufferSize = 4'097;
    const rachis::BufferDesc placedBufferDesc{
        .size = bufferSize,
        .usage = rachis::BufferUsage::copySource | rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::deviceLocal,
        .label = "Heap-resident placed buffer",
    };
    const auto bufferRequirements = device.queryBufferMemoryRequirements(placedBufferDesc);
    REQUIRE(bufferRequirements.has_value());

    const auto automaticHeap = device.createResourceHeap({
        .size = bufferRequirements->size,
        .alignment = bufferRequirements->alignment,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::buffer,
    });
    REQUIRE(automaticHeap.has_value());
    CHECK(device.queryResidency(*automaticHeap) == rachis::ResidencyState::automatic);
    const std::array automaticHeaps{*automaticHeap};
    const auto automaticEviction = device.evict({.heaps = automaticHeaps});
    REQUIRE_FALSE(automaticEviction.has_value());
    CHECK(automaticEviction.error().code == rachis::ErrorCode::invalidArgument);
    REQUIRE(device.destroyResourceHeap(*automaticHeap).has_value());

    const auto invalidUploadHeap = device.createResourceHeap({
        .size = bufferRequirements->size,
        .alignment = bufferRequirements->alignment,
        .memoryType = rachis::MemoryType::upload,
        .resourceClass = rachis::ResourceHeapClass::buffer,
        .allocationFlags = rachis::ResourceAllocationFlag::manualResidency,
    });
    REQUIRE_FALSE(invalidUploadHeap.has_value());
    CHECK(invalidUploadHeap.error().code == rachis::ErrorCode::invalidArgument);

    const auto bufferHeap = device.createResourceHeap({
        .size = bufferRequirements->size,
        .alignment = bufferRequirements->alignment,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::buffer,
        .residencyPriority = rachis::ResidencyPriority::high,
        .allocationFlags = rachis::ResourceAllocationFlag::manualResidency,
        .label = "Manual residency buffer heap",
    });
    REQUIRE(bufferHeap.has_value());
    const auto placedBuffer = device.createPlacedBuffer({.heap = *bufferHeap}, placedBufferDesc);
    REQUIRE(placedBuffer.has_value());
    CHECK(device.queryResidency(*bufferHeap) == rachis::ResidencyState::resident);
    CHECK(device.queryResidency(*placedBuffer) == rachis::ResidencyState::resident);

    const rachis::TextureDesc placedTextureDesc{
        .width = 16,
        .height = 16,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copyDestination,
        .label = "Heap-resident placed texture",
    };
    const auto textureRequirements = device.queryTextureMemoryRequirements(placedTextureDesc);
    REQUIRE(textureRequirements.has_value());
    const auto textureHeap = device.createResourceHeap({
        .size = textureRequirements->size,
        .alignment = textureRequirements->alignment,
        .memoryType = rachis::MemoryType::deviceLocal,
        .resourceClass = rachis::ResourceHeapClass::nonAttachmentTexture,
        .allocationFlags = rachis::ResourceAllocationFlag::manualResidency,
        .label = "Manual residency texture heap",
    });
    REQUIRE(textureHeap.has_value());
    const auto placedTexture = device.createPlacedTexture({.heap = *textureHeap}, placedTextureDesc);
    REQUIRE(placedTexture.has_value());
    CHECK(device.queryResidency(*textureHeap) == rachis::ResidencyState::resident);
    CHECK(device.queryResidency(*placedTexture) == rachis::ResidencyState::resident);

    const std::array heaps{*bufferHeap, *textureHeap};
    const std::array buffers{*placedBuffer};
    const rachis::ResidencySet heapSet{.heaps = heaps};
    const rachis::ResidencySet duplicatePhysicalOwner{.heaps = heaps, .buffers = buffers};
    const auto duplicateEviction = device.evict(duplicatePhysicalOwner);
    REQUIRE_FALSE(duplicateEviction.has_value());
    CHECK(duplicateEviction.error().code == rachis::ErrorCode::invalidArgument);

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
        expected[index] = static_cast<std::byte>((index * 31U + 11U) & 0xFFU);
    }
    REQUIRE(device.writeBuffer(*upload, 0, expected).has_value());

    auto initialize = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
    REQUIRE(initialize.has_value());
    REQUIRE(device
                .bufferBarrier(*initialize,
                               {
                                   .buffer = *placedBuffer,
                                   .stagesBefore = rachis::PipelineStage::all,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::common,
                                   .accessAfter = rachis::Access::copyDestination,
                               })
                .has_value());
    REQUIRE(device.copyBuffer(*initialize, {.source = *upload, .destination = *placedBuffer, .size = bufferSize})
                .has_value());
    REQUIRE(device
                .bufferBarrier(*initialize,
                               {
                                   .buffer = *placedBuffer,
                                   .stagesBefore = rachis::PipelineStage::copy,
                                   .stagesAfter = rachis::PipelineStage::copy,
                                   .accessBefore = rachis::Access::copyDestination,
                                   .accessAfter = rachis::Access::copySource,
                               })
                .has_value());
    REQUIRE(device
                .textureBarrier(*initialize,
                                {
                                    .texture = *placedTexture,
                                    .stagesBefore = rachis::PipelineStage::none,
                                    .stagesAfter = rachis::PipelineStage::copy,
                                    .accessBefore = rachis::Access::none,
                                    .accessAfter = rachis::Access::copyDestination,
                                    .layoutBefore = rachis::TextureLayout::undefined,
                                    .layoutAfter = rachis::TextureLayout::copyDestination,
                                })
                .has_value());
    const auto inUseEviction = device.evict(heapSet);
    REQUIRE_FALSE(inUseEviction.has_value());
    CHECK(inUseEviction.error().code == rachis::ErrorCode::invalidState);
    const auto initialized = device.submit(std::move(*initialize));
    REQUIRE(initialized.has_value());
    REQUIRE(device.wait(*initialized).has_value());
    device.collectGarbage();

    REQUIRE(device.evict(heapSet).has_value());
    CHECK(device.queryResidency(*bufferHeap) == rachis::ResidencyState::evicted);
    CHECK(device.queryResidency(*placedBuffer) == rachis::ResidencyState::evicted);
    CHECK(device.queryResidency(*textureHeap) == rachis::ResidencyState::evicted);
    CHECK(device.queryResidency(*placedTexture) == rachis::ResidencyState::evicted);

    auto rejected = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(rejected.has_value());
    const auto useWhileEvicted =
        device.copyBuffer(*rejected, {.source = *placedBuffer, .destination = *readback, .size = bufferSize});
    REQUIRE_FALSE(useWhileEvicted.has_value());
    CHECK(useWhileEvicted.error().code == rachis::ErrorCode::invalidState);

    const auto ready = device.enqueueMakeResident(rachis::QueueType::copy, heapSet);
    REQUIRE(ready.has_value());
    auto download = rachis::test::acquireCommandBuffer(device, rachis::QueueType::copy);
    REQUIRE(download.has_value());
    REQUIRE(device.copyBuffer(*download, {.source = *placedBuffer, .destination = *readback, .size = bufferSize})
                .has_value());

    REQUIRE(device.destroyTexture(*placedTexture).has_value());
    REQUIRE(device.destroyResourceHeap(*textureHeap).has_value());
    REQUIRE(device.destroyBuffer(*placedBuffer).has_value());
    REQUIRE(device.destroyResourceHeap(*bufferHeap).has_value());
    const auto downloaded = device.submit(std::move(*download));
    REQUIRE(downloaded.has_value());
    REQUIRE(device.wait(*downloaded).has_value());
    device.collectGarbage();

    std::array<std::byte, bufferSize> actual{};
    REQUIRE(device.readBuffer(*readback, 0, actual).has_value());
    CHECK(actual == expected);
    REQUIRE(device.destroyBuffer(*upload).has_value());
    REQUIRE(device.destroyBuffer(*readback).has_value());
    const auto diagnostics = device.diagnosticCounts();
    CHECK(diagnostics.errors == 0);
    CHECK(diagnostics.corruptions == 0);
}
