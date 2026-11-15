#include "Rachis/Device.h"
#include "RhiTestSupport.h"

#include <catch2/catch_test_macros.hpp>

#include <Windows.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace
{

    constexpr std::uint32_t textureWidth = 4;
    constexpr std::uint32_t textureHeight = 4;
    constexpr std::uint32_t bytesPerPixel = 4;
    constexpr std::uint32_t rowPitch = 256;
    constexpr std::uint64_t transferSize = static_cast<std::uint64_t>(rowPitch) * textureHeight;

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

    [[nodiscard]] constexpr std::byte texelByte(std::uint8_t seed,
                                                std::uint32_t x,
                                                std::uint32_t y,
                                                std::uint32_t channel) noexcept
    {
        return static_cast<std::byte>((seed + x * 29U + y * 47U + channel * 61U) & 0xFFU);
    }

    void fillUpload(rachis::Device& device, rachis::BufferHandle buffer, std::uint8_t seed)
    {
        auto mapped = device.mapBuffer(buffer);
        REQUIRE(mapped.has_value());
        std::fill_n(mapped->data, mapped->size, std::byte{0});
        for (std::uint32_t y = 0; y < textureHeight; ++y)
        {
            for (std::uint32_t x = 0; x < textureWidth; ++x)
            {
                for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
                {
                    mapped->data[static_cast<std::uint64_t>(y) * rowPitch +
                                 static_cast<std::uint64_t>(x) * bytesPerPixel + channel] =
                        texelByte(seed, x, y, channel);
                }
            }
        }
        REQUIRE(device.unmapBuffer(buffer).has_value());
    }

    [[nodiscard]] rachis::TimelinePoint uploadTexture(rachis::Device& device,
                                                      rachis::BufferHandle upload,
                                                      rachis::TextureHandle texture)
    {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = texture,
                                        .stagesBefore = rachis::PipelineStage::all,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::common,
                                        .accessAfter = rachis::Access::copyDestination,
                                        .layoutBefore = rachis::TextureLayout::common,
                                        .layoutAfter = rachis::TextureLayout::copyDestination,
                                    })
                    .has_value());
        REQUIRE(device
                    .copyBufferToTexture(*command,
                                         rachis::BufferTextureCopy{
                                             .buffer = upload,
                                             .bytesPerRow = rowPitch,
                                             .rowsPerImage = textureHeight,
                                             .texture = texture,
                                             .extent = {.width = textureWidth, .height = textureHeight},
                                         })
                    .has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = texture,
                                        .stagesBefore = rachis::PipelineStage::copy,
                                        .stagesAfter = rachis::PipelineStage::all,
                                        .accessBefore = rachis::Access::copyDestination,
                                        .accessAfter = rachis::Access::common,
                                        .layoutBefore = rachis::TextureLayout::copyDestination,
                                        .layoutAfter = rachis::TextureLayout::common,
                                    })
                    .has_value());
        auto submitted = device.submit(std::move(*command));
        REQUIRE(submitted.has_value());
        return *submitted;
    }

    struct RecordedReadback final
    {
        rachis::CommandBuffer command;
        rachis::TextureHandle texture;
    };

    [[nodiscard]] RecordedReadback recordReadback(rachis::Device& device,
                                                  rachis::TextureHandle texture,
                                                  rachis::BufferHandle readback)
    {
        auto command = rachis::test::acquireCommandBuffer(device, rachis::QueueType::graphics);
        REQUIRE(command.has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = texture,
                                        .stagesBefore = rachis::PipelineStage::all,
                                        .stagesAfter = rachis::PipelineStage::copy,
                                        .accessBefore = rachis::Access::common,
                                        .accessAfter = rachis::Access::copySource,
                                        .layoutBefore = rachis::TextureLayout::common,
                                        .layoutAfter = rachis::TextureLayout::copySource,
                                    })
                    .has_value());
        REQUIRE(device
                    .copyTextureToBuffer(*command,
                                         rachis::BufferTextureCopy{
                                             .buffer = readback,
                                             .bytesPerRow = rowPitch,
                                             .rowsPerImage = textureHeight,
                                             .texture = texture,
                                             .extent = {.width = textureWidth, .height = textureHeight},
                                         })
                    .has_value());
        REQUIRE(device
                    .textureBarrier(*command,
                                    rachis::TextureBarrier{
                                        .texture = texture,
                                        .stagesBefore = rachis::PipelineStage::copy,
                                        .stagesAfter = rachis::PipelineStage::all,
                                        .accessBefore = rachis::Access::copySource,
                                        .accessAfter = rachis::Access::common,
                                        .layoutBefore = rachis::TextureLayout::copySource,
                                        .layoutAfter = rachis::TextureLayout::common,
                                    })
                    .has_value());
        return {.command = std::move(*command), .texture = texture};
    }

    void requireReadback(rachis::Device& device, rachis::BufferHandle readback, std::uint8_t seed)
    {
        auto mapped = device.mapBuffer(readback);
        REQUIRE(mapped.has_value());
        for (std::uint32_t y = 0; y < textureHeight; ++y)
        {
            for (std::uint32_t x = 0; x < textureWidth; ++x)
            {
                for (std::uint32_t channel = 0; channel < bytesPerPixel; ++channel)
                {
                    const auto offset = static_cast<std::uint64_t>(y) * rowPitch +
                                        static_cast<std::uint64_t>(x) * bytesPerPixel + channel;
                    CAPTURE(seed, x, y, channel, offset);
                    REQUIRE(mapped->data[offset] == texelByte(seed, x, y, channel));
                }
            }
        }
        REQUIRE(device.unmapBuffer(readback).has_value());
    }

    void requireCleanDiagnostics(const rachis::Device& device)
    {
        const auto diagnostics = device.diagnosticCounts();
        CHECK(diagnostics.warnings == 0);
        CHECK(diagnostics.errors == 0);
        CHECK(diagnostics.corruptions == 0);
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

} // namespace

TEST_CASE("shared texture imports reject INVALID_HANDLE_VALUE and unknown handle types",
          "[size:medium][rhi][interop][texture][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).sharedTextureHandles);

    const rachis::TextureDesc textureDesc{
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::renderTarget,
        .label = "Rejected shared interop texture",
    };
    constexpr auto invalidHandleValue = std::numeric_limits<std::uintptr_t>::max();
    const std::array rejected{
        rachis::SharedHandleView{.type = rachis::SharedHandleType::windowsNt, .value = invalidHandleValue},
        rachis::SharedHandleView{.type = static_cast<rachis::SharedHandleType>(2), .value = invalidHandleValue},
    };
    for (const auto handle : rejected)
    {
        CAPTURE(static_cast<unsigned int>(handle.type), handle.value);
        auto imported = device.importSharedTexture(handle, textureDesc);
        REQUIRE_FALSE(imported.has_value());
        CHECK(imported.error().code == rachis::ErrorCode::invalidArgument);
        CHECK(imported.error().nativeCode == 0);
    }
    requireCleanDiagnostics(device);
}

TEST_CASE("shared textures use explicit cross-device timeline handoffs",
          "[size:medium][rhi][interop][texture][timeline][warp]")
{
    auto producer = createWarpDevice();
    auto consumer = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(producer).sharedTextureHandles);
    REQUIRE(rachis::test::requireCapabilities(producer).sharedTimelineHandles);
    REQUIRE(rachis::test::requireCapabilities(consumer).sharedTextureHandles);
    REQUIRE(rachis::test::requireCapabilities(consumer).sharedTimelineHandles);

    const rachis::TextureDesc textureDesc{
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::renderTarget,
        .label = "Shared interop texture",
    };
    auto invalidManualDesc = textureDesc;
    invalidManualDesc.allocationFlags = rachis::ResourceAllocationFlag::manualResidency;
    CHECK_FALSE(producer.createSharedTexture(invalidManualDesc).has_value());
    auto invalidPriorityDesc = textureDesc;
    invalidPriorityDesc.residencyPriority = rachis::ResidencyPriority::high;
    CHECK_FALSE(producer.createSharedTexture(invalidPriorityDesc).has_value());
    auto ordinary = producer.createTexture(textureDesc);
    REQUIRE(ordinary.has_value());
    CHECK_FALSE(producer.exportSharedTexture(*ordinary).has_value());
    REQUIRE(producer.destroyTexture(*ordinary).has_value());
    CHECK_FALSE(consumer.importSharedTexture({}, textureDesc).has_value());
    CHECK_FALSE(consumer
                    .enqueueSharedTimelineWait(rachis::QueueType::graphics,
                                               rachis::SharedTimelinePointView{
                                                   .handle =
                                                       {
                                                           .type = rachis::SharedHandleType::windowsNt,
                                                           .value = 1,
                                                       },
                                               })
                    .has_value());

    auto producerUpload = producer.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Shared interop producer upload",
    });
    auto producerReadback = producer.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Shared interop producer readback",
    });
    auto consumerUpload = consumer.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Shared interop consumer upload",
    });
    auto consumerReadback = consumer.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Shared interop consumer readback",
    });
    auto sharedTexture = producer.createSharedTexture(textureDesc);
    REQUIRE(producerUpload.has_value());
    REQUIRE(producerReadback.has_value());
    REQUIRE(consumerUpload.has_value());
    REQUIRE(consumerReadback.has_value());
    REQUIRE(sharedTexture.has_value());

    auto textureHandle = producer.exportSharedTexture(*sharedTexture);
    REQUIRE(textureHandle.has_value());
    auto mismatchDesc = textureDesc;
    mismatchDesc.width += 1;
    CHECK_FALSE(consumer.importSharedTexture(textureHandle->view(), mismatchDesc).has_value());
    auto incompatibleUsageDesc = textureDesc;
    incompatibleUsageDesc.usage = incompatibleUsageDesc.usage | rachis::TextureUsage::storage;
    CHECK_FALSE(consumer.importSharedTexture(textureHandle->view(), incompatibleUsageDesc).has_value());
    auto explicitClearDesc = textureDesc;
    explicitClearDesc.optimizedClearValue = {
        .kind = rachis::OptimizedClearValueKind::color,
        .color = {0.25F, 0.5F, 0.75F, 1.0F},
    };
    CHECK_FALSE(consumer.importSharedTexture(textureHandle->view(), explicitClearDesc).has_value());
    auto importedTexture = consumer.importSharedTexture(textureHandle->view(), textureDesc);
    REQUIRE(importedTexture.has_value());
    textureHandle->reset();

    constexpr std::uint8_t producerSeed = 17;
    fillUpload(producer, *producerUpload, producerSeed);
    const auto producerPoint = uploadTexture(producer, *producerUpload, *sharedTexture);
    auto sharedProducerPoint = producer.exportTimelinePoint(producerPoint);
    REQUIRE(sharedProducerPoint.has_value());
    auto consumerReady = consumer.enqueueSharedTimelineWait(rachis::QueueType::graphics, sharedProducerPoint->view());
    REQUIRE(consumerReady.has_value());
    sharedProducerPoint->handle.reset();

    auto consumerRead = recordReadback(consumer, *importedTexture, *consumerReadback);
    auto consumerReadPoint = consumer.submit(std::move(consumerRead.command));
    REQUIRE(consumerReadPoint.has_value());
    REQUIRE(consumer.wait(*consumerReadPoint).has_value());
    requireReadback(consumer, *consumerReadback, producerSeed);

    constexpr std::uint8_t consumerSeed = 193;
    fillUpload(consumer, *consumerUpload, consumerSeed);
    const auto consumerPoint = uploadTexture(consumer, *consumerUpload, *importedTexture);
    auto sharedConsumerPoint = consumer.exportTimelinePoint(consumerPoint);
    REQUIRE(sharedConsumerPoint.has_value());
    auto producerReady = producer.enqueueSharedTimelineWait(rachis::QueueType::graphics, sharedConsumerPoint->view());
    REQUIRE(producerReady.has_value());
    sharedConsumerPoint->handle.reset();

    auto producerRead = recordReadback(producer, *sharedTexture, *producerReadback);
    REQUIRE(producer.destroyTexture(*sharedTexture).has_value());
    auto producerReadPoint = producer.submit(std::move(producerRead.command));
    REQUIRE(producerReadPoint.has_value());
    REQUIRE(consumer.destroyTexture(*importedTexture).has_value());
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

TEST_CASE("shared texture exports are independent NT handles", "[size:medium][rhi][interop][texture][warp]")
{
    auto device = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(device).sharedTextureHandles);

    auto sharedTexture = device.createSharedTexture({
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination,
        .label = "Twice-exported shared texture",
    });
    REQUIRE(sharedTexture.has_value());
    auto first = device.exportSharedTexture(*sharedTexture);
    auto second = device.exportSharedTexture(*sharedTexture);
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());
    requireIndependentExports(*first, *second);

    REQUIRE(device.destroyTexture(*sharedTexture).has_value());
    device.collectGarbage();
    requireCleanDiagnostics(device);
}

TEST_CASE("shared texture re-exports name their texture after an earlier export closes",
          "[size:medium][rhi][interop][texture][timeline][warp]")
{
    auto producer = createWarpDevice();
    auto firstConsumer = createWarpDevice();
    auto secondConsumer = createWarpDevice();
    REQUIRE(rachis::test::requireCapabilities(producer).sharedTextureHandles);
    REQUIRE(rachis::test::requireCapabilities(producer).sharedTimelineHandles);

    const rachis::TextureDesc textureDesc{
        .width = textureWidth,
        .height = textureHeight,
        .format = rachis::Format::rgba8Unorm,
        .usage = rachis::TextureUsage::copySource | rachis::TextureUsage::copyDestination |
                 rachis::TextureUsage::renderTarget,
        .label = "Re-exported shared texture",
    };
    auto producerUpload = producer.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Texture re-export producer upload",
    });
    auto producerReadback = producer.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Texture re-export producer readback",
    });
    auto consumerUpload = secondConsumer.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copySource,
        .memoryType = rachis::MemoryType::upload,
        .label = "Texture re-export consumer upload",
    });
    auto consumerReadback = secondConsumer.createBuffer({
        .size = transferSize,
        .usage = rachis::BufferUsage::copyDestination,
        .memoryType = rachis::MemoryType::readback,
        .label = "Texture re-export consumer readback",
    });
    auto exportedTexture = producer.createSharedTexture(textureDesc);
    auto otherTexture = producer.createSharedTexture(textureDesc);
    REQUIRE(producerUpload.has_value());
    REQUIRE(producerReadback.has_value());
    REQUIRE(consumerUpload.has_value());
    REQUIRE(consumerReadback.has_value());
    REQUIRE(exportedTexture.has_value());
    REQUIRE(otherTexture.has_value());

    constexpr std::uint8_t exportedSeed = 41;
    constexpr std::uint8_t otherSeed = 97;
    constexpr std::uint8_t consumerSeed = 211;
    fillUpload(producer, *producerUpload, exportedSeed);
    REQUIRE(producer.wait(uploadTexture(producer, *producerUpload, *exportedTexture)).has_value());
    auto firstExport = producer.exportSharedTexture(*exportedTexture);
    REQUIRE(firstExport.has_value());
    auto firstImport = firstConsumer.importSharedTexture(firstExport->view(), textureDesc);
    REQUIRE(firstImport.has_value());
    fillUpload(producer, *producerUpload, otherSeed);
    const auto producerPoint = uploadTexture(producer, *producerUpload, *otherTexture);
    REQUIRE(producer.wait(producerPoint).has_value());
    firstExport->reset();

    auto otherExport = producer.exportSharedTexture(*otherTexture);
    REQUIRE(otherExport.has_value());
    auto reexport = producer.exportSharedTexture(*exportedTexture);
    REQUIRE(reexport.has_value());
    CHECK(reexport->view().value != otherExport->view().value);
    auto reimported = secondConsumer.importSharedTexture(reexport->view(), textureDesc);
    if (!reimported)
    {
        INFO(reimported.error().message);
    }
    REQUIRE(reimported.has_value());
    reexport->reset();
    otherExport->reset();

    handOff(producer, producerPoint, secondConsumer, rachis::QueueType::graphics);
    auto consumerRead = recordReadback(secondConsumer, *reimported, *consumerReadback);
    auto consumerReadPoint = secondConsumer.submit(std::move(consumerRead.command));
    REQUIRE(consumerReadPoint.has_value());
    REQUIRE(secondConsumer.wait(*consumerReadPoint).has_value());
    requireReadback(secondConsumer, *consumerReadback, exportedSeed);

    fillUpload(secondConsumer, *consumerUpload, consumerSeed);
    const auto consumerPoint = uploadTexture(secondConsumer, *consumerUpload, *reimported);
    handOff(secondConsumer, consumerPoint, producer, rachis::QueueType::graphics);
    auto otherRead = recordReadback(producer, *otherTexture, *producerReadback);
    auto otherReadPoint = producer.submit(std::move(otherRead.command));
    REQUIRE(otherReadPoint.has_value());
    REQUIRE(producer.wait(*otherReadPoint).has_value());
    requireReadback(producer, *producerReadback, otherSeed);
    auto exportedRead = recordReadback(producer, *exportedTexture, *producerReadback);
    auto exportedReadPoint = producer.submit(std::move(exportedRead.command));
    REQUIRE(exportedReadPoint.has_value());
    REQUIRE(producer.wait(*exportedReadPoint).has_value());
    requireReadback(producer, *producerReadback, consumerSeed);

    REQUIRE(secondConsumer.destroyTexture(*reimported).has_value());
    REQUIRE(firstConsumer.destroyTexture(*firstImport).has_value());
    REQUIRE(producer.destroyTexture(*exportedTexture).has_value());
    REQUIRE(producer.destroyTexture(*otherTexture).has_value());
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
